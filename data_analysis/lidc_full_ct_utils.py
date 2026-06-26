"""Utilities for complete CT loading, lung segmentation, and nodule proposals."""

from __future__ import print_function

import math
from pathlib import Path

import numpy as np

from LUNA_process import segment_lung_parenchyma


def require_sitk():
    try:
        import SimpleITK as sitk

        return sitk
    except Exception as exc:
        raise ImportError("SimpleITK is required for full CT series inference.") from exc


def read_dicom_series(directory, series_uid=None):
    sitk = require_sitk()
    directory = Path(directory)
    series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(directory)) or [])
    if not series_ids:
        raise ValueError("No DICOM series found in {}".format(directory))
    if series_uid is None:
        series_uid = max(
            series_ids,
            key=lambda uid: len(
                sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(directory), uid)
            ),
        )
    file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(directory), str(series_uid)
    )
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    return image, str(series_uid), len(file_names)


def read_ct_image(input_path, series_uid=None):
    sitk = require_sitk()
    input_path = Path(input_path)
    if input_path.is_dir():
        image, resolved_uid, file_count = read_dicom_series(input_path, series_uid)
        source_type = "dicom_series"
    elif input_path.suffix.lower() in (".mhd", ".mha", ".nii", ".gz", ".nrrd"):
        image = sitk.ReadImage(str(input_path))
        resolved_uid = input_path.stem
        file_count = 1
        source_type = "volume_file"
    else:
        raise ValueError(
            "Input must be a DICOM directory or a supported volume file: {}".format(
                input_path
            )
        )
    image = sitk.Cast(image, sitk.sitkFloat32)
    return image, {
        "input_path": str(input_path),
        "source_type": source_type,
        "series_uid": resolved_uid,
        "source_file_count": file_count,
        "original_size_xyz": list(image.GetSize()),
        "original_spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def resample_image(image, spacing_xyz=(1.0, 1.0, 1.0), default_hu=-1000.0):
    sitk = require_sitk()
    original_spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    original_size = np.asarray(image.GetSize(), dtype=np.int64)
    target_spacing = np.asarray(spacing_xyz, dtype=np.float64)
    target_size = np.maximum(
        np.rint(original_size * original_spacing / target_spacing).astype(np.int64),
        1,
    )
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(tuple(float(value) for value in target_spacing))
    resampler.SetSize([int(value) for value in target_size])
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetDefaultPixelValue(float(default_hu))
    output = resampler.Execute(image)
    return output


def sitk_to_numpy(image):
    sitk = require_sitk()
    return sitk.GetArrayFromImage(image).astype(np.float32)


def numpy_index_to_world(image, index_zyx):
    z, y, x = [float(value) for value in index_zyx]
    return image.TransformContinuousIndexToPhysicalPoint((x, y, z))


def world_to_numpy_index(image, world_xyz):
    x, y, z = image.TransformPhysicalPointToContinuousIndex(
        tuple(float(value) for value in world_xyz)
    )
    return np.asarray([z, y, x], dtype=np.float64)


def fixed_crop_or_pad(volume, center_zyx, shape, pad_value=0.0):
    shape = tuple(int(value) for value in shape)
    center = [int(round(float(value))) for value in center_zyx]
    output = np.full(shape, float(pad_value), dtype=np.float32)
    source_slices = []
    target_slices = []
    for axis, size in enumerate(shape):
        start = center[axis] - size // 2
        stop = start + size
        source_start = max(start, 0)
        source_stop = min(stop, volume.shape[axis])
        target_start = source_start - start
        target_stop = target_start + max(source_stop - source_start, 0)
        source_slices.append(slice(source_start, source_stop))
        target_slices.append(slice(target_start, target_stop))
    if all(item.start < item.stop for item in source_slices):
        output[tuple(target_slices)] = volume[tuple(source_slices)]
    return output


def normalise_hu(volume, hu_min=-1000.0, hu_max=400.0):
    clipped = np.clip(np.asarray(volume, dtype=np.float32), hu_min, hu_max)
    return (clipped - float(hu_min)) / float(hu_max - hu_min)


def segment_lungs_robust(volume_hu, threshold_hu=-320.0):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise ImportError("scipy is required for lung segmentation.") from exc

    volume_hu = np.asarray(volume_hu, dtype=np.float32)
    primary = segment_lung_parenchyma(
        volume_hu, threshold_hu=float(threshold_hu)
    ).astype(bool)
    fraction = float(primary.mean())
    if 0.01 <= fraction <= 0.65:
        return ndimage.binary_dilation(primary, iterations=2).astype(np.uint8)

    air = volume_hu < float(threshold_hu)
    fallback = np.zeros_like(air, dtype=bool)
    for z in range(air.shape[0]):
        labels, count = ndimage.label(air[z])
        if count == 0:
            continue
        border_labels = set(np.unique(labels[0, :]).tolist())
        border_labels.update(np.unique(labels[-1, :]).tolist())
        border_labels.update(np.unique(labels[:, 0]).tolist())
        border_labels.update(np.unique(labels[:, -1]).tolist())
        border_labels.discard(0)
        internal = air[z] & ~np.isin(labels, list(border_labels))
        component_labels, component_count = ndimage.label(internal)
        if component_count:
            sizes = ndimage.sum(
                internal,
                component_labels,
                index=list(range(1, component_count + 1)),
            )
            keep = [
                index + 1
                for index in np.argsort(sizes)[::-1][:2]
                if sizes[index] >= 100
            ]
            internal = np.isin(component_labels, keep) if keep else internal
        fallback[z] = ndimage.binary_fill_holes(internal)

    structure = ndimage.generate_binary_structure(3, 1)
    fallback = ndimage.binary_closing(fallback, structure=structure, iterations=2)
    fallback = ndimage.binary_opening(fallback, structure=structure, iterations=1)
    labels, count = ndimage.label(fallback)
    if count:
        sizes = ndimage.sum(fallback, labels, index=list(range(1, count + 1)))
        keep = [
            index + 1
            for index in np.argsort(sizes)[::-1][:2]
            if sizes[index] >= 10000
        ]
        if keep:
            fallback = np.isin(labels, keep)
    fallback = ndimage.binary_dilation(fallback, iterations=3)
    return fallback.astype(np.uint8)


def downsample_for_proposals(volume_hu, lung_mask, factor=2):
    try:
        from scipy.ndimage import zoom
    except Exception as exc:
        raise ImportError("scipy is required for full CT candidate proposals.") from exc
    scale = 1.0 / float(factor)
    volume_small = zoom(volume_hu, (scale, scale, scale), order=1, prefilter=False)
    mask_small = zoom(
        lung_mask.astype(np.float32),
        (scale, scale, scale),
        order=0,
        prefilter=False,
    ) > 0.5
    return volume_small.astype(np.float32), mask_small


def non_maximum_suppression(candidates, min_distance_voxels):
    distance = max(float(min_distance_voxels), 1e-6)
    selected = []
    spatial_grid = {}
    for candidate in sorted(candidates, key=lambda row: row["proposal_score"], reverse=True):
        point = np.asarray(candidate["center_zyx"], dtype=np.float64)
        cell = tuple(np.floor(point / distance).astype(np.int64).tolist())
        nearby_indices = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nearby_indices.extend(
                        spatial_grid.get(
                            (cell[0] + dz, cell[1] + dy, cell[2] + dx),
                            (),
                        )
                    )
        if any(
            np.linalg.norm(
                point
                - np.asarray(
                    selected[index]["center_zyx"],
                    dtype=np.float64,
                )
            )
            < distance
            for index in nearby_indices
        ):
            continue
        selected_index = len(selected)
        selected.append(candidate)
        spatial_grid.setdefault(cell, []).append(selected_index)
    return selected


def slice_component_candidates(
    small_hu,
    small_mask,
    proposal_downsample,
):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise ImportError("scipy is required for component proposals.") from exc

    candidates = []
    for z in range(small_hu.shape[0]):
        for threshold_hu in (-600.0, -350.0, -150.0):
            binary = small_mask[z] & (small_hu[z] >= threshold_hu) & (small_hu[z] <= 450.0)
            binary = ndimage.binary_opening(binary, iterations=1)
            labels, count = ndimage.label(binary)
            if not count:
                continue
            objects = ndimage.find_objects(labels)
            for label_id, bounds in enumerate(objects, start=1):
                if bounds is None:
                    continue
                height = bounds[0].stop - bounds[0].start
                width = bounds[1].stop - bounds[1].start
                if height > 24 or width > 24 or min(height, width) < 1:
                    continue
                component = labels[bounds] == label_id
                area = int(component.sum())
                if area < 2 or area > 180:
                    continue
                aspect = max(height, width) / float(max(min(height, width), 1))
                if aspect > 3.0:
                    continue
                coordinates = np.argwhere(component)
                center_yx = coordinates.mean(axis=0)
                y = float(bounds[0].start + center_yx[0])
                x = float(bounds[1].start + center_yx[1])
                covariance = (
                    np.cov(coordinates.T)
                    if coordinates.shape[0] >= 3
                    else np.eye(2, dtype=np.float64)
                )
                eigenvalues = np.linalg.eigvalsh(covariance)
                roundness = float(
                    np.clip(
                        eigenvalues[0] / max(eigenvalues[-1], 1e-6),
                        0.0,
                        1.0,
                    )
                )
                equivalent_diameter = 2.0 * math.sqrt(area / math.pi)
                size_score = float(
                    np.exp(-((equivalent_diameter - 4.0) / 5.0) ** 2)
                )
                values = small_hu[z][bounds][component]
                mean_hu = float(values.mean())
                density_score = float(np.clip((mean_hu + 700.0) / 900.0, 0.0, 1.0))
                score = 0.45 * roundness + 0.30 * size_score + 0.25 * density_score
                candidates.append({
                    "center_zyx": (
                        np.asarray([z, y, x], dtype=np.float64)
                        * int(proposal_downsample)
                    ).tolist(),
                    "proposal_score": float(score),
                    "proposal_source": "axial_component",
                    "log_response": "",
                    "log_sigma_downsampled": "",
                    "estimated_diameter_mm": float(
                        equivalent_diameter * proposal_downsample
                    ),
                    "local_mean_hu": mean_hu,
                    "component_threshold_hu": threshold_hu,
                })
    return candidates


def small_nodule_intensity_candidates(
    volume_hu,
    lung_mask,
    max_candidates=2000,
):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise ImportError("scipy is required for small-nodule proposals.") from exc

    smoothed = ndimage.gaussian_filter(
        np.asarray(volume_hu, dtype=np.float32),
        sigma=0.7,
        mode="nearest",
    )
    valid = (
        np.asarray(lung_mask, dtype=bool)
        & (smoothed >= -750.0)
        & (smoothed <= 450.0)
    )
    local_maximum = smoothed == ndimage.maximum_filter(
        smoothed,
        size=3,
        mode="nearest",
    )
    locations = np.argwhere(valid & local_maximum)
    local_contrast = smoothed - ndimage.uniform_filter(
        smoothed,
        size=9,
        mode="nearest",
    )
    contrast_values = local_contrast[tuple(locations.T)]
    order = np.argsort(contrast_values)[::-1][: int(max_candidates)]
    if len(order):
        selected_contrast = contrast_values[order]
        contrast_low = float(np.percentile(selected_contrast, 5.0))
        contrast_high = float(np.percentile(selected_contrast, 95.0))
    else:
        contrast_low = 0.0
        contrast_high = 1.0
    candidates = []
    for location_index in order:
        z, y, x = locations[location_index]
        local_hu = float(smoothed[z, y, x])
        contrast_score = float(np.clip(
            (
                float(contrast_values[location_index]) - contrast_low
            )
            / max(contrast_high - contrast_low, 1e-6),
            0.0,
            1.0,
        ))
        candidates.append({
            "center_zyx": [float(z), float(y), float(x)],
            "proposal_score": float(0.5 + 0.49 * contrast_score),
            "proposal_source": "full_resolution_intensity_maximum",
            "log_response": "",
            "log_sigma_downsampled": "",
            "estimated_diameter_mm": 3.0,
            "local_mean_hu": local_hu,
            "component_threshold_hu": "",
        })
    return candidates, len(locations)


def propose_nodule_candidates(
    volume_hu,
    lung_mask=None,
    proposal_downsample=2,
    response_percentile=99.7,
    min_distance_mm=3.0,
    max_candidates=128,
    include_small_nodule_maxima=True,
):
    try:
        from scipy import ndimage
    except Exception as exc:
        raise ImportError("scipy is required for LoG nodule proposals.") from exc

    volume_hu = np.asarray(volume_hu, dtype=np.float32)
    if lung_mask is None:
        lung_mask = segment_lungs_robust(volume_hu).astype(bool)
    else:
        lung_mask = np.asarray(lung_mask, dtype=bool)
    if int(lung_mask.sum()) == 0:
        raise RuntimeError("Lung segmentation is empty.")

    small_hu, small_mask = downsample_for_proposals(
        volume_hu,
        lung_mask,
        factor=int(proposal_downsample),
    )
    soft_tissue = np.clip((small_hu + 700.0) / 900.0, 0.0, 1.0)
    soft_tissue *= small_mask.astype(np.float32)
    valid = small_mask & (small_hu > -750.0) & (small_hu < 450.0)
    if int(valid.sum()) < 100:
        raise RuntimeError("Too few valid intrapulmonary voxels for candidate generation.")

    best_response = np.zeros_like(soft_tissue, dtype=np.float32)
    best_sigma = np.zeros_like(soft_tissue, dtype=np.float32)
    for sigma in (0.8, 1.2, 1.8, 2.5, 3.2):
        response = -float(sigma) ** 2 * ndimage.gaussian_laplace(
            soft_tissue,
            sigma=float(sigma),
            mode="nearest",
        )
        update = response > best_response
        best_response[update] = response[update]
        best_sigma[update] = float(sigma)

    threshold = float(np.percentile(best_response[valid], float(response_percentile)))
    local_maximum = best_response == ndimage.maximum_filter(best_response, size=5)
    locations = np.argwhere(valid & local_maximum & (best_response >= threshold))
    raw_candidates = []
    response_scale = max(float(best_response[valid].max()), 1e-6)
    for location in locations:
        z, y, x = [int(value) for value in location]
        source_center = np.asarray([z, y, x], dtype=np.float64) * int(proposal_downsample)
        radius_voxels = max(int(round(best_sigma[z, y, x] * 2.0)), 1)
        z0, z1 = max(z - radius_voxels, 0), min(z + radius_voxels + 1, small_hu.shape[0])
        y0, y1 = max(y - radius_voxels, 0), min(y + radius_voxels + 1, small_hu.shape[1])
        x0, x1 = max(x - radius_voxels, 0), min(x + radius_voxels + 1, small_hu.shape[2])
        local_hu = small_hu[z0:z1, y0:y1, x0:x1]
        mean_hu = float(local_hu.mean()) if local_hu.size else float(small_hu[z, y, x])
        score = float(np.clip(best_response[z, y, x] / response_scale, 0.0, 1.0))
        raw_candidates.append({
            "center_zyx": source_center.tolist(),
            "proposal_score": score,
            "proposal_source": "3d_log",
            "log_response": float(best_response[z, y, x]),
            "log_sigma_downsampled": float(best_sigma[z, y, x]),
            "estimated_diameter_mm": float(
                best_sigma[z, y, x] * math.sqrt(3.0) * 2.0 * proposal_downsample
            ),
            "local_mean_hu": mean_hu,
        })

    raw_candidates.extend(
        slice_component_candidates(
            small_hu,
            small_mask,
            proposal_downsample=int(proposal_downsample),
        )
    )
    base_raw_candidate_count = len(raw_candidates)
    base_candidate_limit = min(int(max_candidates), 1000)
    base_candidates = non_maximum_suppression(
        raw_candidates,
        min_distance_voxels=float(min_distance_mm),
    )[:base_candidate_limit]
    small_candidate_count = 0
    selected_small_candidate_count = 0
    if include_small_nodule_maxima:
        small_limit = max(int(max_candidates) - base_candidate_limit, 0)
        small_candidates, small_candidate_count = small_nodule_intensity_candidates(
            volume_hu,
            lung_mask,
            max_candidates=small_limit,
        )
        selected_small_candidate_count = len(small_candidates)
    else:
        small_candidates = []
    selected = non_maximum_suppression(
        base_candidates + small_candidates,
        min_distance_voxels=float(min_distance_mm),
    )
    return selected[: int(max_candidates)], {
        "lung_voxel_count": int(lung_mask.sum()),
        "lung_fraction": float(lung_mask.mean()),
        "proposal_threshold": threshold,
        "raw_candidate_count": base_raw_candidate_count + small_candidate_count,
        "base_raw_candidate_count": base_raw_candidate_count,
        "small_nodule_candidate_count": small_candidate_count,
        "selected_small_nodule_candidate_count": selected_small_candidate_count,
        "selected_candidate_count": min(len(selected), int(max_candidates)),
        "proposal_downsample": int(proposal_downsample),
        "response_percentile": float(response_percentile),
    }


def match_candidates_to_annotations(candidates, annotations, scan_image):
    matched_annotations = set()
    rows = []
    for candidate_index, candidate in enumerate(candidates):
        world = np.asarray(
            numpy_index_to_world(scan_image, candidate["center_zyx"]),
            dtype=np.float64,
        )
        best_index = None
        best_distance = None
        for annotation_index, annotation in enumerate(annotations):
            center = np.asarray(annotation["world_xyz"], dtype=np.float64)
            distance = float(np.linalg.norm(world - center))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = annotation_index
        matched = False
        if best_index is not None:
            radius = max(float(annotations[best_index].get("diameter_mm", 0.0)) / 2.0, 3.0)
            matched = best_distance <= radius
            if matched:
                matched_annotations.add(best_index)
        out = dict(candidate)
        out.update({
            "candidate_index": candidate_index,
            "world_x": float(world[0]),
            "world_y": float(world[1]),
            "world_z": float(world[2]),
            "nearest_annotation_index": "" if best_index is None else best_index,
            "nearest_annotation_distance_mm": "" if best_distance is None else best_distance,
            "matched_annotation": matched,
        })
        rows.append(out)
    return rows, matched_annotations
