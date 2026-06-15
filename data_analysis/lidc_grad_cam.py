"""
Grad-CAM and Grad-CAM++ visualisation helpers for LIDC Lightning models.
"""

from __future__ import print_function

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lidc_lightning_utils import write_csv


def safe_name(value):
    text = str(value or "sample")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:140]


def target_layer_for_model(model):
    backbone = getattr(model, "model", model)
    if hasattr(backbone, "model"):
        backbone = backbone.model
    if hasattr(backbone, "layer4"):
        return backbone.layer4[-1]
    if hasattr(backbone, "features") and hasattr(backbone.features, "denseblock4"):
        return backbone.features.denseblock4
    if hasattr(backbone, "layers"):
        return backbone.layers[-1]
    if hasattr(backbone, "enc4"):
        return backbone.enc4
    if hasattr(backbone, "axial"):
        return backbone.axial[5]
    raise ValueError("Could not infer a Grad-CAM target layer for {}.".format(backbone.__class__.__name__))


class CamExtractor(object):
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def close(self):
        self.forward_handle.remove()

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._store_gradient)

    def _store_gradient(self, gradient):
        self.gradients = gradient

    def compute(self, image, class_id):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        score = logits[:, int(class_id)].sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        return logits.detach(), self.activations.detach(), self.gradients.detach()


def normalise_cam(cam):
    cam = cam - cam.min()
    maximum = cam.max()
    if maximum > 0:
        cam = cam / maximum
    return cam


def grad_cam_from_tensors(activations, gradients):
    spatial_dims = tuple(range(2, gradients.dim()))
    weights = gradients.mean(dim=spatial_dims, keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    return normalise_cam(cam)


def grad_cam_plus_plus_from_tensors(activations, gradients):
    spatial_dims = tuple(range(2, gradients.dim()))
    grad_2 = gradients.pow(2)
    grad_3 = gradients.pow(3)
    summed_activations = activations.sum(dim=spatial_dims, keepdim=True)
    denominator = 2.0 * grad_2 + summed_activations * grad_3
    denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator))
    alpha = grad_2 / denominator
    weights = (alpha * torch.relu(gradients)).sum(dim=spatial_dims, keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    return normalise_cam(cam)


def resize_cam(cam, image):
    spatial_size = tuple(image.shape[2:])
    if cam.dim() == 4 and len(spatial_size) == 3:
        spatial_size = spatial_size[-2:]
    mode = "trilinear" if len(spatial_size) == 3 else "bilinear"
    return F.interpolate(cam, size=spatial_size, mode=mode, align_corners=False)


def image_to_numpy(image):
    array = image.detach().cpu().float().numpy()[0]
    if array.shape[0] > 1:
        array = array.mean(axis=0)
    else:
        array = array[0]
    array = array - array.min()
    maximum = array.max()
    if maximum > 0:
        array = array / maximum
    return array


def cam_area(cam_array, threshold_quantile=0.85):
    threshold = float(np.quantile(cam_array, threshold_quantile))
    mask = cam_array >= threshold
    coords = np.argwhere(mask)
    peak = np.unravel_index(int(np.argmax(cam_array)), cam_array.shape)
    result = {
        "cam_threshold": threshold,
        "cam_peak_value": float(cam_array[peak]),
    }
    if cam_array.ndim == 2:
        result.update({
            "peak_y": int(peak[0]),
            "peak_x": int(peak[1]),
            "bbox_y_min": int(coords[:, 0].min()) if coords.size else "",
            "bbox_y_max": int(coords[:, 0].max()) if coords.size else "",
            "bbox_x_min": int(coords[:, 1].min()) if coords.size else "",
            "bbox_x_max": int(coords[:, 1].max()) if coords.size else "",
        })
    else:
        result.update({
            "peak_z": int(peak[0]),
            "peak_y": int(peak[1]),
            "peak_x": int(peak[2]),
            "bbox_z_min": int(coords[:, 0].min()) if coords.size else "",
            "bbox_z_max": int(coords[:, 0].max()) if coords.size else "",
            "bbox_y_min": int(coords[:, 1].min()) if coords.size else "",
            "bbox_y_max": int(coords[:, 1].max()) if coords.size else "",
            "bbox_x_min": int(coords[:, 2].min()) if coords.size else "",
            "bbox_x_max": int(coords[:, 2].max()) if coords.size else "",
        })
    return result


def batch_metadata(batch, index):
    meta = batch.get("metadata", {})
    if not isinstance(meta, dict):
        return {}
    out = {}
    for key, value in meta.items():
        if isinstance(value, (list, tuple)):
            out[key] = value[index] if index < len(value) else ""
        else:
            out[key] = value
    return out


def failure_patterns(true_label, predicted_label, class_names, metadata):
    if true_label == predicted_label:
        return []

    patterns = ["misclassified"]
    diameter = metadata.get("median_max_diameter_mm")
    try:
        diameter = float(diameter)
    except Exception:
        diameter = None
    if diameter is not None and diameter < 6.0:
        patterns.append("small_nodule_lt_6mm")

    calcification = str(metadata.get("majority_calcification", "")).strip()
    if calcification:
        patterns.append("calcification_score_{}".format(calcification))

    texture = str(metadata.get("majority_texture", "")).strip()
    if texture:
        patterns.append("texture_score_{}".format(texture))

    true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
    pred_name = class_names[predicted_label] if predicted_label < len(class_names) else str(predicted_label)
    if true_name in ("benign", "low_risk") and pred_name in ("malignant", "high_risk"):
        patterns.append("false_positive_as_malignant_or_high_risk")
        if calcification:
            patterns.append("calcified_nodule_flagged_as_malignant_pattern")
    if true_name in ("malignant", "high_risk") and pred_name in ("benign", "low_risk"):
        patterns.append("false_negative_as_benign_or_low_risk")

    if str(metadata.get("overall_consistency", "")).strip().lower() == "low":
        patterns.append("low_reader_consistency")
    return patterns


def collect_grad_cam_samples(model, data_module, class_names, max_samples, device):
    incorrect = []
    correct = []
    with torch.no_grad():
        for batch in data_module.test_dataloader():
            images = batch["image"].to(device)
            labels = batch["label"]
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            predictions = probabilities.argmax(dim=1)
            for item_idx in range(images.size(0)):
                true_label = int(labels[item_idx].detach().cpu())
                predicted_label = int(predictions[item_idx])
                sample = {
                    "image": images[item_idx:item_idx + 1].detach().cpu(),
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "probabilities": probabilities[item_idx].numpy(),
                    "roi_id": batch["roi_id"][item_idx],
                    "patient_id": batch["patient_id"][item_idx],
                    "metadata": batch_metadata(batch, item_idx),
                }
                if true_label != predicted_label:
                    if len(incorrect) < int(max_samples):
                        incorrect.append(sample)
                elif len(correct) < int(max_samples):
                    correct.append(sample)
    return (incorrect + correct)[:int(max_samples)]


def save_overlay(path, image_array, cam_array, title, slice_index=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if image_array.ndim == 3:
        if cam_array.ndim == 2:
            image_array = image_array[image_array.shape[0] // 2]
        else:
            if slice_index is None:
                slice_index = image_array.shape[0] // 2
            image_array = image_array[slice_index]
            cam_array = cam_array[slice_index]
    elif cam_array.ndim == 3:
        if slice_index is None:
            slice_index = cam_array.shape[0] // 2
        cam_array = cam_array[slice_index]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image_array, cmap="gray")
    ax.imshow(cam_array, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(path), dpi=300)
    plt.close(fig)


def generate_grad_cam_visualizations(
    model,
    data_module,
    output_dir,
    input_dim,
    task,
    class_names,
    max_samples=12,
    device=None,
):
    output_dir = Path(output_dir) / "grad_cam"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    target_layer = target_layer_for_model(model)
    extractor = CamExtractor(model, target_layer)
    rows = []
    failure_rows = []
    samples = collect_grad_cam_samples(model, data_module, class_names, max_samples, device)

    try:
        for sample_count, sample in enumerate(samples):
            image = sample["image"].to(device)
            true_label = sample["true_label"]
            predicted_label = sample["predicted_label"]
            roi_id = sample["roi_id"]
            patient_id = sample["patient_id"]
            metadata = sample.get("metadata", {})

            with torch.enable_grad():
                target_class = predicted_label
                logits, activations, gradients = extractor.compute(image, target_class)
                probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
                cams = {
                    "grad_cam": resize_cam(grad_cam_from_tensors(activations, gradients), image),
                    "grad_cam_plus_plus": resize_cam(grad_cam_plus_plus_from_tensors(activations, gradients), image),
                }

            patterns = failure_patterns(true_label, predicted_label, class_names, metadata)
            image_array = image_to_numpy(image)
            for method, cam_tensor in cams.items():
                cam_array = cam_tensor.detach().cpu().numpy()[0, 0]
                area = cam_area(cam_array)
                slice_index = area.get("peak_z")
                file_stem = "{}__{}__{}".format(sample_count, safe_name(roi_id), method)
                png_path = output_dir / "{}.png".format(file_stem)
                title = "{} target={}".format(method.replace("_", "-"), class_names[target_class])
                save_overlay(png_path, image_array, cam_array, title, slice_index=slice_index)

                row = {
                    "sample_index": sample_count,
                    "method": method,
                    "input_dim": input_dim,
                    "task": task,
                    "roi_id": roi_id,
                    "patient_id": patient_id,
                    "true_label_id": true_label,
                    "true_label": class_names[true_label] if true_label < len(class_names) else str(true_label),
                    "predicted_label_id": predicted_label,
                    "predicted_label": class_names[predicted_label] if predicted_label < len(class_names) else str(predicted_label),
                    "is_correct": true_label == predicted_label,
                    "failure_patterns": ";".join(patterns),
                    "target_class_id": target_class,
                    "target_class": class_names[target_class] if target_class < len(class_names) else str(target_class),
                    "target_probability": float(probabilities[target_class]),
                    "visualization_path": str(png_path),
                }
                row.update(metadata)
                row.update(area)
                rows.append(row)

            if true_label != predicted_label:
                failure_row = {
                    "sample_index": sample_count,
                    "input_dim": input_dim,
                    "task": task,
                    "roi_id": roi_id,
                    "patient_id": patient_id,
                    "true_label": class_names[true_label] if true_label < len(class_names) else str(true_label),
                    "predicted_label": class_names[predicted_label] if predicted_label < len(class_names) else str(predicted_label),
                    "predicted_probability": float(probabilities[predicted_label]),
                    "failure_patterns": ";".join(patterns),
                }
                failure_row.update(metadata)
                failure_rows.append(failure_row)
    finally:
        extractor.close()

    write_csv(output_dir / "grad_cam_interest_areas.csv", rows)
    write_csv(output_dir / "grad_cam_failure_cases.csv", failure_rows)
    write_csv(output_dir / "grad_cam_failure_pattern_summary.csv", summarize_failure_patterns(failure_rows))
    return {
        "grad_cam_dir": str(output_dir),
        "grad_cam_interest_areas": str(output_dir / "grad_cam_interest_areas.csv"),
        "grad_cam_failure_cases": str(output_dir / "grad_cam_failure_cases.csv"),
        "grad_cam_failure_pattern_summary": str(output_dir / "grad_cam_failure_pattern_summary.csv"),
        "grad_cam_sample_count": len(samples),
    }


def summarize_failure_patterns(failure_rows):
    counts = {}
    for row in failure_rows:
        for pattern in str(row.get("failure_patterns", "")).split(";"):
            pattern = pattern.strip()
            if pattern:
                counts[pattern] = counts.get(pattern, 0) + 1
    return [{"failure_pattern": key, "count": value} for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
