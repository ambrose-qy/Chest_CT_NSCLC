"""
LIDC-IDRI datasets and datamodules for Lightning baselines.
"""

from __future__ import print_function

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lidc_lightning_utils import clean_value, import_lightning, read_csv_rows, safe_float, safe_int


HU_MIN = -1000.0
HU_MAX = 400.0
BINARY_LABELS = ["benign", "malignant"]
MULTICLASS_LABELS = ["low_risk", "intermediate_risk", "high_risk"]
METADATA_COLUMNS = [
    "median_max_diameter_mm",
    "majority_calcification",
    "majority_internal_structure",
    "majority_sphericity",
    "majority_margin",
    "majority_lobulation",
    "majority_spiculation",
    "majority_texture",
    "overall_consistency",
    "label_confidence",
]
PCA_METADATA_COLUMNS = [
    "median_max_diameter_mm",
    "majority_calcification",
    "majority_internal_structure",
    "majority_sphericity",
    "majority_margin",
    "majority_lobulation",
    "majority_spiculation",
    "majority_texture",
    "label_confidence",
]


try:
    _PL = import_lightning()
    _BASE_DATA_MODULE = _PL.LightningDataModule
except Exception:
    _BASE_DATA_MODULE = object


def load_dicom_hu(path):
    try:
        import pydicom
    except Exception as exc:
        raise ImportError("pydicom is required for 2D LIDC DICOM slice loading.") from exc

    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return image * slope + intercept


def normalise_hu(image, hu_min=HU_MIN, hu_max=HU_MAX):
    image = np.clip(image, hu_min, hu_max)
    return (image - hu_min) / float(hu_max - hu_min)


def center_crop_or_pad_2d(tensor, size):
    channels, height, width = tensor.shape
    target_h, target_w = size
    out = torch.zeros((channels, target_h, target_w), dtype=tensor.dtype)

    copy_h = min(height, target_h)
    copy_w = min(width, target_w)
    src_y = max((height - copy_h) // 2, 0)
    src_x = max((width - copy_w) // 2, 0)
    dst_y = max((target_h - copy_h) // 2, 0)
    dst_x = max((target_w - copy_w) // 2, 0)
    out[:, dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = tensor[:, src_y:src_y + copy_h, src_x:src_x + copy_w]
    return out


def center_crop_or_pad_3d(tensor, shape):
    channels, depth, height, width = tensor.shape
    target_d, target_h, target_w = shape
    out = torch.zeros((channels, target_d, target_h, target_w), dtype=tensor.dtype)

    copy_d = min(depth, target_d)
    copy_h = min(height, target_h)
    copy_w = min(width, target_w)
    src_z = max((depth - copy_d) // 2, 0)
    src_y = max((height - copy_h) // 2, 0)
    src_x = max((width - copy_w) // 2, 0)
    dst_z = max((target_d - copy_d) // 2, 0)
    dst_y = max((target_h - copy_h) // 2, 0)
    dst_x = max((target_w - copy_w) // 2, 0)
    out[:, dst_z:dst_z + copy_d, dst_y:dst_y + copy_h, dst_x:dst_x + copy_w] = tensor[
        :,
        src_z:src_z + copy_d,
        src_y:src_y + copy_h,
        src_x:src_x + copy_w,
    ]
    return out


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def parse_number_list(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def normalise_tensor(tensor, mean=None, std=None):
    if mean is None or std is None:
        return tensor
    mean_values = parse_number_list(mean)
    std_values = parse_number_list(std)
    if not mean_values or not std_values:
        return tensor
    if len(mean_values) == 1:
        mean_values = mean_values * tensor.shape[0]
    if len(std_values) == 1:
        std_values = std_values * tensor.shape[0]
    shape = [tensor.shape[0]] + [1] * (tensor.dim() - 1)
    mean_tensor = torch.tensor(mean_values[:tensor.shape[0]], dtype=tensor.dtype, device=tensor.device).view(*shape)
    std_tensor = torch.tensor(std_values[:tensor.shape[0]], dtype=tensor.dtype, device=tensor.device).view(*shape)
    return (tensor - mean_tensor) / torch.clamp(std_tensor, min=1e-6)


def metadata_from_row(row):
    return {key: clean_value(row.get(key)) for key in METADATA_COLUMNS}


def roi_stat_feature_vector(row, volume):
    finite = np.asarray(volume, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        finite = np.zeros((1,), dtype=np.float32)

    percentiles = np.percentile(finite, [1, 5, 10, 25, 50, 75, 90, 95, 99]).astype(np.float32)
    features = [
        float(finite.mean()),
        float(finite.std()),
        float(finite.min()),
        float(finite.max()),
        float((finite > 0.10).mean()),
        float((finite > 0.25).mean()),
        float((finite > 0.50).mean()),
        float((finite > 0.75).mean()),
    ]
    features.extend(float(value) for value in percentiles)
    for column in PCA_METADATA_COLUMNS:
        value = safe_float(row.get(column))
        features.append(float(value) if value is not None else 0.0)
    return np.asarray(features, dtype=np.float32)


def transform_pca_features(row, volume, pca_transform):
    if not pca_transform:
        return torch.empty(0, dtype=torch.float32)
    vector = roi_stat_feature_vector(row, volume).reshape(1, -1)
    scaler = pca_transform["scaler"]
    pca = pca_transform["pca"]
    transformed = pca.transform(scaler.transform(vector))[0].astype(np.float32)
    return torch.from_numpy(transformed)


def random_cutout_2d(tensor, fraction):
    if not fraction or fraction <= 0:
        return tensor
    _, height, width = tensor.shape
    cut_h = max(1, int(round(height * fraction)))
    cut_w = max(1, int(round(width * fraction)))
    y0 = random.randint(0, max(height - cut_h, 0))
    x0 = random.randint(0, max(width - cut_w, 0))
    fill = float(tensor.mean())
    tensor[:, y0:y0 + cut_h, x0:x0 + cut_w] = fill
    return tensor


def random_cutout_3d(tensor, fraction):
    if not fraction or fraction <= 0:
        return tensor
    _, depth, height, width = tensor.shape
    cut_d = max(1, int(round(depth * fraction)))
    cut_h = max(1, int(round(height * fraction)))
    cut_w = max(1, int(round(width * fraction)))
    z0 = random.randint(0, max(depth - cut_d, 0))
    y0 = random.randint(0, max(height - cut_h, 0))
    x0 = random.randint(0, max(width - cut_w, 0))
    fill = float(tensor.mean())
    tensor[:, z0:z0 + cut_d, y0:y0 + cut_h, x0:x0 + cut_w] = fill
    return tensor


def augment_2d(
    tensor,
    rotate=True,
    flip=True,
    scale=True,
    scale_range=0.10,
    noise_std=0.02,
    intensity_shift=0.05,
    contrast_range=0.10,
    cutout_fraction=0.0,
):
    if flip and random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(2,))
    if flip and random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(1,))
    if rotate and random.random() < 0.5:
        tensor = torch.rot90(tensor, k=random.choice([1, 2, 3]), dims=(1, 2))
    if scale:
        original = tensor.shape[-2:]
        scale_range = max(float(scale_range), 0.0)
        factor = random.uniform(1.0 - scale_range, 1.0 + scale_range)
        scaled = [max(4, int(round(value * factor))) for value in original]
        tensor = F.interpolate(tensor.unsqueeze(0), size=scaled, mode="bilinear", align_corners=False).squeeze(0)
        tensor = center_crop_or_pad_2d(tensor, original)
    if contrast_range and contrast_range > 0:
        factor = random.uniform(1.0 - contrast_range, 1.0 + contrast_range)
        tensor = (tensor - tensor.mean()) * factor + tensor.mean()
    if intensity_shift and intensity_shift > 0:
        tensor = tensor + random.uniform(-intensity_shift, intensity_shift)
    if noise_std and noise_std > 0:
        tensor = tensor + torch.randn_like(tensor) * float(noise_std)
        tensor = torch.clamp(tensor, 0.0, 1.0)
    if cutout_fraction and cutout_fraction > 0 and random.random() < 0.5:
        tensor = random_cutout_2d(tensor, cutout_fraction)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def augment_3d(
    tensor,
    rotate=True,
    flip=True,
    scale=True,
    scale_range=0.10,
    noise_std=0.02,
    intensity_shift=0.05,
    contrast_range=0.10,
    cutout_fraction=0.0,
):
    if flip and random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(3,))
    if flip and random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(2,))
    if flip and random.random() < 0.25:
        tensor = torch.flip(tensor, dims=(1,))
    if rotate and random.random() < 0.5:
        tensor = torch.rot90(tensor, k=random.choice([1, 2, 3]), dims=(2, 3))
    if scale:
        original = tensor.shape[-3:]
        scale_range = max(float(scale_range), 0.0)
        factor = random.uniform(1.0 - scale_range, 1.0 + scale_range)
        scaled = [max(4, int(round(value * factor))) for value in original]
        tensor = F.interpolate(tensor.unsqueeze(0), size=scaled, mode="trilinear", align_corners=False).squeeze(0)
        tensor = center_crop_or_pad_3d(tensor, original)
    if contrast_range and contrast_range > 0:
        factor = random.uniform(1.0 - contrast_range, 1.0 + contrast_range)
        tensor = (tensor - tensor.mean()) * factor + tensor.mean()
    if intensity_shift and intensity_shift > 0:
        tensor = tensor + random.uniform(-intensity_shift, intensity_shift)
    if noise_std and noise_std > 0:
        tensor = tensor + torch.randn_like(tensor) * float(noise_std)
        tensor = torch.clamp(tensor, 0.0, 1.0)
    if cutout_fraction and cutout_fraction > 0 and random.random() < 0.5:
        tensor = random_cutout_3d(tensor, cutout_fraction)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


class LIDC2DMaxSliceDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        task="binary",
        image_size=224,
        augment=False,
        max_samples=None,
        in_channels=3,
        normalization_mean=None,
        normalization_std=None,
        augment_rotate=True,
        augment_flip=True,
        augment_scale=True,
        augment_scale_range=0.10,
        augment_noise_std=0.02,
        augment_intensity_shift=0.05,
        augment_contrast_range=0.10,
        augment_cutout_fraction=0.0,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.task = task
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.in_channels = int(in_channels)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.augment_rotate = augment_rotate
        self.augment_flip = augment_flip
        self.augment_scale = augment_scale
        self.augment_scale_range = augment_scale_range
        self.augment_noise_std = augment_noise_std
        self.augment_intensity_shift = augment_intensity_shift
        self.augment_contrast_range = augment_contrast_range
        self.augment_cutout_fraction = augment_cutout_fraction
        rows = read_csv_rows(self.manifest_path)

        if task == "binary":
            split_col = "binary_split"
            label_col = "binary_label_id"
        elif task == "multiclass":
            split_col = "multiclass_split"
            label_col = "multiclass_risk_label_id"
        else:
            raise ValueError("Unsupported task '{}'. Use binary or multiclass.".format(task))
        rows = [
            row for row in rows
            if clean_value(row.get(split_col)) == split and clean_value(row.get(label_col)) != ""
        ]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def label_counts(self):
        label_col = "binary_label_id" if self.task == "binary" else "multiclass_risk_label_id"
        return Counter(safe_int(row.get(label_col)) for row in self.rows)

    def labels(self):
        label_col = "binary_label_id" if self.task == "binary" else "multiclass_risk_label_id"
        return [safe_int(row.get(label_col)) for row in self.rows]

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_dicom_hu(row["dicom_path"])
        image = normalise_hu(image)
        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        if self.in_channels > 1:
            tensor = tensor.repeat(self.in_channels, 1, 1)
        if self.augment:
            tensor = augment_2d(
                tensor,
                rotate=as_bool(self.augment_rotate),
                flip=as_bool(self.augment_flip),
                scale=as_bool(self.augment_scale),
                scale_range=float(self.augment_scale_range),
                noise_std=float(self.augment_noise_std),
                intensity_shift=float(self.augment_intensity_shift),
                contrast_range=float(self.augment_contrast_range),
                cutout_fraction=float(self.augment_cutout_fraction),
            )
        tensor = normalise_tensor(tensor, self.normalization_mean, self.normalization_std)
        label_col = "binary_label_id" if self.task == "binary" else "multiclass_risk_label_id"
        label = safe_int(row.get(label_col))
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "roi_id": row.get("roi_id", ""),
            "patient_id": row.get("PatientID") or row.get("patient_folder", ""),
            "metadata": metadata_from_row(row),
        }


class LIDC3DVolumeDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        task="binary",
        augment=False,
        max_samples=None,
        normalization_mean=None,
        normalization_std=None,
        augment_rotate=True,
        augment_flip=True,
        augment_scale=True,
        augment_scale_range=0.10,
        augment_noise_std=0.02,
        augment_intensity_shift=0.05,
        augment_contrast_range=0.10,
        augment_cutout_fraction=0.0,
        pca_transform=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.task = task
        self.augment = bool(augment)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.augment_rotate = augment_rotate
        self.augment_flip = augment_flip
        self.augment_scale = augment_scale
        self.augment_scale_range = augment_scale_range
        self.augment_noise_std = augment_noise_std
        self.augment_intensity_shift = augment_intensity_shift
        self.augment_contrast_range = augment_contrast_range
        self.augment_cutout_fraction = augment_cutout_fraction
        self.pca_transform = pca_transform
        rows = read_csv_rows(self.manifest_path)
        if task == "binary":
            rows = [
                row for row in rows
                if clean_value(row.get("binary_split")) == split and clean_value(row.get("binary_label_id")) != ""
            ]
        elif task == "multiclass":
            rows = [
                row for row in rows
                if clean_value(row.get("multiclass_split")) == split and clean_value(row.get("multiclass_risk_label_id")) != ""
            ]
        else:
            raise ValueError("Unsupported task '{}'. Use binary or multiclass.".format(task))
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def label_counts(self):
        label_col = "binary_label_id" if self.task == "binary" else "multiclass_risk_label_id"
        return Counter(safe_int(row.get(label_col)) for row in self.rows)

    def labels(self):
        label_col = "binary_label_id" if self.task == "binary" else "multiclass_risk_label_id"
        return [safe_int(row.get(label_col)) for row in self.rows]

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(row["volume_path"]) as npz:
            volume = npz["volume"].astype(np.float32)
        tensor = torch.from_numpy(volume).float().unsqueeze(0)
        if self.augment:
            tensor = augment_3d(
                tensor,
                rotate=as_bool(self.augment_rotate),
                flip=as_bool(self.augment_flip),
                scale=as_bool(self.augment_scale),
                scale_range=float(self.augment_scale_range),
                noise_std=float(self.augment_noise_std),
                intensity_shift=float(self.augment_intensity_shift),
                contrast_range=float(self.augment_contrast_range),
                cutout_fraction=float(self.augment_cutout_fraction),
            )
        tensor = normalise_tensor(tensor, self.normalization_mean, self.normalization_std)
        label_col = "binary_label_id" if self.task == "binary" else "multiclass_risk_label_id"
        label = safe_int(row.get(label_col))
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "pca_features": transform_pca_features(row, volume, self.pca_transform),
            "roi_id": row.get("roi_id", ""),
            "patient_id": row.get("PatientID") or row.get("patient_folder", ""),
            "metadata": metadata_from_row(row),
        }


class LIDCDataModule(_BASE_DATA_MODULE):
    def __init__(
        self,
        manifest_path,
        input_dim,
        task="binary",
        batch_size=8,
        image_size=224,
        num_workers=0,
        max_samples_per_split=None,
        in_channels=3,
        normalization_mean=None,
        normalization_std=None,
        normalization_stats_samples=128,
        augment_rotate=True,
        augment_flip=True,
        augment_scale=True,
        augment_scale_range=0.10,
        augment_noise_std=0.02,
        augment_intensity_shift=0.05,
        augment_contrast_range=0.10,
        augment_cutout_fraction=0.0,
        pca_features_enabled=False,
        pca_n_components=0,
        balanced_sampler=False,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.input_dim = input_dim
        self.task = task
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.num_workers = int(num_workers)
        self.max_samples_per_split = max_samples_per_split
        self.in_channels = int(in_channels)
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.normalization_stats_samples = normalization_stats_samples
        self.augment_rotate = augment_rotate
        self.augment_flip = augment_flip
        self.augment_scale = augment_scale
        self.augment_scale_range = augment_scale_range
        self.augment_noise_std = augment_noise_std
        self.augment_intensity_shift = augment_intensity_shift
        self.augment_contrast_range = augment_contrast_range
        self.augment_cutout_fraction = augment_cutout_fraction
        self.pca_features_enabled = bool(pca_features_enabled)
        self.pca_n_components = int(pca_n_components or 0)
        self.balanced_sampler = bool(balanced_sampler)
        self.pca_transform = None
        self.pca_feature_dim = 0
        self.normalization_stats = {}

    def setup(self, stage=None):
        if self.input_dim == "2d":
            dataset_cls = LIDC2DMaxSliceDataset
            kwargs = {"image_size": self.image_size, "in_channels": self.in_channels}
        elif self.input_dim == "3d":
            dataset_cls = LIDC3DVolumeDataset
            kwargs = {}
        else:
            raise ValueError("input_dim must be 2d or 3d.")

        self.train_dataset = dataset_cls(
            self.manifest_path,
            split="train",
            task=self.task,
            augment=True,
            max_samples=self.max_samples_per_split,
            normalization_mean=None,
            normalization_std=None,
            augment_rotate=self.augment_rotate,
            augment_flip=self.augment_flip,
            augment_scale=self.augment_scale,
            augment_scale_range=self.augment_scale_range,
            augment_noise_std=self.augment_noise_std,
            augment_intensity_shift=self.augment_intensity_shift,
            augment_contrast_range=self.augment_contrast_range,
            augment_cutout_fraction=self.augment_cutout_fraction,
            **kwargs
        )
        mean, std = self.resolve_normalization_stats(dataset_cls, kwargs)
        self.train_dataset.normalization_mean = mean
        self.train_dataset.normalization_std = std
        self.val_dataset = dataset_cls(
            self.manifest_path,
            split="val",
            task=self.task,
            augment=False,
            max_samples=self.max_samples_per_split,
            normalization_mean=mean,
            normalization_std=std,
            **kwargs
        )
        self.test_dataset = dataset_cls(
            self.manifest_path,
            split="test",
            task=self.task,
            augment=False,
            max_samples=self.max_samples_per_split,
            normalization_mean=mean,
            normalization_std=std,
            **kwargs
        )
        self.fit_pca_features_if_needed()

    def resolve_normalization_stats(self, dataset_cls, kwargs):
        if self.normalization_mean not in (None, "", "auto") and self.normalization_std not in (None, "", "auto"):
            self.normalization_stats = {
                "mean": self.normalization_mean,
                "std": self.normalization_std,
                "source": "hyperparameters",
            }
            return self.normalization_mean, self.normalization_std
        stats_dataset = dataset_cls(
            self.manifest_path,
            split="train",
            task=self.task,
            augment=False,
            max_samples=self.normalization_stats_samples,
            normalization_mean=None,
            normalization_std=None,
            **kwargs
        )
        mean, std, count = compute_dataset_mean_std(stats_dataset)
        self.normalization_stats = {
            "mean": mean,
            "std": std,
            "source": "train_split_sample",
            "sample_count": count,
        }
        return mean, std

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.balanced_sampler:
            labels = self.train_dataset.labels()
            counts = Counter(label for label in labels if label is not None)
            sample_weights = [
                1.0 / float(max(counts.get(label, 0), 1))
                for label in labels
            ]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True,
            )
            shuffle = False
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def class_counts(self):
        return {
            "train": dict(self.train_dataset.label_counts()),
            "val": dict(self.val_dataset.label_counts()),
            "test": dict(self.test_dataset.label_counts()),
        }

    def fit_pca_features_if_needed(self):
        if self.input_dim != "3d" or not self.pca_features_enabled or self.pca_n_components <= 0:
            return
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            raise ImportError("scikit-learn is required for PCA feature fusion.") from exc

        rows = getattr(self.train_dataset, "rows", [])
        if not rows:
            return
        vectors = []
        for row in rows:
            with np.load(row["volume_path"]) as npz:
                volume = npz["volume"].astype(np.float32)
            vectors.append(roi_stat_feature_vector(row, volume))
        matrix = np.stack(vectors, axis=0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)
        n_components = min(int(self.pca_n_components), scaled.shape[0], scaled.shape[1])
        if n_components <= 0:
            return
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(scaled)
        self.pca_transform = {
            "scaler": scaler,
            "pca": pca,
            "n_components": n_components,
            "source": "train_roi_statistics",
            "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        }
        self.pca_feature_dim = n_components
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            if hasattr(dataset, "pca_transform"):
                dataset.pca_transform = self.pca_transform


def compute_dataset_mean_std(dataset):
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0
    for index in range(len(dataset)):
        item = dataset[index]
        tensor = item["image"].float()
        total_sum += float(tensor.sum())
        total_sq_sum += float((tensor * tensor).sum())
        total_count += tensor.numel()
    if total_count == 0:
        return 0.5, 0.25, 0
    mean = total_sum / float(total_count)
    variance = max(total_sq_sum / float(total_count) - mean * mean, 1e-8)
    return mean, variance ** 0.5, len(dataset)
