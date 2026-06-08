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
from torch.utils.data import DataLoader, Dataset

from lidc_lightning_utils import clean_value, import_lightning, read_csv_rows, safe_int


HU_MIN = -1000.0
HU_MAX = 400.0
BINARY_LABELS = ["benign", "malignant"]
MULTICLASS_LABELS = ["low_risk", "intermediate_risk", "high_risk"]


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


def augment_2d(tensor, rotate=True, flip=True, scale=True, noise_std=0.02):
    if flip and random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(2,))
    if flip and random.random() < 0.5:
        tensor = torch.flip(tensor, dims=(1,))
    if rotate and random.random() < 0.5:
        tensor = torch.rot90(tensor, k=random.choice([1, 2, 3]), dims=(1, 2))
    if scale:
        original = tensor.shape[-2:]
        factor = random.uniform(0.9, 1.1)
        scaled = [max(4, int(round(value * factor))) for value in original]
        tensor = F.interpolate(tensor.unsqueeze(0), size=scaled, mode="bilinear", align_corners=False).squeeze(0)
        tensor = center_crop_or_pad_2d(tensor, original)
    if noise_std and noise_std > 0:
        tensor = tensor + torch.randn_like(tensor) * float(noise_std)
        tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def augment_3d(tensor, rotate=True, flip=True, scale=True, noise_std=0.02):
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
        factor = random.uniform(0.9, 1.1)
        scaled = [max(4, int(round(value * factor))) for value in original]
        tensor = F.interpolate(tensor.unsqueeze(0), size=scaled, mode="trilinear", align_corners=False).squeeze(0)
        tensor = center_crop_or_pad_3d(tensor, original)
    if noise_std and noise_std > 0:
        tensor = tensor + torch.randn_like(tensor) * float(noise_std)
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
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.task = task
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.in_channels = int(in_channels)
        rows = read_csv_rows(self.manifest_path)

        if task != "binary":
            raise ValueError("2D max-slice training currently uses the binary manifest/task.")
        rows = [
            row for row in rows
            if clean_value(row.get("binary_split")) == split and clean_value(row.get("binary_label_id")) != ""
        ]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def label_counts(self):
        return Counter(safe_int(row.get("binary_label_id")) for row in self.rows)

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
        if self.augment:
            tensor = augment_2d(tensor)
        if self.in_channels > 1:
            tensor = tensor.repeat(self.in_channels, 1, 1)
        label = safe_int(row.get("binary_label_id"))
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "roi_id": row.get("roi_id", ""),
            "patient_id": row.get("PatientID") or row.get("patient_folder", ""),
        }


class LIDC3DVolumeDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split,
        task="binary",
        augment=False,
        max_samples=None,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.task = task
        self.augment = bool(augment)
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

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(row["volume_path"]) as npz:
            volume = npz["volume"].astype(np.float32)
        tensor = torch.from_numpy(volume).float().unsqueeze(0)
        if self.augment:
            tensor = augment_3d(tensor)
        label_col = "binary_label_id" if self.task == "binary" else "multiclass_risk_label_id"
        label = safe_int(row.get(label_col))
        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "roi_id": row.get("roi_id", ""),
            "patient_id": row.get("PatientID") or row.get("patient_folder", ""),
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
            **kwargs
        )
        self.val_dataset = dataset_cls(
            self.manifest_path,
            split="val",
            task=self.task,
            augment=False,
            max_samples=self.max_samples_per_split,
            **kwargs
        )
        self.test_dataset = dataset_cls(
            self.manifest_path,
            split="test",
            task=self.task,
            augment=False,
            max_samples=self.max_samples_per_split,
            **kwargs
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
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

