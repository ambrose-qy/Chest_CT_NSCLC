"""
PyTorch dataset for LIDC-IDRI 2D nodule CNN baselines.
"""

from __future__ import print_function

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


HU_MIN = -1000.0
HU_MAX = 400.0


def clean_value(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("none", "nan"):
        return ""
    return text


def safe_float(value):
    text = clean_value(value)
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def safe_int(value):
    value = safe_float(value)
    if value is None:
        return None
    return int(value)


def read_csv_rows(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clamp_int(value, low, high):
    return max(low, min(int(round(value)), high))


def load_dicom_hu(path):
    try:
        import pydicom
    except Exception as exc:
        raise ImportError("pydicom is required for LIDC2DNoduleDataset.") from exc

    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return image * slope + intercept


def window_normalize(image, hu_min=HU_MIN, hu_max=HU_MAX):
    image = np.clip(image, hu_min, hu_max)
    return (image - hu_min) / float(hu_max - hu_min)


def crop_roi(image, row, square=True):
    height, width = image.shape
    x_min = safe_float(row.get("x_min_px_roi"))
    x_max = safe_float(row.get("x_max_px_roi"))
    y_min = safe_float(row.get("y_min_px_roi"))
    y_max = safe_float(row.get("y_max_px_roi"))

    if None in (x_min, x_max, y_min, y_max):
        return image

    x0 = clamp_int(x_min, 0, width - 1)
    x1 = clamp_int(x_max, 0, width - 1)
    y0 = clamp_int(y_min, 0, height - 1)
    y1 = clamp_int(y_max, 0, height - 1)

    if x1 <= x0 or y1 <= y0:
        return image

    if square:
        crop_width = x1 - x0 + 1
        crop_height = y1 - y0 + 1
        side = max(crop_width, crop_height)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        x0 = clamp_int(cx - side / 2.0, 0, width - 1)
        x1 = clamp_int(cx + side / 2.0, 0, width - 1)
        y0 = clamp_int(cy - side / 2.0, 0, height - 1)
        y1 = clamp_int(cy + side / 2.0, 0, height - 1)

    return image[y0:y1 + 1, x0:x1 + 1]


class LIDC2DNoduleDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        split=None,
        image_size=224,
        augment=False,
        hu_min=HU_MIN,
        hu_max=HU_MAX,
        max_samples=None,
    ):
        self.manifest_path = Path(manifest_path)
        rows = read_csv_rows(self.manifest_path)
        if split is not None:
            rows = [row for row in rows if clean_value(row.get("binary_split")) == split]
        if max_samples is not None:
            rows = rows[:max_samples]

        self.rows = rows
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = load_dicom_hu(row["dicom_path"])
        image = crop_roi(image, row, square=True)
        image = window_normalize(image, hu_min=self.hu_min, hu_max=self.hu_max)

        tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)

        if self.augment:
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(2,))
            if random.random() < 0.5:
                tensor = torch.flip(tensor, dims=(1,))
            if random.random() < 0.25:
                tensor = torch.rot90(tensor, k=random.choice([1, 3]), dims=(1, 2))

        tensor = tensor.repeat(3, 1, 1)
        label = safe_int(row.get("binary_label_id"))
        if label is None:
            raise ValueError("Missing binary_label_id for ROI {}".format(row.get("roi_id", index)))

        return {
            "image": tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "roi_id": row.get("roi_id", ""),
            "patient_id": row.get("PatientID") or row.get("patient_folder", ""),
        }


def make_lidc_dataloader(
    manifest_path,
    split,
    batch_size,
    image_size=224,
    augment=False,
    shuffle=None,
    num_workers=0,
    max_samples=None,
):
    dataset = LIDC2DNoduleDataset(
        manifest_path=manifest_path,
        split=split,
        image_size=image_size,
        augment=augment,
        max_samples=max_samples,
    )
    if shuffle is None:
        shuffle = split == "train"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
