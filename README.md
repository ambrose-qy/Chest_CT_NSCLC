# Chest_CT_NSCLC

This repository is organized for exploratory analysis of chest CT datasets used in lung cancer and pulmonary nodule research. The current workspace keeps source notebooks under `data_analysis/`, raw datasets under `data/raw/`, and generated or cleaned outputs under `data/processed/`.

## Current Structure

```text
Chest_CT_NSCLC/
|-- data/
|   |-- raw/
|   |   |-- LIDC/
|   |   |   |-- manifest-1600709154662/
|   |   |   |   `-- LIDC-IDRI/
|   |   |   |-- tcia-lidc-xml/
|   |   |   |-- 161-resubmitted-correction-3-9-12.xml
|   |   |   |-- lidc-idri-nodule-counts-6-23-2015.xlsx
|   |   |   |-- tcia-diagnosis-data-2012-04-20.xls
|   |   |   `-- TCIA_LIDC-IDRI_20200921-nbia-digest.xlsx
|   |   `-- LUNA/
|   |       |-- 3723295.zip
|   |       `-- 4121926.zip
|   `-- processed/
|-- data_analysis/
|   |-- LIDC_analysis.ipynb
|   `-- LUNA16_data_analysis.ipynb
|-- .gitignore
`-- README.md
```

## Data Instructions

- Place original downloaded datasets in `data/raw/`.
- Keep LIDC-IDRI files under `data/raw/LIDC/`.
- Keep LUNA16 archive files or extracted LUNA16 data under `data/raw/LUNA/`.
- Save cleaned tables, derived metadata, extracted image arrays, masks, train/test splits, and other generated files under `data/processed/`.
- Do not commit medical image files, archives, model weights, checkpoints, or generated outputs. These paths and file types are already ignored in `.gitignore`.

## Notebook Instructions

Use the notebooks in `data_analysis/` for dataset inspection and exploratory analysis:

- `data_analysis/LIDC_analysis.ipynb`: intended for LIDC-IDRI data exploration, XML annotation review, metadata checks, and DICOM inspection.
- `data_analysis/LUNA16_data_analysis.ipynb`: intended for LUNA16 archive inspection, metadata loading, and preprocessing experiments.

Run notebooks from the repository root so relative paths resolve consistently:

```powershell
cd F:\AstraZeneca\project\chestCT\Chest_CT_NSCLC
jupyter lab
```

Recommended path pattern inside notebooks:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LIDC_DIR = RAW_DIR / "LIDC"
LUNA_DIR = RAW_DIR / "LUNA"
```

## Workflow

1. Download or prepare raw data in `data/raw/LIDC/` and `data/raw/LUNA/`.
2. Explore dataset structure and metadata in the notebooks under `data_analysis/`.
3. Write any generated intermediate files to `data/processed/`.
4. Keep analysis code, documentation, and lightweight configuration in git.
5. Keep large datasets, images, archives, trained models, and notebook checkpoints out of git.

## Notes

- `data/raw/.gitkeep` and `data/processed/.gitkeep` keep the folder structure available in git.
- The current repository does not include a dependency file. If package requirements become stable, add a `requirements.txt` or environment file at the repository root.
