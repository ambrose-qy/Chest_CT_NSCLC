# Chest_CT_NSCLC

Exploratory analysis workspace for chest CT datasets used in lung cancer and pulmonary nodule research. The current project version is centered on LIDC-IDRI DICOM/XML analysis, with LUNA16 raw data staged for future analysis.

## Repository Layout

```text
Chest_CT_NSCLC/
|-- data/
|   |-- raw/
|   |   |-- LIDC/
|   |   |   |-- manifest-1600709154662/
|   |   |   |   |-- LIDC-IDRI/
|   |   |   |   `-- metadata.csv
|   |   |   |-- tcia-lidc-xml/
|   |   |   |-- 161-resubmitted-correction-3-9-12.xml
|   |   |   |-- lidc-idri-nodule-counts-6-23-2015.xlsx
|   |   |   |-- tcia-diagnosis-data-2012-04-20.xls
|   |   |   `-- TCIA_LIDC-IDRI_20200921-nbia-digest.xlsx
|   |   `-- LUNA/
|   |       |-- annotations.csv
|   |       |-- candidates.csv
|   |       |-- candidates_V2.zip
|   |       |-- evaluationScript.zip
|   |       |-- sampleSubmission.csv
|   |       |-- seg-lungs-LUNA16.zip
|   |       `-- subset0.zip ... subset9.zip
|   `-- processed/
|       |-- tables/
|       `-- figures/
|-- data_analysis/
|   |-- LIDC_analysis.ipynb
|   `-- LUNA16_data_analysis.ipynb
|-- .gitignore
`-- README.md
```

`data/raw/` and `data/processed/` are intentionally ignored by git except for their `.gitkeep` files. Keep downloaded datasets, derived CSVs, figures, model artifacts, and medical images out of version control.

## Current Notebooks

- `data_analysis/LIDC_analysis.ipynb` is the active analysis notebook. It scans LIDC-IDRI DICOM files, parses TCIA XML annotations, builds derived metadata tables, estimates nodule size/location/morphology summaries, clusters multi-reader annotations, and writes QC outputs.
- `data_analysis/LUNA16_data_analysis.ipynb` currently contains only an empty starter cell. LUNA16 archives and metadata are present under `data/raw/LUNA/`, but no LUNA16 analysis workflow has been implemented yet in this checkout.

Run notebooks from the repository root so relative paths and output folders stay consistent:

```powershell
cd F:\AstraZeneca\project\chestCT\Chest_CT_NSCLC
jupyter lab
```

The LIDC notebook currently defines `PROJECT_ROOT` explicitly. If the project is moved, update that path or replace it with:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
```

## Python Environment

The repository does not yet include a pinned `requirements.txt`. The current LIDC notebook uses:

- `pandas`
- `numpy`
- `matplotlib`
- `pydicom`
- `tqdm`
- `openpyxl` for reading `.xlsx` metadata with `pandas.read_excel`
- Jupyter, such as `jupyterlab` or `notebook`

Example setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib pydicom tqdm openpyxl jupyterlab
```

## LIDC Analysis Outputs

`LIDC_analysis.ipynb` writes derived files to `data/processed/tables/` and `data/processed/figures/`.

Main table outputs in the current run include:

- `lidc_series_inventory.csv`: 1,362 DICOM series rows.
- `lidc_main_ct_series.csv`: 1,010 selected main CT series rows.
- `lidc_xml_nodules_raw.csv`: 20,362 raw reader annotation rows parsed from XML.
- `lidc_nodule_size_location_morphology.csv`: 20,362 annotation rows with derived size, location, and morphology fields.
- `lidc_reader_annotations_clustered.csv`: 20,362 annotations with lesion-cluster assignments.
- `lidc_annotation_consistency_by_lesion_cluster.csv`: 7,039 lesion clusters with reader-consistency summaries.
- `lidc_processed_quality_control_report.csv`: downstream analysis risk checks.

Main figure outputs include:

- sample DICOM slice previews
- patient age and sex availability plots
- malignancy and morphology score distributions
- nodule size and coarse location distributions
- reader annotation counts and reader consistency summaries

Current generated QC highlights:

- XML-to-main-CT match rate is about 99.2%.
- Patient age is available for 207 of 1,010 patients.
- Patient sex is available for 286 of 1,010 patients.
- The current clustering summary contains 2,520 four-reader consensus clusters, 1,676 three-reader majority clusters, 1,478 two-reader partial clusters, and 1,365 single-reader clusters.
- Five annotations have estimated maximum diameter greater than 60 mm and should be reviewed before modeling.

## Data Workflow

1. Place original downloaded datasets under `data/raw/LIDC/` and `data/raw/LUNA/`.
2. Run exploratory notebooks from the repository root.
3. Write all derived tables, figures, masks, arrays, train/test splits, and model outputs under `data/processed/` or another ignored output directory.
4. Keep notebooks, documentation, and lightweight configuration in git.
5. Do not commit DICOM files, archives, generated images, large CSV outputs, checkpoints, or trained weights.

## Notes

- `data/raw/.gitkeep` and `data/processed/.gitkeep` preserve the empty folder structure in git.
- `.gitignore` excludes raw data, processed outputs, archives, medical image formats, common model files, Python caches, and notebook checkpoints.
- If the analysis environment stabilizes, add a `requirements.txt` or environment file at the repository root.
