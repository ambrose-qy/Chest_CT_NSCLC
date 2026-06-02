# Chest_CT_NSCLC

Exploratory and preprocessing workspace for chest CT datasets used in lung cancer and pulmonary nodule research. The current project version includes LIDC-IDRI DICOM/XML analysis and LUNA16 subset0-4 preprocessing.

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
|   |-- LIDC_process1.py
|   |-- LIDC_process2.py
|   |-- LIDC_process3.py
|   |-- LIDC_process4.py
|   |-- LUNA_process.py
|   |-- LUNA_process2.py
|   `-- LUNA16_data_analysis.ipynb
|-- .gitignore
`-- README.md
```

`data/raw/` and `data/processed/` are intentionally ignored by git except for their `.gitkeep` files. Keep downloaded datasets, derived CSVs, figures, model artifacts, and medical images out of version control.

## Current Scripts

- `data_analysis/LIDC_process1.py` scans LIDC-IDRI raw DICOM/XML data and builds patient, nodule size, location, and morphology tables.
- `data_analysis/LIDC_process2.py` clusters multi-reader XML annotations and scores annotation consistency.
- `data_analysis/LIDC_process3.py` extracts agreed nodule ROI manifests for nodules >=3 mm approved by at least 3 radiologists.
- `data_analysis/LIDC_process4.py` constructs malignancy labels and patient-level train/validation/test splits using 70/10/20 proportions with risk-label balancing.
- `data_analysis/LUNA_process.py` preprocesses LUNA16 subset0-4 from `E:\LUNA`, including de-identification, 1 mm resampling, HU normalisation, lung parenchyma segmentation, and coordinate validation.
- `data_analysis/LUNA_process2.py` preprocesses LUNA16 subset5-9 from `D:\LUNA` with the same pipeline and project-local output naming.
- `data_analysis/LIDC_analysis.ipynb` and `data_analysis/LUNA16_data_analysis.ipynb` are notebook references used during workflow development.

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

Use the existing conda environment:

```powershell
conda activate torch-gpu
```

or run commands without activating:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_process1.py
```

Verified package versions in `torch-gpu` include:

- `numpy 1.26.4`
- `scipy 1.11.4`

The workflow also uses:

- `pandas`
- `numpy`
- `matplotlib`
- `pydicom`
- `tqdm`
- `openpyxl` for reading `.xlsx` metadata with `pandas.read_excel`
- Jupyter, such as `jupyterlab` or `notebook`

If the environment is missing a package, install it inside `torch-gpu`:

```powershell
conda run -n torch-gpu python -m pip install pandas matplotlib pydicom tqdm openpyxl jupyterlab
```

## LIDC Workflow

Run from the repository root:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_process1.py
conda run -n torch-gpu python data_analysis\LIDC_process2.py --force
conda run -n torch-gpu python data_analysis\LIDC_process3.py
conda run -n torch-gpu python data_analysis\LIDC_process4.py
```

`LIDC_process4.py` defaults to train/validation/test proportions of 70/10/20 and balances benign/malignant and low/intermediate/high-risk labels across splits while keeping patients separated.

## LIDC Outputs

The LIDC scripts write derived files to `data/processed/tables/` and `data/processed/figures/`.

Main table outputs in the current run include:

- `lidc_series_inventory.csv`: 1,362 DICOM series rows.
- `lidc_main_ct_series.csv`: 1,010 selected main CT series rows.
- `lidc_xml_nodules_raw.csv`: 20,362 raw reader annotation rows parsed from XML.
- `lidc_nodule_size_location_morphology.csv`: 20,362 annotation rows with derived size, location, and morphology fields.
- `lidc_reader_annotations_clustered.csv`: 20,362 annotations with lesion-cluster assignments.
- `lidc_annotation_consistency_by_lesion_cluster.csv`: 7,039 lesion clusters with reader-consistency summaries.
- `lidc_roi_consensus_manifest.csv`: ROI-level manifest for nodules >=3 mm approved by at least 3 radiologists.
- `lidc_roi_reader_annotation_manifest.csv`: reader-level ROI annotation manifest.
- `lidc_roi_labeled_manifest.csv`: ROI labels from radiologist malignancy scores.
- `lidc_roi_binary_split_manifest.csv`: benign/malignant split manifest.
- `lidc_roi_multiclass_split_manifest.csv`: low/intermediate/high-risk split manifest.
- `lidc_split_label_balance_diagnostics.csv`: train/validation/test label balance diagnostics.
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
- The current ROI extraction manifest contains 1,593 eligible lesion ROIs.
- The current malignancy labels contain 338 low-risk, 768 intermediate-risk, 284 high-risk, and 203 missing-score ROIs.
- Five annotations have estimated maximum diameter greater than 60 mm and should be reviewed before modeling.

## LUNA16 Workflow

The unzipped LUNA16 subset0-4 source currently lives outside the repository:

```text
E:\LUNA\subset0
E:\LUNA\subset1
E:\LUNA\subset2
E:\LUNA\subset3
E:\LUNA\subset4
E:\LUNA\annotations.csv
```

Run the preprocessing manifest/de-identification check:

```powershell
conda run -n torch-gpu python data_analysis\LUNA_process.py --manifest-only
conda run -n torch-gpu python data_analysis\LUNA_process2.py --manifest-only
```

Run a small preprocessing validation:

```powershell
conda run -n torch-gpu python data_analysis\LUNA_process.py --max-scans 10 --max-qc-png 2
conda run -n torch-gpu python data_analysis\LUNA_process2.py --max-scans 10 --max-qc-png 2
```

Run full subset0-4 preprocessing. Add `--save-volumes` only when there is enough disk space for compressed normalised volumes and lung masks:

```powershell
conda run -n torch-gpu python data_analysis\LUNA_process.py --save-volumes
conda run -n torch-gpu python data_analysis\LUNA_process2.py --save-volumes
```

LUNA outputs are written under:

```text
data/processed/luna_s0_4/
data/processed/luna_s5_9/
```

Main LUNA output tables include:

- `luna_s0_4_manifest.csv`
- `luna_s0_4_processing_manifest.csv`
- `luna_s0_4_deid_map.csv`
- `luna_s0_4_preprocess_summary.csv`
- `luna_s0_4_coord_validation.csv`
- `luna_s0_4_preprocess_report.md`

The subset5-9 script writes the same table names with the `luna_s5_9_` prefix.

## Data Workflow

1. Place original LIDC downloads under `data/raw/LIDC/`; keep large LUNA unzipped subsets on external drives such as `E:\LUNA` for subset0-4 and `D:\LUNA` for subset5-9.
2. Run process scripts or exploratory notebooks from the repository root.
3. Write all derived tables, figures, masks, arrays, train/test splits, and model outputs under `data/processed/` or another ignored output directory.
4. Keep notebooks, documentation, and lightweight configuration in git.
5. Do not commit DICOM files, archives, generated images, large CSV outputs, checkpoints, or trained weights.

## Notes

- `data/raw/.gitkeep` and `data/processed/.gitkeep` preserve the empty folder structure in git.
- `.gitignore` excludes raw data, processed outputs, archives, medical image formats, common model files, Python caches, and notebook checkpoints.
- If the analysis environment stabilizes, add a `requirements.txt` or environment file at the repository root.
