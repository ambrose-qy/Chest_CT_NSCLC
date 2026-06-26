# Chest_CT_NSCLC

End-to-end chest CT workspace for pulmonary nodule preprocessing, 2D/3D classification, LUNA16-domain generalisation, explainability, and complete-CT automatic nodule detection.

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
|   |-- LIDC_process5.py
|   |-- LIDC_process6.py
|   |-- LIDC_2d_resnet.py
|   |-- LIDC_2d_densenet.py
|   |-- LIDC_3d_resnet.py
|   |-- LIDC_3d_vnet.py
|   |-- LIDC_evaluate_lightning.py
|   |-- LIDC_tune_lightning.py
|   |-- LIDC_compare_lightning_experiments.py
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
- `data_analysis/LIDC_process5.py` exports compressed `.npz` 3D nodule ROI volume files as fixed 64 x 64 x 64 voxel-coordinate crops centred on the ROI manifest nodule centre, with label manifests, preprocessing QC, and outlier-handling reports.
- `data_analysis/LIDC_process6.py` writes missing-data, outlier, and conservative imputation-planning tables for report/model use.
- `data_analysis/LIDC_2d_resnet.py`, `LIDC_2d_densenet.py`, `LIDC_3d_resnet.py`, and `LIDC_3d_vnet.py` are model-named training entry points that share the same preprocessing, augmentation, logging, early stopping, scheduling, and evaluation code.
- `data_analysis/LIDC_evaluate_lightning.py` evaluates a Lightning checkpoint on the independent test set and writes metrics, predictions, confusion matrix, and ROC data.
- `data_analysis/LIDC_tune_lightning.py` writes or executes hyperparameter tuning plans and records the best configuration.
- `data_analysis/LIDC_compare_lightning_experiments.py` aggregates completed Lightning runs into a model-comparison report.
- `data_analysis/LUNA_build_labeled_external_rois.py` matches LUNA16 annotations to LIDC lesion labels and exports label-preserving 64 x 64 x 64 ROIs directly from raw MHD scans.
- `data_analysis/LIDC_evaluate_luna_generalization.py` evaluates all selected 2D and 3D models on the LUNA16-domain labeled ROIs.
- `data_analysis/LUNA_prepare_candidate_dataset.py` builds a balanced 32 x 32 x 32 LUNA16 candidate dataset for nodule/non-nodule false-positive reduction.
- `data_analysis/LUNA_train_candidate_detector.py` trains the lightweight 3D CBAM candidate detector and exports its checkpoint, metrics, confusion matrix, and training curve.
- `data_analysis/LUNA_evaluate_candidate_detector.py` reproduces held-out candidate-detector metrics and exports per-candidate probabilities, confusion matrix, and ROC curve.
- `data_analysis/LIDC_full_ct_detection.py` runs complete-CT lung segmentation, high-recall proposal generation, CBAM false-positive reduction, malignancy classification, and risk stratification.
- `data_analysis/LUNA_evaluate_full_ct_proposals.py` evaluates complete-CT proposal sensitivity and post-CBAM detection sensitivity/false positives per scan on held-out LUNA16 subsets.
- `data_analysis/LUNA_analyze_full_ct_detection.py` identifies annotation-level proposal and candidate-filter misses and summarises sensitivity by nodule diameter.
- `data_analysis/LUNA_build_completion_report.py` consolidates external generalisation, candidate-detector, and complete-CT results into a Chinese Markdown report.
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
conda run -n torch-gpu python data_analysis\LIDC_process5.py
conda run -n torch-gpu python data_analysis\LIDC_process6.py
```

`LIDC_process4.py` defaults to train/validation/test proportions of 70/10/20 and balances benign/malignant and low/intermediate/high-risk labels across splits while keeping patients separated.
`LIDC_process5.py` writes fixed-shape 3D ROI volumes as compressed `.npz` files under `data/processed/lidc_roi_3d/volumes/`; use `--max-rois 20 --force` for a small validation run before full extraction.

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
- `lidc_roi_3d_volume_manifest.csv`: standardised 3D ROI volume file manifest.
- `lidc_roi_3d_label_manifest.csv`: volume-level binary and multiclass label file.
- `lidc_roi_3d_preprocessing_qc.csv`: before/after preprocessing statistics for each exported ROI volume.
- `lidc_roi_3d_outlier_report.csv`: outliers, failures, and applied handling during 3D ROI preprocessing.
- `lidc_roi_3d_preprocessing_summary.csv`: dataset-level summary for the standardised 3D ROI export.
- `data/processed/lidc_roi_3d/volumes/*.npz`: compressed standardised 3D ROI volumes with array key `volume`.
- `lidc_data_quality_issue_summary.csv`: missing-data and outlier issue summary with recommended handling.
- `lidc_demographics_imputation_plan.csv`: age/sex handling plan for reports and model covariates.
- `lidc_patient_demographics_model_ready.csv`: patient demographics with age imputation fields, sex Unknown category, and missingness flags.
- `lidc_annotation_outlier_rows.csv`: raw XML annotation rows with spacing, diameter, and score issues.
- `lidc_roi_quality_issue_rows.csv`: ROI-level missing-label, low-consistency, and discordant-label issue rows.

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

## LIDC Report Checklist Artifacts

- Data exploration and analysis report: use `lidc_patient_*`, `lidc_nodule_*`, `lidc_annotation_consistency_*`, `lidc_multireader_consistency_summary.csv`, and the LIDC figures under `data/processed/figures/`.
- Standardised lung nodule ROI dataset: use `data/processed/lidc_roi_3d/volumes/*.npz`, `lidc_roi_3d_volume_manifest.csv`, and `lidc_roi_3d_label_manifest.csv`.
- Data quality assessment report: use `lidc_processed_quality_control_report.csv`, `lidc_data_quality_issue_summary.csv`, `lidc_demographics_imputation_plan.csv`, `lidc_annotation_outlier_rows.csv`, `lidc_roi_quality_issue_rows.csv`, `lidc_roi_3d_preprocessing_qc.csv`, `lidc_roi_3d_outlier_report.csv`, and `lidc_roi_3d_preprocessing_summary.csv`.

## LIDC Lightning Model Workflow

Install Lightning in the training environment if needed:

```powershell
conda run -n torch-gpu pip install lightning
```

Build the full 2D maximum-slice manifest once:

```powershell
conda run -n torch-gpu python data_analysis\lidc_2d_slices.py --output data\processed\tables\lidc_roi_binary_2d_slice_manifest.csv
```

Train 2D baselines:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LIDC_2d_resnet.py --epochs 50 --batch-size 16
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LIDC_2d_densenet.py --epochs 50 --batch-size 16
```

Train 3D baselines:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LIDC_3d_resnet.py --epochs 60 --batch-size 4
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LIDC_3d_vnet.py --epochs 60 --batch-size 4
```

For ordinary tuning, edit `configs/lidc_training.yaml` and run the model-named file directly without command-line overrides. The standalone configuration controls the task, normalization, augmentation, class weighting, attention, fusion, dropout, and gradient clipping for all four model profiles. The training functions can still be imported from another script, for example `from LIDC_3d_resnet import train_3d_resnet`, and dictionary overrides remain supported for programmatic experiments.

The default normalization mode is `normalization_mean = "auto"` and `normalization_std = "auto"`, which computes mean/std from the training split and applies the same statistics to train/validation/test data. Current augmentations include rotation, flipping, scaling, Gaussian noise, intensity shift, contrast jitter, and cutout. Class weighting is configured per model profile; use `class_weight_mode = "custom"` with `custom_class_weights = "w0,w1"` or `"w0,w1,w2"` for manual binary or multiclass weights.

Attention and fusion experiments are available through the shared model factory. Use `--attention cbam` for the recommended CBAM module, or `--attention se` for an SE comparison. For 3D models, `--fusion multiscale` enables multiscale feature fusion in ResNet3D and `--fusion multiview` enables axial/coronal/sagittal multi-view fusion. The model-named defaults now use CBAM, with 3D ResNet defaulting to multiscale fusion.

Audit the selected report checkpoints at configuration and state-dictionary level:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_verify_cbam_results.py
```

The command returns a non-zero exit status if any selected checkpoint is not
CBAM-compliant. Use `--allow-incomplete` only when producing an interim audit.

Each run prints and records the active task and split source, for example `Task: binary using split column 'binary_split'`. For 2D and 3D training, set `task` to `binary` or `multiclass`; the saved `config.json` records `task`, `split_column`, and `label_column`. The 2D maximum-slice workflow writes task-specific binary and multiclass manifests.

Each run writes checkpoints, `config.json`, `hparams.json`, `best_config.json`, `test_metrics.json`, `test_confusion_matrix.csv`, `test_confusion_matrix.json`, `test_confusion_matrix.png`, and Lightning CSV logs under `data/processed/model_results/lidc_lightning/<2d-or-3d>/<model-family>/<run-name>/`. The CSV logs include learning rate, loss, accuracy, precision, recall, F1, AUC-ROC, label counts, dropout and clipping hyperparameters, average absolute gradient, gradient L2 norm, maximum absolute gradient, and parameter L2 norm.

Grad-CAM and Grad-CAM++ outputs are written after training when `enable_grad_cam` is enabled. The visualisation step prioritises incorrectly predicted test cases, writes overlays, records attention peak/bounding-box locations, and summarises failure patterns such as small nodules, calcification-marked false positives, low reader consistency, and false negative/false positive class transitions.

Consolidate the selected model error cases and restore calcification metadata directly from the reader-level ROI annotations without retraining:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_failure_mode_analysis.py
```

On Windows, if `conda run` fails while printing Lightning progress output because of console encoding, run the environment Python directly instead:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LIDC_3d_resnet.py
```

Evaluate a saved checkpoint:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_evaluate_lightning.py --checkpoint <best.ckpt> --config <run_dir>\config.json
```

Aggregate completed runs:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_compare_lightning_experiments.py
```

Plan or run the full required baseline matrix covering 2D ResNet, 2D DenseNet, 3D ResNet, and 3D VNet for both binary and multiclass tasks:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_run_required_experiments.py
conda run -n torch-gpu python data_analysis\LIDC_run_required_experiments.py --execute
```

Plan or run ablations for a selected model/task. The default ablations include no augmentation, individual augmentation removals, no class weights, scheduler variants, and no gradient clipping:

```powershell
conda run -n torch-gpu python data_analysis\LIDC_ablation_lightning.py --input-dim 3d --model resnet3d --task multiclass
conda run -n torch-gpu python data_analysis\LIDC_ablation_lightning.py --input-dim 3d --model resnet3d --task multiclass --execute
```

Build label-matched LUNA16-domain ROIs and evaluate every selected model:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LUNA_build_labeled_external_rois.py
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LIDC_evaluate_luna_generalization.py
```

LUNA16 is derived from LIDC-IDRI, so this is a cross-preprocessing and cross-data-distribution generalisation test, not an independent-hospital validation. The exporter performs one-to-one physical-coordinate matching and retains the original patient-level LIDC test assignment before computing accuracy, precision, recall, F1, AUC-ROC, confusion matrices, and ROC curves.

Current generated outputs are written to:

```text
data/processed/luna16_labeled_external_rois/
data/processed/model_reports/lidc_luna16_generalization/
```

## Complete CT Nodule Detection

Build the LUNA16 candidate patch dataset and train the 3D CBAM false-positive reduction model:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LUNA_prepare_candidate_dataset.py
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LUNA_train_candidate_detector.py
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LUNA_evaluate_candidate_detector.py
```

The candidate model uses `subset0-7` for training, `subset8` for validation, and `subset9` for independent testing. Its inputs are 32 x 32 x 32 HU-normalised patches. CBAM is applied to the deepest 3D feature map to preserve channel and spatial attention without the excessive cost of high-resolution 3D spatial attention.

Edit `configs/luna16_detection.yaml` to maintain candidate-training hyperparameters, proposal settings, detection thresholds, selected checkpoints, and LUNA16 evaluation operating points without changing Python source code.

Run automatic detection and risk classification on a DICOM series directory or a supported volume file:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LIDC_full_ct_detection.py `
  --input D:\path\to\dicom_series `
  --output-dir data\processed\model_reports\full_ct_detection_case
```

The pipeline performs:

1. DICOM/MHD/NIfTI loading and 1 mm isotropic resampling.
2. Robust lung parenchyma segmentation.
3. Multi-scale 3D LoG, axial connected-component, and 1 mm small-nodule local-maximum proposal generation.
4. LUNA16-trained 3D CBAM candidate scoring and false-positive reduction.
5. 3D ResNet binary malignancy prediction and three-class risk stratification.
6. CSV, metadata JSON, ROI NPZ, and annotated overview export.

Evaluate the complete-CT detector on all held-out `subset9` scans:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LUNA_evaluate_full_ct_proposals.py `
  --subsets subset9 `
  --output-dir data\processed\model_reports\luna16_full_ct_detection_subset9
```

The report separates proposal sensitivity from final detection sensitivity and writes multiple nodule-probability operating points with sensitivity, detections per scan, and false positives per scan. The default `0.85` threshold was selected from the complete `subset9` evaluation as a practical sensitivity/false-positive operating point; it remains configurable for sensitivity-first or specificity-first use.

Current held-out `subset9` results: candidate-classifier AUC `0.9870` and F1 `0.8992`; complete-CT proposal sensitivity `0.8095`; final detection sensitivity `0.6667` with `7.60` false positives per scan at threshold `0.85`. The 3-6 mm proposal sensitivity is `0.8222`.

After evaluation, generate annotation-level failure analysis and the consolidated report:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LUNA_analyze_full_ct_detection.py
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe data_analysis\LUNA_build_completion_report.py
```

Launch the interactive ROI/complete-CT interface:

```powershell
C:\Users\Ambro\.conda\envs\torch-gpu\python.exe -m streamlit run data_analysis\LIDC_streamlit_app.py
```

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

The LIDC external-validation ROI exporter requires those saved LUNA full-volume `.npz` files.

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
- `requirements.txt` and `environment.yml` record the verified `torch-gpu` dependency versions used by the current pipeline.
