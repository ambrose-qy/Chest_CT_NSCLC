"""
Minimal Streamlit prototype for LIDC lung nodule risk inference.

Run:

    conda run -n torch-gpu pip install streamlit
    conda run -n torch-gpu streamlit run data_analysis\\LIDC_streamlit_app.py
"""

from __future__ import print_function

import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import torch

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from LIDC_inference import load_model, predict_cases, read_json
from LIDC_full_ct_detection import (
    parse_args as parse_full_ct_args,
    run_full_ct_detection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "data" / "processed" / "model_results" / "lidc_lightning"


def find_run_dirs():
    if not DEFAULT_RESULTS.exists():
        return []
    return sorted([path for path in DEFAULT_RESULTS.rglob("config.json") if "checkpoints" not in path.parts])


def checkpoint_options(run_dir):
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(ckpt_dir.glob("*.ckpt"))


def save_upload(uploaded):
    suffix = Path(uploaded.name).suffix
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded.getbuffer())
    handle.close()
    return Path(handle.name)


def safe_extract_zip(uploaded, destination):
    archive_path = Path(destination) / uploaded.name
    archive_path.write_bytes(uploaded.getbuffer())
    root = Path(destination).resolve()
    with zipfile.ZipFile(str(archive_path), "r") as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError("Unsafe path in uploaded ZIP archive.")
        archive.extractall(str(root))


def find_complete_ct_input(root):
    root = Path(root)
    supported = {".mhd", ".mha", ".nii", ".nrrd"}
    volume_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix.lower() in supported
            or path.name.lower().endswith(".nii.gz")
        )
    ]
    if volume_files:
        return sorted(volume_files)[0]
    directories = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".raw", ".zip"}:
            continue
        directories[path.parent] = directories.get(path.parent, 0) + 1
    if not directories:
        raise ValueError("No CT volume or DICOM series found in the upload.")
    return max(directories, key=directories.get)


def full_ct_interface():
    uploaded = st.file_uploader(
        "Upload a complete CT ZIP or volume",
        type=["zip", "mha", "nii", "gz", "nrrd"],
    )
    threshold = st.sidebar.slider(
        "Nodule probability threshold",
        min_value=0.5,
        max_value=0.99,
        value=0.85,
        step=0.01,
    )
    max_detections = st.sidebar.number_input(
        "Maximum detections",
        min_value=1,
        max_value=256,
        value=64,
        step=1,
    )
    save_rois = st.sidebar.checkbox("Export candidate ROIs", value=False)
    if not st.button("Run complete CT detection"):
        return
    if uploaded is None:
        st.warning("Upload a complete CT first.")
        return

    with tempfile.TemporaryDirectory(prefix="lidc_full_ct_") as temp_dir:
        temp_root = Path(temp_dir)
        if Path(uploaded.name).suffix.lower() == ".zip":
            safe_extract_zip(uploaded, temp_root)
            input_path = find_complete_ct_input(temp_root)
        else:
            input_path = temp_root / uploaded.name
            input_path.write_bytes(uploaded.getbuffer())
        output_dir = temp_root / "output"
        argv = [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--nodule-threshold",
            str(threshold),
            "--max-detections",
            str(int(max_detections)),
        ]
        if save_rois:
            argv.append("--save-rois")
        args = parse_full_ct_args(argv)
        status = st.status("Running complete CT detection", expanded=True)
        try:
            result = run_full_ct_detection(args, progress_callback=status.write)
        except Exception as exc:
            status.update(label="Complete CT detection failed", state="error")
            st.exception(exc)
            return
        status.update(label="Complete CT detection finished", state="complete")

        dataframe = pd.DataFrame(result["rows"])
        st.subheader("Detected Nodules")
        st.dataframe(dataframe, use_container_width=True)
        if Path(result["overview_png"]).exists():
            st.image(str(result["overview_png"]), use_container_width=True)
        st.download_button(
            "Download candidate CSV",
            Path(result["candidate_csv"]).read_bytes(),
            file_name="full_ct_nodule_candidates.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download metadata JSON",
            Path(result["metadata_json"]).read_bytes(),
            file_name="full_ct_detection_metadata.json",
            mime="application/json",
        )
        st.metric("Retained candidates", result["candidate_count"])


def main():
    st.set_page_config(page_title="LIDC Lung Nodule Risk Prototype", layout="wide")
    st.title("LIDC Lung Nodule Risk Prototype")
    mode = st.sidebar.radio(
        "Inference mode",
        ["Complete CT detection", "ROI classification"],
    )
    if mode == "Complete CT detection":
        full_ct_interface()
        return

    run_configs = find_run_dirs()
    run_labels = [str(path.parent.relative_to(PROJECT_ROOT)) for path in run_configs]
    selected_index = st.sidebar.selectbox("Model run", list(range(len(run_configs))), format_func=lambda idx: run_labels[idx] if run_labels else "")
    if not run_configs:
        st.error("No config.json files found under {}".format(DEFAULT_RESULTS))
        return

    config_path = run_configs[selected_index]
    run_dir = config_path.parent
    checkpoints = checkpoint_options(run_dir)
    if not checkpoints:
        st.error("No checkpoints found under {}".format(run_dir / "checkpoints"))
        return
    checkpoint = st.sidebar.selectbox("Checkpoint", checkpoints, format_func=lambda path: path.name)

    uploaded = st.file_uploader("Upload a 3D ROI .npz file or a 2D DICOM file", type=["npz", "dcm", "dicom"])
    batch_file = st.file_uploader("Optional batch CSV", type=["csv"])
    run_button = st.button("Run inference")

    if not run_button:
        return

    config = read_json(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with st.spinner("Loading model..."):
        model = load_model(checkpoint, config, device)

    rows = []
    temp_paths = []
    if batch_file is not None:
        batch_path = save_upload(batch_file)
        temp_paths.append(batch_path)
        rows = pd.read_csv(batch_path).fillna("").to_dict(orient="records")
    elif uploaded is not None:
        input_path = save_upload(uploaded)
        temp_paths.append(input_path)
        if input_path.suffix.lower() == ".npz":
            rows = [{"case_id": Path(uploaded.name).stem, "volume_path": str(input_path)}]
        else:
            rows = [{"case_id": Path(uploaded.name).stem, "dicom_path": str(input_path)}]
    else:
        st.warning("Upload a case or batch CSV first.")
        return

    with st.spinner("Running inference..."):
        predictions = predict_cases(model, rows, config, device)

    df = pd.DataFrame(predictions)
    st.subheader("Prediction Results")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="lidc_inference_predictions.csv",
        mime="text/csv",
    )

    if len(predictions) == 1:
        first = predictions[0]
        metric_cols = st.columns(3)
        metric_cols[0].metric("Predicted label", first.get("predicted_label", ""))
        metric_cols[1].metric("Confidence", "{:.1%}".format(float(first.get("confidence", 0.0))))
        metric_cols[2].metric("Task", first.get("task", ""))

    for path in temp_paths:
        try:
            path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
