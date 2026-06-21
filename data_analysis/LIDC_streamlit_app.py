"""
Minimal Streamlit prototype for LIDC lung nodule risk inference.

Run:

    conda run -n torch-gpu pip install streamlit
    conda run -n torch-gpu streamlit run data_analysis\\LIDC_streamlit_app.py
"""

from __future__ import print_function

import tempfile
from pathlib import Path

import pandas as pd
import torch

import streamlit as st

from LIDC_inference import load_model, predict_cases, read_json


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


def main():
    st.set_page_config(page_title="LIDC Lung Nodule Risk Prototype", layout="wide")
    st.title("LIDC Lung Nodule Risk Prototype")

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
