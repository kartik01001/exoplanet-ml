"""FastAPI server for AstroNet predictions with optional feature plots.

- Builds a TensorFlow Estimator using astronet.models and estimator_util.
- Processes Kepler light curves using astronet.data.preprocess (same as predict.py).
- Exposes /predict endpoint that returns:
    - probability: float
    - global_view_base64: optional base64 PNG
    - local_view_base64: optional base64 PNG
- Exposes /metrics endpoint that returns:
    - training/validation metrics (accuracy, loss)
    - confusion matrix image
"""

from typing import Optional, List, Dict, Any
import os
import io
import base64
import subprocess
import glob

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import matplotlib.pyplot as plt

from astronet import models
from astronet.data import preprocess
from astronet.util import estimator_util
from tf_util import config_util, configdict


# --------------------- Request/response schemas ---------------------


class TCERequest(BaseModel):
    # Model + config
    model: str = "AstroCNNModel"
    config_name: Optional[str] = "local_global"
    config_json: Optional[str] = None

    # Paths inside the container
    model_dir: str
    kepler_data_dir: str

    # TCE parameters
    kepler_id: int
    period: float
    t0: float
    duration: float

    # JSON flag: if true, act like --output_image_file was passed AND
    # also return the image in base64 in the response.
    include_image: bool = False

    # Optional: control image format; used only when include_image is True.
    image_format: str = "png"


class PredictionResponse(BaseModel):
    probability: float
    global_view_base64: Optional[str] = None
    local_view_base64: Optional[str] = None


class MetricsRequest(BaseModel):
    model_dir: str


class MetricPoint(BaseModel):
    step: int
    value: float


class MetricsResponse(BaseModel):
    train_loss: List[MetricPoint] = []
    train_accuracy: List[MetricPoint] = []
    val_loss: List[MetricPoint] = []
    val_accuracy: List[MetricPoint] = []
    confusion_matrix_base64: Optional[str] = None


# --------------------------- FastAPI app -----------------------------


app = FastAPI(title="AstroNet Exoplanet API")


# ---------------------- Internal helper functions -------------------


def _ensure_kepler_data(kepler_id: int):
    """
    Checks if data for the given Kepler ID exists.
    If not, parses get_kepler.sh to find the download command and runs it.
    """
    kic_str = f"{kepler_id:09d}"
    
    # Try to find get_kepler.sh in the current directory or same dir as this script
    script_path = "get_kepler.sh"
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(__file__), "get_kepler.sh")
    
    if not os.path.exists(script_path):
        print(f"Warning: {script_path} not found. Cannot auto-download data.")
        return

    cmd = None
    with open(script_path, "r") as f:
        for line in f:
            if kic_str in line and line.strip().startswith("wget"):
                cmd = line.strip()
                break
    
    if not cmd:
        raise HTTPException(status_code=404, detail=f"Kepler ID {kepler_id} not found in catalog (get_kepler.sh).")
        
    # Check if we need to download
    # The command typically has -P <output_dir>
    parts = cmd.split()
    output_dir = None
    if "-P" in parts:
        try:
            idx = parts.index("-P")
            if idx + 1 < len(parts):
                output_dir = parts[idx+1]
        except ValueError:
            pass
            
    if output_dir and os.path.exists(output_dir) and os.listdir(output_dir):
        # Data seems to exist
        return

    print(f"Downloading data for Kepler ID {kepler_id}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Failed to download data: {e}")


def _build_estimator_and_config(req: TCERequest):
    """Build Estimator and model input config for a given request."""
    model_class = models.get_model_class(req.model)

    # Exactly one of config_name / config_json must be provided.
    assert (req.config_name is None) != (req.config_json is None), (
        "Exactly one of config_name or config_json is required"
    )

    config_data = (
        models.get_model_config(req.model, req.config_name)
        if req.config_name
        else config_util.parse_json(req.config_json)
    )
    config = configdict.ConfigDict(config_data)

    estimator = estimator_util.create_estimator(
        model_class, config.hparams, model_dir=req.model_dir
    )

    return estimator, config.inputs.features


def _plot_single_feature(feature_data, title, image_format):
    """Helper to plot a single feature and return base64 string."""
    # Rectangular figure size (width=8, height=4)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(feature_data[0], ".")
    ax.set_title(title)
    ax.set_xlabel("Bucketized Time (days)")
    ax.set_ylabel("Normalized Flux")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format=image_format, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _process_tce_and_maybe_plot(req: TCERequest, feature_config):
    """Process TCE into features and optionally create/save/encode plots."""
    # This mirrors the check in predict.py
    if not {"global_view", "local_view"}.issuperset(feature_config.keys()):
        raise ValueError("Only 'global_view' and 'local_view' features are supported.")

    # 1) Read and preprocess the light curve (same functions as predict.py)
    all_time, all_flux = preprocess.read_light_curve(
        req.kepler_id, req.kepler_data_dir
    )
    time, flux = preprocess.process_light_curve(all_time, all_flux)
    time, flux = preprocess.phase_fold_and_sort_light_curve(
        time, flux, req.period, req.t0
    )

    # 2) Generate global and local views
    features = {}

    if "global_view" in feature_config:
        global_view = preprocess.global_view(time, flux, req.period)
        features["global_view"] = np.expand_dims(global_view, 0).astype(np.float32)

    if "local_view" in feature_config:
        local_view = preprocess.local_view(time, flux, req.period, req.duration)
        features["local_view"] = np.expand_dims(local_view, 0).astype(np.float32)

    global_b64: Optional[str] = None
    local_b64: Optional[str] = None

    # 3) If include_image is True, generate separate plots
    if req.include_image and features:
        if "global_view" in features:
            global_b64 = _plot_single_feature(features["global_view"], "Global View", req.image_format)
        
        if "local_view" in features:
            local_b64 = _plot_single_feature(features["local_view"], "Local View", req.image_format)

    # The Estimator in predict.py expects a dict: {"time_series_features": features}
    input_features = {"time_series_features": features}
    return input_features, global_b64, local_b64


def _extract_scalars_from_events(event_dir: str, tags: List[str]) -> Dict[str, List[MetricPoint]]:
    """Extracts scalar values for given tags from event files in a directory."""
    data = {tag: [] for tag in tags}
    
    event_files = glob.glob(os.path.join(event_dir, "events.out.tfevents*"))
    # Sort by modification time to process in order
    event_files.sort(key=os.path.getmtime)
    
    for event_file in event_files:
        try:
            for e in tf.train.summary_iterator(event_file):
                for v in e.summary.value:
                    if v.tag in tags:
                        data[v.tag].append(MetricPoint(step=e.step, value=v.simple_value))
        except Exception as e:
            print(f"Error reading event file {event_file}: {e}")
            continue
            
    return data


def _extract_image_from_events(event_dir: str, tag_keyword: str) -> Optional[str]:
    """Extracts the last image matching tag_keyword from event files."""
    event_files = glob.glob(os.path.join(event_dir, "events.out.tfevents*"))
    event_files.sort(key=os.path.getmtime)
    
    last_image_b64 = None
    
    for event_file in event_files:
        try:
            for e in tf.train.summary_iterator(event_file):
                for v in e.summary.value:
                    if tag_keyword in v.tag and v.image.encoded_image_string:
                        # TF stores images as encoded bytes (usually PNG)
                        # We just need to base64 encode it for the API response
                        last_image_b64 = base64.b64encode(v.image.encoded_image_string).decode("ascii")
        except Exception:
            continue
            
    return last_image_b64


# ---------------------------- Endpoint -------------------------------


@app.post("/predict", response_model=PredictionResponse)
def predict_tce(req: TCERequest):
    """Generate a prediction for a TCE using a trained AstroNet model."""
    
    # Ensure data is available
    _ensure_kepler_data(req.kepler_id)
    
    estimator, feature_config = _build_estimator_and_config(req)
    input_features, global_b64, local_b64 = _process_tce_and_maybe_plot(req, feature_config)

    def input_fn():
        # Same dataset construction as in predict.py
        return tf.data.Dataset.from_tensors(input_features)

    # We expect a single prediction (same loop as in predict.py but unrolled)
    pred_iter = estimator.predict(input_fn)
    pred = next(pred_iter)
    assert len(pred) == 1

    return PredictionResponse(
        probability=float(pred[0]),
        global_view_base64=global_b64,
        local_view_base64=local_b64,
    )


@app.post("/metrics", response_model=MetricsResponse)
def get_metrics(req: MetricsRequest):
    """Extracts training and validation metrics from TensorBoard event files."""
    
    if not os.path.exists(req.model_dir):
        raise HTTPException(status_code=404, detail="Model directory not found")

    # 1. Training Metrics (usually in model_dir root)
    # Note: We found that the training event file mainly contains "loss".
    train_data = _extract_scalars_from_events(
        req.model_dir, 
        tags=["loss", "accuracy", "accuracy/accuracy"]
    )
    
    # Normalize tag names
    train_loss = train_data.get("loss", [])
    train_acc = train_data.get("accuracy", []) + train_data.get("accuracy/accuracy", [])
    
    # 2. Validation Metrics (usually in model_dir/eval_val or similar)
    # We'll look for 'eval_val' or 'eval'
    val_dir = os.path.join(req.model_dir, "eval_val")
    if not os.path.exists(val_dir):
        val_dir = os.path.join(req.model_dir, "eval")
    
    val_loss = []
    val_acc = []
    
    if os.path.exists(val_dir):
        val_data = _extract_scalars_from_events(
            val_dir, 
            tags=["loss", "accuracy", "accuracy/accuracy"]
        )
        val_loss = val_data.get("loss", [])
        val_acc = val_data.get("accuracy", []) + val_data.get("accuracy/accuracy", [])
        
    # 3. Confusion Matrix (Image)
    # Often stored in validation dir
    cm_b64 = None
    if os.path.exists(val_dir):
        cm_b64 = _extract_image_from_events(val_dir, "confusion_matrix")
        
    return MetricsResponse(
        train_loss=train_loss,
        train_accuracy=train_acc,
        val_loss=val_loss,
        val_accuracy=val_acc,
        confusion_matrix_base64=cm_b64
    )


@app.post("/ensure_data")
def ensure_data_endpoint(kepler_id: int):
    """Triggers the download of Kepler data for a given ID."""
    _ensure_kepler_data(kepler_id)
    return {"status": "ok", "message": f"Data for {kepler_id} is ready."}
