# backend.py
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import requests
import base64
import config

#BACKEND_URL = "http://localhost:7777/predict"  # FastAPI in Docker


def run_model_via_api(
    model_class: str,
   # df: pd.DataFrame,
    config_name: Optional[str],
    config_json: Optional[str],
    model_dir: str,
    kepler_data_dir: str,
    kic_id: str,
    period: float,
    t0: float,
    duration: float,
    #confidence_threshold: float,
    include_image: bool = True,
    image_format: str = "png",
) -> Dict[str, Any]:
    """
    Call the Astronet backend running in Docker via FastAPI.
    Matches the payload structure from streamlit_app.py.

    Expects the backend /predict endpoint to return JSON like:
      {
        #from csv
        "label": "...",
        "planet_prob": 0.91,
        "other_probs": {"non_planet": 0.09},
        "transit_mask": [...],
        "period": 14.44912,
        "metadata": { ... } 

        #from api
        "probability": 0.91,
        "image_base64": "..." (optional, if include_image=True)
      }
    """

    payload = {
        "model": model_class,
        "config_name": config_name,
        "config_json": config_json,
        "model_dir": model_dir,
        "kepler_data_dir": kepler_data_dir,
        "kepler_id": int(kic_id),
        "period": float(period),
        "t0": float(t0),
        "duration": float(duration),
        "include_image": bool(include_image),
        "image_format": image_format,
    }

    # # ========== DEBUG: REMOVE THIS SECTION LATER ==========
    # import streamlit as st
    # with st.expander("🔍 DEBUG: TCE Request Payload", expanded=False):
    #     st.json(payload)
    #     st.caption(f"Sending to: `{config.API_URL}`")
    # # ======================================================

    # Make API call to backend (matching streamlit_app.py implementation)
    resp = requests.post(config.API_URL, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            # Try to get detailed error message from API
            detail = resp.json().get("detail")
            if detail:
                raise Exception(f"API Error: {detail}") from e
        except ValueError:
            pass # Not JSON
        raise e
    data = resp.json()
    
    # Extract probability from API response
    prob = data.get("probability", 0.0)
    
    # Determine label based on probability
    label = "Planet" if prob >= 0.5 else "Not Planet"
    
    # Decode base64 images if present
    global_bytes = None
    local_bytes = None
    
    global_b64 = data.get("global_view_base64")
    if global_b64:
        global_bytes = base64.b64decode(global_b64)
        
    local_b64 = data.get("local_view_base64")
    if local_b64:
        local_bytes = base64.b64decode(local_b64)

    result: Dict[str, Any] = {
        "label": label,
        "probabilities": {
            "planet": prob,
            "non_planet": 1.0 - prob,
        },
        "global_bytes": global_bytes,
        "local_bytes": local_bytes,
        "period": period,  # Return the input period
        "metadata": {},
    }
    return result


def get_model_metrics(model_dir: str) -> Dict[str, Any]:
    """
    Fetches training and validation metrics from the backend API.
    """
    payload = {"model_dir": model_dir}
    
    try:
        resp = requests.post(f"{config.API_URL.replace('/predict', '/metrics')}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching metrics: {e}")
        return {}


def ensure_backend_data(kic_id: int):
    """
    Triggers the backend to ensure Kepler data is downloaded.
    """
    try:
        # Use a dummy payload or query param depending on how we defined it.
        # We defined it as query param: ensure_data_endpoint(kepler_id: int)
        resp = requests.post(f"{config.API_URL.replace('/predict', '/ensure_data')}?kepler_id={kic_id}", timeout=300)
        resp.raise_for_status()
    except Exception as e:
        print(f"Error ensuring data: {e}")
        raise e


def get_kic_metadata(kic_id: str) -> Dict[str, Any]:
    """
    Fallback dummy metadata if backend does not provide it.
    Prefer metadata from the backend prediction when available.
    """
    if not kic_id:
        kic_id = "Unknown"
    return {
        "KIC": kic_id,
        "Kepmag": 13.4,
        "Teff (K)": 5800,
        "log g": 4.3,
        "Radius (R☉)": 1.1,
        "Quarters": "Q1–Q4",
        "# of data": 34912,
    }
