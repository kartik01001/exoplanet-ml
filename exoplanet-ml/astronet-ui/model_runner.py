# model_runner.py
import pandas as pd
from typing import Dict, Any, Optional
import backend  # Use the existing backend.py

def run_prediction(
    model_class: str,
    config_name: Optional[str],
    config_json: Optional[str],
    model_dir: str,
    kepler_data_dir: str,
    df: pd.DataFrame,
    kic_id: str,
    period: float,
    t0: float,
    duration: float,
    include_image: bool = True,
) -> Dict[str, Any]:
    """
    Runs prediction using the backend FastAPI client.
    """
    return backend.run_model_via_api(
        model_class=model_class,
        config_name=config_name,
        config_json=config_json,
        model_dir=model_dir,
        kepler_data_dir=kepler_data_dir,
        kic_id=kic_id,
        period=period,
        t0=t0,
        duration=duration,
        include_image=include_image,
    )
