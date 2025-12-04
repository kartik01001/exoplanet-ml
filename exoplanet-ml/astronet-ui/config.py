# config.py
import os

# API Configuration
API_URL = "http://localhost:6666/predict"

# Model Presets - maps display name to model configuration
MODEL_PRESETS = {
    "CNN": {
        "model_class": "AstroCNNModel",
        "model_dir": "/tf/celeste/exoplanet-ml/exoplanet-ml/astronet/model",
        "kepler_data_dir": "/tf/celeste/exoplanet-ml/exoplanet-ml/astronet/kepler",
        "description": "Convolutional Neural Network",
        "available_configs": ["local_global"]
    },
    "FC": {
        "model_class": "AstroFCModel",
        "model_dir": "/tf/celeste/exoplanet-ml/exoplanet-ml/astronet/model_fc",
        "kepler_data_dir": "/tf/celeste/exoplanet-ml/exoplanet-ml/astronet/kepler",
        "description": "Feature Pyramid Model",
        "available_configs": ["base", "local_global"]
    },
    "WN": {
        "model_class": "AstroWaveNetModel",
        "model_dir": "/tf/celeste/exoplanet-ml/exoplanet-ml/astronet/model_awn",
        "kepler_data_dir": "/tf/celeste/exoplanet-ml/exoplanet-ml/astronet/kepler",
        "description": "WaveNet Model",
        "available_configs": ["base"]
    }
}

# Default Model Settings
DEFAULT_MODEL_NAME = "CNN"  # Display name
DEFAULT_CONFIG_NAME = "local_global"

# Default Values
DEFAULT_KIC_ID = "11442793"
DEFAULT_SIGMA_CLIP = 3.0
DEFAULT_FLUX_TYPE = "PDCSAP"
DEFAULT_QUARTER = "Any"
DEFAULT_PERIOD = 14.44912
DEFAULT_T0 = 2.2
DEFAULT_DURATION = 0.11267
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_OUTPUT_FILENAME = "astronet_result"

# UI Constants
PAGE_TITLE = "Celeste"
LAYOUT = "wide"
