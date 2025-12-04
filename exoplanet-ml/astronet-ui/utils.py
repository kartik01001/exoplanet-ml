# utils.py
from typing import Dict, Any

def get_kic_metadata(kic_id: str) -> Dict[str, Any]:
    """Dummy Kepler target metadata."""
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
