import pandas as pd
import numpy as np
import os
from typing import Optional
def fetch_kepler_lightcurve(
    kic_id: str,
    quarter: str,
    flux_type: str,
) -> pd.DataFrame:

    """
    Dummy Kepler fetch:
    generates a synthetic light curve with one transit.
    Replace this with real Kepler archive access.
    """
    rng = np.random.default_rng(abs(hash(kic_id)) % (2**32))
    time = np.linspace(0, 90, 4000)

    # base flux
    flux = np.ones_like(time)

    # inject transit
    center = 45
    depth = 0.01
    width = 0.4
    flux -= depth * np.exp(-0.5 * ((time - center) / width) ** 2)

    # noise
    flux += rng.normal(0, 0.0007, size=time.shape)

    df = pd.DataFrame(
        {
            "time": time,
            "flux_raw": flux,
            "flux_clean": flux.copy(),
            "quality": np.zeros_like(time, dtype=int),
        }
    )
    return df


def get_tces_for_kic(kic_id: int) -> pd.DataFrame:
    """
    Queries the TCE CSV for a given KIC ID.
    Returns a DataFrame of matching rows.
    """
    csv_path = "preprocessed_data/exoplanet_ml_combined.csv"
    if not os.path.exists(csv_path):
        # Try absolute path if relative fails (fallback)
        csv_path = "/home/jacksparrow/Downloads/astronet-ui/preprocessed_data/exoplanet_ml_combined.csv"
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"TCE CSV not found at {csv_path}")

    # Read CSV (optimization: could cache this if it's large, but for now just read)
    # We only care about specific columns for display and prediction
    cols = [
        "kepid", "tce_plnt_num", "tce_period", "tce_time0bk", "tce_duration", 
        "tce_depth", "tce_model_snr", "av_pred_class"
    ]
    
    try:
        # Read only necessary columns to speed up
        # Note: 'kepid' in CSV might be int or float, so we handle that
        df = pd.read_csv(csv_path, usecols=lambda c: c in cols or c == "kepid")
        
        # Filter by KIC ID
        # Ensure kepid is treated as int for comparison
        df["kepid"] = df["kepid"].astype(int)
        kic_id = int(kic_id)
        
        matches = df[df["kepid"] == kic_id].copy()
        
        if matches.empty:
            return pd.DataFrame()
            
        return matches
        
    except Exception as e:
        print(f"Error reading TCE CSV: {e}")
        return pd.DataFrame()


def load_uploaded_file(
    uploaded_file, file_type: str, flux_type: str
) -> pd.DataFrame:
    """
    For now: CSV only, with columns 'time' and 'flux'.
    You can extend this to FITS parsing later.
    """
    if uploaded_file is None:
        raise ValueError("No file provided")

    if file_type == "CSV":
        df = pd.read_csv(uploaded_file)
        if not {"time", "flux"}.issubset(df.columns):
            raise ValueError("CSV must contain 'time' and 'flux' columns.")
        df = df.rename(columns={"flux": "flux_raw"})
        df["flux_clean"] = df["flux_raw"]
        df["quality"] = 0
        return df

    # Placeholder for FITS parsing
    raise NotImplementedError("FITS support not implemented yet.")


def preprocess_lightcurve(
    df: pd.DataFrame,
    remove_bad_quality: bool,
    sigma_clip: float,
    normalize: bool = True,
) -> pd.DataFrame:
    """Basic sigma-clip + median normalization."""
    clean = df.copy()

    if remove_bad_quality and "quality" in clean.columns:
        clean = clean[clean["quality"] == 0]

    if sigma_clip > 0:
        m = np.nanmean(clean["flux_raw"])
        s = np.nanstd(clean["flux_raw"])
        mask = (clean["flux_raw"] > m - sigma_clip * s) & (
            clean["flux_raw"] < m + sigma_clip * s
        )
        clean = clean[mask]

    if normalize:
        med = np.nanmedian(clean["flux_raw"])
        clean["flux_clean"] = (clean["flux_raw"] - med) / med
    else:
        clean["flux_clean"] = clean["flux_raw"]

    return clean.reset_index(drop=True)
