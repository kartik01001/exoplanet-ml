# plotter.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

def plot_lightcurve(
    proc_lc: pd.DataFrame,
    raw_lc: Optional[pd.DataFrame] = None,
    show_raw: bool = False,
    show_clean: bool = True,
    show_transits: bool = False,
    pred: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """
    Plots the light curve.
    """
    fig, ax = plt.subplots()
    time = proc_lc["time"].values

    if show_raw and raw_lc is not None:
        ax.scatter(
            raw_lc["time"].values,
            raw_lc["flux_raw"].values,
            s=2,
            alpha=0.25,
            label="Raw",
        )

    if show_clean:
        ax.scatter(
            proc_lc["time"].values,
            proc_lc["flux_clean"].values,
            s=2,
            alpha=0.8,
            label="Cleaned",
        )

    if (
        pred is not None
        and show_transits
        and "transit_mask" in pred
        and any(pred["transit_mask"])
    ):
        mask = np.array(pred["transit_mask"], dtype=bool)
        ax.scatter(
            time[mask],
            proc_lc["flux_clean"].values[mask],
            s=10,
            label="Predicted transit window",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel("Flux")
    ax.legend()
    return fig


def plot_phase_folded(
    proc_lc: pd.DataFrame,
    period: float
) -> plt.Figure:
    """
    Plots the phase-folded light curve.
    """
    time = proc_lc["time"].values
    flux = proc_lc["flux_clean"].values
    phase = (time % period) / period

    fig, ax = plt.subplots()
    ax.scatter(phase, flux, s=3, alpha=0.7)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalized Flux")
    return fig
