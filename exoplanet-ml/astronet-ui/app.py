# app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Import modular components
import config
import utils
import data_manager
import model_runner
import plotter
import ui_components
import backend

# ------------------------------------------------------------
# Streamlit app init
# ------------------------------------------------------------

st.set_page_config(
    page_title=config.PAGE_TITLE,
    layout=config.LAYOUT,
)

# Apply custom CSS
ui_components.apply_custom_css()

# ----- session state init -----
ss = st.session_state
ss.setdefault("raw_lc", None)        # raw lightcurve DataFrame
ss.setdefault("proc_lc", None)       # preprocessed lightcurve
ss.setdefault("prediction", None)    # prediction dict
ss.setdefault("kic_id", "")
# ss.setdefault("history", [])         # list of dicts (commented out)
ss.setdefault("status", "Idle")

# ------------------------------------------------------------
# ------------------------------------------------------------
# Main Inputs (Top)
# ------------------------------------------------------------

inputs = ui_components.render_main_inputs(ss)

# ------------------------------------------------------------
# Data Fetching Logic
# ------------------------------------------------------------

status_placeholder = st.empty()

# Handle Fetch Button
if inputs["fetch_clicked"]:
    try:
        with st.spinner("Fetching data..."):
            # 0. Ensure Backend Data (Download if needed)
            backend.ensure_backend_data(inputs["kic_id"].strip())

            # 1. Fetch Light Curve (Frontend Dummy - kept for compatibility)
            df = data_manager.fetch_kepler_lightcurve(
                kic_id=inputs["kic_id"].strip(),
                quarter=config.DEFAULT_QUARTER,
                flux_type=config.DEFAULT_FLUX_TYPE,
            )
            ss["raw_lc"] = df
            ss["kic_id"] = inputs["kic_id"].strip()
            
            # 2. Preprocess
            proc = data_manager.preprocess_lightcurve(
                df,
                remove_bad_quality=True,
                sigma_clip=config.DEFAULT_SIGMA_CLIP,
                normalize=True,
            )
            ss["proc_lc"] = proc
            
            # 3. Fetch TCEs from CSV
            tces = data_manager.get_tces_for_kic(inputs["kic_id"].strip())
            
            # Sort and Clean TCEs
            if not tces.empty:
                # Sort by TCE number (tce_plnt_num)
                tces = tces.sort_values("tce_plnt_num")
                # Drop av_pred_class if it exists
                if "av_pred_class" in tces.columns:
                    tces = tces.drop(columns=["av_pred_class"])
            
            ss["tces"] = tces
            
            # Reset prediction
            ss["prediction"] = None
            ss["status"] = "Fetched"
            
    except Exception as e:
        ss["status"] = "Error"
        st.error(f"Error fetching data: {e}")

# ------------------------------------------------------------
# TCE Selection & Prediction (Middle)
# ------------------------------------------------------------

pred = ss.get("prediction")
tces = ss.get("tces")

# Only show selection if we have TCEs and haven't run a prediction yet (or if we want to allow re-running)
# User said: "when run prediction button is pressed, it should remove that table"
if ss.get("status") == "Fetched" and tces is not None and not tces.empty and not pred:
    st.success("Light curve downloaded and processed.")
    
    st.markdown("### Select TCE")
    
    # Create a formatted label for the radio button
    tce_labels = []
    for idx, row in tces.iterrows():
        label = f"TCE {row['tce_plnt_num']} | Period: {row['tce_period']:.4f} d | Duration: {row['tce_duration']:.4f} d | SNR: {row['tce_model_snr']:.1f}"
        tce_labels.append(label)
    
    selected_label = st.radio("Choose a TCE to analyze:", tce_labels)
    
    # Find the selected row
    selected_idx = tce_labels.index(selected_label)
    selected_row = tces.iloc[selected_idx]
    
    # Display the full table for reference (No highlighting)
    st.dataframe(tces)
    
    # Update inputs for prediction
    inputs["period"] = selected_row["tce_period"]
    inputs["t0"] = selected_row["tce_time0bk"]
    inputs["duration"] = selected_row["tce_duration"]
    
    st.write("")
    if st.button("Run Prediction", type="primary"):
        try:
            with st.spinner("Running prediction..."):
                pred_result = model_runner.run_prediction(
                    model_class=inputs["model_class"],
                    config_name=inputs["config_name"] if not inputs["use_config_json"] else None,
                    config_json=inputs["config_json"] if inputs["use_config_json"] else None,
                    model_dir=inputs["model_dir"],
                    kepler_data_dir=inputs["kepler_data_dir"],
                    df=ss["proc_lc"],
                    kic_id=inputs["kic_id"].strip(),
                    period=inputs["period"],
                    t0=inputs["t0"],
                    duration=inputs["duration"],
                    include_image=inputs["include_image"],
                )
                ss["prediction"] = pred_result
                ss["selected_tce"] = selected_row.to_dict() # Store selected TCE data
                ss["status"] = "Done"
                st.rerun()
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif ss.get("status") == "Fetched" and (tces is None or tces.empty):
    st.warning("No TCEs found for this Kepler ID in the CSV.")


# ------------------------------------------------------------
# Results Display (Bottom)
# ------------------------------------------------------------

# --- Metrics Fetching Logic ---
# Fetch metrics for the currently selected model
# We use a simple caching mechanism in session state to avoid re-fetching on every rerun
current_model_dir = inputs["model_dir"]
if ss.get("last_model_dir") != current_model_dir:
    with st.spinner("Loading model metrics..."):
        metrics = backend.get_model_metrics(current_model_dir)
        ss["metrics"] = metrics
        ss["last_model_dir"] = current_model_dir

# Create two main columns: Main Content (Left) and Info/Controls (Right)
col_main, col_right = st.columns([2.5, 1.2])

# Create two main columns: Main Content (Left) and Info/Controls (Right)
# Added a spacer column (0.3) to create a gap between main content and sidebar
col_main, col_spacer, col_right = st.columns([2.5, 0.3, 1.2])

with col_right:
    # Kepler Target Info
    st.subheader("Kepler Target")
    
    # Helper to safely format values
    def safe_fmt(val, fmt="{:.2f}"):
        if val is None or pd.isna(val):
            return ""
        try:
            return fmt.format(val)
        except:
            return str(val)

    tce = ss.get("selected_tce", {})
    
    # Display keys and values simply (no borders)
    # Using columns for alignment without table borders
    k1, k2 = st.columns([1, 1])
    with k1:
        st.write("**KIC**")
        st.write("**TCE Num**")
        st.write("**Period**")
        st.write("**Duration**")
        st.write("**T0**")
        st.write("**SNR**")
        st.write("**Depth**")
    
    with k2:
        st.write(f"{safe_fmt(tce.get('kepid'), '{:.0f}')}")
        st.write(f"{safe_fmt(tce.get('tce_plnt_num'), '{}')}")
        st.write(f"{safe_fmt(tce.get('tce_period'), '{:.4f}')} d")
        st.write(f"{safe_fmt(tce.get('tce_duration'), '{:.4f}')} d")
        st.write(f"{safe_fmt(tce.get('tce_time0bk'), '{:.4f}')}")
        st.write(f"{safe_fmt(tce.get('tce_model_snr'), '{:.2f}')}")
        st.write(f"{safe_fmt(tce.get('tce_depth'), '{:.2f}')}")

    
    # Logic to display metrics AND prediction results:
    # - If prediction is DONE (pred exists), show results AND metrics in Sidebar (col_right).
    # - If prediction is NOT done, show metrics in Main Column (col_main).
    
    if pred:
        st.markdown("---")
        st.subheader("Prediction Results")
        ui_components.display_prediction_results(pred)
        
    metrics = ss.get("metrics")
    if pred and metrics:
        with st.expander("Model Metrics", expanded=False):
            ui_components.display_model_metrics(metrics)

with col_main:
    # 1. Model Metrics (Top of Main Column - Only if NO prediction yet)
    if not pred:
        # Removed vertical spacing to align with Kepler Target
        metrics = ss.get("metrics")
        if metrics:
            with st.expander("Model Metrics", expanded=True):
                ui_components.display_model_metrics(metrics)
    
    if pred:
        st.header("Light Curves")
        
        # 1. Global View
        st.subheader("Global View")
        if pred.get("global_bytes"):
            st.image(
                pred["global_bytes"],
                caption="Global View",
                use_container_width=True, # Responsive width
            )
        else:
            st.info("Global view not available.")

        # 2. Local View
        st.subheader("Local View")
        if pred.get("local_bytes"):
            st.image(
                pred["local_bytes"],
                caption="Local View (phase-folded)",
                use_container_width=True, # Responsive width
            )
        else:
            st.info("Local view not available.")



