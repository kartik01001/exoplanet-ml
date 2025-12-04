# ui_components.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

import config
from config import MODEL_PRESETS, DEFAULT_KIC_ID, DEFAULT_QUARTER, DEFAULT_SIGMA_CLIP, DEFAULT_FLUX_TYPE

def apply_custom_css():
    """
    Applies custom CSS to reduce top whitespace.
    """
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    margin-top: 1rem;
                    max-width: 1400px;
                    margin-left: auto;
                    margin-right: auto;
                }
                [data-testid="stSidebar"] {
                    padding-top: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)

def render_main_inputs(ss: Dict[str, Any]) -> Dict[str, Any]:
    """
    Renders the main inputs (Model, KIC ID) and returns a dictionary of user inputs/actions.
    """
    inputs = {}
    
    # --- Model Settings (Top) ---
    st.subheader("Model Settings")
    
    c1, c2 = st.columns(2)
    
    with c1:
        # Model selection dropdown with display names
        inputs["model_name"] = st.selectbox(
            "Model Architecture",
            options=list(MODEL_PRESETS.keys()),
            index=list(MODEL_PRESETS.keys()).index(config.DEFAULT_MODEL_NAME) if config.DEFAULT_MODEL_NAME in MODEL_PRESETS else 0,
            help="CNN: Convolutional Neural Network\nFP: Feature Pyramid\nWN: WaveNet"
        )
    
    # Auto-configure directories based on selected model
    selected_model = MODEL_PRESETS[inputs["model_name"]]
    inputs["model_class"] = selected_model["model_class"]
    inputs["model_dir"] = selected_model["model_dir"]
    inputs["kepler_data_dir"] = selected_model["kepler_data_dir"]
    
    # Config settings - dynamic dropdown based on model
    available_configs = selected_model["available_configs"]
    
    # Set default index based on available configs
    if config.DEFAULT_CONFIG_NAME in available_configs:
        default_index = available_configs.index(config.DEFAULT_CONFIG_NAME)
    else:
        default_index = 0
    
    with c2:
        inputs["config_name"] = st.selectbox(
            "Config Name",
            options=available_configs,
            index=default_index
        )
    inputs["use_config_json"] = False # Simplified for now
    inputs["config_json"] = None

    st.caption(f"🤖 Using: `{selected_model['model_class']}`")
    
    st.markdown("---")

    # --- Target selection ---
    st.subheader("Target Selection")
    
    c_kic, c_btn = st.columns([3, 1])
    with c_kic:
        inputs["kic_id"] = st.text_input("KIC ID", value=ss.get("kic_id", DEFAULT_KIC_ID))
    with c_btn:
        st.write("") # Spacer
        st.write("") # Spacer
        inputs["fetch_clicked"] = st.button("Fetch TCEs", type="primary")

    # Default values for params (will be overwritten by CSV selection)
    inputs["period"] = 0.0
    inputs["t0"] = 0.0
    inputs["duration"] = 0.0
    inputs["include_image"] = True # Always true now

    return inputs

def display_prediction_results(pred: Dict[str, Any]):
    """
    Displays the prediction results in a simplified, premium way.
    """
    if pred is None:
        return

    label = pred["label"]
    probs = pred["probabilities"]
    planet_prob = probs.get("planet", 0.0)
    
    # Styling logic
    if planet_prob >= 0.5:
        verdict = "PLANET CANDIDATE"
        icon = "🪐"
        msg_color = "green"
    else:
        verdict = "FALSE POSITIVE"
        icon = "🌑"
        msg_color = "red"

    # Layout: Metric on left, Verdict on right
    c1, c2 = st.columns([1, 1])
    
    with c1:
        # Vertical spacer to align with the verdict on the right
        st.write("") 
        # Big metric
        st.metric(
            label="Confidence", 
            value=f"{planet_prob:.1%}"
        )
        st.caption(f"Raw: `{planet_prob:.6f}`")
        
    with c2:
        # Verdict with visual emphasis
        st.markdown(f"### {icon} :{msg_color}[{verdict}]")
        st.progress(planet_prob)


import plotly.graph_objects as go

def display_model_metrics(metrics: Dict[str, Any]):
    """
    Displays training metrics (accuracy, loss) and confusion matrix using Plotly.
    """
    if not metrics:
        return

    st.subheader("Model Performance")
    
    # Helper to create Plotly figure
    def plot_metric(train_data, val_data, title, y_label):
        fig = go.Figure()
        
        if train_data:
            df_train = pd.DataFrame(train_data)
            fig.add_trace(go.Scatter(
                x=df_train["step"], 
                y=df_train["value"], 
                mode='lines', 
                name='Train',
                line=dict(color='#00CC96', width=2)
            ))
            
        if val_data:
            df_val = pd.DataFrame(val_data)
            fig.add_trace(go.Scatter(
                x=df_val["step"], 
                y=df_val["value"], 
                mode='lines', 
                name='Validation',
                line=dict(color='#EF553B', width=2)
            ))
            
        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title=y_label,
            margin=dict(l=20, r=20, t=40, b=20),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    # Loss Chart
    train_loss = metrics.get("train_loss", [])
    val_loss = metrics.get("val_loss", [])
    
    if train_loss or val_loss:
        st.plotly_chart(plot_metric(train_loss, val_loss, "Loss Curves", "Loss"), use_container_width=True)

    # Accuracy Chart
    train_acc = metrics.get("train_accuracy", [])
    val_acc = metrics.get("val_accuracy", [])
    
    if train_acc or val_acc:
        st.plotly_chart(plot_metric(train_acc, val_acc, "Accuracy Curves", "Accuracy"), use_container_width=True)

    # 2. Confusion Matrix
    cm_b64 = metrics.get("confusion_matrix_base64")
    if cm_b64:
        st.caption("Confusion Matrix")
        import base64
        cm_bytes = base64.b64decode(cm_b64)
        st.image(cm_bytes, use_container_width=False, width=500)
