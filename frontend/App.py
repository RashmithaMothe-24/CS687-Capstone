# Main entry point for the Fraud Detection Streamlit app.
# It loads each page module from src/ and renders them inside a tab layout.

import os, sys

# Ensure project root is in PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import streamlit as st

# Import page modules
from src import (
    page1_data_preprocessing,
    page2_feature_engineering,
    page3_train_threshold,
    page4_evaluation,
    page5_live_monitor_feedback,
    page6_drift_data_quality,
    page7_admin_export,
)

st.set_page_config(
    page_title="Fraud Detection — Anomaly First",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Fraud Detection System (Anomaly Detection Approach)")

# Define tabs for navigation
tabs = st.tabs([
    "1) Data & Preprocessing",
    "2) Feature Engineering",
    "3) Train & Threshold",
    "4) Evaluation",
    "5) Live Monitor & Feedback",
    "6) Drift & Data Quality",
    "7) Admin & Export",
])

with tabs[0]:
    page1_data_preprocessing.render()
with tabs[1]:
    page2_feature_engineering.render()
with tabs[2]:
    page3_train_threshold.render()
with tabs[3]:
    page4_evaluation.render()
with tabs[4]:
    page5_live_monitor_feedback.render()
with tabs[5]:
    page6_drift_data_quality.render()
with tabs[6]:
    page7_admin_export.render()
