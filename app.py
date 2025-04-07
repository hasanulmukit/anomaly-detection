# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt

# -----------------------------
# Custom CSS for improved styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6; }
    .stButton button {background-color: #4CAF50; color: white; border: none; padding: 10px 24px; font-size: 16px; border-radius: 5px;}
    .stHeader {font-size: 2rem; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True
)

# -----------------------------
# Cache and load the model, scaler, and threshold for faster performance.
# -----------------------------
@st.cache_resource
def load_resources():
    # Use custom_objects to resolve the 'mse' loss function
    model = load_model("autoencoder_model.h5", custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("threshold.pkl", "rb") as f:
        threshold = pickle.load(f)
    return model, scaler, threshold

model, scaler, trained_threshold = load_resources()

# -----------------------------
# Main Title & Overview
# -----------------------------
st.title("Blockchain Block Anomaly Detection")
st.markdown("### Overview")
st.write(
    """
    This dashboard uses an autoencoder model to detect anomalies in blockchain data based on reconstruction errors.
    Uploaded CSV files should have the following columns: 
    `height`, `timestamp`, `size`, `tx_count`, `difficulty`, `median_fee_rate`, `avg_fee_rate`, 
    `total_fees`, `fee_range_min`, `fee_range_max`, `input_count`, `output_count`, `output_amount`.
    An additional engineered feature `fee_spread` (calculated as `fee_range_max - fee_range_min`) is used.
    """
)
st.write("**Loaded autoencoder reconstruction error threshold from training:**", trained_threshold)

# -----------------------------
# Sidebar: File Upload and Configuration
# -----------------------------
st.sidebar.header("Configuration Options")

uploaded_file = st.sidebar.file_uploader("Upload CSV file with blockchain data", type="csv")

# Sidebar: Synthetic Anomaly Injection Options
st.sidebar.markdown("#### Synthetic Anomaly Injection")
inject_anomaly = st.sidebar.checkbox("Inject synthetic anomaly for testing")
if inject_anomaly:
    row_to_modify = st.sidebar.number_input("Row index to modify", min_value=0, value=0)
    feature_to_modify = st.sidebar.selectbox("Feature to modify", 
                                               options=['size', 'tx_count', 'difficulty', 'median_fee_rate', 
                                                        'avg_fee_rate', 'total_fees', 'fee_range_min', 
                                                        'fee_range_max', 'input_count', 'output_count', 
                                                        'output_amount'])
    anomaly_multiplier = st.sidebar.number_input("Multiplier factor for anomaly", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)

# Sidebar: Threshold Adjustment Options
st.sidebar.markdown("#### Threshold Adjustment")
use_adjusted_threshold = st.sidebar.checkbox("Use adjusted threshold based on current data", value=True)
if use_adjusted_threshold:
    threshold_multiplier = st.sidebar.slider("Threshold multiplier", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

# -----------------------------
# Process the Uploaded Data
# -----------------------------
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.markdown("### Data Preview")
    st.dataframe(data.head())
    
    # Preprocess the new data: Convert timestamp and compute engineered feature (fee_spread)
    try:
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    except Exception as e:
        st.error("Error converting timestamp: " + str(e))
    data['fee_spread'] = data['fee_range_max'] - data['fee_range_min']
    
    # Inject synthetic anomaly if requested
    if inject_anomaly:
        try:
            original_value = data.loc[row_to_modify, feature_to_modify]
            data.loc[row_to_modify, feature_to_modify] *= anomaly_multiplier
            st.sidebar.write(f"Modified row {row_to_modify} feature '{feature_to_modify}': {original_value} -> {data.loc[row_to_modify, feature_to_modify]}")
        except Exception as e:
            st.sidebar.error("Error injecting anomaly: " + str(e))
    
    # Select features as used in training.
    features = ['size', 'tx_count', 'difficulty', 'median_fee_rate', 'avg_fee_rate', 
                'total_fees', 'fee_range_min', 'fee_range_max', 'input_count', 'output_count', 
                'output_amount', 'fee_spread']
    try:
        data_scaled = scaler.transform(data[features])
    except Exception as e:
        st.error("Error scaling data: " + str(e))
    
    # Get reconstructions and compute the reconstruction error.
    reconstructions = model.predict(data_scaled)
    mse = np.mean(np.power(data_scaled - reconstructions, 2), axis=1)
    
    # Display reconstruction errors for debugging.
    st.markdown("### Reconstruction Errors")
    error_df = pd.DataFrame({"Row": range(len(mse)), "Reconstruction Error": mse})
    st.dataframe(error_df)
    
    # Calculate threshold: either adjusted or loaded from training.
    if use_adjusted_threshold:
        adjusted_threshold = np.mean(mse) + threshold_multiplier * np.std(mse)
        st.sidebar.write("Adjusted threshold based on current data:", adjusted_threshold)
        threshold = adjusted_threshold
    else:
        threshold = trained_threshold
        st.sidebar.write("Using loaded threshold:", threshold)
    
    # Flag anomalies based on the threshold.
    data['reconstruction_error'] = mse
    data['anomaly'] = data['reconstruction_error'] > threshold
    
    st.markdown("### Results")
    st.write("Data with reconstruction error and anomaly flag:")
    st.dataframe(data.head())
    
    anomalies = data[data['anomaly']]
    st.markdown("### Anomalies Detected")
    if anomalies.empty:
        st.info("No anomalies detected.")
    else:
        st.dataframe(anomalies)
    
    # Plot a histogram of reconstruction errors.
    st.markdown("### Reconstruction Error Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mse, bins=50, alpha=0.7, color='blue')
    ax.axvline(threshold, color='red', linestyle='--', label='Threshold')
    ax.set_title("Reconstruction Error Distribution")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)
