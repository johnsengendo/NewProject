# Importing essential libraries for the digital twin platform
import sys
import os
import streamlit as st
from data_loader import load_data, load_uploaded_data
from training_utils import create_sequences, create_traditional_features, calculate_metrics
from model_architectures import (
    create_lstm_model,
    create_cnn_lstm_model,
    create_conv1d_model,
    create_gru_model,
    create_bidirectional_lstm_model
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta

# Defining custom CSS styling for creating a network monitoring interface
st.markdown("""
<style>
    /* Creating main header styling with gradient background */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Styling header text elements */
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Creating status card base styling */
    .status-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Defining different status card variants */
    .status-card.success {
        border-color: #28a745;
        background: #d4edda;
    }
    
    .status-card.warning {
        border-color: #ffc107;
        background: #fff3cd;
    }
    
    .status-card.error {
        border-color: #dc3545;
        background: #f8d7da;
    }
    
    .status-card.info {
        border-color: #17a2b8;
        background: #d1ecf1;
    }
    
    /* Creating metric display containers */
    .metric-container {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Styling section headers throughout the application */
    .section-header {
        background: #343a40;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    /* Creating configuration section styling */
    .config-section {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Defining digital twin status indicators */
    .twin-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    /* Creating operational status styling */
    .twin-status.operational {
        background: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    
    /* Creating calibrating status styling */
    .twin-status.calibrating {
        background: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    
    /* Creating error status styling */
    .twin-status.error {
        background: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    
    /* Creating performance metrics grid layout */
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Styling prediction interval cards */
    .prediction-interval-card {
        background: #e3f2fd;
        border: 2px solid #2196f3;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Styling training period cards */
    .training-period-card {
        background: #f3e5f5;
        border: 2px solid #9c27b0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Creating comparison table styling */
    .comparison-table {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Styling form input labels */
    .stSelectbox label, .stSlider label, .stMultiSelect label, .stCheckbox label {
        font-weight: 600;
        color: #495057;
    }
    
    /* Styling dual target headers */
    .dual-target-header {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Setting up directory paths and file configurations
DATA_DIR = "Data"
CSV_FILENAME = "543.csv"
CSV_FILEPATH = os.path.join(DATA_DIR, CSV_FILENAME)

# Defining dual target features
TARGET_FEATURES = ['tcp_udp_ratio_packets', 'tcp_udp_ratio_bytes']

# Configuring Streamlit page layout and settings
st.set_page_config(
    page_title="Enhanced Network Digital Twin Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Displaying the main application header with professional styling
st.markdown("""
<div class="main-header">
    <h1>🌐 Enhanced Network Digital Twin Platform</h1>
    <p>TCP/UDP Ratio Prediction: Packets & Bytes Analysis</p>
</div>
""", unsafe_allow_html=True)

# Loading the primary dataset from the CSV file
df_train = load_data(CSV_FILEPATH)

# Validating target features exist in dataset
missing_targets = [target for target in TARGET_FEATURES if target not in df_train.columns]
if missing_targets:
    st.error(f"Missing target features in dataset: {missing_targets}")
    st.error("Please ensure your dataset contains both 'tcp_udp_ratio_packets' and 'tcp_udp_ratio_bytes' columns")
    st.stop()

# Creating utility function for converting data points to human-readable time periods
def convert_to_time_periods(data_points, interval_minutes=10):
    """Converting data points to readable time periods assuming 10-minute intervals"""
    total_minutes = data_points * interval_minutes
    days = total_minutes // (24 * 60)
    hours = (total_minutes % (24 * 60)) // 60
    minutes = total_minutes % 60
    
    # Building readable time string
    time_str = ""
    if days > 0:
        time_str += f"{days} days, "
    if hours > 0:
        time_str += f"{hours} hours, "
    if minutes > 0:
        time_str += f"{minutes} minutes"
    
    return time_str.rstrip(", ")

# Creating modified sequence creation function for dual targets
def create_sequences_dual_target(data, target_features, sequence_length, additional_features=None):
    """Creating sequences for multiple target features"""
    if additional_features is None:
        additional_features = []
    
    # Selecting features for input sequences
    feature_columns = target_features + additional_features
    available_features = [col for col in feature_columns if col in data.columns]
    
    if not available_features:
        raise ValueError("No valid features found in the dataset")
    
    X, y = [], []
    for i in range(sequence_length, len(data)):
        # Creating input sequence
        X.append(data[available_features].iloc[i-sequence_length:i].values)
        # Creating output for both targets
        y.append([data[target].iloc[i] for target in target_features])
    
    return np.array(X), np.array(y)

# Creating modified traditional features function for dual targets
def create_traditional_features_dual_target(data, target_features, window_size=10, additional_features=None):
    """Creating traditional features for multiple target prediction"""
    if additional_features is None:
        additional_features = []
    
    # Combining all feature columns
    all_features = target_features + additional_features
    available_features = [col for col in all_features if col in data.columns]
    
    if not available_features:
        raise ValueError("No valid features found in the dataset")
    
    X_data = []
    y_data = []
    
    for i in range(window_size, len(data)):
        # Creating lagged features
        features = []
        for col in available_features:
            # Adding recent values as features
            recent_values = data[col].iloc[i-window_size:i].values
            features.extend([
                recent_values.mean(),
                recent_values.std() if len(recent_values) > 1 else 0,
                recent_values[-1],  # Most recent value
                recent_values.max(),
                recent_values.min()
            ])
        
        X_data.append(features)
        # Target values for current time step
        y_data.append([data[target].iloc[i] for target in target_features])
    
    # Creating feature names
    feature_names = []
    for col in available_features:
        for stat in ['mean', 'std', 'last', 'max', 'min']:
            feature_names.append(f"{col}_{stat}")
    
    X_df = pd.DataFrame(X_data, columns=feature_names)
    y_df = pd.DataFrame(y_data, columns=target_features)
    
    return X_df, y_df

# Creating modified metrics calculation for dual targets
def calculate_metrics_dual_target(y_true, y_pred, target_names):
    """Calculate metrics for multiple targets"""
    metrics = {}
    
    for i, target in enumerate(target_names):
        y_true_target = y_true[:, i] if y_true.ndim > 1 else y_true
        y_pred_target = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        
        # Ensuring arrays are aligned
        min_len = min(len(y_true_target), len(y_pred_target))
        y_true_aligned = y_true_target[:min_len]
        y_pred_aligned = y_pred_target[:min_len]
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true_aligned - y_pred_aligned) ** 2))
        
        # MAE
        mae = np.mean(np.abs(y_true_aligned - y_pred_aligned))
        
        # MAPE
        non_zero_mask = y_true_aligned != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true_aligned[non_zero_mask] - y_pred_aligned[non_zero_mask]) 
                                / y_true_aligned[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # R²
        ss_res = np.sum((y_true_aligned - y_pred_aligned) ** 2)
        ss_tot = np.sum((y_true_aligned - np.mean(y_true_aligned)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Direction accuracy
        if len(y_true_aligned) > 1:
            true_direction = np.diff(y_true_aligned) > 0
            pred_direction = np.diff(y_pred_aligned) > 0
            direction_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            direction_accuracy = 0
        
        metrics[target] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_true_aligned': y_true_aligned,
            'y_pred_aligned': y_pred_aligned
        }
    
    return metrics

# Creating file upload section for inference mode
st.markdown('<div class="section-header">Inference with New Traffic Data Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Network Traffic Data (CSV format)", 
    type=["csv"], 
    help="CSV file must include both 'tcp_udp_ratio_packets' and 'tcp_udp_ratio_bytes' columns"
)

# Handling file upload and setting inference mode
if uploaded_file is not None:
    df_test = load_uploaded_data(uploaded_file)
    
    # Validating target features in uploaded data
    missing_targets_upload = [target for target in TARGET_FEATURES if target not in df_test.columns]
    if missing_targets_upload:
        st.error(f"Missing target features in uploaded data: {missing_targets_upload}")
        st.error("Please ensure your uploaded data contains both target columns")
        st.stop()
    
    st.markdown('<div class="status-card success">Custom network traffic data loaded successfully. Dual-target inference mode activated.</div>', unsafe_allow_html=True)
    inference_mode = True
else:
    df_test = None
    inference_mode = False

# Creating sidebar configuration panel
st.sidebar.markdown("## Enhanced Control Panel")
st.sidebar.markdown("---")

# Initializing session state parameters for maintaining configuration across interactions
if 'dt_params_applied' not in st.session_state:
    st.session_state.dt_params_applied = False

# Creating enhanced training period configuration section
st.sidebar.markdown("### 🕒 Training Period Configuration")
st.sidebar.markdown("*Configure how much historical data to use for training*")

# Defining training period options with corresponding data point calculations
training_options = {
    "1 Week": 7 * 24 * 6,  # 7 days * 24 hours * 6 (10-min intervals per hour)
    "2 Weeks": 2 * 7 * 24 * 6,
    "1 Month": 30 * 24 * 6,
    "2 Months": 2 * 30 * 24 * 6,
    "3 Months": 3 * 30 * 24 * 6,
    "6 Months": 6 * 30 * 24 * 6,
    "Full Dataset": len(df_train),
    "Custom": -1
}

# Initializing current training period in session state
if 'current_training_period_option' not in st.session_state:
    st.session_state.current_training_period_option = "1 Month"

# Creating training period selection dropdown
temp_training_period_option = st.sidebar.selectbox(
    "Training Period", 
    list(training_options.keys()),
    index=list(training_options.keys()).index(st.session_state.current_training_period_option),
    help="Select the amount of historical data to use for training"
)

# Handling custom training period selection
if temp_training_period_option == "Custom":
    max_training_points = min(len(df_train) - 100, len(df_train))  # Reserving at least 100 points for testing
    temp_custom_training_size = st.sidebar.slider(
        "Custom Training Size (data points)", 
        100, 
        max_training_points, 
        min(10000, max_training_points),
        help="Number of data points to use for training"
    )
    training_size = temp_custom_training_size
else:
    training_size = min(training_options[temp_training_period_option], len(df_train) - 100)

# Displaying selected training data information
st.sidebar.markdown(f"**Training Data**: {training_size:,} points ({convert_to_time_periods(training_size)})")

# Creating enhanced prediction interval configuration section
st.sidebar.markdown("### 📈 Prediction Interval Configuration")
st.sidebar.markdown("*Configure prediction horizons and validation periods*")

# Defining prediction horizon options with corresponding data point calculations
prediction_horizons = {
    "1 Hour": 6,      # 6 intervals of 10 minutes
    "3 Hours": 18,
    "6 Hours": 36,
    "12 Hours": 72,
    "1 Day": 144,     # 144 intervals of 10 minutes = 24 hours
    "2 Days": 288,
    "3 Days": 432,
    "1 Week": 1008,   # 7 * 144
    "2 Weeks": 2016,
    "1 Month": 4320,  # 30 * 144
    "Full Dataset": len(df_train),
    "Custom": -1
}

# Initializing current prediction horizon in session state
if 'current_prediction_horizon' not in st.session_state:
    st.session_state.current_prediction_horizon = "1 Day"

# Creating prediction horizon selection dropdown
temp_prediction_horizon = st.sidebar.selectbox(
    "Prediction Horizon", 
    list(prediction_horizons.keys()),
    index=list(prediction_horizons.keys()).index(st.session_state.current_prediction_horizon),
    help="How far into the future to predict"
)

# Handling custom prediction horizon selection
if temp_prediction_horizon == "Custom":
    remaining_data = len(df_train) - training_size
    max_prediction_horizon = min(remaining_data, 5000)
    temp_custom_prediction_horizon = st.sidebar.slider(
        "Custom Prediction Points", 
        10, 
        max_prediction_horizon, 
        min(144, max_prediction_horizon),
        help="Number of future data points to predict"
    )
    prediction_horizon = temp_custom_prediction_horizon
else:
    remaining_data = len(df_train) - training_size
    prediction_horizon = min(prediction_horizons[temp_prediction_horizon], remaining_data)

# Displaying selected prediction horizon information
st.sidebar.markdown(f"**Prediction Horizon**: {prediction_horizon:,} points ({convert_to_time_periods(prediction_horizon)})")

# Creating comparative analysis section for multiple prediction intervals
st.sidebar.markdown("### 📊 Comparative Analysis")
enable_comparison = st.sidebar.checkbox(
    "Enable Multi-Interval Comparison", 
    value=False,
    help="Compare performance across multiple prediction intervals"
)

# Setting up comparison intervals if comparison is enabled
if enable_comparison:
    comparison_intervals = st.sidebar.multiselect(
        "Comparison Intervals",
        ["1 Hour", "3 Hours", "6 Hours", "12 Hours", "1 Day", "2 Days", "3 Days", "1 Week"],
        default=["1 Hour", "6 Hours", "1 Day", "1 Week"],
        help="Select multiple intervals for performance comparison"
    )
else:
    comparison_intervals = []

# Creating standard digital twin parameter configuration section
st.sidebar.markdown("---")
st.sidebar.markdown("###  Digital Twin Parameters")

# Initializing all session state parameters with default values
if 'current_observation_window' not in st.session_state:
    st.session_state.current_observation_window = 20
if 'current_simulation_cycles' not in st.session_state:
    st.session_state.current_simulation_cycles = 50
if 'current_processing_batch' not in st.session_state:
    st.session_state.current_processing_batch = 32
if 'current_network_features' not in st.session_state:
    # Getting available network parameters from dataset excluding target features
    network_parameters = df_train.columns.tolist()
    for target in TARGET_FEATURES:
        if target in network_parameters:
            network_parameters.remove(target)
    st.session_state.current_network_features = network_parameters[:2] if len(network_parameters) >= 2 else network_parameters
if 'current_twin_model' not in st.session_state:
    st.session_state.current_twin_model = "Neural Network Twin (LSTM)"
if 'current_use_normalization' not in st.session_state:
    st.session_state.current_use_normalization = True
if 'current_normalization_method' not in st.session_state:
    st.session_state.current_normalization_method = "MinMax Scaling"

# Creating observation window slider for pattern recognition
temp_observation_window = st.sidebar.slider(
    "Observation Window", 
    5, 100, 
    st.session_state.current_observation_window, 
    help="Number of historical time steps for pattern recognition"
)

# Creating training epochs slider
temp_simulation_cycles = st.sidebar.slider(
    "Training Epochs", 
    10, 200, 
    st.session_state.current_simulation_cycles, 
    help="Number of training iterations"
)

# Creating batch size selection dropdown
batch_options = [16, 32, 64, 128]
current_batch_index = batch_options.index(st.session_state.current_processing_batch) if st.session_state.current_processing_batch in batch_options else 1
temp_processing_batch = st.sidebar.selectbox(
    "Batch Size", 
    batch_options, 
    index=current_batch_index
)

# Creating network features selection for additional prediction inputs
network_parameters = df_train.columns.tolist()
for target in TARGET_FEATURES:
    if target in network_parameters:
        network_parameters.remove(target)

temp_selected_network_features = st.sidebar.multiselect(
    "Additional Network Features", 
    options=network_parameters, 
    default=st.session_state.current_network_features[:3],  # Limiting to first 3 for performance
    help="Additional network metrics to include in predictions"
)

# Defining available digital twin model architectures
twin_models = {
    "Neural Network Twin (LSTM)": "lstm",
    "Hybrid CNN-LSTM Twin": "cnn_lstm",
    "Convolutional Twin (Conv1D)": "conv1d",
    "Recurrent Twin (GRU)": "gru",
    "Bidirectional Neural Twin": "bilstm",
    "Statistical Twin (ARIMA)": "arima",
    "Adaptive Twin (Exp. Smoothing)": "exp_smoothing",
    "Ensemble Twin (Random Forest)": "rf",
    "Gradient Twin (Boosting)": "gb"
}

# Creating model selection dropdown
current_model_index = list(twin_models.keys()).index(st.session_state.current_twin_model) if st.session_state.current_twin_model in twin_models.keys() else 0
temp_selected_twin_model = st.sidebar.selectbox(
    "Digital Twin Engine", 
    list(twin_models.keys()), 
    index=current_model_index
)

# Creating normalization settings section
temp_use_normalization = st.sidebar.checkbox(
    "Enable Data Normalization", 
    value=st.session_state.current_use_normalization
)

# Creating normalization method selection
normalization_options = ["MinMax Scaling", "Standard Scaling"]
current_norm_index = normalization_options.index(st.session_state.current_normalization_method) if st.session_state.current_normalization_method in normalization_options else 0
temp_normalization_method = st.sidebar.selectbox(
    "Normalization Method", 
    normalization_options, 
    index=current_norm_index
)

# Creating system control buttons section
st.sidebar.markdown("---")
st.sidebar.markdown("### System Controls")
col1, col2 = st.sidebar.columns(2)
with col1:
    reset_config = st.button("Reset All", help="Reset all parameters to defaults", type="secondary")
with col2:
    apply_settings = st.button("Apply & Train", help="Apply settings and train model", type="primary")

# Implementing reset configuration functionality
if reset_config:
    # Clearing all current session state parameters
    for key in list(st.session_state.keys()):
        if key.startswith('current_'):
            del st.session_state[key]
    st.session_state.dt_params_applied = False
    st.rerun()

# Applying settings when button is clicked and updating session state
if apply_settings:
    st.session_state.current_training_period_option = temp_training_period_option
    st.session_state.current_prediction_horizon = temp_prediction_horizon
    st.session_state.current_observation_window = temp_observation_window
    st.session_state.current_simulation_cycles = temp_simulation_cycles
    st.session_state.current_processing_batch = temp_processing_batch
    st.session_state.current_network_features = temp_selected_network_features
    st.session_state.current_twin_model = temp_selected_twin_model
    st.session_state.current_use_normalization = temp_use_normalization
    st.session_state.current_normalization_method = temp_normalization_method
    st.session_state.dt_params_applied = True
    st.success("✅ Configuration applied! Training will begin...")

# Using applied settings from session state for processing
observation_window = st.session_state.current_observation_window
simulation_cycles = st.session_state.current_simulation_cycles
processing_batch = st.session_state.current_processing_batch
selected_network_features = st.session_state.current_network_features
selected_twin_model = st.session_state.current_twin_model
use_normalization = st.session_state.current_use_normalization
normalization_method = st.session_state.current_normalization_method

# Defining comprehensive model training function with flexible parameters for dual targets
def train_model_with_params_dual_target(data, train_size, test_size, model_type, features, obs_window, epochs, batch_size, use_norm, norm_method, target_features):
    """Training model with specific training and testing split parameters for dual targets"""
    
    # Preparing data split and validating data sufficiency
    total_needed = train_size + test_size + obs_window
    if len(data) < total_needed:
        raise ValueError(f"Insufficient data. Need {total_needed}, have {len(data)}")
    
    # Using the most recent data for training and testing
    data_subset = data.iloc[-total_needed:].copy()
    
    try:
        # Processing neural network models that require sequence data
        if model_type in ['lstm', 'cnn_lstm', 'conv1d', 'gru', 'bilstm']:
            # Creating sequences for time series neural networks with dual targets
            X_seq, y_seq = create_sequences_dual_target(data_subset, target_features, obs_window, features)
            
            # Splitting sequences into train and test sets
            X_train = X_seq[:train_size]
            X_test = X_seq[train_size:train_size + test_size]
            y_train = y_seq[:train_size]
            y_test = y_seq[train_size:train_size + test_size]
            
            # Applying normalization if enabled
            scaler = None
            if use_norm:
                if norm_method == "MinMax Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                
                # Reshaping data for normalization
                X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
                X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
                
                # Fitting scaler on training data and transforming both sets
                X_train_scaled = scaler.fit_transform(X_train_reshaped)
                X_test_scaled = scaler.transform(X_test_reshaped)
                
                # Reshaping back to original dimensions
                X_train = X_train_scaled.reshape(X_train.shape)
                X_test = X_test_scaled.reshape(X_test.shape)
        
        else:
            # Processing traditional machine learning models
            X, y = create_traditional_features_dual_target(data_subset, target_features, 10, features)
            
            # Splitting traditional features into train and test sets
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:train_size + test_size]
            y_train = y.iloc[:train_size].values
            y_test = y.iloc[train_size:train_size + test_size].values
            
            # Applying normalization for traditional models if enabled
            scaler = None
            if use_norm and len(X.columns) > 0:
                if norm_method == "MinMax Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()
                
                # Transforming traditional features
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Training models based on selected type (modified for dual output)
        if model_type == 'lstm':
            # Creating and training LSTM model
            model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, callbacks=[early_stopping], verbose=0)
            y_pred = model.predict(X_test, verbose=0).flatten()
        
        elif model_type == 'cnn_lstm':
            # Creating and training CNN-LSTM hybrid model
            model = create_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, callbacks=[early_stopping], verbose=0)
            y_pred = model.predict(X_test, verbose=0).flatten()
            
        elif model_type == 'conv1d':
            # Creating and training 1D Convolutional model
            model = create_conv1d_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, callbacks=[early_stopping], verbose=0)
            y_pred = model.predict(X_test, verbose=0).flatten()
            
        elif model_type == 'gru':
            # Creating and training GRU model
            model = create_gru_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, callbacks=[early_stopping], verbose=0)
            y_pred = model.predict(X_test, verbose=0).flatten()
            
        elif model_type == 'bilstm':
            # Creating and training Bidirectional LSTM model
            model = create_bidirectional_lstm_model((X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, callbacks=[early_stopping], verbose=0)
            y_pred = model.predict(X_test, verbose=0).flatten()
            
        elif model_type == "arima":
            # Creating separate ARIMA models for each target
            y_pred = []
            models = []
            
            for i, target in enumerate(target_features):
                y_train_target = y_train[:, i] if y_train.ndim > 1 else y_train
                
                best_aic = float('inf')
                best_model = None
                
                # Searching for optimal ARIMA parameters
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                arima_model = ARIMA(y_train_target, order=(p, d, q))
                                fitted_model = arima_model.fit()
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_model = fitted_model
                            except:
                                continue
                
                # Using default parameters if optimization fails
                if best_model is None:
                    arima_model = ARIMA(y_train_target, order=(1, 1, 1))
                    best_model = arima_model.fit()
                
                models.append(best_model)
                y_pred.append(best_model.forecast(steps=len(y_test)))
            
            model = models  # Store list of models for dual target
            y_pred = np.array(y_pred).T  # Transpose to match expected shape
            
        elif model_type == "exp_smoothing":
            # Creating separate Exponential Smoothing models for each target
            y_pred = []
            models = []
            
            for i, target in enumerate(target_features):
                y_train_target = y_train[:, i] if y_train.ndim > 1 else y_train
                
                try:
                    exp_model = ExponentialSmoothing(y_train_target, seasonal='add' if len(y_train_target) > 24 else None, 
                                                   seasonal_periods=min(24, len(y_train_target)//2) if len(y_train_target) > 24 else None)
                    fitted_model = exp_model.fit()
                    models.append(fitted_model)
                    y_pred.append(fitted_model.forecast(steps=len(y_test)))
                except:
                    # Falling back to simpler trend model if seasonal fails
                    exp_model = ExponentialSmoothing(y_train_target, trend='add')
                    fitted_model = exp_model.fit()
                    models.append(fitted_model)
                    y_pred.append(fitted_model.forecast(steps=len(y_test)))
            
            model = models  # Store list of models
            y_pred = np.array(y_pred).T  # Transpose to match expected shape
                
        elif model_type == "rf":
            # Creating dual-output Random Forest model
            from sklearn.multioutput import MultiOutputRegressor
            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        elif model_type == "gb":
            # Creating dual-output Gradient Boosting model
            from sklearn.multioutput import MultiOutputRegressor
            model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculating comprehensive performance metrics for both targets
        metrics = calculate_metrics_dual_target(y_test, y_pred, target_features)
        
        # Returning training results and components
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_train': y_train,
            'training_size': train_size,
            'test_size': test_size,
            'target_features': target_features
        }
        
    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")

# Main training and analysis logic
if not inference_mode and st.session_state.dt_params_applied:
    model_type = twin_models[selected_twin_model]
    
    st.markdown('<div class="section-header">Network Digital Twin Training & Analysis</div>', unsafe_allow_html=True)
    
    # Displaying current configuration
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="training-period-card">', unsafe_allow_html=True)
        st.markdown(f"**Training Period**")
        st.markdown(f"Data Points: {training_size:,}")
        st.markdown(f"Time Coverage: {convert_to_time_periods(training_size)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="prediction-interval-card">', unsafe_allow_html=True)
        st.markdown(f"**Prediction Horizon**")
        st.markdown(f"Data Points: {prediction_horizon:,}")
        st.markdown(f"Time Coverage: {convert_to_time_periods(prediction_horizon)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        st.markdown(f"**Digital Twin Model Configuration**")
        st.markdown(f"Architecture: {selected_twin_model}")
        st.markdown(f"Target Features: {len(TARGET_FEATURES)}")
        st.markdown(f"Additional Features: {len(selected_network_features)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Training primary model with dual targets
    with st.spinner(f"Training {selected_twin_model} for dual-target prediction..."):
        try:
            primary_results = train_model_with_params_dual_target(
                df_train, training_size, prediction_horizon, model_type, 
                selected_network_features, observation_window, simulation_cycles, 
                processing_batch, use_normalization, normalization_method, TARGET_FEATURES
            )
            
            st.session_state.trained_model = primary_results['model']
            st.session_state.trained_scaler = primary_results['scaler']
            st.session_state.trained_model_type = model_type
            st.session_state.model_config = {
                'observation_window': observation_window,
                'selected_network_features': selected_network_features,
                'use_normalization': use_normalization,
                'normalization_method': normalization_method,
                'selected_twin_model': selected_twin_model,
                'training_size': training_size,
                'prediction_horizon': prediction_horizon,
                'target_features': TARGET_FEATURES
            }
            
            st.markdown('<div class="status-card success">✅ Dual-Target Digital Twin Successfully Calibrated and Operational</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'<div class="status-card error">Training failed: {str(e)}</div>', unsafe_allow_html=True)
            st.stop()
    
    # Displaying primary results for both targets
    st.markdown('<div class="dual-target-header">Primary Model Performance - Dual Target Analysis</div>', unsafe_allow_html=True)
    
    # Creating performance metrics for both targets
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📦 TCP/UDP Ratio - Packets")
        metrics_packets = primary_results['metrics'][TARGET_FEATURES[0]]
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.metric("RMSE", f"{metrics_packets['rmse']:.4f}")
            st.metric("R² Score", f"{metrics_packets['r2']:.4f}")
        with subcol2:
            st.metric("MAE", f"{metrics_packets['mae']:.4f}")
            if metrics_packets['mape'] != float('inf'):
                st.metric("MAPE (%)", f"{metrics_packets['mape']:.2f}")
            else:
                st.metric("MAPE (%)", "N/A")
    
    with col2:
        st.markdown("### 📊 TCP/UDP Ratio - Bytes")
        metrics_bytes = primary_results['metrics'][TARGET_FEATURES[1]]
        
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            st.metric("RMSE", f"{metrics_bytes['rmse']:.4f}")
            st.metric("R² Score", f"{metrics_bytes['r2']:.4f}")
        with subcol2:
            st.metric("MAE", f"{metrics_bytes['mae']:.4f}")
            if metrics_bytes['mape'] != float('inf'):
                st.metric("MAPE (%)", f"{metrics_bytes['mape']:.2f}")
            else:
                st.metric("MAPE (%)", "N/A")
    
    # Visualization of primary results - Dual target plots
    st.markdown('<div class="section-header">Dual-Target Network Prediction Visualization</div>', unsafe_allow_html=True)
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')
    
    # Training data range
    train_range = np.arange(len(primary_results['y_train']))
    test_range = np.arange(len(primary_results['y_train']), 
                          len(primary_results['y_train']) + len(primary_results['y_test']))
    
    # Plottings for TCP/UDP Ratio - Packets
    y_train_packets = primary_results['y_train'][:, 0] if primary_results['y_train'].ndim > 1 else primary_results['y_train']
    y_test_packets = primary_results['metrics'][TARGET_FEATURES[0]]['y_true_aligned']
    y_pred_packets = primary_results['metrics'][TARGET_FEATURES[0]]['y_pred_aligned']
    
    ax1.plot(train_range, y_train_packets, label='Training Data', 
            color='#023a6c', alpha=0.7, linewidth=1)
    ax1.plot(test_range[:len(y_test_packets)], y_test_packets, label='Actual Ratio', 
            color='#28a745', linewidth=2)
    ax1.plot(test_range[:len(y_pred_packets)], y_pred_packets, label='Digital Twin Prediction', 
            color='#dc3545', linewidth=2, linestyle='--')
    ax1.axvline(x=len(primary_results['y_train']), color='#fd7e14', 
               linestyle=':', linewidth=2, label='Training/Prediction Split')
    
    ax1.set_title('TCP/UDP Ratio Prediction - Packets', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps (10-minute intervals)')
    ax1.set_ylabel('TCP/UDP Ratio (Packets)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot for TCP/UDP Ratio - Bytes
    y_train_bytes = primary_results['y_train'][:, 1] if primary_results['y_train'].ndim > 1 else primary_results['y_train']
    y_test_bytes = primary_results['metrics'][TARGET_FEATURES[1]]['y_true_aligned']
    y_pred_bytes = primary_results['metrics'][TARGET_FEATURES[1]]['y_pred_aligned']
    
    ax2.plot(train_range, y_train_bytes, label='Training Data', 
            color='#023a6c', alpha=0.7, linewidth=1)
    ax2.plot(test_range[:len(y_test_bytes)], y_test_bytes, label='Actual Ratio', 
            color='#28a745', linewidth=2)
    ax2.plot(test_range[:len(y_pred_bytes)], y_pred_bytes, label='Digital Twin Prediction', 
            color='#dc3545', linewidth=2, linestyle='--')
    ax2.axvline(x=len(primary_results['y_train']), color='#fd7e14', 
               linestyle=':', linewidth=2, label='Training/Prediction Split')
    
    ax2.set_title('TCP/UDP Ratio Prediction - Bytes', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Steps (10-minute intervals)')
    ax2.set_ylabel('TCP/UDP Ratio (Bytes)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Error analysis for both targets
    st.markdown('<div class="section-header">Prediction Error Analysis</div>', unsafe_allow_html=True)
    
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor('white')
    
    # Error plot for packets
    errors_packets = y_test_packets - y_pred_packets
    time_indices_packets = np.arange(len(errors_packets))
    ax1.plot(time_indices_packets, errors_packets, color='red', alpha=0.7, label="Prediction Errors")
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between(time_indices_packets, errors_packets, alpha=0.3, color='red')
    ax1.set_title('Prediction Errors - TCP/UDP Ratio (Packets)', fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Error (Actual - Predicted)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Error plot for bytes
    errors_bytes = y_test_bytes - y_pred_bytes
    time_indices_bytes = np.arange(len(errors_bytes))
    ax2.plot(time_indices_bytes, errors_bytes, color='blue', alpha=0.7, label="Prediction Errors")
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(time_indices_bytes, errors_bytes, alpha=0.3, color='blue')
    ax2.set_title('Prediction Errors - TCP/UDP Ratio (Bytes)', fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Error (Actual - Predicted)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.session_state.dt_params_applied = False

# Inference mode (when file is uploaded) - Modified for dual targets
elif inference_mode:
    st.header("Dual-Target Digital Twin Inference on Uploaded Data")

    if not hasattr(st.session_state, 'trained_model'):
        st.error("❌ No trained model found! Please train the digital twin first by:")
        st.markdown("1. Remove the uploaded file temporarily")
        st.markdown("2. Configure and train the dual-target model")
        st.markdown("3. Re-upload your file for inference")
        st.stop()
    
    if 'model_config' not in st.session_state:
        st.error("❌ Digital Twin Model configuration not found! Please train the model first.")
        st.stop()

    try:
        trained_model = st.session_state.trained_model
        trained_scaler = st.session_state.trained_scaler
        trained_model_type = st.session_state.trained_model_type
        model_config = st.session_state.model_config

        inference_observation_window = model_config['observation_window']
        inference_network_features = model_config['selected_network_features']
        inference_use_normalization = model_config['use_normalization']
        inference_target_features = model_config['target_features']

        with st.spinner("🔄 Running dual-target inference on uploaded data..."):
            if trained_model_type in ['lstm', 'cnn_lstm', 'gru', 'conv1d', 'bilstm']:
                X_new, y_new = create_sequences_dual_target(df_test, inference_target_features, inference_observation_window, inference_network_features)

                if len(X_new) == 0:
                    st.error(f"❌ Uploaded data is too short. Need at least {inference_observation_window + 1} observations for sequence creation.")
                    st.stop()

                if trained_scaler is not None:
                    X_new_reshaped = X_new.reshape(-1, X_new.shape[-1])
                    X_new_scaled = trained_scaler.transform(X_new_reshaped)
                    X_new = X_new_scaled.reshape(X_new.shape)

                y_pred_new = trained_model.predict(X_new, verbose=0)

            else:
                X_new, y_new = create_traditional_features_dual_target(df_test, inference_target_features, 10, inference_network_features)

                if len(X_new) == 0:
                    st.error("❌ Could not create features from uploaded data. Please check data format.")
                    st.stop()

                if trained_scaler is not None and len(X_new.columns) > 0:
                    X_new_scaled = trained_scaler.transform(X_new)
                    X_new = pd.DataFrame(X_new_scaled, columns=X_new.columns, index=X_new.index)

                if trained_model_type in ['arima', 'exp_smoothing']:
                    y_pred_new = []
                    for i, model in enumerate(trained_model):
                        y_pred_new.append(model.forecast(steps=len(y_new)))
                    y_pred_new = np.array(y_pred_new).T
                else:
                    y_pred_new = trained_model.predict(X_new)

        inference_metrics = calculate_metrics_dual_target(y_new, y_pred_new, inference_target_features)

        st.markdown('<div class="dual-target-header">📊 Dual-Target Inference Results</div>', unsafe_allow_html=True)
        
        # Performance metrics for both targets
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📦 TCP/UDP Ratio - Packets Performance")
            packets_metrics = inference_metrics[TARGET_FEATURES[0]]
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("🎯 RMSE", f"{packets_metrics['rmse']:.4f}")
                st.metric("📈 R² Score", f"{packets_metrics['r2']:.4f}")
            with subcol2:
                st.metric("📏 MAE", f"{packets_metrics['mae']:.4f}")
                st.metric("🎯 Direction Accuracy", f"{packets_metrics['direction_accuracy']:.1f}%")
        
        with col2:
            st.markdown("### 📊 TCP/UDP Ratio - Bytes Performance")
            bytes_metrics = inference_metrics[TARGET_FEATURES[1]]
            
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("🎯 RMSE", f"{bytes_metrics['rmse']:.4f}")
                st.metric("📈 R² Score", f"{bytes_metrics['r2']:.4f}")
            with subcol2:
                st.metric("📏 MAE", f"{bytes_metrics['mae']:.4f}")
                st.metric("🎯 Direction Accuracy", f"{bytes_metrics['direction_accuracy']:.1f}%")

        st.markdown('<div class="section-header">📈 Dual-Target Predictions on Uploaded Data</div>', unsafe_allow_html=True)
        
        # Creating dual-target visualization for inference
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 8))
        fig.patch.set_facecolor('white')
        
        # Plotting for TCP/UDP Ratio - Packets
        time_indices = np.arange(len(packets_metrics['y_true_aligned']))
        ax1.plot(time_indices, packets_metrics['y_true_aligned'], label="Actual Ratio", color="green", linewidth=2)
        ax1.plot(time_indices, packets_metrics['y_pred_aligned'], label="Digital Twin Prediction", color="red", linestyle="--", linewidth=2)
        ax1.set_title(f"TCP/UDP Ratio Inference - Packets\n{model_config['selected_twin_model']}")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("TCP/UDP Ratio (Packets)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')

        # plotting for TCP/UDP Ratio - Bytes
        time_indices = np.arange(len(bytes_metrics['y_true_aligned']))
        ax2.plot(time_indices, bytes_metrics['y_true_aligned'], label="Actual Ratio", color="blue", linewidth=2)
        ax2.plot(time_indices, bytes_metrics['y_pred_aligned'], label="Digital Twin Prediction", color="orange", linestyle="--", linewidth=2)
        ax2.set_title(f"TCP/UDP Ratio Inference - Bytes\n{model_config['selected_twin_model']}")
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("TCP/UDP Ratio (Bytes)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)

        # Error analysis for inference
        st.markdown('<div class="section-header">🔍 Inference Error Analysis</div>', unsafe_allow_html=True)
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 6))
        fig.patch.set_facecolor('white')
        
        # Error analysis for packets
        errors_packets = packets_metrics['y_true_aligned'] - packets_metrics['y_pred_aligned']
        time_indices = np.arange(len(errors_packets))
        ax1.plot(time_indices, errors_packets, color='red', alpha=0.7, label="Prediction Errors")
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.fill_between(time_indices, errors_packets, alpha=0.3, color='red')
        ax1.set_title('Prediction Errors - TCP/UDP Ratio (Packets)')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Error (Actual - Predicted)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # Error analysis for bytes
        errors_bytes = bytes_metrics['y_true_aligned'] - bytes_metrics['y_pred_aligned']
        ax2.plot(time_indices, errors_bytes, color='blue', alpha=0.7, label="Prediction Errors")
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(time_indices, errors_bytes, alpha=0.3, color='blue')
        ax2.set_title('Prediction Errors - TCP/UDP Ratio (Bytes)')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Error (Actual - Predicted)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('<div class="section-header">🎯 Dual-Target Quality Assessment</div>', unsafe_allow_html=True)
        
        # Combined performance assessment
        avg_r2 = (packets_metrics['r2'] + bytes_metrics['r2']) / 2
        avg_mape = (packets_metrics['mape'] + bytes_metrics['mape']) / 2 if packets_metrics['mape'] != float('inf') and bytes_metrics['mape'] != float('inf') else float('inf')
        
        if avg_r2 >= 0.7 and (avg_mape <= 10 or avg_mape == float('inf')):
            st.success("🎉 **Excellent Dual-Target Performance!** The digital twin performs very well on both TCP/UDP ratio predictions.")
        elif avg_r2 >= 0.5:
            st.info("👍 **Good Dual-Target Performance** - The digital twin provides reliable predictions for both targets.")
        elif avg_r2 >= 0.3:
            st.warning("⚠️ **Moderate Performance** - Some predictive capability shown, but may need domain adaptation.")
        else:
            st.error("🔴 **Poor Performance** - Consider retraining with more similar data.")

        st.markdown('<div class="section-header">🔍 Data Comparison Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Data Statistics:**")
            for target in TARGET_FEATURES:
                st.write(f"**{target}:**")
                st.write(f"- Mean: {df_train[target].mean():.4f}")
                st.write(f"- Std: {df_train[target].std():.4f}")
                st.write(f"- Min: {df_train[target].min():.4f}")
                st.write(f"- Max: {df_train[target].max():.4f}")

        with col2:
            st.write("**Uploaded Data Statistics:**")
            for target in TARGET_FEATURES:
                st.write(f"**{target}:**")
                st.write(f"- Mean: {df_test[target].mean():.4f}")
                st.write(f"- Std: {df_test[target].std():.4f}")
                st.write(f"- Min: {df_test[target].min():.4f}")
                st.write(f"- Max: {df_test[target].max():.4f}")

    except Exception as e:
        st.error(f"❌ Error during dual-target inference: {str(e)}")
        st.error("Please ensure your uploaded data has the same format and both target features.")

elif hasattr(st.session_state, 'trained_model'):
    st.markdown('<div class="twin-status operational">Dual-Target Digital Twin Model Ready for Deployment</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-card info"><strong>System Status:</strong> Pre-trained dual-target model is loaded and ready for inference. Upload a CSV file with both TCP/UDP ratio features to generate network predictions.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-header">Configuration Required</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-card warning"><strong>Configuration Pending:</strong> Please configure the dual-target digital twin parameters and click "Apply & Train" to begin training for TCP/UDP ratio prediction.</div>', unsafe_allow_html=True)
