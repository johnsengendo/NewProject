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

# Defining custom CSS styling for creating a professional network monitoring interface
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
</style>
""", unsafe_allow_html=True)

# Setting up directory paths and file configurations
DATA_DIR = "Data"
CSV_FILENAME = "543.csv"
CSV_FILEPATH = os.path.join(DATA_DIR, CSV_FILENAME)

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
    <p>Flexible Prediction Intervals & Training Period Analysis</p>
</div>
""", unsafe_allow_html=True)

# Loading the primary dataset from the CSV file
df_train = load_data(CSV_FILEPATH)

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

# Creating file upload section for inference mode
st.markdown('<div class="section-header">Inference with New Data Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Network Traffic Data (CSV format)", 
    type=["csv"], 
    help="CSV file must include 'n_flows' column for flow predictions"
)

# Handling file upload and setting inference mode
if uploaded_file is not None:
    df_test = load_uploaded_data(uploaded_file)
    st.markdown('<div class="status-card success">Custom network traffic data loaded successfully. Inference mode activated.</div>', unsafe_allow_html=True)
    inference_mode = True
else:
    df_test = None
    inference_mode = False

# Creating sidebar configuration panel
st.sidebar.markdown("## Enhanced Digital Twin Control Panel")
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
st.sidebar.markdown("### Standard Twin Parameters")

# Initializing all session state parameters with default values
if 'current_observation_window' not in st.session_state:
    st.session_state.current_observation_window = 20
if 'current_simulation_cycles' not in st.session_state:
    st.session_state.current_simulation_cycles = 50
if 'current_processing_batch' not in st.session_state:
    st.session_state.current_processing_batch = 32
if 'current_network_features' not in st.session_state:
    # Getting available network parameters from dataset
    network_parameters = df_train.columns.tolist()
    if "n_flows" in network_parameters:
        network_parameters.remove("n_flows")
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
if "n_flows" in network_parameters:
    network_parameters.remove("n_flows")
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
normalization_options = ["Standard Scaling"]
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

# Defining comprehensive model training function with flexible parameters
def train_model_with_params(data, train_size, test_size, model_type, features, obs_window, epochs, batch_size, use_norm, norm_method):
    """Training model with specific training and testing split parameters"""
    
    # Preparing data split and validating data sufficiency
    total_needed = train_size + test_size + obs_window
    if len(data) < total_needed:
        raise ValueError(f"Insufficient data. Need {total_needed}, have {len(data)}")
    
    # Using the most recent data for training and testing
    data_subset = data.iloc[-total_needed:].copy()
    
    try:
        # Processing neural network models that require sequence data
        if model_type in ['lstm', 'cnn_lstm', 'conv1d', 'gru', 'bilstm']:
            # Creating sequences for time series neural networks
            X_seq, y_seq = create_sequences(data_subset, 'n_flows', obs_window, features)
            
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
            X, y = create_traditional_features(data_subset, 'n_flows', 10, features)
            
            # Splitting traditional features into train and test sets
            X_train = X.iloc[:train_size]
            X_test = X.iloc[train_size:train_size + test_size]
            y_train = y.iloc[:train_size]
            y_test = y.iloc[train_size:train_size + test_size]
            
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
        
        # Training models based on selected type
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
            # Creating and training ARIMA model with parameter optimization
            best_aic = float('inf')
            best_model = None
            
            # Searching for optimal ARIMA parameters
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            arima_model = ARIMA(y_train, order=(p, d, q))
                            fitted_model = arima_model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                        except:
                            continue
            
            # Using default parameters if optimization fails
            if best_model is None:
                arima_model = ARIMA(y_train, order=(1, 1, 1))
                best_model = arima_model.fit()
            
            model = best_model
            y_pred = model.forecast(steps=len(y_test))
            
        elif model_type == "exp_smoothing":
            # Creating and training Exponential Smoothing model
            try:
                exp_model = ExponentialSmoothing(y_train, seasonal='add' if len(y_train) > 24 else None, 
                                               seasonal_periods=min(24, len(y_train)//2) if len(y_train) > 24 else None)
                model = exp_model.fit()
                y_pred = model.forecast(steps=len(y_test))
            except:
                # Falling back to simpler trend model if seasonal fails
                exp_model = ExponentialSmoothing(y_train, trend='add')
                model = exp_model.fit()
                y_pred = model.forecast(steps=len(y_test))
                
        elif model_type == "rf":
            # Creating and training Random Forest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        elif model_type == "gb":
            # Creating and training Gradient Boosting model
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculating comprehensive performance metrics
        metrics, y_test_aligned, y_pred_aligned = calculate_metrics(y_test, y_pred)
        
        # Returning training results and components
        return {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'y_test': y_test_aligned,
            'y_pred': y_pred_aligned,
            'y_train': y_train,
            'training_size': train_size,
            'test_size': test_size
        }
        
    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")
# Main training and analysis logic
if not inference_mode and st.session_state.dt_params_applied:
    model_type = twin_models[selected_twin_model]
    
    st.markdown('<div class="section-header">Digital Twin Training & Analysis</div>', unsafe_allow_html=True)
    
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
        st.markdown(f"**Model Configuration**")
        st.markdown(f"Architecture: {selected_twin_model}")
        st.markdown(f"Observation Window: {observation_window}")
        st.markdown(f"Features: {len(selected_network_features) + 1}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Training primary model
    with st.spinner(f"Training {selected_twin_model} with {convert_to_time_periods(training_size)} of data..."):
        try:
            primary_results = train_model_with_params(
                df_train, training_size, prediction_horizon, model_type, 
                selected_network_features, observation_window, simulation_cycles, 
                processing_batch, use_normalization, normalization_method
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
                'prediction_horizon': prediction_horizon
            }
            
            st.markdown('<div class="status-card success">✅ Digital Twin Successfully Calibrated and Operational</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'<div class="status-card error">Training failed: {str(e)}</div>', unsafe_allow_html=True)
            st.stop()
    
    # Displaying primary results
    st.markdown('### Primary Model Performance')
    metrics = primary_results['metrics']
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("RMSE", f"{metrics['rmse']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("MAE", f"{metrics['mae']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    # Comparison analysis
    if enable_comparison and comparison_intervals:
        st.markdown('<div class="section-header">Multi-Interval Performance Comparison</div>', unsafe_allow_html=True)
        
        comparison_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, interval_name in enumerate(comparison_intervals):
            if interval_name in prediction_horizons:
                interval_size = prediction_horizons[interval_name]
                # Ensuring we don't exceed available data
                interval_size = min(interval_size, len(df_train) - training_size)
                
                if interval_size > 10:  # Only train if we have enough data
                    status_text.text(f"Training for {interval_name} interval...")
                    
                    try:
                        result = train_model_with_params(
                            df_train, training_size, interval_size, model_type,
                            selected_network_features, observation_window, 
                            simulation_cycles, processing_batch, use_normalization, 
                            normalization_method
                        )
                        
                        comparison_results.append({
                            'interval': interval_name,
                            'data_points': interval_size,
                            'time_period': convert_to_time_periods(interval_size),
                            'rmse': result['metrics']['rmse'],
                            'mae': result['metrics']['mae'],
                            'mape': result['metrics']['mape'],
                            'r2': result['metrics']['r2'],
                            'direction_accuracy': result['metrics']['direction_accuracy'],
                            'y_test': result['y_test'],
                            'y_pred': result['y_pred']
                        })
                        
                    except Exception as e:
                        st.warning(f"Failed to train for {interval_name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(comparison_intervals))
        
        status_text.empty()
        progress_bar.empty()
        
        if comparison_results:
            # Creating comparison table
            st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
            st.markdown("### Performance Comparison Across Prediction Intervals")
            
            df_comparison = pd.DataFrame(comparison_results)
            df_comparison = df_comparison.round(3)
            
            # Formating the dataframe for better display
            df_display = df_comparison[['interval', 'time_period', 'rmse', 'mae', 'mape', 'r2', 'direction_accuracy']].copy()
            df_display.columns = ['Prediction Interval', 'Time Coverage', 'RMSE', 'MAE', 'MAPE (%)', 'R²', 'Direction Acc (%)']
            
            st.dataframe(df_display, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Performance degradation analysis
            st.markdown("### Performance Degradation Analysis")
            
            # Sorting by data points for degradation analysis
            df_comparison_sorted = df_comparison.sort_values('data_points')
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.patch.set_facecolor('white')
            
            # RMSE vs Prediction Horizon
            ax1.plot(df_comparison_sorted['data_points'], df_comparison_sorted['rmse'], 
                    marker='o', linewidth=2, markersize=8, color='#dc3545')
            ax1.set_xlabel('Prediction Horizon (Data Points)')
            ax1.set_ylabel('RMSE')
            ax1.set_title('RMSE vs Prediction Horizon', fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#f8f9fa')
            
            # R² vs Prediction Horizon
            ax2.plot(df_comparison_sorted['data_points'], df_comparison_sorted['r2'], 
                    marker='s', linewidth=2, markersize=8, color='#28a745')
            ax2.set_xlabel('Prediction Horizon (Data Points)')
            ax2.set_ylabel('R² Score')
            ax2.set_title('Model Accuracy vs Prediction Horizon', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#f8f9fa')
            
            # MAPE vs Prediction Horizon
            valid_mape = df_comparison_sorted[df_comparison_sorted['mape'] != float('inf')]
            if len(valid_mape) > 0:
                ax3.plot(valid_mape['data_points'], valid_mape['mape'], 
                        marker='^', linewidth=2, markersize=8, color='#fd7e14')
                ax3.set_xlabel('Prediction Horizon (Data Points)')
                ax3.set_ylabel('MAPE (%)')
                ax3.set_title('Relative Error vs Prediction Horizon', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.set_facecolor('#f8f9fa')
            
            # Direction Accuracy vs Prediction Horizon
            ax4.plot(df_comparison_sorted['data_points'], df_comparison_sorted['direction_accuracy'], 
                    marker='d', linewidth=2, markersize=8, color='#6f42c1')
            ax4.set_xlabel('Prediction Horizon (Data Points)')
            ax4.set_ylabel('Direction Accuracy (%)')
            ax4.set_title('Trend Prediction Accuracy vs Horizon', fontweight='bold')
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('#f8f9fa')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recommendations based on analysis
            st.markdown("### Trade-off Analysis & Recommendations")
            
            # Finding optimal trade-off points
            df_comparison_sorted['score'] = (df_comparison_sorted['r2'] * 0.4 + 
                                           (100 - df_comparison_sorted['mape'].clip(upper=100)) / 100 * 0.3 + 
                                           df_comparison_sorted['direction_accuracy'] / 100 * 0.3)
            
            best_overall = df_comparison_sorted.loc[df_comparison_sorted['score'].idxmax()]
            best_short_term = df_comparison_sorted[df_comparison_sorted['data_points'] <= 144].loc[df_comparison_sorted[df_comparison_sorted['data_points'] <= 144]['score'].idxmax()] if len(df_comparison_sorted[df_comparison_sorted['data_points'] <= 144]) > 0 else None
            best_long_term = df_comparison_sorted[df_comparison_sorted['data_points'] >= 1008].loc[df_comparison_sorted[df_comparison_sorted['data_points'] >= 1008]['score'].idxmax()] if len(df_comparison_sorted[df_comparison_sorted['data_points'] >= 1008]) > 0 else None
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="status-card info">', unsafe_allow_html=True)
                st.markdown(f"**Optimal Overall Performance**")
                st.markdown(f"Interval: {best_overall['interval']}")
                st.markdown(f"R²: {best_overall['r2']:.3f}")
                st.markdown(f"RMSE: {best_overall['rmse']:.3f}")
                st.markdown(f"MAPE: {best_overall['mape']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if best_short_term is not None:
                with col2:
                    st.markdown('<div class="status-card success">', unsafe_allow_html=True)
                    st.markdown(f"**Best Short-term (≤1 Day)**")
                    st.markdown(f"Interval: {best_short_term['interval']}")
                    st.markdown(f"R²: {best_short_term['r2']:.3f}")
                    st.markdown(f"RMSE: {best_short_term['rmse']:.3f}")
                    st.markdown(f"MAPE: {best_short_term['mape']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            if best_long_term is not None:
                with col3:
                    st.markdown('<div class="status-card warning">', unsafe_allow_html=True)
                    st.markdown(f"**Best Long-term (≥1 Week)**")
                    st.markdown(f"Interval: {best_long_term['interval']}")
                    st.markdown(f"R²: {best_long_term['r2']:.3f}")
                    st.markdown(f"RMSE: {best_long_term['rmse']:.3f}")
                    st.markdown(f"MAPE: {best_long_term['mape']:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualization of primary results - Only show main prediction plot without error analysis
    st.markdown('<div class="section-header">Network Flow Prediction Visualization</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    
    # Main prediction plot
    train_range = np.arange(len(primary_results['y_train']))
    test_range = np.arange(len(primary_results['y_train']), 
                          len(primary_results['y_train']) + len(primary_results['y_test']))
    
    ax.plot(train_range, primary_results['y_train'], label='Training Data', 
            color='#023a6c', alpha=0.7, linewidth=1)
    ax.plot(test_range, primary_results['y_test'], label='Actual Flow', 
            color='#28a745', linewidth=2)
    ax.plot(test_range, primary_results['y_pred'], label='Digital Twin Prediction', 
            color='#dc3545', linewidth=2, linestyle='--')
    ax.axvline(x=len(primary_results['y_train']), color='#fd7e14', 
               linestyle=':', linewidth=2, label='Training/Prediction Split')
    
    ax.set_title(f'Digital Twin Performance: {selected_twin_model} ({convert_to_time_periods(prediction_horizon)} Prediction)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Time Steps (10-minute intervals)')
    ax.set_ylabel('Network Flow Count')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Training period impact analysis
    st.markdown('<div class="section-header">Training Period Impact Analysis</div>', unsafe_allow_html=True)
    
    # Testing different training periods
    training_period_analysis = st.checkbox("Enable Training Period Analysis", 
                                         value=False, 
                                         help="Compare performance with different training period lengths")
    
    if training_period_analysis:
        st.markdown("### Training Data Size vs Performance")
        
        # Defining training sizes to test
        max_training = len(df_train) - prediction_horizon - observation_window
        training_sizes_to_test = [
            min(7 * 24 * 6, max_training),      # 1 week
            min(14 * 24 * 6, max_training),     # 2 weeks  
            min(30 * 24 * 6, max_training),     # 1 month
            min(60 * 24 * 6, max_training),     # 2 months
            min(90 * 24 * 6, max_training),     # 3 months
            max_training                         # All available
        ]
        
        training_sizes_to_test = [size for size in training_sizes_to_test if size > observation_window + 100]
        
        if len(training_sizes_to_test) > 1:
            training_analysis_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, train_size in enumerate(training_sizes_to_test):
                status_text.text(f"Testing training size: {convert_to_time_periods(train_size)}")
                
                try:
                    result = train_model_with_params(
                        df_train, train_size, prediction_horizon, model_type,
                        selected_network_features, observation_window, simulation_cycles,
                        processing_batch, use_normalization, normalization_method
                    )
                    
                    training_analysis_results.append({
                        'training_size': train_size,
                        'training_period': convert_to_time_periods(train_size),
                        'rmse': result['metrics']['rmse'],
                        'mae': result['metrics']['mae'],
                        'mape': result['metrics']['mape'],
                        'r2': result['metrics']['r2'],
                        'direction_accuracy': result['metrics']['direction_accuracy']
                    })
                    
                except Exception as e:
                    st.warning(f"Failed training with {convert_to_time_periods(train_size)}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(training_sizes_to_test))
            
            status_text.empty()
            progress_bar.empty()
            
            if len(training_analysis_results) > 1:
                df_training_analysis = pd.DataFrame(training_analysis_results)
                
                # Displaying results table
                st.markdown("### Training Period Performance Comparison")
                df_display = df_training_analysis[['training_period', 'rmse', 'mae', 'mape', 'r2', 'direction_accuracy']].copy()
                df_display.columns = ['Training Period', 'RMSE', 'MAE', 'MAPE (%)', 'R²', 'Direction Acc (%)']
                df_display = df_display.round(3)
                st.dataframe(df_display, use_container_width=True)
                
                # Visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                fig.patch.set_facecolor('white')
                
                training_days = [size / (24 * 6) for size in df_training_analysis['training_size']]
                
                ax1.plot(training_days, df_training_analysis['rmse'], 
                        marker='o', linewidth=2, markersize=8, color='#dc3545')
                ax1.set_xlabel('Training Period (Days)')
                ax1.set_ylabel('RMSE')
                ax1.set_title('RMSE vs Training Period', fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.set_facecolor('#f8f9fa')
                
                ax2.plot(training_days, df_training_analysis['r2'], 
                        marker='s', linewidth=2, markersize=8, color='#28a745')
                ax2.set_xlabel('Training Period (Days)')
                ax2.set_ylabel('R² Score')
                ax2.set_title('Model Accuracy vs Training Period', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_facecolor('#f8f9fa')
                
                valid_mape_training = df_training_analysis[df_training_analysis['mape'] != float('inf')]
                if len(valid_mape_training) > 0:
                    ax3.plot([size / (24 * 6) for size in valid_mape_training['training_size']], 
                            valid_mape_training['mape'], 
                            marker='^', linewidth=2, markersize=8, color='#fd7e14')
                    ax3.set_xlabel('Training Period (Days)')
                    ax3.set_ylabel('MAPE (%)')
                    ax3.set_title('Relative Error vs Training Period', fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                    ax3.set_facecolor('#f8f9fa')
                
                ax4.plot(training_days, df_training_analysis['direction_accuracy'], 
                        marker='d', linewidth=2, markersize=8, color='#6f42c1')
                ax4.set_xlabel('Training Period (Days)')
                ax4.set_ylabel('Direction Accuracy (%)')
                ax4.set_title('Trend Prediction vs Training Period', fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.set_facecolor('#f8f9fa')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Finding optimal training period
                df_training_analysis['training_score'] = (
                    df_training_analysis['r2'] * 0.4 + 
                    (100 - df_training_analysis['mape'].clip(upper=100)) / 100 * 0.3 + 
                    df_training_analysis['direction_accuracy'] / 100 * 0.3
                )
                
                optimal_training = df_training_analysis.loc[df_training_analysis['training_score'].idxmax()]
                
                st.markdown("### Optimal Training Period Recommendation")
                st.markdown('<div class="status-card info">', unsafe_allow_html=True)
                st.markdown(f"**Recommended Training Period**: {optimal_training['training_period']}")
                st.markdown(f"**Performance Metrics**:")
                st.markdown(f"- R² Score: {optimal_training['r2']:.3f}")
                st.markdown(f"- RMSE: {optimal_training['rmse']:.3f}")
                st.markdown(f"- MAPE: {optimal_training['mape']:.1f}%")
                st.markdown(f"- Direction Accuracy: {optimal_training['direction_accuracy']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.session_state.dt_params_applied = False

# Inference mode (when file is uploaded)
elif inference_mode:
    st.header("Digital Twin Inference on Uploaded Data")

    if not hasattr(st.session_state, 'trained_model'):
        st.error("❌ No trained model found! Please train the digital twin first by:")
        st.markdown("1. Remove the uploaded file temporarily")
        st.markdown("2. Configure and train the model")
        st.markdown("3. Re-upload your file for inference")
        st.stop()
    if 'model_config' not in st.session_state:
        st.error("❌ Model configuration not found! Please train the model first.")
        st.stop()

    try:
        trained_model = st.session_state.trained_model
        trained_scaler = st.session_state.trained_scaler
        trained_model_type = st.session_state.trained_model_type
        model_config = st.session_state.model_config

        inference_observation_window = model_config['observation_window']
        inference_network_features = model_config['selected_network_features']
        inference_use_normalization = model_config['use_normalization']

        with st.spinner("🔄 Running inference on uploaded data..."):
            if trained_model_type in ['lstm', 'cnn_lstm', 'gru', 'conv1d', 'bilstm']:
                X_new, y_new = create_sequences(df_test, 'n_flows', inference_observation_window, inference_network_features)

                if len(X_new) == 0:
                    st.error(f"❌ Uploaded data is too short. Need at least {inference_observation_window + 1} observations for sequence creation.")
                    st.stop()

                if trained_scaler is not None:
                    X_new_reshaped = X_new.reshape(-1, X_new.shape[-1])
                    X_new_scaled = trained_scaler.transform(X_new_reshaped)
                    X_new = X_new_scaled.reshape(X_new.shape)

                y_pred_new = trained_model.predict(X_new, verbose=0).flatten()

            else:
                X_new, y_new = create_traditional_features(df_test, 'n_flows', 10, inference_network_features)

                if len(X_new) == 0:
                    st.error("❌ Could not create features from uploaded data. Please check data format.")
                    st.stop()

                if trained_scaler is not None and len(X_new.columns) > 0:
                    X_new_scaled = trained_scaler.transform(X_new)
                    X_new = pd.DataFrame(X_new_scaled, columns=X_new.columns, index=X_new.index)

                if trained_model_type in ['arima', 'exp_smoothing']:
                    y_pred_new = trained_model.forecast(steps=len(y_new))
                else:
                    y_pred_new = trained_model.predict(X_new)

        inference_metrics, y_new_aligned, y_pred_new_aligned = calculate_metrics(y_new, y_pred_new)

        st.subheader("📊 Inference Results")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🎯 Prediction Error (RMSE)", f"{inference_metrics['rmse']:.3f}")
        with col2:
            st.metric("📏 Average Deviation (MAE)", f"{inference_metrics['mae']:.3f}")
        with col3:
            st.metric("📊 Relative Error (%)", f"{inference_metrics['mape']:.1f}%" if inference_metrics['mape'] != float('inf') else "N/A")
        with col4:
            st.metric("📈 Model Performance (R²)", f"{inference_metrics['r2']:.3f}")
        with col5:
            st.metric("🎯 Trend Accuracy", f"{inference_metrics['direction_accuracy']:.1f}%")

        st.subheader("📈 Predictions on Uploaded Network Data")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        time_indices = np.arange(len(y_new_aligned))
        ax1.plot(time_indices, y_new_aligned, label="Actual Flow", color="green", linewidth=2)
        ax1.plot(time_indices, y_pred_new_aligned, label="Digital Twin Prediction", color="red", linestyle="--", linewidth=2)
        ax1.set_title(f"Digital Twin Inference: {model_config['selected_twin_model']} on Uploaded Data")
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Network Flow Count")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        errors_new = y_new_aligned - y_pred_new_aligned
        ax2.plot(time_indices, errors_new, color='red', alpha=0.7, label="Prediction Errors")
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(time_indices, errors_new, alpha=0.3, color='red')
        ax2.set_title('Prediction Errors on Uploaded Data')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Error (Actual - Predicted)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("🎯 Inference Quality Assessment")
        if inference_metrics['mape'] <= 10 and inference_metrics['r2'] >= 0.7:
            st.success("🎉 **Excellent Inference Results!** The digital twin performs very well on your uploaded data.")
        elif inference_metrics['mape'] <= 20 and inference_metrics['r2'] >= 0.5:
            st.info("👍 **Good Inference Performance** - The digital twin provides reliable predictions on your data.")
        elif inference_metrics['r2'] >= 0.3:
            st.warning("⚠️ **Moderate Performance** - The digital twin shows some predictive capability but may need domain adaptation.")
        else:
            st.error("🔴 **Poor Performance** - The uploaded data may be significantly different from training data. Consider retraining with similar data.")

        st.subheader("🔍 Data Comparison Insights")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training Data Statistics:**")
            st.write(f"- Mean: {df_train['n_flows'].mean():.2f}")
            st.write(f"- Std: {df_train['n_flows'].std():.2f}")
            st.write(f"- Min: {df_train['n_flows'].min():.2f}")
            st.write(f"- Max: {df_train['n_flows'].max():.2f}")

        with col2:
            st.write("**Uploaded Data Statistics:**")
            st.write(f"- Mean: {df_test['n_flows'].mean():.2f}")
            st.write(f"- Std: {df_test['n_flows'].std():.2f}")
            st.write(f"- Min: {df_test['n_flows'].min():.2f}")
            st.write(f"- Max: {df_test['n_flows'].max():.2f}")

        train_mean = df_train['n_flows'].mean()
        test_mean = df_test['n_flows'].mean()
        mean_diff_pct = abs(test_mean - train_mean) / train_mean * 100

        if mean_diff_pct <= 10:
            st.success(f"✅ **Data Similarity**: Uploaded data is very similar to training data (mean difference: {mean_diff_pct:.1f}%)")
        elif mean_diff_pct <= 25:
            st.info(f"ℹ️ **Moderate Similarity**: Some differences detected (mean difference: {mean_diff_pct:.1f}%)")
        else:
            st.warning(f"⚠️ **Data Drift Detected**: Significant differences from training data (mean difference: {mean_diff_pct:.1f}%). Consider model adaptation.")

    except Exception as e:
        st.error(f"❌ Error during inference on uploaded data: {str(e)}")
        st.error("Please ensure your uploaded data has the same format and features as the training data.")


elif hasattr(st.session_state, 'trained_model'):
    st.markdown('<div class="twin-status operational">Digital Twin Model Ready for Deployment</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-card info"><strong>System Status:</strong> Pre-trained model is loaded and ready for inference. Upload a CSV file to generate network flow predictions.</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="section-header">Configuration Required</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-card warning"><strong>Configuration Pending:</strong> Please configure the digital twin parameters and click "Apply & Train" to begin training.</div>', unsafe_allow_html=True)