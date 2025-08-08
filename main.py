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

# Custom CSS for professional network theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
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
    
    .status-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
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
    
    .metric-container {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        background: #343a40;
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .config-section {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .twin-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .twin-status.operational {
        background: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    
    .twin-status.calibrating {
        background: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    
    .twin-status.error {
        background: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stSelectbox label, .stSlider label, .stMultiSelect label, .stCheckbox label {
        font-weight: 600;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

# Configurations and Setup
DATA_DIR = "Data"
CSV_FILENAME = "543.csv"
CSV_FILEPATH = os.path.join(DATA_DIR, CSV_FILENAME)

# Page setup
st.set_page_config(
    page_title="🌐Network Digital Twin Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main header with professional styling
st.markdown("""
<div class="main-header">
    <h1>🌐Network Digital Twin Platform</h1>
    <p>Advanced Network Flow Modeling and Predictive Analytics System</p>
</div>
""", unsafe_allow_html=True)

# Loading data
df_train = load_data(CSV_FILEPATH)

# Initializing uploaded file variable
uploaded_file = None

if uploaded_file is not None:
    df_test = load_uploaded_data(uploaded_file)
    st.markdown('<div class="status-card success">Custom network traffic data loaded successfully. Inference mode activated.</div>', unsafe_allow_html=True)
    inference_mode = True
else:
    df_test = None
    inference_mode = False

# Sidebar configuration with styling
st.sidebar.markdown("## Digital Twin Control Panel")
st.sidebar.markdown("---")

# Initializing session state for parameters if not exists
if 'dt_params_applied' not in st.session_state:
    st.session_state.dt_params_applied = False
if 'current_observation_window' not in st.session_state:
    st.session_state.current_observation_window = 20
if 'current_simulation_cycles' not in st.session_state:
    st.session_state.current_simulation_cycles = 50
if 'current_processing_batch' not in st.session_state:
    st.session_state.current_processing_batch = 32
if 'current_network_features' not in st.session_state:
    network_parameters = df_train.columns.tolist()
    if "n_flows" in network_parameters:
        network_parameters.remove("n_flows")
    st.session_state.current_network_features = network_parameters[:2] if len(network_parameters) >= 2 else network_parameters
if 'current_twin_model' not in st.session_state:
    st.session_state.current_twin_model = "Neural Network Twin (LSTM)"
if 'current_validation_period' not in st.session_state:
    st.session_state.current_validation_period = min(200, len(df_train)//4)
if 'current_use_normalization' not in st.session_state:
    st.session_state.current_use_normalization = True
if 'current_normalization_method' not in st.session_state:
    st.session_state.current_normalization_method = "MinMax Scaling"

# Control buttons
st.sidebar.markdown("### System Controls")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Reset Configuration", help="Reset all parameters to default values", type="secondary"):
        st.session_state.current_observation_window = 20
        st.session_state.current_simulation_cycles = 50
        st.session_state.current_processing_batch = 32
        network_parameters = df_train.columns.tolist()
        if "n_flows" in network_parameters:
            network_parameters.remove("n_flows")
        st.session_state.current_network_features = network_parameters[:2] if len(network_parameters) >= 2 else network_parameters
        st.session_state.current_twin_model = "Neural Network Twin (LSTM)"
        st.session_state.current_validation_period = min(200, len(df_train)//4)
        st.session_state.current_use_normalization = True
        st.session_state.current_normalization_method = "MinMax Scaling"
        st.session_state.dt_params_applied = False
        st.experimental_rerun()
with col2:
    apply_settings = st.button("Apply Configuration", help="Apply current settings to digital twin", type="primary")

# Network Digital Twin specific parameters
st.sidebar.markdown("### Twin Architecture Parameters")
temp_observation_window = st.sidebar.slider(
    "Observation Window", 
    5, 100, 
    st.session_state.current_observation_window, 
    help="Number of historical time steps for pattern recognition"
)
temp_simulation_cycles = st.sidebar.slider(
    "Simulation_cycles", 
    10, 200, 
    st.session_state.current_simulation_cycles, 
    help="Number of training iterations for twin calibration"
)
batch_options = [16, 32, 64, 128]
current_batch_index = batch_options.index(st.session_state.current_processing_batch) if st.session_state.current_processing_batch in batch_options else 1
temp_processing_batch = st.sidebar.selectbox(
    "Processing_batch", 
    batch_options, 
    index=current_batch_index, 
    help="Data batch size for twin processing"
)

# Network parameter selection
st.sidebar.markdown("### Network Feature Selection")
network_parameters = df_train.columns.tolist()
if "n_flows" in network_parameters:
    network_parameters.remove("n_flows")
temp_selected_network_features = st.sidebar.multiselect(
    "Additional Network Parameters", 
    options=network_parameters, 
    default=st.session_state.current_network_features, 
    help="Additional network metrics to include in the digital twin model"
)

# Digital Twin Model selection
st.sidebar.markdown("### Twin Model Architecture")
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
current_model_index = list(twin_models.keys()).index(st.session_state.current_twin_model) if st.session_state.current_twin_model in twin_models.keys() else 0
temp_selected_twin_model = st.sidebar.selectbox(
    "Digital Twin Engine", 
    list(twin_models.keys()), 
    index=current_model_index
)

# Setting Validation configuration
st.sidebar.markdown("### Validation Configuration")
temp_validation_period = st.sidebar.slider(
    "Validation Period", 
    500, 
    min(1000, len(df_train)//2), 
    st.session_state.current_validation_period, 
    help="Number of recent observations for twin validation"
)

# Data preprocessing
st.sidebar.markdown("### Data Preprocessing")
temp_use_normalization = st.sidebar.checkbox(
    "Enable Data Normalization", 
    value=st.session_state.current_use_normalization, 
    help="Normalize input data for optimal twin performance"
)
normalization_options = ["MinMax Scaling", "Standard Scaling"]
current_norm_index = normalization_options.index(st.session_state.current_normalization_method) if st.session_state.current_normalization_method in normalization_options else 0
temp_normalization_method = st.sidebar.selectbox(
    "Normalization Method", 
    normalization_options, 
    index=current_norm_index, 
    help="Method for data preprocessing"
)

# Applying settings when the button is clicked
if apply_settings:
    st.session_state.current_observation_window = temp_observation_window
    st.session_state.current_simulation_cycles = temp_simulation_cycles
    st.session_state.current_processing_batch = temp_processing_batch
    st.session_state.current_network_features = temp_selected_network_features
    st.session_state.current_twin_model = temp_selected_twin_model
    st.session_state.current_validation_period = temp_validation_period
    st.session_state.current_use_normalization = temp_use_normalization
    st.session_state.current_normalization_method = temp_normalization_method
    st.session_state.dt_params_applied = True
    st.success("✅ Digital Twin settings applied successfully!")

# Using applied settings for processing
observation_window = st.session_state.current_observation_window
simulation_cycles = st.session_state.current_simulation_cycles
processing_batch = st.session_state.current_processing_batch
selected_network_features = st.session_state.current_network_features
selected_twin_model = st.session_state.current_twin_model
validation_period = st.session_state.current_validation_period
use_normalization = st.session_state.current_use_normalization
normalization_method = st.session_state.current_normalization_method

def check_model_compatibility():
    if 'model_config' not in st.session_state:
        return False

    config = st.session_state.model_config
    return (
        config['observation_window'] == observation_window and
        config['selected_network_features'] == selected_network_features and
        config['use_normalization'] == use_normalization and
        config['normalization_method'] == normalization_method and
        config['selected_twin_model'] == selected_twin_model
    )

# Main Logic: Training or Inference
need_training = (
    not inference_mode and
    (not hasattr(st.session_state, 'trained_model') or
     not check_model_compatibility() or
     st.session_state.dt_params_applied)
)

if need_training and (st.session_state.dt_params_applied or not any([key.startswith('current_') for key in st.session_state.keys()])):
    model_type = twin_models[selected_twin_model]
    
    # Showing training status
    st.markdown('<div class="twin-status calibrating">Digital Twin Calibration in Progress</div>', unsafe_allow_html=True)

    if model_type in ['lstm', 'cnn_lstm', 'conv1d', 'gru', 'bilstm']:
        with st.spinner("✅Calibrating neural network architecture..."):
            X_seq, y_seq = create_sequences(df_train, 'n_flows', observation_window, selected_network_features)
            train_size = len(X_seq) - validation_period
            X_train, X_test = X_seq[:train_size], X_seq[train_size:]
            y_train, y_test = y_seq[:train_size], y_seq[train_size:]

            scaler = None
            if use_normalization:
                if normalization_method == "MinMax Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()

                X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
                X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

                X_train_scaled = scaler.fit_transform(X_train_reshaped)
                X_test_scaled = scaler.transform(X_test_reshaped)

                X_train = X_train_scaled.reshape(X_train.shape)
                X_test = X_test_scaled.reshape(X_test.shape)

    else:
        with st.spinner("Configuring statistical model architecture..."):
            X, y = create_traditional_features(df_train, 'n_flows', 10, selected_network_features)
            train_size = len(X) - validation_period
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

            scaler = None
            if use_normalization and len(X.columns) > 0:
                if normalization_method == "MinMax Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()

                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    with st.spinner(f"Training {selected_twin_model} architecture..."):
        try:
            if model_type == 'lstm':
                model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(X_train, y_train, epochs=simulation_cycles, batch_size=processing_batch, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test, verbose=0).flatten()

            elif model_type == 'cnn_lstm':
                model = create_cnn_lstm_model((X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(X_train, y_train, epochs=simulation_cycles, batch_size=processing_batch, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test, verbose=0).flatten()

            elif model_type == 'conv1d':
                model = create_conv1d_model((X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(X_train, y_train, epochs=simulation_cycles, batch_size=processing_batch, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test, verbose=0).flatten()

            elif model_type == 'gru':
                model = create_gru_model((X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(X_train, y_train, epochs=simulation_cycles, batch_size=processing_batch, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test, verbose=0).flatten()

            elif model_type == 'bilstm':
                model = create_bidirectional_lstm_model((X_train.shape[1], X_train.shape[2]))
                early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
                history = model.fit(X_train, y_train, epochs=simulation_cycles, batch_size=processing_batch, validation_split=0.2, callbacks=[early_stopping], verbose=0)
                y_pred = model.predict(X_test, verbose=0).flatten()

            elif model_type == "arima":
                best_aic = float('inf')
                best_model = None

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

                if best_model is None:
                    arima_model = ARIMA(y_train, order=(1, 1, 1))
                    best_model = arima_model.fit()

                model = best_model
                y_pred = model.forecast(steps=len(y_test))

            elif model_type == "exp_smoothing":
                try:
                    exp_model = ExponentialSmoothing(y_train, seasonal='add' if len(y_train) > 24 else None, seasonal_periods=min(24, len(y_train)//2) if len(y_train) > 24 else None)
                    model = exp_model.fit()
                    y_pred = model.forecast(steps=len(y_test))
                except:
                    exp_model = ExponentialSmoothing(y_train, trend='add')
                    model = exp_model.fit()
                    y_pred = model.forecast(steps=len(y_test))

            elif model_type == "rf":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            elif model_type == "gb":
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

        except Exception as e:
            st.markdown(f'<div class="status-card error">Digital twin calibration failed: {e}</div>', unsafe_allow_html=True)
            st.stop()

    # Storing model and configurations
    st.session_state.trained_model = model
    st.session_state.trained_scaler = scaler
    st.session_state.trained_model_type = model_type
    st.session_state.model_config = {
        'observation_window': observation_window,
        'selected_network_features': selected_network_features,
        'use_normalization': use_normalization,
        'normalization_method': normalization_method,
        'selected_twin_model': selected_twin_model
    }
    
    st.markdown('<div class="twin-status operational">Digital Twin Successfully Calibrated and Operational</div>', unsafe_allow_html=True)

    # Calculating metrics
    metrics, y_test_aligned, y_pred_aligned = calculate_metrics(y_test, y_pred)

    # Performance Dashboard
    st.markdown('<div class="section-header">Digital Twin Performance Dashboard</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Prediction Error (RMSE)", f"{metrics['rmse']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Average Deviation (MAE)", f"{metrics['mae']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Relative Error (%)", f"{metrics['mape']:.1f}%" if metrics['mape'] != float('inf') else "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Twin Fidelity (R²)", f"{metrics['r2']:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Traffic Trend Accuracy", f"{metrics['direction_accuracy']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    # Performance Assessment
    if metrics['r2'] >= 0.8 and metrics['mape'] <= 10:
        st.markdown('<div class="status-card success"><strong>Production Ready:</strong> High fidelity network representation achieved. Digital twin is ready for operational deployment.</div>', unsafe_allow_html=True)
    elif metrics['r2'] >= 0.6 and metrics['mape'] <= 20:
        st.markdown('<div class="status-card info"><strong>Operational Status:</strong> Digital twin is suitable for network monitoring and basic predictions.</div>', unsafe_allow_html=True)
    elif metrics['r2'] >= 0.4:
        st.markdown('<div class="status-card warning"><strong>Calibration Required:</strong> Consider parameter tuning or additional network features for improved performance.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-card error"><strong>Reconfiguration Needed:</strong> Significant improvements required for reliable operation.</div>', unsafe_allow_html=True)

    # Visualization
    st.markdown('<div class="section-header">Network Flow Prediction Analysis</div>', unsafe_allow_html=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Setting professional styling for plots
    plt.style.use('default')
    fig.patch.set_facecolor('white')
    
    total_len = len(y_train) + len(y_test_aligned)
    train_range = np.arange(len(y_train))
    test_range = np.arange(len(y_train), len(y_train) + len(y_test_aligned))
    
    ax1.plot(train_range, y_train, label='Historical Network Data', color="#080c79", alpha=0.7, linewidth=1)
    ax1.plot(test_range, y_test_aligned, label='Actual Network Flow', color='#28a745', linewidth=2)
    ax1.plot(test_range, y_pred_aligned, label=f'Digital Twin Prediction', color='#dc3545', linewidth=2, linestyle='--')
    ax1.axvline(x=len(y_train), color='#fd7e14', linestyle=':', linewidth=2, label='Training/Validation Split')
    ax1.set_title(f'Digital Twin Performance: {selected_twin_model}', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Network Flow Count')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')

    test_start = max(0, len(y_train) - 50)
    zoom_train_range = np.arange(test_start, len(y_train))
    zoom_train_data = y_train[test_start:] if hasattr(y_train, '__getitem__') else y_train.iloc[test_start:]
    ax2.plot(zoom_train_range, zoom_train_data, label='Recent Historical Data', color='#6c757d', alpha=0.5, linewidth=1)
    ax2.plot(test_range, y_test_aligned, label='Actual Network Flow', color='#28a745', linewidth=2)
    ax2.plot(test_range, y_pred_aligned, label=f'Twin Prediction', color='#dc3545', linewidth=2, linestyle='--')
    ax2.axvline(x=len(y_train), color='#fd7e14', linestyle=':', linewidth=2, label='Validation Period Start')
    ax2.set_title('Detailed Validation Period Analysis', fontsize=12, pad=20)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Network Flow Count')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    st.pyplot(fig)

    # Error Analysis
    st.markdown('<div class="section-header">Prediction Accuracy Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        errors = y_test_aligned - y_pred_aligned
        error_range = np.arange(len(errors))

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.plot(error_range, errors, color='#dc3545', alpha=0.7)
        ax.axhline(y=0, color='#343a40', linestyle='--', alpha=0.5)
        ax.fill_between(error_range, errors, alpha=0.3, color='#dc3545')
        ax.set_title('Prediction Errors Over Time', fontweight='bold', pad=20)
        ax.set_xlabel('Validation Time Steps')
        ax.set_ylabel('Error (Actual - Predicted)')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)
        
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.hist(errors, bins=30, alpha=0.7, color='#17a2b8', edgecolor='#343a40')
        ax.axvline(x=0, color='#dc3545', linestyle='--', linewidth=2)
        ax.set_title('Error Distribution Analysis', fontweight='bold', pad=20)
        ax.set_xlabel('Error (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#f8f9fa')
        st.pyplot(fig)

    st.session_state.dt_params_applied = False

elif inference_mode:
    st.markdown('<div class="section-header">Digital Twin Inference Mode</div>', unsafe_allow_html=True)

    if not hasattr(st.session_state, 'trained_model'):
        st.markdown('<div class="status-card error"><strong>No Trained Model Available</strong><br>Please train the digital twin first by:<br>1. Remove the uploaded file temporarily<br>2. Configure and train the model<br>3. Re-upload your file for inference</div>', unsafe_allow_html=True)
        st.stop()
        
    if 'model_config' not in st.session_state:
        st.markdown('<div class="status-card error"><strong>Model Configuration Missing</strong><br>Please train the model first.</div>', unsafe_allow_html=True)
        st.stop()

    try:
        trained_model = st.session_state.trained_model
        trained_scaler = st.session_state.trained_scaler
        trained_model_type = st.session_state.trained_model_type
        model_config = st.session_state.model_config

        inference_observation_window = model_config['observation_window']
        inference_network_features = model_config['selected_network_features']
        inference_use_normalization = model_config['use_normalization']

        with st.spinner("Processing inference on uploaded network data..."):
            if trained_model_type in ['lstm', 'cnn_lstm', 'gru', 'conv1d', 'bilstm']:
                X_new, y_new = create_sequences(df_test, 'n_flows', inference_observation_window, inference_network_features)

                if len(X_new) == 0:
                    st.markdown(f'<div class="status-card error"><strong>Insufficient Data</strong><br>Uploaded data is too short. Need at least {inference_observation_window + 1} observations for sequence creation.</div>', unsafe_allow_html=True)
                    st.stop()

                if trained_scaler is not None:
                    X_new_reshaped = X_new.reshape(-1, X_new.shape[-1])
                    X_new_scaled = trained_scaler.transform(X_new_reshaped)
                    X_new = X_new_scaled.reshape(X_new.shape)

                y_pred_new = trained_model.predict(X_new, verbose=0).flatten()

            else:
                X_new, y_new = create_traditional_features(df_test, 'n_flows', 10, inference_network_features)

                if len(X_new) == 0:
                    st.markdown('<div class="status-card error"><strong>Feature Creation Failed</strong><br>Could not create features from uploaded data. Please check data format.</div>', unsafe_allow_html=True)
                    st.stop()

                if trained_scaler is not None and len(X_new.columns) > 0:
                    X_new_scaled = trained_scaler.transform(X_new)
                    X_new = pd.DataFrame(X_new_scaled, columns=X_new.columns, index=X_new.index)

                if trained_model_type in ['arima', 'exp_smoothing']:
                    y_pred_new = trained_model.forecast(steps=len(y_new))
                else:
                    y_pred_new = trained_model.predict(X_new)

        inference_metrics, y_new_aligned, y_pred_new_aligned = calculate_metrics(y_new, y_pred_new)

        # Showing Inference Results Dashboard
        st.markdown("### Inference Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Prediction Error (RMSE)", f"{inference_metrics['rmse']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Average Deviation (MAE)", f"{inference_metrics['mae']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Relative Error (%)", f"{inference_metrics['mape']:.1f}%" if inference_metrics['mape'] != float('inf') else "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Model Performance (R²)", f"{inference_metrics['r2']:.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Trend Accuracy", f"{inference_metrics['direction_accuracy']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        # Defining Inference Visualization
        st.markdown("### Network Flow Predictions on Uploaded Data")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        fig.patch.set_facecolor('white')
        
        time_indices = np.arange(len(y_new_aligned))
        ax1.plot(time_indices, y_new_aligned, label="Actual Flow", color="#28a745", linewidth=2)
        ax1.plot(time_indices, y_pred_new_aligned, label="Digital Twin Prediction", color="#dc3545", linestyle="--", linewidth=2)
        ax1.set_title(f"Digital Twin Inference: {model_config['selected_twin_model']}", fontweight='bold', pad=20)
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Network Flow Count")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')

        errors_new = y_new_aligned - y_pred_new_aligned
        ax2.plot(time_indices, errors_new, color='#dc3545', alpha=0.7, label="Prediction Errors")
        ax2.axhline(y=0, color='#343a40', linestyle='--', alpha=0.5)
        ax2.fill_between(time_indices, errors_new, alpha=0.3, color='#dc3545')
        ax2.set_title('Prediction Error Analysis', fontweight='bold', pad=20)
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Error (Actual - Predicted)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        st.pyplot(fig)

        # Doing Performance Assessment for Inference
        st.markdown("### Inference Quality Assessment")
        if inference_metrics['mape'] <= 10 and inference_metrics['r2'] >= 0.7:
            st.markdown('<div class="status-card success"><strong>Excellent Inference Performance</strong><br>The digital twin performs exceptionally well on your uploaded data with high accuracy and reliability.</div>', unsafe_allow_html=True)
        elif inference_metrics['mape'] <= 20 and inference_metrics['r2'] >= 0.5:
            st.markdown('<div class="status-card info"><strong>Good Inference Performance</strong><br>The digital twin provides reliable predictions on your data with acceptable accuracy levels.</div>', unsafe_allow_html=True)
        elif inference_metrics['r2'] >= 0.3:
            st.markdown('<div class="status-card warning"><strong>Moderate Performance</strong><br>The digital twin shows predictive capability but may require domain adaptation for optimal results.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card error"><strong>Performance Issues Detected</strong><br>The uploaded data appears significantly different from training data. Consider retraining with similar network characteristics.</div>', unsafe_allow_html=True)

        # Data Comparison Analysis
        st.markdown("### Network Data Comparison Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.markdown("**Training Data Characteristics**")
            st.write(f"Mean Flow: {df_train['n_flows'].mean():.2f}")
            st.write(f"Standard Deviation: {df_train['n_flows'].std():.2f}")
            st.write(f"Minimum Flow: {df_train['n_flows'].min():.2f}")
            st.write(f"Maximum Flow: {df_train['n_flows'].max():.2f}")
            st.write(f"Data Points: {len(df_train):,}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.markdown("**Uploaded Data Characteristics**")
            st.write(f"Mean Flow: {df_test['n_flows'].mean():.2f}")
            st.write(f"Standard Deviation: {df_test['n_flows'].std():.2f}")
            st.write(f"Minimum Flow: {df_test['n_flows'].min():.2f}")
            st.write(f"Maximum Flow: {df_test['n_flows'].max():.2f}")
            st.write(f"Data Points: {len(df_test):,}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Data Drift Analysis
        train_mean = df_train['n_flows'].mean()
        test_mean = df_test['n_flows'].mean()
        mean_diff_pct = abs(test_mean - train_mean) / train_mean * 100

        if mean_diff_pct <= 10:
            st.markdown(f'<div class="status-card success"><strong>Data Compatibility Confirmed</strong><br>Uploaded data characteristics are very similar to training data (mean difference: {mean_diff_pct:.1f}%). High confidence in predictions.</div>', unsafe_allow_html=True)
        elif mean_diff_pct <= 25:
            st.markdown(f'<div class="status-card info"><strong>Acceptable Data Variance</strong><br>Some differences detected between datasets (mean difference: {mean_diff_pct:.1f}%). Predictions should be reliable with moderate confidence.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="status-card warning"><strong>Significant Data Drift Detected</strong><br>Substantial differences from training data (mean difference: {mean_diff_pct:.1f}%). Consider model retraining or domain adaptation for optimal performance.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<div class="status-card error"><strong>Inference Processing Error</strong><br>Error during inference: {str(e)}<br><br>Please ensure your uploaded data has the same format and features as the training data.</div>', unsafe_allow_html=True)

elif hasattr(st.session_state, 'trained_model'):
    st.markdown('<div class="twin-status operational">Digital Twin Model Ready for Deployment</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-card info"><strong>System Status:</strong> Pre-trained model is loaded and ready for inference. Upload a CSV file to generate network flow predictions.</div>', unsafe_allow_html=True)
    
    # Showing current model configuration
    if 'model_config' in st.session_state:
        config = st.session_state.model_config
        st.markdown('<div class="section-header">Current Digital Twin Configuration</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.markdown("**Model Architecture**")
            st.write(f"Twin Engine: {config['selected_twin_model']}")
            st.write(f"Observation Window: {config['observation_window']} time steps")
            st.write(f"Data Normalization: {'Enabled' if config['use_normalization'] else 'Disabled'}")
            if config['use_normalization']:
                st.write(f"Normalization Method: {config['normalization_method']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="config-section">', unsafe_allow_html=True)
            st.markdown("**Network Features**")
            if config['selected_network_features']:
                for feature in config['selected_network_features']:
                    st.write(f"• {feature}")
            else:
                st.write("• Primary flow data only")
            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('Digital Twin Configuration Required', unsafe_allow_html=True)
    st.markdown('<div class="status-card warning"><strong>Configuration Pending:</strong> Please configure the digital twin parameters in the control panel and click "Apply Configuration" to begin calibrating.</div>', unsafe_allow_html=True)
    
    # Showing available data information
    st.markdown('<div class="section-header">Available Training Data</div>', unsafe_allow_html=True)
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown("**Dataset Information**")
    st.write(f"Total Observations: {len(df_train):,}")
    st.write(f"Available Features: {', '.join(df_train.columns.tolist())}")
    st.write(f"Time Range: {len(df_train)} time steps")
    st.write(f"Primary Target: Network Flow Count (n_flows)")
    st.markdown('</div>', unsafe_allow_html=True)

# Testing New Data Section
st.markdown('<div class="section-header">Test/Inference Digital Twin with New Network Data</div>', unsafe_allow_html=True)
st.markdown('<div class="status-card info"><strong>Inference Testing:</strong> Upload new network traffic data to test your trained digital twin model on unseen data and evaluate its real-world performance.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload New Network Traffic for Inference Testing", 
    type=["csv"], 
    help="Upload CSV file with same format as training data. Must include 'n_flows' column for predictions.",
    key="inference_uploader"
)
