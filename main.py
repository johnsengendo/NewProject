import sys
import os

# Add the directory containing your modules to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

# Configuration and Setup
CSV_FILENAME = "543.csv"

# Page setup
st.set_page_config(page_title="Network Flow Digital Twin Analytics", layout="wide")
st.title("🌐 Network Flow Digital Twin Analytics Platform")
st.markdown("*Real-time network behavior modeling and predictive analytics for traffic flow optimization*")

# Load data
df_train = load_data(CSV_FILENAME)

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"], help="Must include 'n_flows' column for predictions.")

if uploaded_file is not None:
    df_test = load_uploaded_data(uploaded_file)
    st.success("Custom network traffic data uploaded successfully! Inference-only mode enabled.")
    inference_mode = True
else:
    df_test = None
    inference_mode = False

# Sidebar configuration
st.sidebar.header("⚙️ Digital Twin Configuration Panel")

# Initialize session state for parameters if not exists
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
st.sidebar.subheader("🎛️ Digital Twin Controls")
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Reset DT Settings", help="Reset all parameters to default values"):
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
    apply_settings = st.button("Apply", help="Apply current settings to digital twin")

# Network Digital Twin specific parameters
st.sidebar.subheader("Network Twin Parameters")
temp_observation_window = st.sidebar.slider("Observation Window (time steps)", 5, 100, st.session_state.current_observation_window, help="Historical time steps for pattern recognition")
temp_simulation_cycles = st.sidebar.slider("Simulation Cycles", 10, 200, st.session_state.current_simulation_cycles, help="Number of training iterations for twin calibration")
batch_options = [16, 32, 64, 128]
current_batch_index = batch_options.index(st.session_state.current_processing_batch) if st.session_state.current_processing_batch in batch_options else 1
temp_processing_batch = st.sidebar.selectbox("Processing Batch Size", batch_options, index=current_batch_index, help="Data batch size for twin processing")

# Network parameter selection
network_parameters = df_train.columns.tolist()
if "n_flows" in network_parameters:
    network_parameters.remove("n_flows")
temp_selected_network_features = st.sidebar.multiselect("🔗 Network Parameters for Twin Model:", options=network_parameters, default=st.session_state.current_network_features, help="Additional network metrics to include in the digital twin")

# Digital Twin Model selection
st.sidebar.subheader("🤖 Twin Intelligence Engine")
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
temp_selected_twin_model = st.sidebar.selectbox("Choose Digital Twin Engine", list(twin_models.keys()), index=current_model_index)

# Validation configuration
st.sidebar.subheader("🔬 Twin Validation Setup")
temp_validation_period = st.sidebar.slider("Validation Period (recent observations)", 50, min(500, len(df_train)//2), st.session_state.current_validation_period, help="Number of recent observations for twin validation")

# Data preprocessing for digital twin
temp_use_normalization = st.sidebar.checkbox("Apply Data Normalization", value=st.session_state.current_use_normalization, help="Normalize input data for optimal twin performance")
normalization_options = ["MinMax Scaling", "Standard Scaling"]
current_norm_index = normalization_options.index(st.session_state.current_normalization_method) if st.session_state.current_normalization_method in normalization_options else 0
temp_normalization_method = st.sidebar.selectbox("Normalization Method", normalization_options, index=current_norm_index, help="Method for data preprocessing")

# Apply settings when button is clicked
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

# Use applied settings for processing
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
    st.info("🔄 Training new digital twin model...")

    if model_type in ['lstm', 'cnn_lstm', 'conv1d', 'gru', 'bilstm']:
        with st.spinner("🔄 Calibrating neural digital twin..."):
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
        with st.spinner("🔄 Calibrating statistical digital twin..."):
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

    with st.spinner(f"🔄 Training {selected_twin_model} digital twin..."):
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
            st.error(f"Digital twin calibration failed: {e}")
            st.stop()

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
    st.success("✅ Digital twin model trained and saved successfully!")

    metrics, y_test_aligned, y_pred_aligned = calculate_metrics(y_test, y_pred)

    st.header("Digital Twin Performance Dashboard")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Prediction Error (RMSE)", f"{metrics['rmse']:.3f}")
    with col2:
        st.metric("Average Deviation (MAE)", f"{metrics['mae']:.3f}")
    with col3:
        st.metric("Relative Error (%)", f"{metrics['mape']:.1f}%" if metrics['mape'] != float('inf') else "N/A")
    with col4:
        st.metric("Twin Fidelity (R²)", f"{metrics['r2']:.3f}")
    with col5:
        st.metric("Traffic Trend Accuracy", f"{metrics['direction_accuracy']:.1f}%")

    if metrics['r2'] >= 0.8 and metrics['mape'] <= 10:
        st.success("🎉 **Digital Twin Ready for Production!** High fidelity network representation achieved.")
    elif metrics['r2'] >= 0.6 and metrics['mape'] <= 20:
        st.info("👍 **Digital Twin Operational** - Suitable for network monitoring and basic predictions.")
    elif metrics['r2'] >= 0.4:
        st.warning("⚠️ **Twin Requires Calibration** - Consider parameter tuning or additional network features.")
    else:
        st.error("🔴 **Twin Needs Reconfiguration** - Significant improvements required for reliable operation.")

    st.header("📈 Network Flow Digital Twin Predictions")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    total_len = len(y_train) + len(y_test_aligned)
    train_range = np.arange(len(y_train))
    test_range = np.arange(len(y_train), len(y_train) + len(y_test_aligned))
    ax1.plot(train_range, y_train, label='Historical Network Data', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(test_range, y_test_aligned, label='Actual Network Flow', color='green', linewidth=2)
    ax1.plot(test_range, y_pred_aligned, label=f'Digital Twin Prediction', color='red', linewidth=2, linestyle='--')
    ax1.axvline(x=len(y_train), color='orange', linestyle=':', linewidth=2, label='Calibration/Validation Split')
    ax1.set_title(f'Network Digital Twin: {selected_twin_model} - Flow Prediction Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Network Flow Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    test_start = max(0, len(y_train) - 50)
    zoom_train_range = np.arange(test_start, len(y_train))
    zoom_train_data = y_train[test_start:] if hasattr(y_train, '__getitem__') else y_train.iloc[test_start:]
    ax2.plot(zoom_train_range, zoom_train_data, label='Recent Historical Data', color='blue', alpha=0.5, linewidth=1)
    ax2.plot(test_range, y_test_aligned, label='Actual Network Flow', color='green', linewidth=2)
    ax2.plot(test_range, y_pred_aligned, label=f'Twin Prediction', color='red', linewidth=2, linestyle='--')
    ax2.axvline(x=len(y_train), color='orange', linestyle=':', linewidth=2, label='Validation Period Start')
    ax2.set_title('Detailed View: Digital Twin Validation Period', fontsize=12)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Network Flow Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    st.header("🔍 Digital Twin Accuracy Analysis")
    col1, col2 = st.columns(2)
    with col1:
        errors = y_test_aligned - y_pred_aligned
        error_range = np.arange(len(errors))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(error_range, errors, color='red', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.fill_between(error_range, errors, alpha=0.3, color='red')
        ax.set_title('Digital Twin Prediction Errors Over Time')
        ax.set_xlabel('Validation Time Steps')
        ax.set_ylabel('Error (Actual - Predicted)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Distribution of Twin Prediction Errors')
        ax.set_xlabel('Error (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.session_state.dt_params_applied = False

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
    st.info("✅ **Pre-trained Model Available**: Digital twin is ready for inference. Upload a CSV file to make predictions.")

else:
    st.warning("⚠️ Please click 'Apply' to train the digital twin with your selected settings.")
