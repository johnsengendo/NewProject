import matplotlib.pyplot as plt
import streamlit as st

def plot_network_traffic(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_data = df["n_flows"].iloc[-1000:] if len(df) > 1000 else df["n_flows"]
    ax.plot(plot_data.index, plot_data.values, linewidth=1.5, color='navy')
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Flow Count")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def plot_predictions(y_train, y_test, y_pred, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    total_len = len(y_train) + len(y_test)
    train_range = np.arange(len(y_train))
    test_range = np.arange(len(y_train), total_len)
    ax1.plot(train_range, y_train, label='Historical Network Data', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(test_range, y_test, label='Actual Network Flow', color='green', linewidth=2)
    ax1.plot(test_range, y_pred, label=f'Digital Twin Prediction', color='red', linewidth=2, linestyle='--')
    ax1.axvline(x=len(y_train), color='orange', linestyle=':', linewidth=2, label='Calibration/Validation Split')
    ax1.set_title(f'Network Digital Twin: {model_name} - Flow Prediction Analysis', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Network Flow Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    test_start = max(0, len(y_train) - 50)
    zoom_train_range = np.arange(test_start, len(y_train))
    zoom_train_data = y_train[test_start:]
    ax2.plot(zoom_train_range, zoom_train_data, label='Recent Historical Data', color='blue', alpha=0.5, linewidth=1)
    ax2.plot(test_range, y_test, label='Actual Network Flow', color='green', linewidth=2)
    ax2.plot(test_range, y_pred, label=f'Twin Prediction', color='red', linewidth=2, linestyle='--')
    ax2.axvline(x=len(y_train), color='orange', linestyle=':', linewidth=2, label='Validation Period Start')
    ax2.set_title('Detailed View: Digital Twin Validation Period', fontsize=12)
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Network Flow Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)