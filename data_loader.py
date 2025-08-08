import pandas as pd
import streamlit as st

@st.cache_data
def load_data(filename):
    try:
        df = pd.read_csv(filename)
        if 'id_time' in df.columns:
            try:
                df['id_time'] = pd.to_datetime(df['id_time'])
                df = df.set_index('id_time').sort_index()
            except:
                st.warning("⚠️ Could not parse 'id_time' as datetime. Using sequential indexing instead.")
        return df
    except FileNotFoundError:
        st.error(f"❌ Network data file '{filename}' not found in the current directory.")
        st.stop()

def load_uploaded_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if 'id_time' in df.columns:
        df['id_time'] = pd.to_datetime(df['id_time'], errors='coerce')
        df = df.set_index('id_time').sort_index()
    return df