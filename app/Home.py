"""
MPCIM Thesis - Interactive Dashboard
Multi-Dimensional Performance-Career Integration Model

Author: Deni Sulaeman
Date: October 22, 2025
"""

import os
import io
import urllib.request

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# caching helper for data
@st.cache_data(ttl=60 * 60)
def load_csv_from_path(path_or_buffer):
    return pd.read_csv(path_or_buffer)

# Page configuration
st.set_page_config(
    page_title="MPCIM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 MPCIM Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Dimensional Performance-Career Integration Model</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=MPCIM+Thesis", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📚 Navigation")
    st.markdown("""
    Gunakan menu di atas untuk navigasi:
    - **🏠 Home**: Overview & ringkasan
    - **📊 Data Explorer**: Eksplorasi dataset
    - **📈 EDA Results**: Hasil analisis eksploratori
    - **🤖 Model Performance**: Performa model ML
    - **🔮 Prediction**: Prediksi promosi
    """)
    st.markdown("---")
    st.markdown("### 👨‍🎓 About")
    st.markdown("""
    **Author**: Deni Sulaeman  
    **Program**: Master in Information Systems  
    **Topic**: Predictive Analytics for Career Progression
    """)

# Main content
st.markdown("## 🎯 Tentang Penelitian")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### Multi-Dimensional Performance-Career Integration Model (MPCIM)
    
    Penelitian ini mengembangkan model prediktif untuk **career progression** menggunakan 
    pendekatan **multi-dimensional** yang mengintegrasikan:
    
    1. **📊 Performance Metrics**: Skor kinerja karyawan
    2. **🎭 Behavioral Competencies**: Kompetensi perilaku
    3. **👤 Demographic Factors**: Faktor demografis
    
    #### 🎓 Kontribusi Penelitian
    
    - ✅ **Pendekatan Holistik**: Mengatasi keterbatasan metrik tunggal
    - ✅ **Machine Learning**: Menggunakan algoritma ML advanced (XGBoost, Neural Networks)
    - ✅ **Interpretability**: Analisis feature importance dengan SHAP
    - ✅ **Practical Impact**: Aplikasi nyata untuk HR decision-making
    
    #### 📈 Metodologi
    
    - **Data**: 1,500+ employee records
    - **Models**: Logistic Regression, Random Forest, XGBoost, Neural Networks
    - **Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    - **Validation**: Stratified cross-validation
    """)

with col2:
    st.markdown("### 📊 Quick Stats")
    
    # Robust data loading: prefer repository relative path, else DATA_URL env var
    repo_root = Path(__file__).resolve().parents[1]
    default_data = repo_root / "data" / "final" / "integrated_performance_behavioral.csv"
    data_url = os.environ.get("DATA_URL")  # allow deploy to point to an external CSV

    df = None
    data_load_error = None

    try:
        if default_data.exists():
            df = load_csv_from_path(default_data)
        elif data_url:
            st.info("Mengunduh data dari DATA_URL environment variable...")
            # stream download into BytesIO to avoid writing to disk
            with urllib.request.urlopen(data_url) as resp:
                raw = resp.read()
            df = load_csv_from_path(io.BytesIO(raw))
        else:
            data_load_error = (
                "Data tidak ditemukan. Tambahkan file CSV di: 'data/final/integrated_performance_behavioral.csv' "
                "atau set environment variable DATA_URL ke URL file CSV."
            )
    except Exception as e:
        data_load_error = f"Gagal memuat data: {e}"

    if df is not None:
        st.metric("Total Employees", f"{len(df):,}")
        if 'has_promotion' in df.columns:
            st.metric("Promotion Rate", f"{df['has_promotion'].mean()*100:.2f}%")
        else:
            st.metric("Promotion Rate", "N/A")
        st.metric("Features", len(df.columns))

        # Promotion distribution
        if 'has_promotion' in df.columns:
            promo_counts = df['has_promotion'].value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=['Not Promoted', 'Promoted'],
                values=promo_counts.values,
                hole=0.4,
                marker_colors=['#e74c3c', '#2ecc71']
            )])
            fig.update_layout(
                title="Promotion Distribution",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download buttons: dataset CSV and EDA report (if exists)
        with st.expander("🔽 Download & Export"):
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download dataset (CSV)",
                data=csv_bytes,
                file_name="integrated_performance_behavioral.csv",
                mime="text/csv"
            )

            # EDA summary file (optional)
            eda_summary = repo_root / "results" / "EDA_Summary_Report.txt"
            if eda_summary.exists():
                with open(eda_summary, "rb") as f:
                    st.download_button(
                        label="Download EDA Summary",
                        data=f,
                        file_name="EDA_Summary_Report.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Tidak ditemukan EDA_Summary_Report.txt di folder results/. Jalankan skrip EDA untuk membuatnya.")

    else:
        st.warning(f"⚠️ {data_load_error}")

# Key Features
st.markdown("---")
st.markdown("## 🚀 Fitur Dashboard")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Data Explorer
    - Upload & view dataset
    - Filter dan search
    - Export hasil filter
    - Statistik deskriptif
    """)

with col2:
    st.markdown("""
    ### 📈 EDA Results
    - Visualisasi distribusi
    - Correlation analysis
    - Statistical tests
    - Comparative analysis
    """)

with col3:
    st.markdown("""
    ### 🤖 Model Performance
    - Confusion matrix
    - ROC curves
    - Feature importance
    - Model comparison
    """)

# Instructions
st.markdown("---")
st.markdown("## 📖 Cara Menggunakan")

with st.expander("🔍 Klik untuk melihat panduan lengkap"):
    st.markdown("""
    ### 1️⃣ Data Explorer
    - Navigasi ke halaman **Data Explorer**
    - Upload file CSV atau gunakan data default
    - Eksplorasi data dengan filter dan search
    - Download hasil filter jika diperlukan
    
    ### 2️⃣ EDA Results
    - Lihat hasil Exploratory Data Analysis
    - Analisis distribusi variabel
    - Correlation heatmap
    - Statistical significance tests
    
    ### 3️⃣ Model Performance
    - Bandingkan performa berbagai model ML
    - Lihat confusion matrix dan metrics
    - Analisis feature importance
    - ROC curves untuk setiap model
    
    ### 4️⃣ Prediction
    - Input data karyawan baru
    - Dapatkan prediksi promosi
    - Lihat probability score
    - Interpretasi hasil prediksi
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 2rem 0;'>
    <p><strong>MPCIM Thesis Dashboard</strong> | Master Program in Information Systems</p>
    <p>© 2025 Deni Sulaeman | Built with Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
