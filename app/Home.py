"""
MPCIM Thesis - Interactive Dashboard
Multi-Dimensional Performance-Career Integration Model

Author: Denis Ulaeman
Date: October 22, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

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
    **Author**: Denis Ulaeman  
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
    
    # Try to load data for quick stats
    data_path = Path("/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/integrated_performance_behavioral.csv")
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        
        st.metric("Total Employees", f"{len(df):,}")
        st.metric("Promotion Rate", f"{df['has_promotion'].mean()*100:.2f}%")
        st.metric("Features", len(df.columns))
        
        # Promotion distribution
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
    else:
        st.warning("⚠️ Data belum tersedia. Silakan upload data terlebih dahulu.")

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
    <p>© 2025 Denis Ulaeman | Built with Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
