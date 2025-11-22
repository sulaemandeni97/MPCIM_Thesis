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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui import apply_styles, page_header

# caching helper for data
@st.cache_data(ttl=60 * 60)
def load_csv_from_path(path_or_buffer):
    return pd.read_csv(path_or_buffer)

# Page configuration
st.set_page_config(
    page_title="MPCIM Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global styles
apply_styles()

# Paths and dataset loader helpers
repo_root = Path(__file__).resolve().parents[1]
default_data_qa = repo_root / "data" / "final" / "integrated_full_dataset.csv"
default_data = repo_root / "data" / "final" / "integrated_performance_behavioral.csv"
data_url = os.environ.get("DATA_URL")

# Helper to load whichever dataset is active (Data Explorer session > integrated_full > integrated_performance)
def get_current_dataset():
    df = None
    has_qa_data = False
    data_load_error = None
    source = None

    try:
        if 'mpcim_df' in st.session_state and st.session_state['mpcim_df'] is not None:
            df = st.session_state['mpcim_df']
            source = "Data Explorer"
        elif default_data_qa.exists():
            df = load_csv_from_path(default_data_qa)
            source = "integrated_full_dataset.csv"
        elif default_data.exists():
            df = load_csv_from_path(default_data)
            source = "integrated_performance_behavioral.csv"
        elif data_url:
            with urllib.request.urlopen(data_url) as resp:
                raw = resp.read()
            df = load_csv_from_path(io.BytesIO(raw))
            source = "DATA_URL"
        else:
            data_load_error = (
                "Data tidak ditemukan. Tambahkan file CSV di: 'data/final/integrated_full_dataset.csv' "
                "atau set environment variable DATA_URL ke URL file CSV."
            )

        if df is not None:
            has_qa_data = 'has_quick_assessment' in df.columns
    except Exception as e:
        data_load_error = f"Gagal memuat data: {e}"

    return df, has_qa_data, data_load_error, source

# Load dataset once for reuse across sections
df, has_qa_data, data_load_error, data_source = get_current_dataset()

# Enhanced Title with gradient
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 2.5em;">🎓 MPCIM Dashboard</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0; font-size: 1.2em;">
        Multi-Dimensional Performance-Career Integration Model
    </p>
    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 0.9em;">
        Prediksi Promosi Karyawan Berbasis Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
        <h2 style="color: white; margin: 0;">🎓 MPCIM</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 0.9em;">
            Thesis Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
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

# Main content with enhanced cards
st.markdown("""
<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h2 style="color: #667eea; margin-top: 0;">📚 Tentang Penelitian</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div style="background: white; padding: 25px; border-radius: 10px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h3 style="color: #667eea; margin-top: 0;">
            🎯 Multi-Dimensional Performance-Career Integration Model (MPCIM)
        </h3>
        <p style="font-size: 1.05em; line-height: 1.6;">
            Penelitian ini mengembangkan model prediktif untuk <strong>career progression</strong> 
            menggunakan pendekatan <strong>multi-dimensional</strong> yang mengintegrasikan:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 2.5em;">📊</h1>
            <h4 style="margin: 10px 0;">Performance</h4>
            <p style="margin: 0; font-size: 0.9em;">Skor kinerja karyawan</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 2.5em;">🎭</h1>
            <h4 style="margin: 10px 0;">Behavioral</h4>
            <p style="margin: 0; font-size: 0.9em;">Kompetensi perilaku</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 20px; border-radius: 10px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 2.5em;">🧠</h1>
            <h4 style="margin: 10px 0;">Psychological</h4>
            <p style="margin: 0; font-size: 0.9em;">Quick Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Contributions
    st.markdown("""
    <div style="background: #e8f5e9; padding: 20px; border-radius: 10px; 
                border-left: 4px solid #4CAF50; margin-bottom: 15px;">
        <h4 style="color: #2e7d32; margin-top: 0;">✅ Kontribusi Penelitian</h4>
        <ul style="margin-bottom: 0;">
            <li><strong>Pendekatan 3-Dimensi</strong>: Performance + Behavioral + Psychological</li>
            <li><strong>Quick Assessment Integration</strong>: 12 komponen psikologis untuk prediksi lebih akurat</li>
            <li><strong>Machine Learning</strong>: Algoritma ML advanced (XGBoost, Neural Networks)</li>
            <li><strong>Interpretability</strong>: Analisis feature importance dengan SHAP</li>
            <li><strong>Practical Impact</strong>: Aplikasi nyata untuk HR decision-making</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Methodology
    if df is not None:
        total_employees_text = f"{len(df):,} employee records"
        qa_records = df['has_quick_assessment'].sum() if 'has_quick_assessment' in df.columns else None
        qa_text = f" + {int(qa_records):,} Quick Assessment records" if qa_records is not None else ""
        data_bullet = f"<li><strong>Data</strong>: {total_employees_text}{qa_text}</li>"
    else:
        data_bullet = "<li><strong>Data</strong>: (data belum dimuat)</li>"

    st.markdown("""
    <div style="background: #e3f2fd; padding: 20px; border-radius: 10px; 
                border-left: 4px solid #2196F3;">
        <h4 style="color: #1565c0; margin-top: 0;">🔬 Metodologi</h4>
        <ul style="margin-bottom: 0;">
            {data_bullet}
            <li><strong>Features</strong>: 23 features (14 traditional + 9 psychological)</li>
            <li><strong>Models</strong>: Logistic Regression, Random Forest, XGBoost, Neural Networks</li>
            <li><strong>Evaluation</strong>: Accuracy, Precision, Recall, F1-Score, ROC-AUC</li>
            <li><strong>Validation</strong>: Stratified cross-validation + SMOTE balancing</li>
        </ul>
    </div>
    """.format(data_bullet=data_bullet), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
        <h3 style="color: #667eea; margin: 0;">📊 Quick Stats</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if df is not None:
        if data_source == "Data Explorer":
            st.info("📊 Menggunakan dataset dari Data Explorer")
        elif data_source:
            st.info(f"📁 Menggunakan dataset default: {data_source}")

        st.metric("Total Employees", f"{len(df):,}")
        if 'has_promotion' in df.columns:
            st.metric("Promotion Rate", f"{df['has_promotion'].mean()*100:.2f}%")
        else:
            st.metric("Promotion Rate", "N/A")
        st.metric("Features", len(df.columns))
        
        # Quick Assessment stats
        if has_qa_data and 'has_quick_assessment' in df.columns:
            qa_coverage = df['has_quick_assessment'].sum()
            st.metric("QA Coverage", f"{qa_coverage}/{len(df)} ({qa_coverage/len(df)*100:.1f}%)",
                     help="Jumlah karyawan dengan data Quick Assessment")
            
            if 'psychological_score' in df.columns:
                avg_psych = df['psychological_score'].mean()
                st.metric("Avg Psychological Score", f"{avg_psych:.1f}",
                         help="Rata-rata skor psikologis dari Quick Assessment")
            
            if 'leadership_potential' in df.columns:
                avg_leadership = df['leadership_potential'].mean()
                st.metric("Avg Leadership Potential", f"{avg_leadership:.1f}",
                         help="Rata-rata potensi kepemimpinan")

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
        st.warning(f"⚠️ {data_load_error or 'Data tidak ditemukan.'}")

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

# Quick Assessment Overview
if df is not None and has_qa_data:
    st.markdown("---")
    st.markdown("## 🧠 Quick Assessment Overview")
    
    qa_col1, qa_col2 = st.columns([2, 1])
    
    with qa_col1:
        st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <h3 style="color: #667eea; margin-top: 0;">📊 Komponen Psikologis</h3>
            <p>Quick Assessment menambahkan dimensi psikologis untuk prediksi yang lebih akurat:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show QA component distributions
        qa_components = ['drive_score', 'mental_strength_score', 'adaptability_score', 
                        'collaboration_score', 'psychological_score', 'leadership_potential']
        
        available_components = [col for col in qa_components if col in df.columns]
        
        if available_components:
            # Create bar chart for average scores
            avg_scores = {col.replace('_', ' ').title(): df[col].mean() 
                         for col in available_components}
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(avg_scores.keys()),
                    y=list(avg_scores.values()),
                    marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#fa709a', '#fee140'],
                    text=[f"{v:.1f}" for v in avg_scores.values()],
                    textposition='outside'
                )
            ])
            fig.update_layout(
                title="Average Quick Assessment Scores",
                xaxis_title="Component",
                yaxis_title="Average Score",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with qa_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; color: white; margin-bottom: 15px;">
            <h4 style="margin: 0;">🎯 QA Impact</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Show QA impact metrics
        if 'has_quick_assessment' in df.columns and 'has_promotion' in df.columns:
            qa_promo_rate = df[df['has_quick_assessment'] == 1]['has_promotion'].mean()
            no_qa_promo_rate = df[df['has_quick_assessment'] == 0]['has_promotion'].mean()
            
            st.metric("Promotion Rate (With QA)", f"{qa_promo_rate*100:.1f}%")
            st.metric("Promotion Rate (Without QA)", f"{no_qa_promo_rate*100:.1f}%")
            
            if qa_promo_rate > no_qa_promo_rate:
                diff = (qa_promo_rate - no_qa_promo_rate) * 100
                st.success(f"✅ QA data shows +{diff:.1f}% higher promotion rate")
            else:
                st.info("ℹ️ QA data provides additional insights for prediction")
        
        # Feature importance from QA
        st.markdown("""
        <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin-top: 15px;">
            <h5 style="color: #2e7d32; margin-top: 0;">🔑 Key QA Features</h5>
            <ul style="margin-bottom: 0; font-size: 0.9em;">
                <li><strong>Collaboration</strong>: 3-4% importance</li>
                <li><strong>Mental Strength</strong>: 4% importance</li>
                <li><strong>Leadership Potential</strong>: 3% importance</li>
                <li><strong>Drive Score</strong>: 2-3% importance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

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
