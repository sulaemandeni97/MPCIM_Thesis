"""
Data Explorer Page
Interactive data exploration and filtering
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Data Explorer")
st.markdown("Eksplorasi dan analisis dataset MPCIM")

# Load data
@st.cache_data
def load_data():
    data_path = Path("/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/integrated_performance_behavioral.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_data()

if df is None:
    st.error("❌ Data tidak ditemukan. Pastikan file CSV tersedia di lokasi yang benar.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### 🔍 Filters")

# Promotion filter
promotion_filter = st.sidebar.multiselect(
    "Promotion Status",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "Promoted" if x == 1 else "Not Promoted"
)

# Gender filter
if 'gender' in df.columns:
    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=df['gender'].unique(),
        default=df['gender'].unique()
    )
else:
    gender_filter = None

# Performance score range
if 'performance_score' in df.columns:
    perf_min, perf_max = float(df['performance_score'].min()), float(df['performance_score'].max())
    perf_range = st.sidebar.slider(
        "Performance Score Range",
        min_value=perf_min,
        max_value=perf_max,
        value=(perf_min, perf_max)
    )
else:
    perf_range = None

# Behavioral score range
if 'behavior_avg' in df.columns:
    beh_min, beh_max = float(df['behavior_avg'].min()), float(df['behavior_avg'].max())
    beh_range = st.sidebar.slider(
        "Behavioral Score Range",
        min_value=beh_min,
        max_value=beh_max,
        value=(beh_min, beh_max)
    )
else:
    beh_range = None

# Apply filters
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['has_promotion'].isin(promotion_filter)]

if gender_filter is not None:
    filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]

if perf_range is not None:
    filtered_df = filtered_df[
        (filtered_df['performance_score'] >= perf_range[0]) &
        (filtered_df['performance_score'] <= perf_range[1])
    ]

if beh_range is not None:
    filtered_df = filtered_df[
        (filtered_df['behavior_avg'] >= beh_range[0]) &
        (filtered_df['behavior_avg'] <= beh_range[1])
    ]

# Display metrics
st.markdown("### 📈 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(filtered_df):,}", 
              delta=f"{len(filtered_df) - len(df):,}" if len(filtered_df) != len(df) else None)

with col2:
    promo_rate = filtered_df['has_promotion'].mean() * 100
    st.metric("Promotion Rate", f"{promo_rate:.2f}%")

with col3:
    if 'performance_score' in filtered_df.columns:
        avg_perf = filtered_df['performance_score'].mean()
        st.metric("Avg Performance", f"{avg_perf:.2f}")

with col4:
    if 'behavior_avg' in filtered_df.columns:
        avg_beh = filtered_df['behavior_avg'].mean()
        st.metric("Avg Behavioral", f"{avg_beh:.2f}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📊 Statistics", "📈 Distributions", "🔗 Relationships"])

with tab1:
    st.markdown("### 📋 Data Table")
    
    # Search functionality
    search_col = st.selectbox("Search in column:", filtered_df.columns)
    search_term = st.text_input("Search term:")
    
    if search_term:
        mask = filtered_df[search_col].astype(str).str.contains(search_term, case=False, na=False)
        display_df = filtered_df[mask]
    else:
        display_df = filtered_df
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download button
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name="mpcim_filtered_data.csv",
        mime="text/csv"
    )

with tab2:
    st.markdown("### 📊 Descriptive Statistics")
    
    # Numeric columns
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Overall Statistics")
        st.dataframe(filtered_df[numeric_cols].describe(), use_container_width=True)
    
    with col2:
        st.markdown("#### By Promotion Status")
        
        promoted = filtered_df[filtered_df['has_promotion'] == 1]
        not_promoted = filtered_df[filtered_df['has_promotion'] == 0]
        
        comparison_data = {
            'Metric': [],
            'Promoted': [],
            'Not Promoted': [],
            'Difference': []
        }
        
        for col in ['performance_score', 'behavior_avg', 'tenure_years']:
            if col in filtered_df.columns:
                comparison_data['Metric'].append(col)
                promoted_mean = promoted[col].mean()
                not_promoted_mean = not_promoted[col].mean()
                comparison_data['Promoted'].append(f"{promoted_mean:.2f}")
                comparison_data['Not Promoted'].append(f"{not_promoted_mean:.2f}")
                comparison_data['Difference'].append(f"{promoted_mean - not_promoted_mean:.2f}")
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📈 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'performance_score' in filtered_df.columns:
            fig = px.histogram(
                filtered_df,
                x='performance_score',
                color='has_promotion',
                nbins=30,
                title="Performance Score Distribution",
                labels={'has_promotion': 'Promotion Status'},
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'behavior_avg' in filtered_df.columns:
            fig = px.histogram(
                filtered_df,
                x='behavior_avg',
                color='has_promotion',
                nbins=30,
                title="Behavioral Score Distribution",
                labels={'has_promotion': 'Promotion Status'},
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        if 'tenure_years' in filtered_df.columns:
            fig = px.box(
                filtered_df,
                x='has_promotion',
                y='tenure_years',
                color='has_promotion',
                title="Tenure by Promotion Status",
                labels={'has_promotion': 'Promotion Status', 'tenure_years': 'Tenure (years)'},
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        if 'gender' in filtered_df.columns:
            gender_promo = filtered_df.groupby(['gender', 'has_promotion']).size().reset_index(name='count')
            fig = px.bar(
                gender_promo,
                x='gender',
                y='count',
                color='has_promotion',
                title="Gender Distribution by Promotion",
                labels={'has_promotion': 'Promotion Status'},
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### 🔗 Relationship Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'performance_score' in filtered_df.columns and 'behavior_avg' in filtered_df.columns:
            fig = px.scatter(
                filtered_df,
                x='performance_score',
                y='behavior_avg',
                color='has_promotion',
                title="Performance vs Behavioral Score",
                labels={'has_promotion': 'Promotion Status'},
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                opacity=0.6
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'tenure_years' in filtered_df.columns and 'performance_score' in filtered_df.columns:
            fig = px.scatter(
                filtered_df,
                x='tenure_years',
                y='performance_score',
                color='has_promotion',
                title="Tenure vs Performance Score",
                labels={'has_promotion': 'Promotion Status'},
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                opacity=0.6
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("#### Correlation Matrix")
    numeric_cols = ['tenure_years', 'performance_score', 'behavior_avg', 'has_promotion']
    available_cols = [col for col in numeric_cols if col in filtered_df.columns]
    
    if len(available_cols) > 1:
        corr_matrix = filtered_df[available_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(3),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Correlation Heatmap",
            height=500,
            xaxis_title="",
            yaxis_title=""
        )
        
        st.plotly_chart(fig, use_container_width=True)
