"""
EDA Results Page
Display exploratory data analysis results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from pathlib import Path

st.set_page_config(page_title="EDA Results", page_icon="📈", layout="wide")

st.title("📈 Exploratory Data Analysis Results")
st.markdown("Hasil analisis eksploratori data MPCIM")

# Load data
@st.cache_data
def load_data():
    data_path = Path("/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/integrated_performance_behavioral.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None

df = load_data()

if df is None:
    st.error("❌ Data tidak ditemukan.")
    st.stop()

# Key findings
st.markdown("## 🔍 Key Findings")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **📊 Performance Score**
    - Mean: {:.2f}
    - Std: {:.2f}
    - Range: {:.2f} - {:.2f}
    """.format(
        df['performance_score'].mean(),
        df['performance_score'].std(),
        df['performance_score'].min(),
        df['performance_score'].max()
    ))

with col2:
    st.info("""
    **🎭 Behavioral Score**
    - Mean: {:.2f}
    - Std: {:.2f}
    - Range: {:.2f} - {:.2f}
    """.format(
        df['behavior_avg'].mean(),
        df['behavior_avg'].std(),
        df['behavior_avg'].min(),
        df['behavior_avg'].max()
    ))

with col3:
    promo_rate = df['has_promotion'].mean() * 100
    st.info("""
    **🎯 Promotion Rate**
    - Rate: {:.2f}%
    - Promoted: {:,}
    - Not Promoted: {:,}
    """.format(
        promo_rate,
        df['has_promotion'].sum(),
        len(df) - df['has_promotion'].sum()
    ))

# Statistical Tests
st.markdown("---")
st.markdown("## 📊 Statistical Significance Tests")

promoted = df[df['has_promotion'] == 1]
not_promoted = df[df['has_promotion'] == 0]

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Performance Score T-Test")
    
    t_stat_perf, p_val_perf = stats.ttest_ind(
        promoted['performance_score'],
        not_promoted['performance_score']
    )
    
    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((len(promoted) - 1) * promoted['performance_score'].std()**2 + 
         (len(not_promoted) - 1) * not_promoted['performance_score'].std()**2) / 
        (len(promoted) + len(not_promoted) - 2)
    )
    cohens_d_perf = (promoted['performance_score'].mean() - not_promoted['performance_score'].mean()) / pooled_std
    
    results_df = pd.DataFrame({
        'Group': ['Promoted', 'Not Promoted'],
        'Mean': [promoted['performance_score'].mean(), not_promoted['performance_score'].mean()],
        'Std': [promoted['performance_score'].std(), not_promoted['performance_score'].std()],
        'N': [len(promoted), len(not_promoted)]
    })
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    - **t-statistic**: {t_stat_perf:.4f}
    - **p-value**: {p_val_perf:.4f}
    - **Cohen's d**: {cohens_d_perf:.4f}
    - **Significance**: {'✅ Significant (p < 0.05)' if p_val_perf < 0.05 else '❌ Not Significant (p ≥ 0.05)'}
    """)
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=not_promoted['performance_score'],
        name='Not Promoted',
        marker_color='#e74c3c',
        boxmean='sd'
    ))
    fig.add_trace(go.Box(
        y=promoted['performance_score'],
        name='Promoted',
        marker_color='#2ecc71',
        boxmean='sd'
    ))
    fig.update_layout(
        title="Performance Score Distribution",
        yaxis_title="Performance Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Behavioral Score T-Test")
    
    t_stat_beh, p_val_beh = stats.ttest_ind(
        promoted['behavior_avg'],
        not_promoted['behavior_avg']
    )
    
    # Cohen's d effect size
    pooled_std = np.sqrt(
        ((len(promoted) - 1) * promoted['behavior_avg'].std()**2 + 
         (len(not_promoted) - 1) * not_promoted['behavior_avg'].std()**2) / 
        (len(promoted) + len(not_promoted) - 2)
    )
    cohens_d_beh = (promoted['behavior_avg'].mean() - not_promoted['behavior_avg'].mean()) / pooled_std
    
    results_df = pd.DataFrame({
        'Group': ['Promoted', 'Not Promoted'],
        'Mean': [promoted['behavior_avg'].mean(), not_promoted['behavior_avg'].mean()],
        'Std': [promoted['behavior_avg'].std(), not_promoted['behavior_avg'].std()],
        'N': [len(promoted), len(not_promoted)]
    })
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    st.markdown(f"""
    - **t-statistic**: {t_stat_beh:.4f}
    - **p-value**: {p_val_beh:.4f}
    - **Cohen's d**: {cohens_d_beh:.4f}
    - **Significance**: {'✅ Significant (p < 0.05)' if p_val_beh < 0.05 else '❌ Not Significant (p ≥ 0.05)'}
    """)
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=not_promoted['behavior_avg'],
        name='Not Promoted',
        marker_color='#e74c3c',
        boxmean='sd'
    ))
    fig.add_trace(go.Box(
        y=promoted['behavior_avg'],
        name='Promoted',
        marker_color='#2ecc71',
        boxmean='sd'
    ))
    fig.update_layout(
        title="Behavioral Score Distribution",
        yaxis_title="Behavioral Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Correlation Analysis
st.markdown("---")
st.markdown("## 🔗 Correlation Analysis")

numeric_cols = ['tenure_years', 'performance_score', 'behavior_avg', 'has_promotion']
corr_matrix = df[numeric_cols].corr()

col1, col2 = st.columns([2, 1])

with col1:
    # Heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        height=500,
        xaxis_title="",
        yaxis_title=""
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Correlation with Promotion")
    
    promo_corr = corr_matrix['has_promotion'].sort_values(ascending=False)
    
    corr_data = []
    for col, val in promo_corr.items():
        if col != 'has_promotion':
            corr_data.append({
                'Feature': col,
                'Correlation': f"{val:.3f}",
                'Strength': 'Strong' if abs(val) > 0.5 else 'Moderate' if abs(val) > 0.3 else 'Weak'
            })
    
    corr_df = pd.DataFrame(corr_data)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Interpretation:**
    - |r| > 0.5: Strong correlation
    - 0.3 < |r| ≤ 0.5: Moderate correlation
    - |r| ≤ 0.3: Weak correlation
    """)

# Distribution Comparisons
st.markdown("---")
st.markdown("## 📊 Distribution Comparisons")

tab1, tab2, tab3 = st.tabs(["Performance Score", "Behavioral Score", "Combined Analysis"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='performance_score',
            color='has_promotion',
            nbins=30,
            title="Performance Score Distribution by Promotion",
            labels={'has_promotion': 'Promotion Status'},
            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.violin(
            df,
            x='has_promotion',
            y='performance_score',
            color='has_promotion',
            title="Performance Score Violin Plot",
            labels={'has_promotion': 'Promotion Status'},
            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
            box=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df,
            x='behavior_avg',
            color='has_promotion',
            nbins=30,
            title="Behavioral Score Distribution by Promotion",
            labels={'has_promotion': 'Promotion Status'},
            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
            barmode='overlay',
            opacity=0.7
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.violin(
            df,
            x='has_promotion',
            y='behavior_avg',
            color='has_promotion',
            title="Behavioral Score Violin Plot",
            labels={'has_promotion': 'Promotion Status'},
            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
            box=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # 3D scatter plot
    fig = px.scatter_3d(
        df,
        x='performance_score',
        y='behavior_avg',
        z='tenure_years',
        color='has_promotion',
        title="3D Scatter: Performance vs Behavioral vs Tenure",
        labels={'has_promotion': 'Promotion Status'},
        color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
        opacity=0.7
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

# Summary
st.markdown("---")
st.markdown("## 📝 Summary & Insights")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    ### ✅ Key Insights
    
    1. **Behavioral Score** shows significant difference between promoted and non-promoted employees
    2. **Performance Score** alone may not be sufficient for promotion prediction
    3. **Multi-dimensional approach** is justified by the data
    4. **Moderate class imbalance** (~9% promotion rate) requires special handling
    """)

with col2:
    st.info("""
    ### 💡 Recommendations
    
    1. Use **SMOTE** or similar techniques for class imbalance
    2. Include **behavioral metrics** as key features
    3. Consider **interaction features** between performance and behavior
    4. Apply **ensemble methods** for better prediction
    """)
