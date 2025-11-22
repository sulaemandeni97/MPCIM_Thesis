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
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import apply_styles, page_header

st.set_page_config(page_title="EDA Results", page_icon="📈", layout="wide")

apply_styles()

# Enhanced header
st.markdown("""
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
            padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 2.2em;">EDA Results</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 1.1em;">
        Exploratory Data Analysis - Hasil Analisis Data
    </p>
</div>
""", unsafe_allow_html=True)

# Allow user to upload a CSV for EDA, or use dataset from Data Explorer (session), or fallback to default/sample
uploaded = st.sidebar.file_uploader("Upload CSV dataset for EDA (optional)", type=["csv"])


@st.cache_data
def load_data_from_paths():
    repo_root = Path(__file__).resolve().parents[2]
    # Try integrated_full_dataset first (with QA), fallback to old dataset
    default_path_qa = repo_root / "data" / "final" / "integrated_full_dataset.csv"
    default_path = repo_root / "data" / "final" / "integrated_performance_behavioral.csv"
    sample_path = repo_root / "data" / "final" / "integrated_performance_behavioral_sample.csv"

    if default_path_qa.exists():
        return pd.read_csv(default_path_qa)
    
    if default_path.exists():
        return pd.read_csv(default_path)

    if sample_path.exists():
        return pd.read_csv(sample_path)

    return None


df = None

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("✅ File uploaded untuk EDA")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat file upload: {e}")

if df is None and 'mpcim_df' in st.session_state:
    df = st.session_state['mpcim_df']
    st.sidebar.info("✅ Menggunakan dataset yang dimuat di Data Explorer")

if df is None:
    df = load_data_from_paths()
    if df is not None:
        st.sidebar.info("✅ Menggunakan data default dari repository")

if df is None:
    st.error("❌ Data tidak ditemukan. Upload CSV di sidebar atau muat data di halaman Data Explorer.")
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

# Quick Assessment Key Findings (if available)
has_qa_data = 'psychological_score' in df.columns

if has_qa_data:
    st.markdown("---")
    st.markdown("## 🧠 Quick Assessment Key Findings")
    
    qa_col1, qa_col2, qa_col3 = st.columns(3)
    
    with qa_col1:
        st.info("""
        **🧠 Psychological Score**
        - Mean: {:.2f}
        - Std: {:.2f}
        - Range: {:.2f} - {:.2f}
        """.format(
            df['psychological_score'].mean(),
            df['psychological_score'].std(),
            df['psychological_score'].min(),
            df['psychological_score'].max()
        ))
    
    with qa_col2:
        if 'leadership_potential' in df.columns:
            st.info("""
            **👔 Leadership Potential**
            - Mean: {:.2f}
            - Std: {:.2f}
            - Range: {:.2f} - {:.2f}
            """.format(
                df['leadership_potential'].mean(),
                df['leadership_potential'].std(),
                df['leadership_potential'].min(),
                df['leadership_potential'].max()
            ))
    
    with qa_col3:
        if 'has_quick_assessment' in df.columns:
            qa_count = df['has_quick_assessment'].sum()
            qa_rate = (qa_count / len(df)) * 100
            st.info("""
            **📊 QA Coverage**
            - Coverage: {:.1f}%
            - With QA: {:,}
            - Without QA: {:,}
            """.format(
                qa_rate,
                qa_count,
                len(df) - qa_count
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

# Quick Assessment Visualizations (if available)
if has_qa_data:
    st.markdown("---")
    st.markdown("## 🧠 Quick Assessment Analysis")
    
    qa_tab1, qa_tab2, qa_tab3, qa_tab4 = st.tabs([
        "📊 QA Distributions", 
        "🔗 QA Correlations", 
        "📈 QA vs Promotion",
        "🎯 3D Holistic View"
    ])
    
    with qa_tab1:
        # QA component distributions
        qa_features = ['psychological_score', 'drive_score', 'mental_strength_score', 
                      'adaptability_score', 'collaboration_score', 'leadership_potential']
        available_qa = [f for f in qa_features if f in df.columns]
        
        if available_qa:
            cols = st.columns(2)
            for idx, feature in enumerate(available_qa[:6]):
                with cols[idx % 2]:
                    fig = px.histogram(
                        df,
                        x=feature,
                        color='has_promotion',
                        nbins=30,
                        title=f"{feature.replace('_', ' ').title()} Distribution",
                        labels={'has_promotion': 'Promotion Status'},
                        color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                        barmode='overlay',
                        opacity=0.7
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
    
    with qa_tab2:
        # Correlation heatmap for QA features
        if available_qa:
            corr_features = available_qa + ['has_promotion']
            corr_matrix = df[corr_features].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title="Quick Assessment Features Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation with promotion
            st.markdown("### Correlation with Promotion")
            promo_corr = df[available_qa + ['has_promotion']].corr()['has_promotion'].drop('has_promotion').sort_values(ascending=False)
            
            fig = px.bar(
                x=promo_corr.index,
                y=promo_corr.values,
                title="QA Features Correlation with Promotion",
                labels={'x': 'Feature', 'y': 'Correlation'},
                color=promo_corr.values,
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with qa_tab3:
        # QA features vs Promotion comparison
        if available_qa:
            promoted_qa = df[df['has_promotion'] == 1]
            not_promoted_qa = df[df['has_promotion'] == 0]
            
            comparison_data = []
            for feature in available_qa:
                comparison_data.append({
                    'Feature': feature.replace('_', ' ').title(),
                    'Promoted': promoted_qa[feature].mean(),
                    'Not Promoted': not_promoted_qa[feature].mean(),
                    'Difference': promoted_qa[feature].mean() - not_promoted_qa[feature].mean()
                })
            
            comp_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comp_df,
                x='Feature',
                y=['Promoted', 'Not Promoted'],
                title="Average QA Scores: Promoted vs Not Promoted",
                barmode='group',
                color_discrete_map={'Promoted': '#2ecc71', 'Not Promoted': '#e74c3c'}
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show difference
            st.markdown("### Difference Analysis")
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
    with qa_tab4:
        # 3D scatter with holistic view
        if 'holistic_score' in df.columns:
            fig = px.scatter_3d(
                df,
                x='performance_score',
                y='behavior_avg',
                z='psychological_score',
                color='has_promotion',
                size='holistic_score',
                title="3D Holistic View: Performance + Behavioral + Psychological",
                labels={'has_promotion': 'Promotion Status'},
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
                opacity=0.7
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Holistic Score Formula:**  
            40% Performance + 30% Behavioral + 30% Psychological
            
            Bubble size represents the holistic score - larger bubbles indicate higher overall scores.
            """)

# Summary
st.markdown("---")
st.markdown("## 📝 Summary & Insights")

col1, col2 = st.columns(2)

with col1:
    insights_text = """
    ### ✅ Key Insights
    
    1. **Behavioral Score** shows significant difference between promoted and non-promoted employees
    2. **Performance Score** alone may not be sufficient for promotion prediction
    3. **Multi-dimensional approach** is justified by the data
    4. **Moderate class imbalance** (~9% promotion rate) requires special handling
    """
    
    if has_qa_data:
        insights_text += """
    5. **Quick Assessment** adds psychological dimension for comprehensive evaluation
    6. **Collaboration & Mental Strength** show strong correlation with promotion
    7. **Holistic Score** (3D assessment) provides better prediction accuracy
    """
    
    st.success(insights_text)

with col2:
    recommendations_text = """
    ### 💡 Recommendations
    
    1. Use **SMOTE** or similar techniques for class imbalance
    2. Include **behavioral metrics** as key features
    3. Consider **interaction features** between performance and behavior
    4. Apply **ensemble methods** for better prediction
    """
    
    if has_qa_data:
        recommendations_text += """
    5. Leverage **Quick Assessment** data for 12-20% improvement
    6. Focus on **psychological factors** (collaboration, mental strength)
    7. Use **holistic score** for comprehensive employee evaluation
    """
    
    st.info(recommendations_text)

# AI Analysis Section
st.markdown("---")
st.markdown("## 🤖 AI Analysis & Insights")

with st.expander("📈 Gemini AI Analysis - Interpretasi EDA", expanded=False):
    st.markdown("""
    Gemini AI akan menganalisis hasil EDA dan memberikan interpretasi mendalam tentang:
    - Distribusi dan pola data
    - Perbedaan antara kelompok promoted dan not promoted
    - Korelasi dan feature importance
    - Implikasi untuk modeling
    """)
    
    if st.button("🔍 Generate AI Analysis", key="eda_ai"):
        with st.spinner("🤖 Gemini AI sedang menganalisis hasil EDA..."):
            try:
                from services.page_analysis_service import create_page_analysis_service
                
                # Create analysis service
                analysis_service = create_page_analysis_service()
                
                if not analysis_service.is_enabled():
                    st.warning("⚠️ Gemini AI tidak tersedia. Pastikan GEMINI_API_KEY sudah dikonfigurasi.")
                    st.info("💡 Lihat STREAMLIT_DEPLOY_GUIDE.md untuk setup instructions.")
                else:
                    # Prepare statistics for AI analysis
                    promoted = df[df['has_promotion'] == 1]
                    not_promoted = df[df['has_promotion'] == 0]
                    
                    stats = {
                        'promoted_pct': (len(promoted) / len(df) * 100),
                        'not_promoted_pct': (len(not_promoted) / len(df) * 100),
                        'imbalance_ratio': len(not_promoted) / len(promoted) if len(promoted) > 0 else 0,
                        
                        # Performance scores
                        'promoted_perf_mean': promoted['performance_score'].mean() if 'performance_score' in promoted.columns else 0,
                        'promoted_perf_std': promoted['performance_score'].std() if 'performance_score' in promoted.columns else 0,
                        'not_promoted_perf_mean': not_promoted['performance_score'].mean() if 'performance_score' in not_promoted.columns else 0,
                        'not_promoted_perf_std': not_promoted['performance_score'].std() if 'performance_score' in not_promoted.columns else 0,
                        
                        # Behavioral scores
                        'promoted_behav_mean': promoted['behavior_avg'].mean() if 'behavior_avg' in promoted.columns else 0,
                        'promoted_behav_std': promoted['behavior_avg'].std() if 'behavior_avg' in promoted.columns else 0,
                        'not_promoted_behav_mean': not_promoted['behavior_avg'].mean() if 'behavior_avg' in not_promoted.columns else 0,
                        'not_promoted_behav_std': not_promoted['behavior_avg'].std() if 'behavior_avg' in not_promoted.columns else 0,
                        
                        # Psychological scores (if available)
                        'promoted_psych_mean': promoted['psychological_score'].mean() if 'psychological_score' in promoted.columns else 0,
                        'promoted_psych_std': promoted['psychological_score'].std() if 'psychological_score' in promoted.columns else 0,
                        'not_promoted_psych_mean': not_promoted['psychological_score'].mean() if 'psychological_score' in not_promoted.columns else 0,
                        'not_promoted_psych_std': not_promoted['psychological_score'].std() if 'psychological_score' in not_promoted.columns else 0,
                        
                        # Correlations
                        'corr_performance': df[['performance_score', 'has_promotion']].corr().iloc[0, 1] if 'performance_score' in df.columns else 0,
                        'corr_behavioral': df[['behavior_avg', 'has_promotion']].corr().iloc[0, 1] if 'behavior_avg' in df.columns else 0,
                        'corr_psychological': df[['psychological_score', 'has_promotion']].corr().iloc[0, 1] if 'psychological_score' in df.columns else 0,
                    }
                    
                    # Get AI analysis
                    analysis = analysis_service.analyze_eda_results(stats)
                    
                    # Display analysis
                    st.markdown(analysis)
                    
                    st.success("✅ Analisis EDA selesai!")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Menggunakan analisis fallback...")
                
                # Show fallback analysis
                promoted = df[df['has_promotion'] == 1]
                not_promoted = df[df['has_promotion'] == 0]
                
                stats = {
                    'promoted_pct': (len(promoted) / len(df) * 100),
                    'not_promoted_pct': (len(not_promoted) / len(df) * 100),
                    'imbalance_ratio': len(not_promoted) / len(promoted) if len(promoted) > 0 else 0,
                    'promoted_perf_mean': promoted['performance_score'].mean() if 'performance_score' in promoted.columns else 0,
                    'promoted_perf_std': promoted['performance_score'].std() if 'performance_score' in promoted.columns else 0,
                    'not_promoted_perf_mean': not_promoted['performance_score'].mean() if 'performance_score' in not_promoted.columns else 0,
                    'not_promoted_perf_std': not_promoted['performance_score'].std() if 'performance_score' in not_promoted.columns else 0,
                    'promoted_behav_mean': promoted['behavior_avg'].mean() if 'behavior_avg' in promoted.columns else 0,
                    'promoted_behav_std': promoted['behavior_avg'].std() if 'behavior_avg' in promoted.columns else 0,
                    'not_promoted_behav_mean': not_promoted['behavior_avg'].mean() if 'behavior_avg' in not_promoted.columns else 0,
                    'not_promoted_behav_std': not_promoted['behavior_avg'].std() if 'behavior_avg' in not_promoted.columns else 0,
                    'promoted_psych_mean': promoted['psychological_score'].mean() if 'psychological_score' in promoted.columns else 0,
                    'promoted_psych_std': promoted['psychological_score'].std() if 'psychological_score' in promoted.columns else 0,
                    'not_promoted_psych_mean': not_promoted['psychological_score'].mean() if 'psychological_score' in not_promoted.columns else 0,
                    'not_promoted_psych_std': not_promoted['psychological_score'].std() if 'psychological_score' in not_promoted.columns else 0,
                    'corr_performance': df[['performance_score', 'has_promotion']].corr().iloc[0, 1] if 'performance_score' in df.columns else 0,
                    'corr_behavioral': df[['behavior_avg', 'has_promotion']].corr().iloc[0, 1] if 'behavior_avg' in df.columns else 0,
                    'corr_psychological': df[['psychological_score', 'has_promotion']].corr().iloc[0, 1] if 'psychological_score' in df.columns else 0,
                }
                
                from services.page_analysis_service import PageAnalysisService
                fallback_service = PageAnalysisService()
                fallback_analysis = fallback_service._fallback_eda_analysis(stats)
                st.markdown(fallback_analysis)
