"""
Data Explorer Page
Interactive data exploration and filtering
"""

import os
import io
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import apply_styles, page_header

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

apply_styles()

# Enhanced header
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center;">
    <h1 style="color: white; margin: 0; font-size: 2.2em;">📊 Data Explorer</h1>
    <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 1.1em;">
        Eksplorasi dan Analisis Dataset MPCIM
    </p>
</div>
""", unsafe_allow_html=True)


# Required columns for validation
REQUIRED_COLUMNS = [
    'employee_id_hash', 'company_id', 'tenure_years', 'gender', 'marital_status', 
    'is_permanent', 'performance_score', 'performance_rating', 'has_promotion', 
    'behavior_avg', 'psychological_score', 'drive_score', 'mental_strength_score', 
    'adaptability_score', 'collaboration_score', 'has_quick_assessment', 
    'holistic_score', 'score_alignment', 'leadership_potential'
]

def validate_dataset(df):
    """Validate uploaded dataset structure and content."""
    errors = []
    warnings = []
    
    # Check required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for duplicate employee IDs
    if 'employee_id_hash' in df.columns:
        duplicates = df['employee_id_hash'].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate employee IDs")
    
    # Check data types and ranges
    if 'performance_score' in df.columns:
        if df['performance_score'].min() < 0 or df['performance_score'].max() > 100:
            warnings.append("Performance scores outside 0-100 range")
    
    if 'behavior_avg' in df.columns:
        if df['behavior_avg'].min() < 0 or df['behavior_avg'].max() > 100:
            warnings.append("Behavioral scores outside 0-100 range")
    
    if 'psychological_score' in df.columns:
        if df['psychological_score'].min() < 0 or df['psychological_score'].max() > 100:
            warnings.append("Psychological scores outside 0-100 range")
    
    if 'has_promotion' in df.columns:
        if not df['has_promotion'].isin([0, 1]).all():
            errors.append("has_promotion must be 0 or 1")
    
    if 'has_quick_assessment' in df.columns:
        if not df['has_quick_assessment'].isin([0, 1]).all():
            errors.append("has_quick_assessment must be 0 or 1")
    
    # Check for missing values in critical columns
    critical_cols = ['employee_id_hash', 'performance_score', 'behavior_avg', 'has_promotion']
    for col in critical_cols:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Missing values found in critical column: {col}")
    
    return errors, warnings

# Robust data loader with uploader and sample fallback
@st.cache_data(ttl=60 * 60)
def load_data(uploaded_file=None):
    repo_root = Path(__file__).resolve().parents[2]
    # Try integrated_full_dataset first (with QA), fallback to old dataset
    default_path_qa = repo_root / "data" / "final" / "integrated_full_dataset.csv"
    default_path = repo_root / "data" / "final" / "integrated_performance_behavioral.csv"
    sample_path = repo_root / "data" / "final" / "integrated_performance_behavioral_sample.csv"
    sample_100_path = repo_root / "data" / "final" / "sample_dataset_100.csv"

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            errors, warnings = validate_dataset(df)
            
            if errors:
                st.error("❌ Dataset validation failed:")
                for error in errors:
                    st.error(f"  • {error}")
                st.info("💡 Please check DATASET_UPLOAD_GUIDE.md for proper format")
                return None
            
            if warnings:
                st.warning("⚠️ Dataset warnings:")
                for warning in warnings:
                    st.warning(f"  • {warning}")
            
            st.success("✅ Dataset validated successfully!")
            return df
        except Exception as e:
            st.error(f"❌ Error reading CSV: {str(e)}")
            return None

    # Try balanced sample first for quick demo (70% promoted, 30% not promoted)
    sample_100_balanced_path = repo_root / "data" / "final" / "sample_dataset_100_balanced.csv"
    if sample_100_balanced_path.exists():
        return pd.read_csv(sample_100_balanced_path)
    
    # Fallback to original sample
    if sample_100_path.exists():
        return pd.read_csv(sample_100_path)
    
    if default_path_qa.exists():
        return pd.read_csv(default_path_qa)
    
    if default_path.exists():
        return pd.read_csv(default_path)

    if sample_path.exists():
        return pd.read_csv(sample_path)

    # Try DATA_URL env var
    data_url = os.environ.get("DATA_URL")
    if data_url:
        try:
            import urllib.request
            with urllib.request.urlopen(data_url) as resp:
                raw = resp.read()
            return pd.read_csv(io.BytesIO(raw))
        except Exception:
            return None

    return None


data_state = st.session_state.setdefault('data_explorer_state', {})

# Dataset info section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Options")

with st.sidebar.expander("ℹ️ Available Datasets", expanded=False):
    st.markdown("""
    **Sample Datasets:**
    - `sample_dataset_100_balanced.csv` ⭐ **DEFAULT**
      - 100 rows (70% promoted, 30% not)
      - QA scores in 0-100 range
      - Perfect for demos!
    - `sample_dataset_100.csv` (100 rows, 10% promoted)
    - `integrated_full_dataset.csv` (712 rows, 9.3% promoted)
    
    **Template:**
    - `UPLOAD_TEMPLATE.csv` (structure reference)
    
    **Guide:**
    - See `DATASET_UPLOAD_GUIDE.md` for details
    
    **Required Columns:** 19 columns
    - Basic Info (6)
    - Performance (3)
    - Behavioral (1)
    - Quick Assessment (9)
    """)

uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"], key="data_explorer_upload")

if data_state.get("source") == "uploaded":
    st.sidebar.success(f"📂 Menggunakan data upload: {data_state.get('uploaded_name', 'custom_upload')}")
    if st.sidebar.button("Reset ke data default", key="reset_uploaded_dataset"):
        data_state.clear()
        st.experimental_rerun()
else:
    st.sidebar.info("📂 Menggunakan data default: sample_dataset_100_balanced.csv (70% promoted)")

df = None

if uploaded is not None:
    df = load_data(uploaded_file=uploaded)
    if df is not None:
        try:
            uploaded_bytes = uploaded.getvalue()
        except AttributeError:
            uploaded.seek(0)
            uploaded_bytes = uploaded.read()
        data_state["uploaded_bytes"] = uploaded_bytes
        data_state["uploaded_name"] = uploaded.name
        data_state["source"] = "uploaded"
elif data_state.get("source") == "uploaded" and data_state.get("uploaded_bytes"):
    try:
        df = pd.read_csv(io.BytesIO(data_state["uploaded_bytes"]))
    except Exception:
        data_state.clear()

if df is None:
    df = load_data()
    data_state.setdefault("source", "default")

if df is None:
    st.error("❌ Data tidak ditemukan atau gagal dimuat. Upload CSV atau tambahkan data ke folder data/final/ atau set DATA_URL.")
    st.stop()

# Store loaded dataset in session_state so other pages (EDA, Model) can reuse it
try:
    st.session_state['mpcim_df'] = df
except Exception:
    # session_state may not be available in very old Streamlit versions; ignore silently
    pass

# Normalize column names (common variations)
col_map = {}
if 'behavior_avg' not in df.columns and 'behavioral_score' in df.columns:
    df['behavior_avg'] = df['behavioral_score']
if 'performance_score' not in df.columns and 'performance' in df.columns:
    df['performance_score'] = df['performance']

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

# Quick Assessment filters
has_qa_data = 'psychological_score' in df.columns

if has_qa_data:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Quick Assessment Filters")
    
    # QA data availability filter
    if 'has_quick_assessment' in df.columns:
        qa_filter = st.sidebar.multiselect(
            "QA Data Availability",
            options=[0, 1],
            default=[0, 1],
            format_func=lambda x: "Has QA Data" if x == 1 else "No QA Data"
        )
    else:
        qa_filter = None
    
    # Psychological score range
    if 'psychological_score' in df.columns:
        psych_min, psych_max = float(df['psychological_score'].min()), float(df['psychological_score'].max())
        psych_range = st.sidebar.slider(
            "Psychological Score Range",
            min_value=psych_min,
            max_value=psych_max,
            value=(psych_min, psych_max)
        )
    else:
        psych_range = None
    
    # Leadership potential range
    if 'leadership_potential' in df.columns:
        lead_min, lead_max = float(df['leadership_potential'].min()), float(df['leadership_potential'].max())
        lead_range = st.sidebar.slider(
            "Leadership Potential Range",
            min_value=lead_min,
            max_value=lead_max,
            value=(lead_min, lead_max)
        )
    else:
        lead_range = None
else:
    qa_filter = None
    psych_range = None
    lead_range = None

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

# Apply QA filters
if qa_filter is not None:
    filtered_df = filtered_df[filtered_df['has_quick_assessment'].isin(qa_filter)]

if psych_range is not None:
    filtered_df = filtered_df[
        (filtered_df['psychological_score'] >= psych_range[0]) &
        (filtered_df['psychological_score'] <= psych_range[1])
    ]

if lead_range is not None:
    filtered_df = filtered_df[
        (filtered_df['leadership_potential'] >= lead_range[0]) &
        (filtered_df['leadership_potential'] <= lead_range[1])
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

# Additional QA metrics if available
if has_qa_data:
    st.markdown("#### 🧠 Quick Assessment Metrics")
    qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)
    
    with qa_col1:
        if 'psychological_score' in filtered_df.columns:
            avg_psych = filtered_df['psychological_score'].mean()
            st.metric("Avg Psychological", f"{avg_psych:.2f}")
    
    with qa_col2:
        if 'leadership_potential' in filtered_df.columns:
            avg_lead = filtered_df['leadership_potential'].mean()
            st.metric("Avg Leadership", f"{avg_lead:.2f}")
    
    with qa_col3:
        if 'has_quick_assessment' in filtered_df.columns:
            qa_count = filtered_df['has_quick_assessment'].sum()
            st.metric("QA Coverage", f"{qa_count}/{len(filtered_df)}")
    
    with qa_col4:
        if 'holistic_score' in filtered_df.columns:
            avg_holistic = filtered_df['holistic_score'].mean()
            st.metric("Avg Holistic Score", f"{avg_holistic:.2f}")

# Model utilities: try import xgboost, fallback to sklearn
try:
    import xgboost as xgb
    xgb_available = True
except Exception:
    xgb_available = False

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

models_dir = Path(__file__).resolve().parents[2] / "models"
models_dir.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

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

    # Model actions: train or load
    st.markdown("---")
    st.markdown("### 🤖 Model Actions")
    model_path = models_dir / "xgb_or_sklearn_model.joblib"
    existing_model = load_model(model_path) if model_path.exists() else None

    if existing_model is not None:
        st.success("Model ditemukan: tersedia untuk download dan evaluasi.")
        with open(model_path, "rb") as mfile:
            st.download_button("📥 Download Trained Model", data=mfile, file_name=model_path.name)
    else:
        st.info("Belum ada model tersimpan. Anda bisa melatih model cepat di sini (XGBoost jika tersedia, lainnya sklearn).")
        if st.button("Train quick model (default features)"):
            # prepare data
            features = [c for c in ['performance_score', 'behavior_avg', 'tenure_years'] if c in filtered_df.columns]
            if 'has_promotion' not in filtered_df.columns or len(features) == 0:
                st.error("Tidak ada target 'has_promotion' atau fitur yang cukup untuk melatih model.")
            else:
                X = filtered_df[features].dropna()
                y = filtered_df.loc[X.index, 'has_promotion']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                if xgb_available:
                    st.write("Melatih XGBoostClassifier...")
                    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3)
                else:
                    st.write("XGBoost tidak tersedia — fallback ke sklearn.GradientBoostingClassifier")
                    clf = GradientBoostingClassifier(n_estimators=50, max_depth=3)

                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                try:
                    proba = clf.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, proba)
                except Exception:
                    auc = None

                joblib.dump(clf, model_path)
                st.success(f"Training selesai — Accuracy: {acc:.3f}" + (f", AUC: {auc:.3f}" if auc is not None else ""))
                with open(model_path, "rb") as mfile:
                    st.download_button("📥 Download Trained Model", data=mfile, file_name=model_path.name)

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
