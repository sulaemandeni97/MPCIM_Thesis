"""
Prediction Page
Interactive promotion prediction tool backed by trained models
"""

from pathlib import Path
import sys
from typing import Dict, Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import apply_styles, page_header
from services.prediction_service import load_service
from services.ai_service import create_ai_service

st.set_page_config(page_title="Promotion Prediction", page_icon="🔮", layout="wide")

apply_styles()

page_header(
    "Promotion Prediction Tool",
    "Prediksi probabilitas promosi berdasarkan data karyawan dan model terlatih",
    icon="🔮"
)

# Show AI status at top
st.sidebar.markdown("### 🤖 AI Analysis Status")

service = load_service()

# Initialize AI service (supports OpenAI and Gemini)
try:
    ai_service, ai_provider, ai_enabled = create_ai_service()  # Returns tuple: (service, name, enabled)
    if ai_enabled:
        print(f"✅ AI Service initialized: {ai_provider}")
        st.sidebar.success(f"✅ {ai_provider} Ready")
    else:
        print(f"⚠️ AI Service not configured")
        # Get appropriate message based on environment
        from services.ai_service import get_ai_status
        status = get_ai_status()
        st.sidebar.warning(status['message'])
except Exception as e:
    ai_service = None
    ai_enabled = False
    ai_provider = 'none'
    print(f"❌ AI Service error: {e}")
    st.sidebar.error(f"❌ Error\n\n{str(e)}")
    import traceback
    traceback.print_exc()

BATCH_REQUIRED_COLUMNS = [
    "performance_score",
    "behavior_avg",
    "tenure_years",
    "gender",
    "marital_status",
    "is_permanent",
    "performance_rating",
]

COLUMN_ALIASES = {
    "behavioral_score": "behavior_avg",
    "behavior_score": "behavior_avg",
    "behavior": "behavior_avg",
}


reference_assets = service.reference_assets
available_models = service.list_models()

if service.scaler is None or not service.feature_columns:
    st.error("Scaler atau fitur hasil feature engineering belum tersedia. Jalankan pipeline feature_engineering.py terlebih dahulu.")
    st.stop()

if not available_models:
    st.error("Tidak ada model terlatih yang ditemukan di folder results/. Jalankan skrip modeling sebelum menggunakan halaman ini.")
    st.stop()

model_choice = st.selectbox(
    "Pilih model untuk inferensi",
    options=list(available_models.keys()),
    format_func=lambda key: f"{available_models[key]['label']} ({available_models[key]['type']})",
    index=list(available_models.keys()).index("xgboost") if "xgboost" in available_models else 0
)

# Warnings for suboptimal models
if model_choice == "neural_network":
    st.warning("⚠️ **Note:** Neural Network model mungkin memberikan hasil yang tidak optimal. Disarankan menggunakan **XGBoost** atau **Random Forest** untuk hasil terbaik.")
elif model_choice == "dual_logistic":
    st.warning("⚠️ **Note:** Dual Logistic model memiliki akurasi rendah untuk data imbalanced. **Sangat disarankan** menggunakan **XGBoost** (default) untuk hasil terbaik.")

# Show recommended models
if model_choice not in ["xgboost", "random_forest"]:
    st.info("💡 **Rekomendasi**: Gunakan **XGBoost** (akurasi tinggi) atau **Random Forest** (balanced) untuk prediksi yang lebih akurat.")

slider_defaults = reference_assets.get("slider_defaults")
default_perf = float(slider_defaults["performance_score"]) if slider_defaults is not None else 75.0
default_beh = float(slider_defaults["behavior_avg"]) if slider_defaults is not None else 80.0
default_tenure = float(slider_defaults["tenure_years"]) if slider_defaults is not None else 5.0

st.markdown("## 📝 Input Data Karyawan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Performance & Behavioral Metrics")
    performance_score = st.slider(
        "Performance Score",
        min_value=0.0,
        max_value=100.0,
        value=float(round(default_perf, 2)),
        step=0.5,
        help="Skor kinerja karyawan (0-100)"
    )

    behavioral_score = st.slider(
        "Behavioral Score",
        min_value=0.0,
        max_value=100.0,
        value=float(round(default_beh, 2)),
        step=0.5,
        help="Rata-rata skor kompetensi perilaku (0-100)"
    )

    tenure_years = st.number_input(
        "Tenure (Years)",
        min_value=0.0,
        max_value=40.0,
        value=float(round(default_tenure, 1)),
        step=0.5,
        help="⭐ Sweet spot untuk promosi: 4-6 tahun. Tenure < 3 atau > 7 tahun memiliki peluang promosi lebih rendah."
    )
    
    # Tenure guidance
    if tenure_years < 3:
        st.info("💡 **Tip**: Tenure terlalu pendek. Optimal: 4-6 tahun untuk peluang promosi tertinggi.")
    elif tenure_years > 7:
        st.warning("⚠️ **Note**: Tenure > 7 tahun mengurangi peluang promosi (tenure paradox). Optimal: 4-6 tahun.")

with col2:
    st.markdown("### Demographic & Employment Information")
    cat_options = reference_assets["category_options"]

    gender_options = cat_options.get("gender", ["M", "O"])
    gender = st.selectbox(
        "Gender",
        options=gender_options,
        format_func=lambda x: {"M": "Male", "O": "Female/Other"}.get(x, x)
    )

    marital_options = cat_options.get("marital_status", ["single", "married", "widow", "widower"])
    marital_status = st.selectbox(
        "Marital Status",
        options=marital_options,
        format_func=lambda x: x.title()
    )

    is_permanent = st.selectbox(
        "Employment Type",
        options=["t", "f"],
        format_func=lambda x: "Permanent" if x == "t" else "Contract"
    )

    perf_rating_options = cat_options.get("performance_rating", ["Good", "Excellent", "Average", "Need Improvement", "unknown"])
    performance_rating = st.selectbox(
        "Performance Rating",
        options=perf_rating_options,
        format_func=lambda x: x.title() if x != "unknown" else "Unknown",
        index=0,  # Default to "Good"
        help="⭐ Untuk score tinggi (80+), pilih 'Good' atau 'Excellent'. Rating 'Average' akan menurunkan peluang promosi meskipun score tinggi."
    )
    
    # Rating guidance
    if performance_rating == "Average" and (performance_score >= 80 or behavioral_score >= 80):
        st.warning("⚠️ **Perhatian**: Score Anda tinggi (80+) tapi rating 'Average'. Ini **inconsistent** dan menurunkan peluang promosi. Ubah ke 'Good' atau 'Excellent' untuk hasil lebih baik.")

# Quick Assessment Section
st.markdown("---")
st.markdown("## 🧠 Quick Assessment (Psychological Components)")

# Expandable section for QA
with st.expander("📊 **Tambahkan Data Quick Assessment** (Opsional - Meningkatkan Akurasi Prediksi)", expanded=False):
    st.info("💡 **Quick Assessment** menambahkan dimensi psikologis untuk prediksi yang lebih akurat. Jika tidak diisi, akan menggunakan nilai rata-rata.")
    
    qa_col1, qa_col2 = st.columns(2)
    
    with qa_col1:
        st.markdown("#### 🎯 Drive & Ambition")
        
        drive_score = st.slider(
            "Drive Score",
            min_value=0.0,
            max_value=100.0,
            value=43.0,
            step=1.0,
            help="Tingkat motivasi dan ambisi karyawan"
        )
        
        self_ambition = st.slider(
            "Self Ambition",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Ambisi pribadi untuk berkembang"
        )
        
        st.markdown("#### 💪 Mental Strength")
        
        mental_strength_score = st.slider(
            "Mental Strength",
            min_value=0.0,
            max_value=100.0,
            value=43.1,
            step=1.0,
            help="Ketahanan mental dalam menghadapi tantangan"
        )
        
        resilience = st.slider(
            "Resilience",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Kemampuan bangkit dari kegagalan"
        )
    
    with qa_col2:
        st.markdown("#### 🤝 Collaboration & Adaptability")
        
        collaboration_score = st.slider(
            "Collaboration Score",
            min_value=0.0,
            max_value=20.0,
            value=7.4,
            step=0.1,
            help="Kemampuan bekerja sama dalam tim"
        )
        
        adaptability_score = st.slider(
            "Functional Adaptability",
            min_value=0.0,
            max_value=100.0,
            value=43.2,
            step=1.0,
            help="Kemampuan beradaptasi dengan perubahan"
        )
        
        st.markdown("#### 🧩 Problem Solving")
        
        problem_solving = st.slider(
            "Problem Solving",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Kemampuan menyelesaikan masalah"
        )
        
        learner_orientation = st.slider(
            "Learner Orientation",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help="Orientasi untuk terus belajar"
        )
    
    # Calculate psychological score (average of components)
    psychological_components = [
        self_ambition, learner_orientation, resilience, 
        problem_solving, drive_score, mental_strength_score,
        adaptability_score, collaboration_score
    ]
    psychological_score = sum(psychological_components) / len(psychological_components)
    
    has_quick_assessment = 1  # User provided QA data
    
    # Show calculated metrics
    st.markdown("---")
    st.markdown("#### 📊 Calculated Psychological Metrics")
    
    qa_metrics_col1, qa_metrics_col2, qa_metrics_col3 = st.columns(3)
    
    with qa_metrics_col1:
        st.metric("Psychological Score", f"{psychological_score:.1f}", 
                 help="Rata-rata dari semua komponen psikologis")
    
    with qa_metrics_col2:
        leadership_potential = (drive_score * 0.4 + mental_strength_score * 0.3 + collaboration_score * 0.3)
        st.metric("Leadership Potential", f"{leadership_potential:.1f}",
                 help="Potensi kepemimpinan berdasarkan drive, mental strength, dan collaboration")
    
    with qa_metrics_col3:
        holistic_score = (performance_score * 0.4 + behavioral_score * 0.3 + psychological_score * 0.3)
        st.metric("Holistic Score", f"{holistic_score:.1f}",
                 help="Skor gabungan: 40% performance + 30% behavioral + 30% psychological")

# If QA not filled, use defaults
if 'psychological_score' not in locals():
    psychological_score = 23.5
    drive_score = 43.0
    mental_strength_score = 43.1
    adaptability_score = 43.2
    collaboration_score = 7.4
    has_quick_assessment = 0

# OpenAI Analysis toggle
st.markdown("---")
col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    prediction_ready = st.button("🎯 Predict Promotion Probability", type="primary", use_container_width=True)
with col_btn2:
    if ai_enabled:
        provider_emoji = "🤖" if ai_provider == 'gemini' else "🧠"
        provider_name = ai_provider.title()
        use_ai = st.checkbox(f"{provider_emoji} AI Analysis ({provider_name})", value=True, 
                             help=f"Gunakan {provider_name} untuk analisis mendalam")
    else:
        use_ai = False
        st.info("💡 Set GEMINI_API_KEY atau OPENAI_API_KEY")

if prediction_ready:
    with st.spinner("Menjalankan inferensi model..."):
        try:
            raw_inputs = {
                "performance_score": performance_score,
                "behavior_avg": behavioral_score,
                "tenure_years": tenure_years,
                "gender": gender,
                "marital_status": marital_status,
                "is_permanent": is_permanent,
                "performance_rating": performance_rating,
                # Quick Assessment features
                "psychological_score": psychological_score,
                "drive_score": drive_score,
                "mental_strength_score": mental_strength_score,
                "adaptability_score": adaptability_score,
                "collaboration_score": collaboration_score,
                "has_quick_assessment": has_quick_assessment,
            }
            prediction_result = service.predict_single(raw_inputs, model_choice)
            prediction = prediction_result.prediction
            probability = prediction_result.probability
            derived = prediction_result.derived_signals
        except Exception as exc:
            st.error(f"Gagal melakukan prediksi: {exc}")
            st.stop()

    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", "✅ PROMOTED" if prediction == 1 else "❌ NOT PROMOTED")

    with col2:
        st.metric("Probability", f"{probability:.2%}")

    with col3:
        confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
        st.metric("Confidence", confidence)

    st.markdown("### 📊 Promotion Probability")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Promotion Probability (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧩 Feature Signals")
    
    # Traditional signals
    col_sig1, col_sig2 = st.columns(2)
    
    with col_sig1:
        st.markdown("#### 📊 Traditional Metrics")
        traditional_df = pd.DataFrame({
            "Signal": [
                "Perf/Beh Ratio",
                "Combined Score",
                "Score Difference",
                "Tenure Category",
                "Performance Level",
                "Behavioral Level",
            ],
            "Value": [
                f"{derived['perf_beh_ratio']:.2f}",
                f"{derived['combined_score']:.2f}",
                f"{derived['score_difference']:.2f}",
                derived["tenure_category"].title(),
                derived["performance_level"].title(),
                derived["behavioral_level"].title(),
            ]
        })
        st.dataframe(traditional_df, hide_index=True, use_container_width=True)
    
    with col_sig2:
        st.markdown("#### 🧠 Psychological Metrics")
        qa_df = pd.DataFrame({
            "Signal": [
                "Psychological Score",
                "Drive Score",
                "Mental Strength",
                "Adaptability",
                "Collaboration",
                "Leadership Potential",
                "Holistic Score",
                "Score Alignment",
            ],
            "Value": [
                f"{derived['psychological_score']:.2f}",
                f"{derived['drive_score']:.2f}",
                f"{derived['mental_strength_score']:.2f}",
                f"{derived['adaptability_score']:.2f}",
                f"{derived['collaboration_score']:.2f}",
                f"{derived['leadership_potential']:.2f}",
                f"{derived['holistic_score']:.2f}",
                f"{derived['score_alignment']:.2f}",
            ]
        })
        st.dataframe(qa_df, hide_index=True, use_container_width=True)
        
        if derived['has_quick_assessment'] == 1:
            st.success("✅ Quick Assessment data provided - Enhanced prediction accuracy!")
        else:
            st.info("ℹ️ Using average Quick Assessment values - Add QA data for better accuracy")

    st.markdown("### 💡 Recommendations")
    improvements = []
    benchmark = service.get_benchmark()
    if benchmark is not None:
        if performance_score < benchmark["promoted"]["performance_score"]:
            improvements.append("📊 Tingkatkan konsistensi performance agar menyamai rerata karyawan promoted.")
        if behavioral_score < benchmark["promoted"]["behavior_avg"]:
            improvements.append("🎭 Fokus pada penguatan kompetensi perilaku (kolaborasi, kepemimpinan).")
        if tenure_years < benchmark["promoted"]["tenure_years"]:
            improvements.append("⏱️ Perlu pengalaman lebih panjang atau pencapaian luar biasa untuk mengimbangi masa kerja.")

    if prediction == 1:
        st.success(
            "Karyawan menunjukkan potensi promosi tinggi. Pertimbangkan program leadership, succession planning, "
            "dan tetap monitor konsistensi kinerja."
        )
    else:
        if not improvements:
            improvements.append("Tetap tingkatkan kompetensi inti dan cari feedback berkala.")
        st.warning("Perkuat area berikut untuk meningkatkan peluang promosi:\n\n- " + "\n- ".join(improvements))

    st.markdown("### 📊 Comparison with Promoted Employees")
    if benchmark is not None:
        benchmark_data = pd.DataFrame({
            "Metric": ["Performance Score", "Behavioral Score", "Tenure (years)"],
            "Your Score": [performance_score, behavioral_score, tenure_years],
            "Promoted Avg": [
                benchmark["promoted"]["performance_score"],
                benchmark["promoted"]["behavior_avg"],
                benchmark["promoted"]["tenure_years"],
            ],
            "Not Promoted Avg": [
                benchmark["not_promoted"]["performance_score"],
                benchmark["not_promoted"]["behavior_avg"],
                benchmark["not_promoted"]["tenure_years"],
            ],
        })
    else:
        benchmark_data = pd.DataFrame({
            "Metric": ["Performance Score", "Behavioral Score", "Tenure (years)"],
            "Your Score": [performance_score, behavioral_score, tenure_years],
        })

    col1, col2 = st.columns([2, 1])
    with col1:
        fig = go.Figure()
        for column, color in [
            ("Your Score", "#3498db"),
            ("Promoted Avg", "#2ecc71"),
            ("Not Promoted Avg", "#e74c3c"),
        ]:
            if column in benchmark_data.columns:
                fig.add_trace(go.Bar(
                    name=column,
                    x=benchmark_data["Metric"],
                    y=benchmark_data[column],
                    marker_color=color
                ))
        fig.update_layout(
            title="Benchmark Comparison",
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode="group",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.dataframe(benchmark_data, use_container_width=True, hide_index=True)

    # AI Analysis Section - Enhanced Professional UI
    if use_ai and ai_enabled and ai_service:
        st.markdown("---")
        
        # Header with gradient-style container
        provider_icon = "🤖" if ai_provider == 'gemini' else "🧠"
        provider_name = "Gemini Pro" if ai_provider == 'gemini' else "OpenAI GPT-4"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">
                {provider_icon} AI-Powered Analysis
            </h2>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 14px;">
                Powered by {provider_name}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        spinner_text = f"🧠 Menganalisis hasil prediksi dengan {provider_name}..."
        with st.spinner(spinner_text):
            try:
                prediction_data = {
                    "label": prediction_result.label,
                    "prediction": prediction_result.prediction,
                    "probability": prediction_result.probability,
                    "derived_signals": prediction_result.derived_signals,
                    "model_key": model_choice
                }
                
                employee_data = raw_inputs
                benchmark_data_dict = service.get_benchmark()
                
                analysis = ai_service.analyze_prediction(
                    prediction_data,
                    employee_data,
                    benchmark_data_dict
                )
                
                # Executive Summary with enhanced styling
                st.markdown("""
                <div style="background-color: #f0f7ff; border-left: 4px solid #2196F3; 
                            padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="color: #1976D2; margin-top: 0;">📋 Ringkasan Eksekutif</h3>
                """, unsafe_allow_html=True)
                st.markdown(analysis.summary)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Create tabs with icons
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Analisis Detail", 
                    "💪 Kekuatan", 
                    "⚠️ Faktor Risiko", 
                    "🎯 Rekomendasi"
                ])
                
                with tab1:
                    st.markdown("""
                    <div style="background-color: #fafafa; padding: 20px; 
                                border-radius: 8px; border: 1px solid #e0e0e0;">
                    """, unsafe_allow_html=True)
                    st.markdown(analysis.detailed_analysis)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tab2:
                    if analysis.strengths:
                        st.markdown("### 💪 Kekuatan Karyawan")
                        for i, strength in enumerate(analysis.strengths, 1):
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 12px; 
                                        border-radius: 6px; margin-bottom: 10px; 
                                        border-left: 3px solid #4CAF50;">
                                <strong>{i}.</strong> {strength}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Tidak ada kekuatan spesifik yang diidentifikasi.")
                
                with tab3:
                    if analysis.risk_factors:
                        st.markdown("### ⚠️ Area yang Perlu Perhatian")
                        for i, risk in enumerate(analysis.risk_factors, 1):
                            st.markdown(f"""
                            <div style="background-color: #fff3e0; padding: 12px; 
                                        border-radius: 6px; margin-bottom: 10px; 
                                        border-left: 3px solid #FF9800;">
                                <strong>{i}.</strong> {risk}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ Tidak ada faktor risiko signifikan yang teridentifikasi.")
                
                with tab4:
                    if analysis.recommendations:
                        st.markdown("### 🎯 Rekomendasi Actionable")
                        for i, rec in enumerate(analysis.recommendations, 1):
                            st.markdown(f"""
                            <div style="background-color: #e3f2fd; padding: 12px; 
                                        border-radius: 6px; margin-bottom: 10px; 
                                        border-left: 3px solid #2196F3;">
                                <strong>{i}.</strong> {rec}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("Tidak ada rekomendasi khusus saat ini.")
                
                # Show raw response in expander with better styling
                with st.expander("🔍 Lihat Analisis Lengkap (Raw Response)"):
                    st.code(analysis.raw_response, language="text")
                
            except Exception as e:
                st.markdown("""
                <div style="background-color: #ffebee; border-left: 4px solid #f44336; 
                            padding: 15px; border-radius: 5px;">
                    <h4 style="color: #c62828; margin-top: 0;">❌ Error dalam Analisis AI</h4>
                """, unsafe_allow_html=True)
                st.error(f"**Detail Error:** {str(e)}")
                st.info(f"💡 **Solusi:** Pastikan `{ai_provider.upper()}_API_KEY` sudah diset dengan benar di file `.env`")
                st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 📁 Batch Prediction")

st.info("""
**Upload CSV file** untuk prediksi batch multiple karyawan. 
Kolom wajib: `performance_score`, `behavior_avg`, `tenure_years`, `gender`, `marital_status`, `is_permanent`, `performance_rating`.
Opsional: gunakan `behavioral_score` sebagai alias `behavior_avg`.
""")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

def normalize_batch_inputs(row: pd.Series) -> Dict[str, Any]:
    data = {}
    for col in BATCH_REQUIRED_COLUMNS:
        data[col] = row[col]

    data["performance_score"] = float(data["performance_score"])
    data["behavior_avg"] = float(data["behavior_avg"])
    data["tenure_years"] = float(data["tenure_years"])

    gender_raw = str(data["gender"]).strip().lower()
    gender_map = {"male": "M", "m": "M", "female": "O", "f": "O", "o": "O"}
    data["gender"] = gender_map.get(gender_raw, data["gender"])

    data["marital_status"] = str(data["marital_status"]).strip().lower()

    emp_raw = str(data["is_permanent"]).strip().lower()
    emp_map = {"permanent": "t", "contract": "f", "t": "t", "f": "f", "true": "t", "false": "f", "1": "t", "0": "f"}
    data["is_permanent"] = emp_map.get(emp_raw, data["is_permanent"])

    data["performance_rating"] = str(data["performance_rating"]).strip()
    return data


if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        batch_df.rename(columns=COLUMN_ALIASES, inplace=True)
        missing_cols = [col for col in BATCH_REQUIRED_COLUMNS if col not in batch_df.columns]
        if missing_cols:
            st.error(f"Kolom berikut wajib ada: {', '.join(missing_cols)}")
        else:
            st.success(f"✅ File uploaded successfully! {len(batch_df)} records found.")
            st.markdown("### Preview Data")
            st.dataframe(batch_df.head(10), use_container_width=True)

            if st.button("🎯 Run Batch Prediction"):
                preds, probs = [], []
                for _, row in batch_df.iterrows():
                    try:
                        normalized = normalize_batch_inputs(row)
                        result = service.predict_single(normalized, model_choice)
                        preds.append(result.prediction)
                        probs.append(result.probability)
                    except Exception as exc:
                        preds.append(float("nan"))
                        probs.append(float("nan"))
                        st.warning(f"Baris gagal diproses: {exc}")

                batch_df['prediction'] = preds
                batch_df['probability'] = probs
                batch_df['prediction_label'] = batch_df['prediction'].map({0: 'Not Promoted', 1: 'Promoted'})

                st.markdown("### 🎯 Prediction Results")
                st.dataframe(batch_df, use_container_width=True)

                valid_probs = batch_df['probability'].dropna()
                total_valid = batch_df['prediction'].dropna().shape[0]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", total_valid)
                with col2:
                    promoted_count = int(batch_df['prediction'].fillna(0).sum())
                    st.metric("Predicted Promoted", promoted_count)
                with col3:
                    avg_prob = valid_probs.mean() if not valid_probs.empty else 0.0
                    st.metric("Avg Probability", f"{avg_prob:.2%}")

                csv_bytes = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv_bytes,
                    file_name="promotion_predictions.csv",
                    mime="text/csv"
                )

    except Exception as exc:
        st.error(f"❌ Error processing file: {exc}")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem 0;'>
    <p><strong>Note:</strong> Gunakan hasil prediksi sebagai bahan pertimbangan, bukan keputusan tunggal.</p>
</div>
""", unsafe_allow_html=True)
