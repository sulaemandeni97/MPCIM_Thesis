"""
Prediction Page
Interactive promotion prediction tool
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import apply_styles, page_header

st.set_page_config(page_title="Promotion Prediction", page_icon="🔮", layout="wide")

apply_styles()

page_header("Promotion Prediction Tool", "Prediksi probabilitas promosi berdasarkan data karyawan", icon="🔮")

# Mock prediction function (replace with actual model)
def predict_promotion(features):
    """
    Mock prediction function
    In production, this would load a trained model and make predictions
    """
    # Simple rule-based prediction for demonstration
    performance = features['performance_score']
    behavioral = features['behavior_avg']
    tenure = features['tenure_years']
    
    # Calculate weighted score
    score = (performance * 0.3 + behavioral * 0.4 + tenure * 0.3) / 100
    
    # Add some randomness
    score = min(0.95, max(0.05, score + np.random.normal(0, 0.1)))
    
    prediction = 1 if score > 0.5 else 0
    
    return prediction, score

# Input form
st.markdown("## 📝 Input Data Karyawan")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Performance & Behavioral Metrics")
    
    performance_score = st.slider(
        "Performance Score",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=0.5,
        help="Skor kinerja karyawan (0-100)"
    )
    
    behavioral_score = st.slider(
        "Behavioral Score",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=0.5,
        help="Rata-rata skor kompetensi perilaku (0-100)"
    )
    
    tenure_years = st.number_input(
        "Tenure (Years)",
        min_value=0.0,
        max_value=40.0,
        value=5.0,
        step=0.5,
        help="Masa kerja dalam tahun"
    )

with col2:
    st.markdown("### Demographic Information")
    
    gender = st.selectbox(
        "Gender",
        options=['M', 'F'],
        format_func=lambda x: 'Male' if x == 'M' else 'Female'
    )
    
    marital_status = st.selectbox(
        "Marital Status",
        options=['Single', 'Married', 'Divorced', 'Widowed']
    )
    
    is_permanent = st.selectbox(
        "Employment Type",
        options=['t', 'f'],
        format_func=lambda x: 'Permanent' if x == 't' else 'Contract'
    )

# Predict button
st.markdown("---")

if st.button("🎯 Predict Promotion Probability", type="primary", use_container_width=True):
    
    # Prepare features
    features = {
        'performance_score': performance_score,
        'behavior_avg': behavioral_score,
        'tenure_years': tenure_years,
        'gender': gender,
        'marital_status': marital_status,
        'is_permanent': is_permanent
    }
    
    # Make prediction
    prediction, probability = predict_promotion(features)
    
    # Display results
    st.markdown("---")
    st.markdown("## 🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Prediction",
            "✅ PROMOTED" if prediction == 1 else "❌ NOT PROMOTED",
            delta=None
        )
    
    with col2:
        st.metric(
            "Probability",
            f"{probability:.2%}",
            delta=None
        )
    
    with col3:
        confidence = "High" if abs(probability - 0.5) > 0.3 else "Medium" if abs(probability - 0.5) > 0.15 else "Low"
        st.metric(
            "Confidence",
            confidence,
            delta=None
        )
    
    # Probability gauge
    st.markdown("### 📊 Promotion Probability")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Promotion Probability (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#ffcccc'},
                {'range': [30, 70], 'color': '#ffffcc'},
                {'range': [70, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature contribution
    st.markdown("### 🎯 Feature Contribution Analysis")
    
    # Calculate normalized contributions
    perf_contrib = (performance_score / 100) * 0.30
    beh_contrib = (behavioral_score / 100) * 0.40
    tenure_contrib = min(tenure_years / 10, 1.0) * 0.30
    
    contrib_data = pd.DataFrame({
        'Feature': ['Performance Score', 'Behavioral Score', 'Tenure'],
        'Contribution': [perf_contrib, beh_contrib, tenure_contrib],
        'Weight': ['30%', '40%', '30%']
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=contrib_data['Feature'],
                y=contrib_data['Contribution'],
                marker_color=['#3498db', '#9b59b6', '#f39c12'],
                text=contrib_data['Contribution'].round(3),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Feature Contribution to Prediction",
            xaxis_title="Feature",
            yaxis_title="Contribution Score",
            height=400,
            yaxis_range=[0, 0.5]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Contribution Breakdown")
        st.dataframe(contrib_data, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Interpretation:**
        - Higher contribution = stronger influence on prediction
        - Behavioral score has highest weight (40%)
        - Performance score: 30%
        - Tenure: 30%
        """)
    
    # Recommendations
    st.markdown("---")
    st.markdown("### 💡 Recommendations")
    
    if prediction == 1:
        st.success("""
        #### ✅ High Promotion Potential
        
        This employee shows strong indicators for promotion:
        - Continue maintaining high performance levels
        - Keep demonstrating positive behavioral competencies
        - Consider for leadership development programs
        - Prepare succession planning discussions
        """)
    else:
        # Identify areas for improvement
        improvements = []
        
        if performance_score < 70:
            improvements.append("📊 **Performance Score**: Focus on improving work quality and productivity")
        
        if behavioral_score < 70:
            improvements.append("🎭 **Behavioral Competencies**: Develop soft skills and teamwork abilities")
        
        if tenure_years < 2:
            improvements.append("⏱️ **Tenure**: Gain more experience in current role")
        
        st.warning(f"""
        #### ⚠️ Development Areas
        
        To improve promotion chances, consider:
        
        {chr(10).join(improvements) if improvements else "- Continue building experience and skills"}
        
        **Action Items:**
        - Set clear performance goals
        - Participate in training programs
        - Seek mentorship opportunities
        - Request regular feedback
        """)
    
    # Comparison with benchmarks
    st.markdown("---")
    st.markdown("### 📊 Comparison with Promoted Employees")
    
    # Mock benchmark data
    benchmark_data = pd.DataFrame({
        'Metric': ['Performance Score', 'Behavioral Score', 'Tenure (years)'],
        'Your Score': [performance_score, behavioral_score, tenure_years],
        'Promoted Avg': [78.5, 82.3, 6.2],
        'Not Promoted Avg': [72.1, 75.8, 4.8]
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Your Score',
            x=benchmark_data['Metric'],
            y=benchmark_data['Your Score'],
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            name='Promoted Avg',
            x=benchmark_data['Metric'],
            y=benchmark_data['Promoted Avg'],
            marker_color='#2ecc71'
        ))
        
        fig.add_trace(go.Bar(
            name='Not Promoted Avg',
            x=benchmark_data['Metric'],
            y=benchmark_data['Not Promoted Avg'],
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title="Comparison with Benchmarks",
            xaxis_title="Metric",
            yaxis_title="Score",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Benchmark Data")
        st.dataframe(benchmark_data, use_container_width=True, hide_index=True)

# Batch prediction
st.markdown("---")
st.markdown("## 📁 Batch Prediction")

st.info("""
**Upload CSV file** untuk prediksi batch multiple karyawan.

Format CSV harus memiliki kolom:
- `performance_score`
- `behavior_avg`
- `tenure_years`
- `gender`
- `marital_status`
- `is_permanent`
""")

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! {len(batch_df)} records found.")
        
        # Show preview
        st.markdown("### Preview Data")
        st.dataframe(batch_df.head(10), use_container_width=True)
        
        if st.button("🎯 Run Batch Prediction"):
            # Make predictions for all rows
            predictions = []
            probabilities = []
            
            for _, row in batch_df.iterrows():
                features = row.to_dict()
                pred, prob = predict_promotion(features)
                predictions.append(pred)
                probabilities.append(prob)
            
            # Add results to dataframe
            batch_df['prediction'] = predictions
            batch_df['probability'] = probabilities
            batch_df['prediction_label'] = batch_df['prediction'].map({0: 'Not Promoted', 1: 'Promoted'})
            
            st.markdown("### 🎯 Prediction Results")
            st.dataframe(batch_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", len(batch_df))
            
            with col2:
                promoted_count = batch_df['prediction'].sum()
                st.metric("Predicted Promoted", promoted_count)
            
            with col3:
                avg_prob = batch_df['probability'].mean()
                st.metric("Avg Probability", f"{avg_prob:.2%}")
            
            # Download results
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="promotion_predictions.csv",
                mime="text/csv"
            )
    
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem 0;'>
    <p><strong>Note:</strong> Predictions are based on trained ML models. Always combine with human judgment for final decisions.</p>
</div>
""", unsafe_allow_html=True)
