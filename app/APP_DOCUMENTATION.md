# 📚 MPCIM Dashboard - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Installation](#installation)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Customization](#customization)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)

---

## Overview

MPCIM Dashboard adalah aplikasi web interaktif yang dibangun dengan **Streamlit** untuk visualisasi dan analisis hasil penelitian Multi-Dimensional Performance-Career Integration Model.

### Technology Stack
- **Frontend**: Streamlit 1.29.0
- **Visualization**: Plotly 5.18.0
- **Data Processing**: Pandas 2.1.4, NumPy 1.26.2
- **Statistics**: SciPy 1.11.4
- **Machine Learning**: Scikit-learn 1.3.2, XGBoost 2.0.3

### Key Capabilities
- Interactive data exploration
- Statistical analysis with hypothesis testing
- ML model performance comparison
- Real-time promotion prediction
- Batch processing
- Export functionality

---

## Architecture

### Application Structure

```
app/
├── Home.py                          # Main entry point
├── pages/                           # Multi-page app structure
│   ├── 1_📊_Data_Explorer.py       # Data exploration
│   ├── 2_📈_EDA_Results.py         # Statistical analysis
│   ├── 3_🤖_Model_Performance.py   # ML model metrics
│   └── 4_🔮_Prediction.py          # Prediction tool
├── .streamlit/                      # Configuration
│   ├── config.toml                  # App settings
│   └── secrets.toml.example         # Secrets template
├── requirements.txt                 # Dependencies
├── README.md                        # Main documentation
├── QUICK_START.md                   # Quick start guide
└── run_app.sh                       # Startup script
```

### Data Flow

```
User Input → Streamlit UI → Data Processing (Pandas) → 
Analysis/Prediction → Visualization (Plotly) → Display Results
```

---

## Features

### 1. Home Page (`Home.py`)

**Purpose**: Landing page dengan overview penelitian

**Components**:
- Research overview
- Quick statistics dashboard
- Navigation guide
- Feature highlights

**Key Functions**:
```python
@st.cache_data
def load_data():
    """Load and cache dataset"""
    return pd.read_csv(data_path)
```

### 2. Data Explorer (`1_📊_Data_Explorer.py`)

**Purpose**: Interactive data exploration dan filtering

**Features**:
- Multi-criteria filtering (promotion, gender, scores)
- Search functionality
- Descriptive statistics
- Distribution visualizations
- Correlation analysis
- Data export (CSV)

**Filters Available**:
- Promotion Status (0/1)
- Gender (M/F)
- Performance Score Range (slider)
- Behavioral Score Range (slider)

**Visualizations**:
- Histograms (performance, behavioral scores)
- Box plots by promotion status
- Scatter plots (relationships)
- Correlation heatmap

### 3. EDA Results (`2_📈_EDA_Results.py`)

**Purpose**: Comprehensive statistical analysis

**Statistical Tests**:
- Independent t-tests
- Cohen's d effect size
- Correlation analysis
- Distribution comparisons

**Visualizations**:
- Box plots with mean/SD
- Violin plots
- Overlapping histograms
- 3D scatter plots
- Correlation heatmaps

**Key Metrics**:
```python
# T-test example
t_stat, p_val = stats.ttest_ind(promoted, not_promoted)

# Cohen's d
pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
cohens_d = (mean1 - mean2) / pooled_std
```

### 4. Model Performance (`3_🤖_Model_Performance.py`)

**Purpose**: ML model comparison dan evaluation

**Models Compared**:
1. Logistic Regression (Baseline)
2. Random Forest (Baseline)
3. XGBoost (Advanced)
4. Neural Network (Advanced)

**Metrics Displayed**:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

**Visualizations**:
- Metrics comparison table (styled)
- Grouped bar charts
- Radar charts
- ROC curves
- Confusion matrix
- Feature importance

**Example Metrics Structure**:
```python
results = {
    'Model Name': {
        'accuracy': 0.92,
        'precision': 0.85,
        'recall': 0.64,
        'f1_score': 0.73,
        'roc_auc': 0.94
    }
}
```

### 5. Prediction Tool (`4_🔮_Prediction.py`)

**Purpose**: Interactive promotion prediction

**Input Features**:
- Performance Score (0-100)
- Behavioral Score (0-100)
- Tenure (years)
- Gender (M/F)
- Marital Status
- Employment Type (Permanent/Contract)

**Output**:
- Prediction (Promoted/Not Promoted)
- Probability (0-100%)
- Confidence Level (High/Medium/Low)
- Feature contributions
- Recommendations
- Benchmark comparison

**Batch Prediction**:
- Upload CSV file
- Process multiple records
- Download results

**Prediction Function**:
```python
def predict_promotion(features):
    """
    Make promotion prediction
    
    Args:
        features (dict): Employee features
    
    Returns:
        tuple: (prediction, probability)
    """
    # Feature preparation
    # Model prediction
    # Return results
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum
- Modern web browser

### Step-by-Step Installation

1. **Navigate to app directory**:
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/app
```

2. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
streamlit --version
```

### Dependencies

```
streamlit==1.29.0      # Web framework
pandas==2.1.4          # Data manipulation
numpy==1.26.2          # Numerical computing
plotly==5.18.0         # Interactive visualizations
scipy==1.11.4          # Statistical functions
scikit-learn==1.3.2    # ML utilities
xgboost==2.0.3         # Gradient boosting
```

---

## Usage Guide

### Running the Application

**Method 1: Direct Command**
```bash
streamlit run Home.py
```

**Method 2: Custom Port**
```bash
streamlit run Home.py --server.port 8080
```

**Method 3: Network Access**
```bash
streamlit run Home.py --server.address 0.0.0.0
```

### Navigation

1. **Sidebar**: Use for page navigation and filters
2. **Tabs**: Within pages for different views
3. **Expandable Sections**: Click to reveal more details
4. **Interactive Elements**: Sliders, dropdowns, buttons

### Data Upload

**Supported Formats**: CSV

**Required Columns**:
- `performance_score` (float)
- `behavior_avg` (float)
- `tenure_years` (float)
- `gender` (string: M/F)
- `marital_status` (string)
- `is_permanent` (string: t/f)
- `has_promotion` (int: 0/1)

**Upload Process**:
1. Navigate to Prediction page
2. Scroll to "Batch Prediction"
3. Click "Upload CSV file"
4. Select file
5. Click "Run Batch Prediction"
6. Download results

---

## API Reference

### Caching Functions

```python
@st.cache_data
def load_data():
    """
    Load and cache dataset
    Cached to improve performance
    """
    data_path = Path("...")
    return pd.read_csv(data_path)
```

### Prediction Functions

```python
def predict_promotion(features: dict) -> tuple:
    """
    Predict promotion probability
    
    Args:
        features: Dictionary with employee data
        
    Returns:
        (prediction, probability) tuple
    """
```

### Visualization Functions

```python
def create_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """
    Create interactive confusion matrix
    
    Args:
        cm: 2x2 confusion matrix array
        
    Returns:
        Plotly figure object
    """
```

---

## Customization

### Changing Colors

**Edit in each page file**:
```python
color_discrete_map = {
    0: '#e74c3c',  # Red for not promoted
    1: '#2ecc71'   # Green for promoted
}
```

### Modifying Metrics

**Add new metrics in Model Performance**:
```python
comparison_data.append({
    'Model': model_name,
    'Accuracy': metrics['accuracy'],
    'Your_New_Metric': metrics['new_metric']
})
```

### Custom Styling

**Add CSS in Home.py**:
```python
st.markdown("""
<style>
    .custom-class {
        /* Your styles */
    }
</style>
""", unsafe_allow_html=True)
```

### Theme Configuration

**Edit `.streamlit/config.toml`**:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "monospace"
```

---

## Deployment

### Streamlit Cloud (Free)

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy**:
- Go to https://share.streamlit.io
- Connect GitHub repository
- Select `app/Home.py` as main file
- Deploy!

### Heroku

1. **Create `Procfile`**:
```
web: sh setup.sh && streamlit run Home.py
```

2. **Create `setup.sh`**:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

3. **Deploy**:
```bash
heroku create your-app-name
git push heroku main
```

### Docker

**Create `Dockerfile`**:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "Home.py"]
```

**Build and run**:
```bash
docker build -t mpcim-dashboard .
docker run -p 8501:8501 mpcim-dashboard
```

---

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Solution
pip install -r requirements.txt --upgrade
```

**2. Port Already in Use**
```bash
# Solution
streamlit run Home.py --server.port 8502
```

**3. Data Not Loading**
- Check file path
- Verify CSV format
- Check file permissions

**4. Slow Performance**
- Use smaller dataset for testing
- Clear cache: Settings → Clear cache
- Restart application

**5. Visualization Not Showing**
- Check browser console for errors
- Try different browser
- Clear browser cache

### Debug Mode

**Enable debug logging**:
```bash
streamlit run Home.py --logger.level=debug
```

### Performance Optimization

1. **Use caching**:
```python
@st.cache_data
def expensive_computation():
    # Your code
```

2. **Limit data size**:
```python
df = df.sample(n=1000)  # Sample for testing
```

3. **Lazy loading**:
```python
if st.button("Load Data"):
    df = load_data()
```

---

## Best Practices

### Code Organization
- Keep functions small and focused
- Use type hints
- Add docstrings
- Follow PEP 8 style guide

### Performance
- Cache expensive operations
- Use vectorized operations (Pandas/NumPy)
- Limit data size for visualizations
- Optimize imports

### Security
- Never hardcode credentials
- Use environment variables
- Validate user inputs
- Sanitize file uploads

### User Experience
- Provide clear instructions
- Show loading indicators
- Handle errors gracefully
- Give meaningful feedback

---

## Future Enhancements

- [ ] User authentication
- [ ] Database integration (PostgreSQL)
- [ ] Real-time model training
- [ ] PDF report generation
- [ ] Email notifications
- [ ] Multi-language support
- [ ] Dark mode
- [ ] Mobile optimization
- [ ] API endpoints
- [ ] Advanced analytics

---

## Support

For issues or questions:
1. Check this documentation
2. Review error messages
3. Check Streamlit logs
4. Consult Streamlit documentation

---

**Last Updated**: October 22, 2025  
**Version**: 1.0.0  
**Author**: Denis Ulaeman
