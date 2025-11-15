# 🤖 Model Integration Guide

**Panduan untuk mengintegrasikan trained ML models ke dalam MPCIM Dashboard**

---

## 📋 Overview

Saat ini, aplikasi menggunakan **mock predictions** untuk demonstrasi. Panduan ini akan membantu Anda mengintegrasikan model ML yang sudah dilatih untuk prediksi yang sebenarnya.

---

## 🎯 Prerequisites

### Model Files Needed
- ✅ Trained model file (`.pkl`, `.joblib`, `.h5`, dll)
- ✅ Scaler/preprocessor (jika ada)
- ✅ Feature names list
- ✅ Model metadata (accuracy, metrics, dll)

### Python Libraries
```bash
pip install joblib  # For sklearn models
pip install tensorflow  # For neural networks (if needed)
```

---

## 🔧 Step-by-Step Integration

### Step 1: Save Your Trained Model

**For Scikit-learn models:**
```python
import joblib

# After training
model = your_trained_model
joblib.dump(model, 'models/xgboost_model.pkl')

# Save scaler too
joblib.dump(scaler, 'models/scaler.pkl')
```

**For TensorFlow/Keras:**
```python
model.save('models/neural_network.h5')
```

### Step 2: Create Models Directory

```bash
mkdir -p /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/app/models
```

### Step 3: Copy Model Files

```bash
cp /path/to/your/model.pkl app/models/
cp /path/to/your/scaler.pkl app/models/
```

---

## 📝 Code Integration

### Update Prediction Page

**File**: `pages/4_🔮_Prediction.py`

#### Current Code (Mock):
```python
def predict_promotion(features):
    """Mock prediction function"""
    performance = features['performance_score']
    behavioral = features['behavior_avg']
    tenure = features['tenure_years']
    
    score = (performance * 0.3 + behavioral * 0.4 + tenure * 0.3) / 100
    score = min(0.95, max(0.05, score + np.random.normal(0, 0.1)))
    
    prediction = 1 if score > 0.5 else 0
    return prediction, score
```

#### Updated Code (Real Model):
```python
import joblib
from pathlib import Path

# Load model once (at top of file, outside function)
@st.cache_resource
def load_model():
    """Load trained model and scaler"""
    model_path = Path(__file__).parent.parent / 'models' / 'xgboost_model.pkl'
    scaler_path = Path(__file__).parent.parent / 'models' / 'scaler.pkl'
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    return model, scaler

# Load model
model, scaler = load_model()

def predict_promotion(features):
    """Real prediction using trained model"""
    
    # Prepare features in correct order
    feature_order = [
        'performance_score',
        'behavior_avg', 
        'tenure_years',
        'gender_encoded',  # Adjust based on your features
        'marital_status_encoded',
        'is_permanent_encoded'
    ]
    
    # Encode categorical variables
    gender_encoded = 1 if features['gender'] == 'M' else 0
    is_permanent_encoded = 1 if features['is_permanent'] == 't' else 0
    
    # Marital status encoding (adjust based on your encoding)
    marital_mapping = {'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3}
    marital_encoded = marital_mapping.get(features['marital_status'], 0)
    
    # Create feature array
    X = np.array([[
        features['performance_score'],
        features['behavior_avg'],
        features['tenure_years'],
        gender_encoded,
        marital_encoded,
        is_permanent_encoded
    ]])
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]  # Probability of class 1
    
    return int(prediction), float(probability)
```

---

## 🔄 Feature Engineering Integration

If your model uses engineered features:

```python
def create_engineered_features(features):
    """Create engineered features matching training"""
    
    # Example: Performance-Behavioral interaction
    perf_beh_interaction = (
        features['performance_score'] * features['behavior_avg']
    )
    
    # Example: Performance category
    if features['performance_score'] >= 80:
        perf_category = 2  # High
    elif features['performance_score'] >= 60:
        perf_category = 1  # Medium
    else:
        perf_category = 0  # Low
    
    # Add to features
    features['perf_beh_interaction'] = perf_beh_interaction
    features['perf_category'] = perf_category
    
    return features

# Use in prediction function
def predict_promotion(features):
    # Add engineered features
    features = create_engineered_features(features)
    
    # Continue with prediction...
```

---

## 📊 Update Model Performance Page

**File**: `pages/3_🤖_Model_Performance.py`

### Load Real Model Results

```python
import json

@st.cache_data
def load_model_results():
    """Load actual model performance results"""
    
    results_path = Path(__file__).parent.parent.parent / 'results'
    
    results = {
        'baseline': {},
        'advanced': {}
    }
    
    # Load from your results files
    # Example: Load from JSON
    with open(results_path / 'baseline_results.json', 'r') as f:
        results['baseline'] = json.load(f)
    
    with open(results_path / 'advanced_results.json', 'r') as f:
        results['advanced'] = json.load(f)
    
    return results
```

### Load Confusion Matrix

```python
@st.cache_data
def load_confusion_matrix(model_name):
    """Load actual confusion matrix for model"""
    
    cm_path = Path(__file__).parent.parent.parent / 'results' / 'advanced_models' / f'{model_name}_confusion_matrix.npy'
    
    if cm_path.exists():
        return np.load(cm_path)
    else:
        # Return mock data if not available
        return np.array([[120, 10], [5, 15]])
```

### Load ROC Curve Data

```python
@st.cache_data
def load_roc_data(model_name):
    """Load actual ROC curve data"""
    
    roc_path = Path(__file__).parent.parent.parent / 'results' / 'advanced_models' / f'{model_name}_roc.npz'
    
    if roc_path.exists():
        data = np.load(roc_path)
        return data['fpr'], data['tpr'], data['auc']
    else:
        # Return mock data
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr) * 0.95
        auc = 0.94
        return fpr, tpr, auc
```

---

## 💾 Saving Model Results

When training your models, save results in this format:

```python
import json
import numpy as np

# After training and evaluation
results = {
    'XGBoost': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    }
}

# Save metrics
with open('results/advanced_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save confusion matrix
np.save('results/advanced_models/XGBoost_confusion_matrix.npy', cm)

# Save ROC data
np.savez('results/advanced_models/XGBoost_roc.npz', 
         fpr=fpr, tpr=tpr, auc=roc_auc)

# Save feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
})
feature_importance.to_csv('results/advanced_models/XGBoost_feature_importance.csv', index=False)
```

---

## 🎯 Feature Importance Integration

Update feature importance section:

```python
@st.cache_data
def load_feature_importance(model_name='XGBoost'):
    """Load actual feature importance"""
    
    fi_path = Path(__file__).parent.parent.parent / 'results' / 'advanced_models' / f'{model_name}_feature_importance.csv'
    
    if fi_path.exists():
        return pd.read_csv(fi_path)
    else:
        # Return mock data
        return pd.DataFrame({
            'Feature': ['behavior_avg', 'performance_score', 'tenure_years'],
            'Importance': [0.35, 0.28, 0.18]
        })

# Use in visualization
feature_importance = load_feature_importance('XGBoost')
```

---

## 🔍 Model Metadata

Create a model info file:

**File**: `models/model_info.json`

```json
{
  "XGBoost": {
    "version": "1.0.0",
    "trained_date": "2025-10-21",
    "training_samples": 1200,
    "test_samples": 300,
    "features": [
      "performance_score",
      "behavior_avg",
      "tenure_years",
      "gender",
      "marital_status",
      "is_permanent"
    ],
    "hyperparameters": {
      "n_estimators": 100,
      "max_depth": 5,
      "learning_rate": 0.1
    },
    "best_threshold": 0.5
  }
}
```

Load and display:

```python
@st.cache_data
def load_model_info():
    """Load model metadata"""
    with open('models/model_info.json', 'r') as f:
        return json.load(f)

# Display in sidebar
model_info = load_model_info()
st.sidebar.markdown(f"""
**Model Info:**
- Version: {model_info['XGBoost']['version']}
- Trained: {model_info['XGBoost']['trained_date']}
- Samples: {model_info['XGBoost']['training_samples']}
""")
```

---

## 🧪 Testing Integration

Create a test script:

**File**: `test_model_integration.py`

```python
"""Test model integration"""

import joblib
import numpy as np
import pandas as pd

def test_model_loading():
    """Test if model loads correctly"""
    try:
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model and scaler loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_prediction():
    """Test if prediction works"""
    try:
        model = joblib.load('models/xgboost_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Test data
        X_test = np.array([[75.0, 80.0, 5.0, 1, 1, 1]])
        X_scaled = scaler.transform(X_test)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        print(f"✅ Prediction: {prediction}")
        print(f"✅ Probability: {probability:.4f}")
        return True
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return False

if __name__ == "__main__":
    print("Testing Model Integration...")
    print("-" * 50)
    
    test_model_loading()
    test_prediction()
    
    print("-" * 50)
    print("Testing complete!")
```

Run test:
```bash
python test_model_integration.py
```

---

## 📦 Complete Integration Checklist

- [ ] Train and save model (`.pkl` or `.joblib`)
- [ ] Save scaler/preprocessor
- [ ] Save model metrics (JSON)
- [ ] Save confusion matrix (`.npy`)
- [ ] Save ROC data (`.npz`)
- [ ] Save feature importance (CSV)
- [ ] Create model info file
- [ ] Update `predict_promotion()` function
- [ ] Update `load_model_results()` function
- [ ] Update feature importance loading
- [ ] Test model loading
- [ ] Test predictions
- [ ] Update documentation

---

## 🚀 Deployment Considerations

### For Streamlit Cloud

1. **Add model files to repo** (if < 100MB):
```bash
git add models/*.pkl
git commit -m "Add trained models"
```

2. **Or use Git LFS** for large files:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
git add models/*.pkl
```

3. **Or download from cloud storage**:
```python
@st.cache_resource
def download_model():
    """Download model from cloud storage"""
    import urllib.request
    
    url = "https://your-storage.com/model.pkl"
    urllib.request.urlretrieve(url, "models/model.pkl")
    
    return joblib.load("models/model.pkl")
```

---

## 💡 Best Practices

1. **Version Control**: Track model versions
2. **Caching**: Use `@st.cache_resource` for models
3. **Error Handling**: Add try-except blocks
4. **Validation**: Validate input features
5. **Logging**: Log predictions for monitoring
6. **Testing**: Test with various inputs
7. **Documentation**: Document feature engineering

---

## 🔒 Security Notes

- ❌ Don't commit large model files to Git
- ✅ Use Git LFS or cloud storage
- ❌ Don't hardcode API keys
- ✅ Use environment variables
- ✅ Validate all user inputs
- ✅ Sanitize file uploads

---

## 📚 Additional Resources

- [Joblib Documentation](https://joblib.readthedocs.io/)
- [Streamlit Caching](https://docs.streamlit.io/library/advanced-features/caching)
- [Model Deployment Best Practices](https://ml-ops.org/)

---

**Ready to integrate your models! 🚀**

*Last Updated: October 22, 2025*
