# 📑 MPCIM Dashboard - File Index

**Quick Reference untuk semua file dalam aplikasi**

---

## 🎯 Start Here

| File | Purpose | When to Use |
|------|---------|-------------|
| **QUICK_START.md** | Panduan cepat memulai | Pertama kali menjalankan app |
| **APP_SUMMARY.md** | Ringkasan lengkap aplikasi | Untuk overview fitur |
| **README.md** | Dokumentasi utama | Untuk instalasi & setup |
| **APP_DOCUMENTATION.md** | Dokumentasi teknis lengkap | Untuk customization & development |

---

## 🚀 Application Files

### Main Application
| File | Description | Lines |
|------|-------------|-------|
| `Home.py` | Landing page, overview penelitian | ~200 |

### Pages (Multi-page App)
| File | Description | Lines |
|------|-------------|-------|
| `pages/1_📊_Data_Explorer.py` | Data exploration & filtering | ~400 |
| `pages/2_📈_EDA_Results.py` | Statistical analysis results | ~450 |
| `pages/3_🤖_Model_Performance.py` | ML model comparison | ~500 |
| `pages/4_🔮_Prediction.py` | Prediction tool | ~550 |

**Total Application Code**: ~2,100 lines

---

## ⚙️ Configuration Files

| File | Purpose | Edit? |
|------|---------|-------|
| `.streamlit/config.toml` | App configuration (theme, port, etc.) | ✅ Yes |
| `.streamlit/secrets.toml.example` | Template for secrets | ✅ Copy & edit |
| `requirements.txt` | Python dependencies | ⚠️ Careful |

---

## 📚 Documentation Files

| File | Content | Pages |
|------|---------|-------|
| `README.md` | Main documentation, installation, features | 5 |
| `QUICK_START.md` | Quick start guide, troubleshooting | 3 |
| `APP_DOCUMENTATION.md` | Complete technical documentation | 12 |
| `APP_SUMMARY.md` | Application summary & statistics | 6 |
| `INDEX.md` | This file - file index | 2 |

**Total Documentation**: ~28 pages

---

## 🛠️ Utility Files

| File | Purpose | Usage |
|------|---------|-------|
| `run_app.sh` | Startup script | `./run_app.sh` |

---

## 📊 File Structure Tree

```
app/
│
├── 📄 Home.py                          # Main entry point
│
├── 📁 pages/                           # Multi-page structure
│   ├── 1_📊_Data_Explorer.py
│   ├── 2_📈_EDA_Results.py
│   ├── 3_🤖_Model_Performance.py
│   └── 4_🔮_Prediction.py
│
├── 📁 .streamlit/                      # Configuration
│   ├── config.toml
│   └── secrets.toml.example
│
├── 📚 Documentation/
│   ├── README.md
│   ├── QUICK_START.md
│   ├── APP_DOCUMENTATION.md
│   ├── APP_SUMMARY.md
│   └── INDEX.md                        # This file
│
├── ⚙️ Configuration/
│   ├── requirements.txt
│   └── run_app.sh
│
└── 📊 Data/ (external)
    └── /Users/.../data/final/integrated_performance_behavioral.csv
```

---

## 🎯 Quick Navigation Guide

### I want to...

**...run the app for the first time**
→ Read `QUICK_START.md`

**...understand what the app does**
→ Read `APP_SUMMARY.md`

**...install and configure**
→ Read `README.md`

**...customize or develop**
→ Read `APP_DOCUMENTATION.md`

**...find a specific file**
→ Read `INDEX.md` (this file)

**...change theme/colors**
→ Edit `.streamlit/config.toml`

**...add new features**
→ Edit files in `pages/` directory

**...fix errors**
→ Check `QUICK_START.md` → Troubleshooting section

---

## 📝 File Descriptions

### Home.py
**Purpose**: Main landing page  
**Features**:
- Research overview
- Quick statistics
- Navigation guide
- Feature highlights

**Key Functions**:
- `load_data()` - Load dataset with caching
- Page layout and styling
- Metrics display

---

### 1_📊_Data_Explorer.py
**Purpose**: Interactive data exploration  
**Features**:
- Data table with search
- Multi-criteria filtering
- Descriptive statistics
- Visualizations (histograms, scatter, correlation)
- CSV export

**Key Functions**:
- `load_data()` - Load and cache data
- Filter logic
- Plotly visualizations

---

### 2_📈_EDA_Results.py
**Purpose**: Statistical analysis results  
**Features**:
- T-tests with p-values
- Cohen's d effect size
- Correlation analysis
- Distribution comparisons
- 3D visualizations

**Key Functions**:
- Statistical tests (scipy.stats)
- Plotly charts (box, violin, 3D scatter)
- Correlation heatmap

---

### 3_🤖_Model_Performance.py
**Purpose**: ML model comparison  
**Features**:
- 4 model comparison
- Performance metrics table
- ROC curves
- Confusion matrix
- Feature importance

**Key Functions**:
- `load_model_results()` - Load model metrics
- Radar charts
- Styled dataframes

---

### 4_🔮_Prediction.py
**Purpose**: Promotion prediction tool  
**Features**:
- Individual prediction form
- Batch prediction (CSV upload)
- Probability gauge
- Feature contribution
- Recommendations

**Key Functions**:
- `predict_promotion()` - Make predictions
- Gauge visualization
- Benchmark comparison

---

## 🔍 Code Statistics

| Metric | Value |
|--------|-------|
| Total Files | 13 |
| Python Files | 5 |
| Config Files | 2 |
| Documentation Files | 5 |
| Script Files | 1 |
| Total Lines of Code | ~2,500 |
| Total Documentation | ~28 pages |

---

## 📦 Dependencies (requirements.txt)

```
streamlit==1.29.0      # Web framework
pandas==2.1.4          # Data processing
numpy==1.26.2          # Numerical computing
plotly==5.18.0         # Visualizations
scipy==1.11.4          # Statistics
scikit-learn==1.3.2    # ML utilities
xgboost==2.0.3         # Gradient boosting
```

---

## 🎨 Customization Quick Reference

### Change Colors
**File**: Any page file  
**Line**: `color_discrete_map = {...}`

### Change Theme
**File**: `.streamlit/config.toml`  
**Section**: `[theme]`

### Change Port
**File**: `.streamlit/config.toml`  
**Line**: `port = 8501`

### Add New Page
**Location**: `pages/`  
**Naming**: `X_emoji_PageName.py` (X = number)

### Modify Data Path
**Files**: All page files  
**Function**: `load_data()`

---

## 🚀 Deployment Files

### For Streamlit Cloud
- `requirements.txt` ✅
- `Home.py` ✅
- `pages/` directory ✅

### For Heroku
- `requirements.txt` ✅
- `Procfile` (create)
- `setup.sh` (create)

### For Docker
- `requirements.txt` ✅
- `Dockerfile` (create)

---

## 📞 Getting Help

| Issue | Check File |
|-------|-----------|
| Installation problems | `README.md` |
| Running the app | `QUICK_START.md` |
| Error messages | `QUICK_START.md` → Troubleshooting |
| Customization | `APP_DOCUMENTATION.md` |
| Feature questions | `APP_SUMMARY.md` |
| File locations | `INDEX.md` (this file) |

---

## ✅ Checklist

Before running the app, ensure:
- [ ] All files present (13 files)
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Data file available
- [ ] Port 8501 available

---

## 🎯 Quick Commands

```bash
# Navigate to app
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/app

# Install dependencies
pip3 install -r requirements.txt

# Run app
streamlit run Home.py

# Run on different port
streamlit run Home.py --server.port 8080

# Run with network access
streamlit run Home.py --server.address 0.0.0.0
```

---

**Last Updated**: October 22, 2025  
**Version**: 1.0.0
