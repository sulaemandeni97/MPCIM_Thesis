# MPCIM Thesis Project - File Organization

## 📁 Project Structure

```
/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/
├── README.md                          # This file
├── data/                              # Data files (will be copied here)
│   ├── raw/                          # Raw exported data
│   ├── processed/                    # Processed/cleaned data
│   └── final/                        # Final integrated dataset
├── scripts/                          # Python scripts
│   ├── export/                       # Data export scripts
│   ├── analysis/                     # Analysis scripts
│   └── modeling/                     # ML modeling scripts
├── notebooks/                        # Jupyter notebooks (optional)
├── docs/                             # Documentation
│   ├── proposal/                     # Thesis proposal
│   ├── analysis/                     # Analysis reports
│   └── references/                   # Literature references
└── results/                          # Model results, plots, etc.
```

---

## 📂 Current File Locations

### **1. Data Files (CSV)**
**Current Location**: `/tmp/mpcim_export_cna/`

**Files**:
- `integrated_performance_behavioral.csv` - **MAIN DATASET** (712 employees, 2 dimensions)
- `00_integrated_dataset.csv` - All dimensions from database
- `01_employee_master.csv` - Employee demographics
- `02_performance_contract.csv` - Performance assessments
- `03_performance_kpi_items.csv` - Detailed KPI data
- `04_competency_assessment.csv` - Competency scores
- `05_talent_scorecard.csv` - Talent assessments
- `06_promotion_history.csv` - Promotion history (TARGET)
- `employee_nik_mapping.csv` - NIK to hash mapping
- `ref_*.csv` - Reference tables

**Size**: ~8.6 MB total

---

### **2. Scripts**
**Current Location**: `/tmp/`

**Export Scripts**:
- `export_cna_anonymized.sh` - Main export script
- `mpcim_export_queries.sql` - SQL queries for export
- `anonymize_example.sql` - Anonymization examples

**Analysis Scripts**:
- `merge_with_nik.py` - **FINAL MERGE SCRIPT** (Performance + Behavioral)
- `analyze_exported_data.py` - Data analysis
- `deep_analysis_mpcim.py` - Deep analysis

**Documentation**:
- `mpcim_database_analysis.md` - Database schema analysis
- `PANDUAN_EXPORT_DATA.md` - Export guide

---

## 🎯 Main Dataset

**File**: `integrated_performance_behavioral.csv`
**Location**: `/tmp/mpcim_export_cna/integrated_performance_behavioral.csv`

**Contents**:
- 712 employees
- 10 columns:
  1. employee_id_hash (anonymized)
  2. company_id
  3. tenure_years
  4. gender
  5. marital_status
  6. is_permanent
  7. performance_score (Dimension 1)
  8. performance_rating
  9. has_promotion (TARGET)
  10. behavior_avg (Dimension 2)

**Statistics**:
- Performance Score: Mean 81.88, Std 34.94
- Behavioral Score: Mean 89.72, Std 8.71
- Promotion Rate: 9.27% (66 promoted, 646 not promoted)

---

## 🔧 Key Scripts to Use

### **1. Data Export**
```bash
# Export data from database
/tmp/export_cna_anonymized.sh
```

### **2. Data Integration**
```bash
# Merge Performance + Behavioral
python3 /tmp/merge_with_nik.py
```

### **3. Data Analysis**
```bash
# Analyze integrated dataset
python3 /tmp/analyze_exported_data.py
```

---

## 📊 Next Steps

### **Immediate**:
1. Copy files to permanent location
2. Setup virtual environment
3. Install dependencies

### **Short-term**:
1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Baseline Model Development

### **Medium-term**:
1. MPCIM Model Development
2. Model Evaluation
3. SHAP Analysis

---

## 🚀 Quick Start

### **1. Copy Data to Project Folder**
```bash
# Copy main dataset
cp /tmp/mpcim_export_cna/integrated_performance_behavioral.csv \
   /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/

# Copy all raw data
cp -r /tmp/mpcim_export_cna/* \
   /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/raw/
```

### **2. Copy Scripts**
```bash
# Copy analysis scripts
cp /tmp/merge_with_nik.py \
   /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/scripts/analysis/

cp /tmp/analyze_exported_data.py \
   /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/scripts/analysis/
```

### **3. Setup Environment**
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost shap
```

---

## 📝 Important Notes

### **Data Security**:
- ✅ All employee IDs are MD5 hashed (anonymized)
- ✅ No names, emails, or personal identifiers
- ✅ Safe for research purposes

### **Data Quality**:
- ✅ 100% complete for both dimensions
- ✅ Only 14 missing values in performance_rating (2%)
- ✅ Ready for ML modeling

### **Class Imbalance**:
- ⚠️ 9.27% promotion rate (moderate imbalance)
- Solution: Use SMOTE, class weights, or ensemble methods

---

## 🎓 Thesis Information

**Title**: 
"Dual-Dimensional Predictive Analytics untuk Career Progression: Integrating Performance dan Behavioral Assessment dalam Imbalanced Dataset"

**Dimensions**:
1. Performance Assessment (from database)
2. Behavioral Assessment (from Excel)

**Target**: Promotion Prediction (has_promotion)

**Sample Size**: 712 employees (sufficient for ML)

**Timeline**: 6 months

---

## 📧 Contact

For questions about this project, refer to the analysis scripts or documentation in `/tmp/`.

---

**Last Updated**: October 21, 2025
**Status**: Data Integration Complete ✅
**Next**: EDA & Baseline Model Development
