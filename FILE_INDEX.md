# MPCIM Thesis - Complete File Index

## 📍 Project Location
**Main Folder**: `/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/`

---

## 📁 Folder Structure

```
MPCIM_Thesis/
├── README.md                          # Project overview
├── FILE_INDEX.md                      # This file
│
├── data/
│   ├── raw/                          # Raw exported data from database
│   │   ├── 00_integrated_dataset.csv              # All dimensions (15,184 records)
│   │   ├── 01_employee_master.csv                 # Employee demographics (3,234)
│   │   ├── 02_performance_contract.csv            # Performance data (13,478)
│   │   ├── 03_performance_kpi_items.csv           # KPI details (127,579)
│   │   ├── 04_competency_assessment.csv           # Competency (15)
│   │   ├── 05_talent_scorecard.csv                # Talent (47)
│   │   ├── 06_promotion_history.csv               # Promotions (130) - TARGET
│   │   ├── employee_nik_mapping.csv               # NIK to hash mapping
│   │   └── ref_*.csv                              # Reference tables
│   │
│   ├── processed/                    # Processed/cleaned data (future)
│   │
│   └── final/                        # Final integrated dataset
│       └── integrated_performance_behavioral.csv  # **MAIN DATASET** (712 employees)
│
├── scripts/
│   ├── export/                       # Database export scripts
│   │   ├── export_cna_anonymized.sh               # Main export script
│   │   ├── mpcim_export_queries.sql               # SQL queries
│   │   └── anonymize_example.sql                  # Anonymization examples
│   │
│   ├── analysis/                     # Data analysis scripts
│   │   ├── merge_with_nik.py                      # **MAIN MERGE SCRIPT**
│   │   ├── analyze_exported_data.py               # Data analysis
│   │   └── deep_analysis_mpcim.py                 # Deep analysis
│   │
│   └── modeling/                     # ML modeling scripts (future)
│
├── docs/
│   ├── proposal/                     # Thesis proposal (future)
│   ├── analysis/                     # Analysis documentation
│   │   ├── mpcim_database_analysis.md             # Database schema analysis
│   │   └── PANDUAN_EXPORT_DATA.md                 # Export guide
│   └── references/                   # Literature references (future)
│
├── results/                          # Model results, plots (future)
│
└── notebooks/                        # Jupyter notebooks (future)
```

---

## 🎯 Key Files

### **1. Main Dataset** ⭐
**File**: `data/final/integrated_performance_behavioral.csv`
**Size**: 57 KB
**Records**: 712 employees
**Columns**: 10
- employee_id_hash (anonymized)
- company_id
- tenure_years
- gender
- marital_status
- is_permanent
- performance_score (Dimension 1)
- performance_rating
- has_promotion (TARGET - 9.27% positive)
- behavior_avg (Dimension 2)

**Status**: ✅ Ready for ML modeling

---

### **2. Main Scripts**

#### **Data Integration**
**File**: `scripts/analysis/merge_with_nik.py`
**Purpose**: Merge Performance (database) + Behavioral (Excel) data
**Output**: integrated_performance_behavioral.csv
**Usage**:
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis
python3 scripts/analysis/merge_with_nik.py
```

#### **Data Analysis**
**File**: `scripts/analysis/analyze_exported_data.py`
**Purpose**: Comprehensive data analysis
**Features**:
- Descriptive statistics
- Correlation analysis
- Promoted vs Not Promoted comparison
- Statistical tests

#### **Database Export**
**File**: `scripts/export/export_cna_anonymized.sh`
**Purpose**: Export data from PostgreSQL database
**Database**: db_cna_digispace_august_132025 (port 5433)

---

### **3. Documentation**

#### **Database Analysis**
**File**: `docs/analysis/mpcim_database_analysis.md`
**Contents**:
- Database schema overview (195 tables)
- Relevant tables for MPCIM (60 tables)
- Data dimensions available
- Recommended approach

#### **Export Guide**
**File**: `docs/analysis/PANDUAN_EXPORT_DATA.md`
**Contents**:
- Step-by-step export instructions
- Anonymization techniques
- Troubleshooting guide

---

## 📊 Data Summary

### **Raw Data (from Database)**
| File | Records | Description |
|------|---------|-------------|
| 00_integrated_dataset.csv | 15,184 | All dimensions combined |
| 01_employee_master.csv | 3,234 | Employee demographics |
| 02_performance_contract.csv | 13,478 | Performance assessments |
| 03_performance_kpi_items.csv | 127,579 | Detailed KPI data |
| 06_promotion_history.csv | 130 | Promotion history (TARGET) |

### **Final Integrated Dataset**
| Metric | Value |
|--------|-------|
| Total Employees | 712 |
| With Performance | 712 (100%) |
| With Behavioral | 712 (100%) |
| Promoted | 66 (9.27%) |
| Not Promoted | 646 (90.73%) |

### **Data Quality**
- ✅ 100% complete for both dimensions
- ✅ Only 14 missing values (2%)
- ✅ All employee IDs anonymized (MD5 hash)
- ✅ No personal identifiers (names, emails, etc.)

---

## 🔧 How to Use

### **1. Navigate to Project**
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis
```

### **2. View Main Dataset**
```bash
head data/final/integrated_performance_behavioral.csv
```

### **3. Run Analysis**
```bash
# Setup virtual environment (first time only)
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scipy matplotlib seaborn

# Run analysis
python3 scripts/analysis/merge_with_nik.py
```

### **4. Open in IDE**
```bash
# Open entire project in VS Code
code /Users/denisulaeman/CascadeProjects/MPCIM_Thesis
```

---

## 📈 Next Steps

### **Immediate**:
- [x] Data export from database
- [x] Data integration (Performance + Behavioral)
- [x] Data quality check
- [ ] Setup virtual environment
- [ ] Install ML dependencies

### **Short-term**:
- [ ] Exploratory Data Analysis (EDA)
- [ ] Feature engineering
- [ ] Baseline model development
- [ ] Handle class imbalance

### **Medium-term**:
- [ ] MPCIM model development
- [ ] Model evaluation & comparison
- [ ] SHAP analysis for explainability
- [ ] Prototype dashboard

---

## 🎓 Thesis Information

**Title**: 
"Dual-Dimensional Predictive Analytics untuk Career Progression: Integrating Performance dan Behavioral Assessment dalam Imbalanced Dataset"

**Key Findings (from preliminary analysis)**:
- ✅ Behavioral score is **significant** (p=0.037) for promotion
- ⚠️ Performance score is **not significant** (p=0.083) alone
- 📊 This supports the need for multi-dimensional approach!

**Research Questions**:
1. Is dual-dimensional (Performance + Behavioral) more accurate than single-dimension?
2. Which dimension is more influential for promotion?
3. How to handle class imbalance effectively?
4. How to provide explainable recommendations?

---

## 🔒 Data Security

**Anonymization Applied**:
- ✅ Employee IDs → MD5 hash (irreversible)
- ✅ No names, emails, phone numbers
- ✅ No addresses, NIK/KTP
- ✅ No salary or financial data
- ✅ Only assessment scores and metadata

**Safe for**:
- ✅ Research purposes
- ✅ Thesis publication
- ✅ Academic presentations

---

## 📞 Quick Reference

**Database Connection**:
- Host: localhost
- Port: 5433
- Database: db_cna_digispace_august_132025
- User: denisulaeman

**Excel Source**:
- File: /Users/denisulaeman/Downloads/MPCIM Dataset.xlsx
- Sheet: "Behavior atau Prilaku"
- Records: 19,929 (766 unique employees)

**Python Environment**:
- Virtual env: /tmp/mpcim_venv (temporary)
- Recommended: Create new venv in project folder

---

**Last Updated**: October 21, 2025, 5:05 PM
**Status**: ✅ Data Integration Complete
**Next**: EDA & Baseline Model Development
