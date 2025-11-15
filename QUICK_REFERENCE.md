# MPCIM Thesis - Quick Reference Card

## 📍 Lokasi Semua File

### **Folder Utama**
```
/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/
```

### **Dataset Utama** ⭐
```
/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/integrated_performance_behavioral.csv
```

### **File Penting**
| File | Lokasi | Deskripsi |
|------|--------|-----------|
| **README.md** | Root folder | Overview project |
| **FILE_INDEX.md** | Root folder | Index lengkap semua file |
| **QUICK_REFERENCE.md** | Root folder | Quick reference (file ini) |
| **Main Dataset** | data/final/ | Dataset utama (712 employees) |
| **Raw Data** | data/raw/ | Data mentah dari database (14 files) |
| **Merge Script** | scripts/analysis/merge_with_nik.py | Script merge data |
| **Analysis Script** | scripts/analysis/analyze_exported_data.py | Script analisis |
| **Export Script** | scripts/export/export_cna_anonymized.sh | Script export database |

---

## 🚀 Command Cepat

### **1. Navigate ke Project**
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis
```

### **2. Lihat Dataset**
```bash
# Lihat 10 baris pertama
head -10 data/final/integrated_performance_behavioral.csv

# Hitung jumlah baris
wc -l data/final/integrated_performance_behavioral.csv

# Lihat struktur
head -1 data/final/integrated_performance_behavioral.csv
```

### **3. Setup Environment**
```bash
# Buat virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install pandas numpy scipy matplotlib seaborn scikit-learn xgboost shap jupyter
```

### **4. Run Analysis**
```bash
# Activate venv first
source venv/bin/activate

# Run merge script (if needed)
python3 scripts/analysis/merge_with_nik.py

# Run analysis
python3 scripts/analysis/analyze_exported_data.py
```

### **5. Open in IDE**
```bash
# VS Code
code /Users/denisulaeman/CascadeProjects/MPCIM_Thesis

# PyCharm
open -a "PyCharm" /Users/denisulaeman/CascadeProjects/MPCIM_Thesis
```

---

## 📊 Dataset Info

### **Main Dataset**: `integrated_performance_behavioral.csv`

**Columns (10)**:
1. `employee_id_hash` - Anonymized employee ID (MD5)
2. `company_id` - Company identifier
3. `tenure_years` - Years of service
4. `gender` - M/F/O
5. `marital_status` - Marital status
6. `is_permanent` - Employment type
7. `performance_score` - **Dimension 1** (mean: 81.88)
8. `performance_rating` - Good/Excellent/Average/Need Improvement
9. `has_promotion` - **TARGET** (0/1, 9.27% positive)
10. `behavior_avg` - **Dimension 2** (mean: 89.72)

**Statistics**:
- Total: 712 employees
- Promoted: 66 (9.27%)
- Not Promoted: 646 (90.73%)
- Complete data: 100% for both dimensions

---

## 🔧 Troubleshooting

### **Problem: Virtual environment not found**
```bash
# Create new venv
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis
python3 -m venv venv
source venv/bin/activate
```

### **Problem: Module not found**
```bash
# Install required packages
pip install pandas numpy scipy matplotlib seaborn scikit-learn
```

### **Problem: File not found**
```bash
# Check if you're in the right directory
pwd
# Should show: /Users/denisulaeman/CascadeProjects/MPCIM_Thesis

# List files
ls -la data/final/
```

### **Problem: Permission denied**
```bash
# Make script executable
chmod +x scripts/export/export_cna_anonymized.sh
```

---

## 📝 Common Tasks

### **Task 1: View Data Summary**
```python
import pandas as pd

df = pd.read_csv('data/final/integrated_performance_behavioral.csv')
print(df.info())
print(df.describe())
print(df.head())
```

### **Task 2: Check Missing Values**
```python
import pandas as pd

df = pd.read_csv('data/final/integrated_performance_behavioral.csv')
print(df.isnull().sum())
```

### **Task 3: Check Class Distribution**
```python
import pandas as pd

df = pd.read_csv('data/final/integrated_performance_behavioral.csv')
print(df['has_promotion'].value_counts())
print(f"Promotion rate: {df['has_promotion'].mean()*100:.2f}%")
```

### **Task 4: Export to Excel**
```python
import pandas as pd

df = pd.read_csv('data/final/integrated_performance_behavioral.csv')
df.to_excel('data/final/integrated_dataset.xlsx', index=False)
```

---

## 🎯 Next Steps Checklist

### **Phase 1: Setup** ✅
- [x] Export data from database
- [x] Merge Performance + Behavioral
- [x] Organize files in project folder
- [ ] Setup virtual environment
- [ ] Install dependencies

### **Phase 2: EDA** (Next)
- [ ] Load and explore data
- [ ] Visualize distributions
- [ ] Check correlations
- [ ] Identify outliers
- [ ] Feature engineering

### **Phase 3: Modeling**
- [ ] Train/test split
- [ ] Baseline models
- [ ] MPCIM model
- [ ] Handle class imbalance
- [ ] Model evaluation

### **Phase 4: Analysis**
- [ ] SHAP analysis
- [ ] Feature importance
- [ ] Model comparison
- [ ] Generate insights

### **Phase 5: Documentation**
- [ ] Write thesis proposal
- [ ] Document methodology
- [ ] Create visualizations
- [ ] Write results

---

## 📞 Database Connection (if needed)

**Database**: db_cna_digispace_august_132025
**Host**: localhost
**Port**: 5433
**User**: denisulaeman

**Connect**:
```bash
psql -U denisulaeman -p 5433 -d db_cna_digispace_august_132025
```

**Export data** (if needed):
```bash
scripts/export/export_cna_anonymized.sh
```

---

## 💡 Tips

1. **Always activate venv** before running Python scripts
2. **Use relative paths** in scripts (from project root)
3. **Save results** in `results/` folder
4. **Document changes** in notebooks or markdown files
5. **Backup data** regularly (especially final dataset)

---

## 🆘 Need Help?

1. Check **README.md** for project overview
2. Check **FILE_INDEX.md** for complete file list
3. Check **docs/analysis/** for detailed documentation
4. Review scripts in **scripts/analysis/** for examples

---

**Last Updated**: October 21, 2025
**Status**: ✅ Ready for Development
**Current Phase**: Setup Complete → Start EDA
