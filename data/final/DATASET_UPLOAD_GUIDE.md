# 📊 Dataset Upload Guide

## 📁 Available Sample Datasets

### 1. **sample_dataset_100.csv** (100 rows)
- **Purpose**: Quick testing and demo
- **Size**: 100 employee records
- **QA Coverage**: 100%
- **Promotion Rate**: 10%
- **Use Case**: Testing upload functionality, quick analysis

### 2. **integrated_full_dataset.csv** (712 rows)
- **Purpose**: Full analysis and production
- **Size**: 712 employee records
- **QA Coverage**: 99.7%
- **Promotion Rate**: 9.3%
- **Use Case**: Complete analysis, model training

### 3. **UPLOAD_TEMPLATE.csv** (Template)
- **Purpose**: Template for creating new datasets
- **Size**: 3 example rows
- **Use Case**: Reference for column structure

---

## 📋 Required Columns (19 columns)

### 1. **Basic Information** (6 columns)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `employee_id_hash` | string | Unique employee identifier (hashed) | `00003e3b9e5336685200ae85d21b4f5e` |
| `company_id` | integer | Company/department ID | `82`, `101`, `72` |
| `tenure_years` | integer | Years of service | `1`, `5`, `10` |
| `gender` | string | Gender (O/L) | `O` (Other), `L` (Laki-laki) |
| `marital_status` | string | Marital status | `married`, `single` |
| `is_permanent` | boolean | Permanent employee status | `t` (true), `f` (false) |

### 2. **Performance Metrics** (3 columns)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `performance_score` | float | Performance score (0-100) | `85.5`, `92.1` |
| `performance_rating` | string | Performance rating | `Excellent`, `Good`, `Average` |
| `has_promotion` | integer | Promotion status (0/1) | `0` (No), `1` (Yes) |

### 3. **Behavioral Metrics** (1 column)
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `behavior_avg` | float | Average behavioral score (0-100) | `88.3`, `91.5` |

### 4. **Quick Assessment (QA) Features** (8 columns)
| Column | Type | Description | Range | Example |
|--------|------|-------------|-------|---------|
| `psychological_score` | float | Overall psychological score | 0-100 | `23.5` |
| `drive_score` | float | Drive & ambition score | 0-100 | `75.2` |
| `mental_strength_score` | float | Mental strength score | 0-100 | `78.5` |
| `adaptability_score` | float | Adaptability score | 0-100 | `76.8` |
| `collaboration_score` | float | Collaboration score | 0-100 | `82.1` |
| `has_quick_assessment` | integer | QA data availability (0/1) | 0 or 1 | `1` |
| `holistic_score` | float | Holistic score (calculated) | 0-100 | `65.2` |
| `score_alignment` | float | Score alignment metric | 0-1 | `0.65` |
| `leadership_potential` | float | Leadership potential score | 0-100 | `58.3` |

---

## 🔢 Data Formats & Rules

### Data Types:
```
employee_id_hash:          string (32 characters, hexadecimal)
company_id:                integer (1-999)
tenure_years:              integer (0-50)
gender:                    string ('O' or 'L')
marital_status:            string ('married' or 'single')
is_permanent:              boolean ('t' or 'f')
performance_score:         float (0.0-100.0)
performance_rating:        string ('Excellent', 'Good', 'Average', 'Poor')
has_promotion:             integer (0 or 1)
behavior_avg:              float (0.0-100.0)
psychological_score:       float (0.0-100.0)
drive_score:               float (0.0-100.0)
mental_strength_score:     float (0.0-100.0)
adaptability_score:        float (0.0-100.0)
collaboration_score:       float (0.0-100.0)
has_quick_assessment:      integer (0 or 1)
holistic_score:            float (0.0-100.0)
score_alignment:           float (0.0-1.0)
leadership_potential:      float (0.0-100.0)
```

### Validation Rules:
1. **No missing values** in required columns
2. **employee_id_hash** must be unique
3. **Scores** must be within valid ranges
4. **Boolean fields** must be 't' or 'f' (or 0/1)
5. **has_promotion** must be 0 or 1
6. **has_quick_assessment** must be 0 or 1

---

## 📝 Calculated Fields

### 1. **holistic_score**
```
Formula: (0.4 × performance_score) + (0.3 × behavior_avg) + (0.3 × psychological_score)
Example: (0.4 × 85.5) + (0.3 × 88.3) + (0.3 × 23.5) = 67.74
```

### 2. **score_alignment**
```
Formula: 1 - (|performance_score - behavior_avg| / 100)
Example: 1 - (|85.5 - 88.3| / 100) = 0.972
```

### 3. **leadership_potential**
```
Formula: (0.3 × drive_score) + (0.25 × mental_strength_score) + 
         (0.25 × adaptability_score) + (0.2 × collaboration_score)
Example: (0.3 × 75.2) + (0.25 × 78.5) + (0.25 × 76.8) + (0.2 × 82.1) = 77.735
```

**Note**: These can be calculated automatically if you provide the base scores.

---

## 🚀 How to Upload Dataset

### Step 1: Prepare Your CSV File
1. Use **UPLOAD_TEMPLATE.csv** as reference
2. Ensure all 19 columns are present
3. Follow data format rules
4. Validate data ranges

### Step 2: Upload in Streamlit App
1. Open **Data Explorer** page
2. Click **"Browse files"** in sidebar
3. Select your CSV file
4. Wait for validation

### Step 3: Validation
The app will automatically check:
- ✅ All required columns present
- ✅ Data types correct
- ✅ No missing values
- ✅ Scores within valid ranges
- ✅ Unique employee IDs

### Step 4: Use Dataset
Once validated:
- ✅ Apply filters (including QA filters)
- ✅ View metrics (including QA metrics)
- ✅ Explore data
- ✅ Use for predictions

---

## 📊 Sample Data Examples

### Example 1: Employee with QA Data
```csv
employee_id_hash,company_id,tenure_years,gender,marital_status,is_permanent,performance_score,performance_rating,has_promotion,behavior_avg,psychological_score,drive_score,mental_strength_score,adaptability_score,collaboration_score,has_quick_assessment,holistic_score,score_alignment,leadership_potential
emp001,101,5,O,married,t,85.5,Good,1,88.3,23.5,75.2,78.5,76.8,82.1,1,65.2,0.65,58.3
```

### Example 2: Employee without QA Data
```csv
employee_id_hash,company_id,tenure_years,gender,marital_status,is_permanent,performance_score,performance_rating,has_promotion,behavior_avg,psychological_score,drive_score,mental_strength_score,adaptability_score,collaboration_score,has_quick_assessment,holistic_score,score_alignment,leadership_potential
emp002,82,3,O,single,t,78.2,Average,0,82.1,0,0,0,0,0,0,61.5,0.58,0
```

**Note**: For employees without QA data, set:
- `has_quick_assessment = 0`
- All QA scores = 0
- `holistic_score` will be calculated from performance + behavioral only
- `leadership_potential = 0`

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Missing columns"
**Solution**: Ensure all 19 columns are present in exact order

### Issue 2: "Invalid data type"
**Solution**: Check data types match the specification

### Issue 3: "Score out of range"
**Solution**: Ensure all scores are within valid ranges (0-100 or 0-1)

### Issue 4: "Duplicate employee_id_hash"
**Solution**: Each employee must have unique ID

### Issue 5: "Missing values"
**Solution**: Fill all required fields (use 0 for missing QA data)

---

## 🔧 Data Preparation Script

If you need to prepare data programmatically:

```python
import pandas as pd
import numpy as np

# Load your raw data
df = pd.read_csv('your_data.csv')

# Add QA columns if missing
qa_columns = ['psychological_score', 'drive_score', 'mental_strength_score', 
              'adaptability_score', 'collaboration_score', 'has_quick_assessment']

for col in qa_columns:
    if col not in df.columns:
        df[col] = 0  # Default to 0 if no QA data

# Calculate holistic_score
df['holistic_score'] = (
    0.4 * df['performance_score'] + 
    0.3 * df['behavior_avg'] + 
    0.3 * df['psychological_score']
)

# Calculate score_alignment
df['score_alignment'] = 1 - (
    abs(df['performance_score'] - df['behavior_avg']) / 100
)

# Calculate leadership_potential
df['leadership_potential'] = (
    0.3 * df['drive_score'] + 
    0.25 * df['mental_strength_score'] + 
    0.25 * df['adaptability_score'] + 
    0.2 * df['collaboration_score']
)

# Save prepared data
df.to_csv('prepared_data.csv', index=False)
```

---

## 📈 Dataset Statistics

### sample_dataset_100.csv:
- **Rows**: 100
- **QA Coverage**: 100%
- **Promotion Rate**: 10%
- **Avg Performance**: 82.5
- **Avg Behavioral**: 85.3
- **Avg Psychological**: 23.8
- **Avg Leadership**: 57.2

### integrated_full_dataset.csv:
- **Rows**: 712
- **QA Coverage**: 99.7%
- **Promotion Rate**: 9.3%
- **Avg Performance**: 82.3
- **Avg Behavioral**: 85.1
- **Avg Psychological**: 23.5
- **Avg Leadership**: 57.1

---

## 🎯 Best Practices

1. **Always validate** your CSV before uploading
2. **Use template** as reference for structure
3. **Test with sample** dataset first
4. **Backup** your data before modifications
5. **Document** any custom calculations
6. **Check ranges** for all scores
7. **Ensure uniqueness** of employee IDs

---

## 📞 Support

If you encounter issues:
1. Check this guide first
2. Validate your CSV format
3. Test with sample_dataset_100.csv
4. Review error messages in app
5. Check data types and ranges

---

**Last Updated**: November 17, 2025  
**Version**: 1.0  
**Compatible with**: MPCIM Thesis App v2.0 (with QA Integration)
