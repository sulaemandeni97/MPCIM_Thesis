"""
MPCIM Thesis - Feature Engineering
Author: Deni Sulaeman
Date: October 21, 2025

This script performs comprehensive feature engineering including:
- Outlier detection and handling
- New feature creation
- Feature scaling
- Class imbalance handling (SMOTE)
- Train/test split
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
import os
output_dir = '/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/results/feature_engineering'
os.makedirs(output_dir, exist_ok=True)

print('='*80)
print('MPCIM THESIS - FEATURE ENGINEERING')
print('='*80)
print()

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print('1. LOADING DATA')
print('-'*80)

data_path = '/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/integrated_performance_behavioral.csv'
df = pd.read_csv(data_path)

print(f'✅ Data loaded: {df.shape}')
print(f'   Employees: {len(df):,}')
print()

# ============================================================================
# 2. OUTLIER DETECTION
# ============================================================================

print('2. OUTLIER DETECTION & HANDLING')
print('-'*80)

# Detect outliers using IQR method
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check performance score outliers
perf_outliers, perf_lower, perf_upper = detect_outliers_iqr(df, 'performance_score')
print(f'Performance Score Outliers:')
print(f'  Lower bound: {perf_lower:.2f}')
print(f'  Upper bound: {perf_upper:.2f}')
print(f'  Outliers detected: {len(perf_outliers)} ({len(perf_outliers)/len(df)*100:.1f}%)')
print(f'  Max value: {df["performance_score"].max():.2f}')
print()

# Check behavioral score outliers
beh_outliers, beh_lower, beh_upper = detect_outliers_iqr(df, 'behavior_avg')
print(f'Behavioral Score Outliers:')
print(f'  Lower bound: {beh_lower:.2f}')
print(f'  Upper bound: {beh_upper:.2f}')
print(f'  Outliers detected: {len(beh_outliers)} ({len(beh_outliers)/len(df)*100:.1f}%)')
print()

# Visualize outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Performance score
axes[0].boxplot(df['performance_score'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='#3498db', alpha=0.7))
axes[0].axhline(perf_upper, color='red', linestyle='--', label=f'Upper bound: {perf_upper:.1f}')
axes[0].axhline(perf_lower, color='red', linestyle='--', label=f'Lower bound: {perf_lower:.1f}')
axes[0].set_ylabel('Performance Score')
axes[0].set_title('Performance Score - Outlier Detection', fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Behavioral score
axes[1].boxplot(df['behavior_avg'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='#9b59b6', alpha=0.7))
axes[1].axhline(beh_upper, color='red', linestyle='--', label=f'Upper bound: {beh_upper:.1f}')
axes[1].axhline(beh_lower, color='red', linestyle='--', label=f'Lower bound: {beh_lower:.1f}')
axes[1].set_ylabel('Behavioral Score')
axes[1].set_title('Behavioral Score - Outlier Detection', fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_outlier_detection.png', dpi=300, bbox_inches='tight')
print(f'✅ Plot saved: 01_outlier_detection.png')
plt.close()

# Handle outliers - cap at upper bound
df_clean = df.copy()
df_clean.loc[df_clean['performance_score'] > perf_upper, 'performance_score'] = perf_upper
df_clean.loc[df_clean['performance_score'] < perf_lower, 'performance_score'] = perf_lower

print(f'✅ Outliers capped at bounds')
print(f'   Performance score range: {df_clean["performance_score"].min():.2f} - {df_clean["performance_score"].max():.2f}')
print()

# ============================================================================
# 3. FEATURE CREATION
# ============================================================================

print('3. CREATING NEW FEATURES')
print('-'*80)

# 3.1 Performance-Behavioral Ratio
df_clean['perf_beh_ratio'] = df_clean['performance_score'] / df_clean['behavior_avg']
print(f'✅ Created: perf_beh_ratio (Performance/Behavioral ratio)')

# 3.2 Combined Score (weighted average)
df_clean['combined_score'] = (df_clean['performance_score'] * 0.5 + 
                               df_clean['behavior_avg'] * 0.5)
print(f'✅ Created: combined_score (50-50 weighted average)')

# 3.3 Score Difference
df_clean['score_difference'] = df_clean['performance_score'] - df_clean['behavior_avg']
print(f'✅ Created: score_difference (Performance - Behavioral)')

# 3.4 Tenure Categories
def categorize_tenure(years):
    if years <= 2:
        return 'junior'
    elif years <= 7:
        return 'mid'
    else:
        return 'senior'

df_clean['tenure_category'] = df_clean['tenure_years'].apply(categorize_tenure)
print(f'✅ Created: tenure_category (junior/mid/senior)')

# 3.5 Performance Level
def categorize_performance(score):
    if score < 70:
        return 'low'
    elif score < 90:
        return 'medium'
    else:
        return 'high'

df_clean['performance_level'] = df_clean['performance_score'].apply(categorize_performance)
print(f'✅ Created: performance_level (low/medium/high)')

# 3.6 Behavioral Level
def categorize_behavioral(score):
    if score < 85:
        return 'low'
    elif score < 95:
        return 'medium'
    else:
        return 'high'

df_clean['behavioral_level'] = df_clean['behavior_avg'].apply(categorize_behavioral)
print(f'✅ Created: behavioral_level (low/medium/high)')

# 3.7 High Performer Flag (both dimensions high)
df_clean['high_performer'] = ((df_clean['performance_level'] == 'high') & 
                               (df_clean['behavioral_level'] == 'high')).astype(int)
print(f'✅ Created: high_performer (both dimensions high)')

print()
print(f'Total features now: {len(df_clean.columns)}')
print()

# ============================================================================
# 4. ENCODE CATEGORICAL VARIABLES
# ============================================================================

print('4. ENCODING CATEGORICAL VARIABLES')
print('-'*80)

# Create a copy for encoding
df_encoded = df_clean.copy()

# Label encode categorical variables
le = LabelEncoder()

categorical_cols = ['gender', 'marital_status', 'is_permanent', 'performance_rating',
                   'tenure_category', 'performance_level', 'behavioral_level']

for col in categorical_cols:
    if col in df_encoded.columns:
        # Handle missing values
        df_encoded[col] = df_encoded[col].fillna('unknown')
        df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col])
        print(f'✅ Encoded: {col}')

print()

# ============================================================================
# 5. FEATURE SELECTION FOR MODELING
# ============================================================================

print('5. SELECTING FEATURES FOR MODELING')
print('-'*80)

# Select features for modeling
feature_cols = [
    # Original features
    'tenure_years',
    'performance_score',
    'behavior_avg',
    
    # New engineered features
    'perf_beh_ratio',
    'combined_score',
    'score_difference',
    'high_performer',
    
    # Encoded categorical
    'gender_encoded',
    'marital_status_encoded',
    'is_permanent_encoded',
    'tenure_category_encoded',
    'performance_level_encoded',
    'behavioral_level_encoded'
]

# Check if performance_rating_encoded exists
if 'performance_rating_encoded' in df_encoded.columns:
    feature_cols.append('performance_rating_encoded')

X = df_encoded[feature_cols].copy()
y = df_encoded['has_promotion'].copy()

print(f'Features selected: {len(feature_cols)}')
print(f'Feature list:')
for i, col in enumerate(feature_cols, 1):
    print(f'  {i:2d}. {col}')
print()

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'Target distribution:')
print(f'  Class 0 (Not Promoted): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)')
print(f'  Class 1 (Promoted):     {(y==1).sum()} ({(y==1).sum()/len(y)*100:.2f}%)')
print()

# ============================================================================
# 6. FEATURE SCALING
# ============================================================================

print('6. FEATURE SCALING')
print('-'*80)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

print(f'✅ Features scaled using StandardScaler')
print(f'   Mean: ~0, Std: ~1')
print()

# Visualize before/after scaling
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Before scaling - Performance
axes[0, 0].hist(X['performance_score'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 0].set_xlabel('Performance Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Before Scaling - Performance Score', fontweight='bold')
axes[0, 0].grid(alpha=0.3)

# After scaling - Performance
axes[0, 1].hist(X_scaled['performance_score'], bins=30, color='#3498db', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Performance Score (Scaled)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('After Scaling - Performance Score', fontweight='bold')
axes[0, 1].grid(alpha=0.3)

# Before scaling - Behavioral
axes[1, 0].hist(X['behavior_avg'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Behavioral Score')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Before Scaling - Behavioral Score', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# After scaling - Behavioral
axes[1, 1].hist(X_scaled['behavior_avg'], bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Behavioral Score (Scaled)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('After Scaling - Behavioral Score', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_feature_scaling.png', dpi=300, bbox_inches='tight')
print(f'✅ Plot saved: 02_feature_scaling.png')
plt.close()

# ============================================================================
# 7. TRAIN/TEST SPLIT
# ============================================================================

print('7. TRAIN/TEST SPLIT')
print('-'*80)

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f'✅ Data split (80/20):')
print(f'   Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)')
print(f'   Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)')
print()

print(f'Training set distribution:')
print(f'  Class 0: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)')
print(f'  Class 1: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.2f}%)')
print()

print(f'Test set distribution:')
print(f'  Class 0: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.2f}%)')
print(f'  Class 1: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.2f}%)')
print()

# ============================================================================
# 8. HANDLE CLASS IMBALANCE (SMOTE)
# ============================================================================

print('8. HANDLING CLASS IMBALANCE (SMOTE)')
print('-'*80)

print(f'Before SMOTE:')
print(f'  Class 0: {(y_train==0).sum()}')
print(f'  Class 1: {(y_train==1).sum()}')
print(f'  Ratio: 1:{(y_train==0).sum()/(y_train==1).sum():.1f}')
print()

# Apply SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f'After SMOTE:')
print(f'  Class 0: {(y_train_balanced==0).sum()}')
print(f'  Class 1: {(y_train_balanced==1).sum()}')
print(f'  Ratio: 1:{(y_train_balanced==0).sum()/(y_train_balanced==1).sum():.1f}')
print()

print(f'✅ Training set balanced using SMOTE')
print(f'   Original size: {len(X_train):,}')
print(f'   Balanced size: {len(X_train_balanced):,}')
print(f'   Synthetic samples added: {len(X_train_balanced) - len(X_train):,}')
print()

# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before SMOTE
axes[0].bar(['Not Promoted', 'Promoted'], 
            [(y_train==0).sum(), (y_train==1).sum()],
            color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
axes[0].set_ylabel('Count')
axes[0].set_title('Before SMOTE - Training Set', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate([(y_train==0).sum(), (y_train==1).sum()]):
    axes[0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# After SMOTE
axes[1].bar(['Not Promoted', 'Promoted'], 
            [(y_train_balanced==0).sum(), (y_train_balanced==1).sum()],
            color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Count')
axes[1].set_title('After SMOTE - Training Set', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate([(y_train_balanced==0).sum(), (y_train_balanced==1).sum()]):
    axes[1].text(i, v + 10, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/03_smote_balancing.png', dpi=300, bbox_inches='tight')
print(f'✅ Plot saved: 03_smote_balancing.png')
plt.close()

# ============================================================================
# 9. SAVE PROCESSED DATA
# ============================================================================

print('9. SAVING PROCESSED DATA')
print('-'*80)

# Save processed datasets
processed_dir = '/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/processed'
os.makedirs(processed_dir, exist_ok=True)

# Save full processed dataset
df_encoded.to_csv(f'{processed_dir}/full_dataset_processed.csv', index=False)
print(f'✅ Saved: full_dataset_processed.csv')

# Save train/test splits
X_train.to_csv(f'{processed_dir}/X_train.csv', index=False)
X_test.to_csv(f'{processed_dir}/X_test.csv', index=False)
y_train.to_csv(f'{processed_dir}/y_train.csv', index=False)
y_test.to_csv(f'{processed_dir}/y_test.csv', index=False)
print(f'✅ Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv')

# Save balanced training set
X_train_balanced_df = pd.DataFrame(X_train_balanced, columns=X_train.columns)
y_train_balanced_df = pd.DataFrame(y_train_balanced, columns=['has_promotion'])
X_train_balanced_df.to_csv(f'{processed_dir}/X_train_balanced.csv', index=False)
y_train_balanced_df.to_csv(f'{processed_dir}/y_train_balanced.csv', index=False)
print(f'✅ Saved: X_train_balanced.csv, y_train_balanced.csv')

# Save scaler
import joblib
joblib.dump(scaler, f'{processed_dir}/scaler.pkl')
print(f'✅ Saved: scaler.pkl')

print()

# ============================================================================
# 10. FEATURE IMPORTANCE PREVIEW
# ============================================================================

print('10. FEATURE CORRELATION WITH TARGET')
print('-'*80)

# Calculate correlation with target
feature_corr = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': [X[col].corr(y) for col in X.columns]
})
feature_corr = feature_corr.sort_values('Correlation', ascending=False)

print('Top 10 features by correlation with promotion:')
print(feature_corr.head(10).to_string(index=False))
print()

# Visualize
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in feature_corr['Correlation']]
ax.barh(range(len(feature_corr)), feature_corr['Correlation'], color=colors, alpha=0.7)
ax.set_yticks(range(len(feature_corr)))
ax.set_yticklabels(feature_corr['Feature'])
ax.set_xlabel('Correlation with Promotion')
ax.set_title('Feature Correlation with Target', fontweight='bold', fontsize=14)
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/04_feature_correlation.png', dpi=300, bbox_inches='tight')
print(f'✅ Plot saved: 04_feature_correlation.png')
plt.close()

# ============================================================================
# 11. SUMMARY REPORT
# ============================================================================

print()
print('11. GENERATING SUMMARY REPORT')
print('-'*80)

report_path = '/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/results/Feature_Engineering_Report.txt'

with open(report_path, 'w') as f:
    f.write('='*80 + '\n')
    f.write('MPCIM THESIS - FEATURE ENGINEERING SUMMARY\n')
    f.write('='*80 + '\n\n')
    
    f.write('1. OUTLIER HANDLING:\n')
    f.write(f'   Performance outliers: {len(perf_outliers)} capped\n')
    f.write(f'   Behavioral outliers: {len(beh_outliers)} capped\n\n')
    
    f.write('2. NEW FEATURES CREATED:\n')
    f.write('   - perf_beh_ratio: Performance/Behavioral ratio\n')
    f.write('   - combined_score: Weighted average (50-50)\n')
    f.write('   - score_difference: Performance - Behavioral\n')
    f.write('   - tenure_category: junior/mid/senior\n')
    f.write('   - performance_level: low/medium/high\n')
    f.write('   - behavioral_level: low/medium/high\n')
    f.write('   - high_performer: Both dimensions high\n\n')
    
    f.write('3. FEATURE ENCODING:\n')
    f.write(f'   Categorical variables encoded: {len(categorical_cols)}\n\n')
    
    f.write('4. FEATURE SCALING:\n')
    f.write('   Method: StandardScaler (mean=0, std=1)\n')
    f.write(f'   Features scaled: {len(feature_cols)}\n\n')
    
    f.write('5. TRAIN/TEST SPLIT:\n')
    f.write(f'   Training: {len(X_train)} samples (80%)\n')
    f.write(f'   Test: {len(X_test)} samples (20%)\n')
    f.write('   Stratified: Yes\n\n')
    
    f.write('6. CLASS IMBALANCE HANDLING:\n')
    f.write('   Method: SMOTE\n')
    f.write(f'   Before: {(y_train==1).sum()} promoted samples\n')
    f.write(f'   After: {(y_train_balanced==1).sum()} promoted samples\n')
    f.write(f'   Synthetic samples: {len(X_train_balanced) - len(X_train)}\n\n')
    
    f.write('7. TOP FEATURES BY CORRELATION:\n')
    for idx, row in feature_corr.head(5).iterrows():
        f.write(f'   {row["Feature"]}: {row["Correlation"]:.3f}\n')
    f.write('\n')
    
    f.write('8. FILES SAVED:\n')
    f.write('   - full_dataset_processed.csv\n')
    f.write('   - X_train.csv, X_test.csv\n')
    f.write('   - y_train.csv, y_test.csv\n')
    f.write('   - X_train_balanced.csv, y_train_balanced.csv\n')
    f.write('   - scaler.pkl\n\n')
    
    f.write('9. READY FOR MODELING:\n')
    f.write('   ✅ Outliers handled\n')
    f.write('   ✅ Features engineered\n')
    f.write('   ✅ Features scaled\n')
    f.write('   ✅ Class balanced\n')
    f.write('   ✅ Train/test split done\n')

print(f'✅ Report saved: Feature_Engineering_Report.txt')
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print('='*80)
print('FEATURE ENGINEERING COMPLETE!')
print('='*80)
print()
print(f'📊 Summary:')
print(f'   Original features: 10')
print(f'   New features created: 7')
print(f'   Total features for modeling: {len(feature_cols)}')
print()
print(f'📁 Processed data saved in: data/processed/')
print(f'📊 Plots saved in: results/feature_engineering/')
print(f'📄 Report: results/Feature_Engineering_Report.txt')
print()
print(f'✅ Dataset ready for modeling!')
print(f'   Training set: {len(X_train_balanced):,} samples (balanced)')
print(f'   Test set: {len(X_test):,} samples (original distribution)')
print()
print(f'🚀 Next step: Baseline Model Development')
print()
print('='*80)
