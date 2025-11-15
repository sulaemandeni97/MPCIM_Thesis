import pandas as pd
import numpy as np
from scipy import stats

print('='*80)
print('MPCIM DATA INTEGRATION: Performance + Behavioral (Using NIK)')
print('='*80)
print()

# ============================================================================
# STEP 1: Load Performance Data
# ============================================================================

print('STEP 1: Loading Performance Data')
print('-'*80)

df_performance = pd.read_csv('/tmp/mpcim_export_cna/00_integrated_dataset.csv')
df_perf_clean = df_performance[df_performance['performance_score'].notna()].copy()

# Aggregate per employee
df_perf_agg = df_perf_clean.groupby('employee_id_hash').agg({
    'company_id': 'first',
    'tenure_years': 'first',
    'gender': 'first',
    'marital_status': 'first',
    'is_permanent': 'first',
    'performance_score': 'mean',
    'performance_rating': 'last',
    'has_promotion': 'max'
}).reset_index()

print(f'✅ Performance data: {len(df_perf_agg):,} employees')
print()

# ============================================================================
# STEP 2: Load Behavioral Data from Excel
# ============================================================================

print('STEP 2: Loading Behavioral Data from Excel')
print('-'*80)

xl_file = pd.ExcelFile('/Users/denisulaeman/Downloads/MPCIM Dataset.xlsx')
df_behavior = pd.read_excel(xl_file, sheet_name='Behavior atau Prilaku')

print(f'✅ Behavioral data: {len(df_behavior):,} records')
print(f'   Unique employees (NIK): {df_behavior["employee_id"].nunique():,}')
print()

# Pivot and aggregate
behavior_pivot = df_behavior.pivot_table(
    index='employee_id',
    columns='category',
    values='value',
    aggfunc='mean'
)
behavior_pivot['behavior_avg'] = behavior_pivot.mean(axis=1)
behavior_pivot = behavior_pivot.reset_index()

print(f'   Behavioral scores aggregated: {len(behavior_pivot):,} employees')
print()

# ============================================================================
# STEP 3: Load NIK Mapping
# ============================================================================

print('STEP 3: Loading NIK to Hash Mapping')
print('-'*80)

df_nik_mapping = pd.read_csv('/tmp/mpcim_export_cna/employee_nik_mapping.csv')

print(f'✅ NIK mapping loaded: {len(df_nik_mapping):,} records')
print()

# Convert to string for matching
df_nik_mapping['employee_id'] = df_nik_mapping['employee_id'].astype(str)
behavior_pivot['employee_id'] = behavior_pivot['employee_id'].astype(str)

print('Sample NIK from mapping:')
print(df_nik_mapping['employee_id'].head(5).tolist())
print()

print('Sample NIK from Excel:')
print(behavior_pivot['employee_id'].head(5).tolist())
print()

# ============================================================================
# STEP 4: Merge Behavioral with NIK Mapping
# ============================================================================

print('STEP 4: Merging Behavioral Data with NIK Mapping')
print('-'*80)

df_behavior_mapped = behavior_pivot.merge(
    df_nik_mapping,
    on='employee_id',
    how='inner'
)

print(f'✅ Matched employees: {len(df_behavior_mapped):,}')
print(f'   Match rate: {(len(df_behavior_mapped) / len(behavior_pivot)) * 100:.1f}%')
print()

# ============================================================================
# STEP 5: Merge Performance + Behavioral
# ============================================================================

print('STEP 5: Merging Performance + Behavioral')
print('-'*80)

df_integrated = df_perf_agg.merge(
    df_behavior_mapped[['employee_id_hash', 'behavior_avg']],
    on='employee_id_hash',
    how='inner'
)

print(f'✅ INTEGRATED DATASET CREATED!')
print(f'   Total employees with BOTH dimensions: {len(df_integrated):,}')
print()

# ============================================================================
# STEP 6: Data Quality Check
# ============================================================================

print('='*80)
print('DATA QUALITY CHECK')
print('='*80)
print()

print(f'Dataset shape: {df_integrated.shape}')
print()

print('Columns:')
for i, col in enumerate(df_integrated.columns, 1):
    print(f'  {i}. {col}')
print()

print('Missing values:')
missing = df_integrated.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print('  ✅ No missing values!')
print()

print('Sample data (first 5 rows):')
print(df_integrated.head())
print()

# ============================================================================
# STEP 7: Descriptive Statistics
# ============================================================================

print('='*80)
print('DESCRIPTIVE STATISTICS')
print('='*80)
print()

print('Performance Score:')
print(df_integrated['performance_score'].describe())
print()

print('Behavioral Score:')
print(df_integrated['behavior_avg'].describe())
print()

print('Tenure:')
print(df_integrated['tenure_years'].describe())
print()

print('Target Variable (has_promotion):')
print(df_integrated['has_promotion'].value_counts())
promotion_rate = (df_integrated['has_promotion'].sum() / len(df_integrated)) * 100
print(f'Promotion rate: {promotion_rate:.2f}%')
print()

# ============================================================================
# STEP 8: Correlation Analysis
# ============================================================================

print('='*80)
print('CORRELATION ANALYSIS')
print('='*80)
print()

corr_cols = ['tenure_years', 'performance_score', 'behavior_avg', 'has_promotion']
correlation = df_integrated[corr_cols].corr()

print('Correlation Matrix:')
print(correlation.round(3))
print()

print('Correlation with Promotion:')
promo_corr = correlation['has_promotion'].sort_values(ascending=False)
for col, val in promo_corr.items():
    if col != 'has_promotion':
        print(f'  {col}: {val:.3f}')
print()

# ============================================================================
# STEP 9: Promoted vs Not Promoted Comparison
# ============================================================================

print('='*80)
print('PROMOTED vs NOT PROMOTED COMPARISON')
print('='*80)
print()

promoted = df_integrated[df_integrated['has_promotion'] == 1]
not_promoted = df_integrated[df_integrated['has_promotion'] == 0]

print(f'Promoted: {len(promoted):,} employees')
print(f'Not Promoted: {len(not_promoted):,} employees')
print()

print('Performance Score:')
print(f'  Promoted     - Mean: {promoted["performance_score"].mean():.2f}, Median: {promoted["performance_score"].median():.2f}, Std: {promoted["performance_score"].std():.2f}')
print(f'  Not Promoted - Mean: {not_promoted["performance_score"].mean():.2f}, Median: {not_promoted["performance_score"].median():.2f}, Std: {not_promoted["performance_score"].std():.2f}')

if len(promoted) > 1 and len(not_promoted) > 1:
    t_stat, p_val = stats.ttest_ind(promoted['performance_score'], not_promoted['performance_score'])
    print(f'  T-test: t={t_stat:.3f}, p={p_val:.4f}', end='')
    if p_val < 0.05:
        print(' ✅ Significant')
    else:
        print(' ⚠️  Not significant')
print()

print('Behavioral Score:')
print(f'  Promoted     - Mean: {promoted["behavior_avg"].mean():.2f}, Median: {promoted["behavior_avg"].median():.2f}, Std: {promoted["behavior_avg"].std():.2f}')
print(f'  Not Promoted - Mean: {not_promoted["behavior_avg"].mean():.2f}, Median: {not_promoted["behavior_avg"].median():.2f}, Std: {not_promoted["behavior_avg"].std():.2f}')

if len(promoted) > 1 and len(not_promoted) > 1:
    t_stat, p_val = stats.ttest_ind(promoted['behavior_avg'], not_promoted['behavior_avg'])
    print(f'  T-test: t={t_stat:.3f}, p={p_val:.4f}', end='')
    if p_val < 0.05:
        print(' ✅ Significant')
    else:
        print(' ⚠️  Not significant')
print()

print('Tenure:')
print(f'  Promoted     - Mean: {promoted["tenure_years"].mean():.2f} years')
print(f'  Not Promoted - Mean: {not_promoted["tenure_years"].mean():.2f} years')
print()

# ============================================================================
# STEP 10: Save Integrated Dataset
# ============================================================================

print('='*80)
print('SAVING INTEGRATED DATASET')
print('='*80)
print()

output_file = '/tmp/mpcim_export_cna/integrated_performance_behavioral.csv'
df_integrated.to_csv(output_file, index=False)

print(f'✅ Dataset saved to: {output_file}')
print(f'   Records: {len(df_integrated):,}')
print(f'   Columns: {len(df_integrated.columns)}')
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print('='*80)
print('FINAL SUMMARY - MPCIM INTEGRATED DATASET')
print('='*80)
print()

print(f'📊 DATASET OVERVIEW:')
print(f'   Total Employees: {len(df_integrated):,}')
print(f'   With Performance: {len(df_integrated):,} (100%)')
print(f'   With Behavioral: {len(df_integrated):,} (100%)')
print(f'   With Promotion: {promoted.shape[0]:,} ({promotion_rate:.2f}%)')
print()

print(f'📈 DIMENSIONS:')
print(f'   1. Performance Score (mean: {df_integrated["performance_score"].mean():.2f})')
print(f'   2. Behavioral Score (mean: {df_integrated["behavior_avg"].mean():.2f})')
print(f'   3. Demographics (tenure, gender, marital status)')
print(f'   4. Target: Promotion')
print()

print(f'✅ READY FOR ML MODEL!')
print()

# Check if sample size is sufficient
if len(df_integrated) >= 200:
    print(f'✅ Sample size ({len(df_integrated):,}) is SUFFICIENT for ML modeling')
elif len(df_integrated) >= 100:
    print(f'⚠️  Sample size ({len(df_integrated):,}) is MODERATE - consider simple models')
else:
    print(f'❌ Sample size ({len(df_integrated):,}) is TOO SMALL for robust ML')
print()

if promotion_rate < 5:
    print(f'⚠️  Class imbalance is HIGH ({promotion_rate:.2f}%) - will need special handling')
elif promotion_rate < 20:
    print(f'⚠️  Class imbalance is MODERATE ({promotion_rate:.2f}%) - consider SMOTE or class weights')
else:
    print(f'✅ Class balance is REASONABLE ({promotion_rate:.2f}%)')
print()

print('='*80)
print('INTEGRATION COMPLETE!')
print('='*80)
