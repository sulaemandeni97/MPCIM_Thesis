import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print('='*80)
print('MPCIM DATA ANALYSIS - CNA Database')
print('='*80)
print()

# Load all datasets
print('Loading datasets...')
df_integrated = pd.read_csv('/tmp/mpcim_export_cna/00_integrated_dataset.csv')
df_employee = pd.read_csv('/tmp/mpcim_export_cna/01_employee_master.csv')
df_performance = pd.read_csv('/tmp/mpcim_export_cna/02_performance_contract.csv')
df_kpi = pd.read_csv('/tmp/mpcim_export_cna/03_performance_kpi_items.csv')
df_competency = pd.read_csv('/tmp/mpcim_export_cna/04_competency_assessment.csv')
df_talent = pd.read_csv('/tmp/mpcim_export_cna/05_talent_scorecard.csv')
df_promotion = pd.read_csv('/tmp/mpcim_export_cna/06_promotion_history.csv')

print('✅ All datasets loaded successfully!')
print()

# ============================================================================
# ANALYSIS 1: INTEGRATED DATASET OVERVIEW
# ============================================================================

print('='*80)
print('ANALYSIS 1: INTEGRATED DATASET OVERVIEW')
print('='*80)
print()

print(f'Total records: {len(df_integrated):,}')
print(f'Unique employees: {df_integrated["employee_id_hash"].nunique():,}')
print()

print('Dataset shape:', df_integrated.shape)
print()

print('Columns:')
for i, col in enumerate(df_integrated.columns, 1):
    print(f'  {i}. {col}')
print()

print('Data types:')
print(df_integrated.dtypes)
print()

print('Missing values:')
missing = df_integrated.isnull().sum()
missing_pct = (missing / len(df_integrated)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))
print()

# ============================================================================
# ANALYSIS 2: PERFORMANCE DIMENSION
# ============================================================================

print('='*80)
print('ANALYSIS 2: PERFORMANCE DIMENSION ANALYSIS')
print('='*80)
print()

# Performance scores
perf_data = df_integrated[df_integrated['performance_score'].notna()]
print(f'Employees with performance data: {len(perf_data):,}')
print(f'Unique employees: {perf_data["employee_id_hash"].nunique():,}')
print()

print('Performance Score Statistics:')
print(perf_data['performance_score'].describe())
print()

print('Performance Rating Distribution:')
print(perf_data['performance_rating'].value_counts())
print()

# Performance by tenure
print('Performance by Tenure:')
tenure_perf = perf_data.groupby('tenure_years')['performance_score'].agg(['mean', 'count'])
print(tenure_perf.head(10))
print()

# ============================================================================
# ANALYSIS 3: PROMOTION ANALYSIS (TARGET VARIABLE)
# ============================================================================

print('='*80)
print('ANALYSIS 3: PROMOTION ANALYSIS (TARGET VARIABLE)')
print('='*80)
print()

print(f'Total promotion records: {len(df_promotion):,}')
print(f'Unique employees promoted: {df_promotion["employee_id_hash"].nunique():,}')
print()

print('Promotion status distribution:')
print(df_promotion['status'].value_counts())
print()

# Promotion rate in integrated dataset
promotion_rate = (df_integrated['has_promotion'].sum() / len(df_integrated)) * 100
print(f'Overall promotion rate: {promotion_rate:.2f}%')
print(f'Promoted: {df_integrated["has_promotion"].sum():,}')
print(f'Not promoted: {(len(df_integrated) - df_integrated["has_promotion"].sum()):,}')
print()

# ============================================================================
# ANALYSIS 4: PERFORMANCE vs PROMOTION
# ============================================================================

print('='*80)
print('ANALYSIS 4: PERFORMANCE vs PROMOTION CORRELATION')
print('='*80)
print()

# Compare performance scores between promoted and non-promoted
promoted = df_integrated[df_integrated['has_promotion'] == 1]['performance_score'].dropna()
not_promoted = df_integrated[df_integrated['has_promotion'] == 0]['performance_score'].dropna()

print('Performance Score Comparison:')
print(f'Promoted employees:')
print(f'  Mean: {promoted.mean():.2f}')
print(f'  Median: {promoted.median():.2f}')
print(f'  Std: {promoted.std():.2f}')
print(f'  Count: {len(promoted):,}')
print()

print(f'Not promoted employees:')
print(f'  Mean: {not_promoted.mean():.2f}')
print(f'  Median: {not_promoted.median():.2f}')
print(f'  Std: {not_promoted.std():.2f}')
print(f'  Count: {len(not_promoted):,}')
print()

# Statistical test
from scipy import stats
if len(promoted) > 0 and len(not_promoted) > 0:
    t_stat, p_value = stats.ttest_ind(promoted, not_promoted)
    print(f'T-test results:')
    print(f'  t-statistic: {t_stat:.4f}')
    print(f'  p-value: {p_value:.4f}')
    if p_value < 0.05:
        print('  ✅ Significant difference (p < 0.05)')
    else:
        print('  ⚠️  No significant difference (p >= 0.05)')
print()

# ============================================================================
# ANALYSIS 5: DATA COMPLETENESS FOR ML
# ============================================================================

print('='*80)
print('ANALYSIS 5: DATA COMPLETENESS FOR ML MODEL')
print('='*80)
print()

# Check employees with complete data
complete_data = df_integrated[
    df_integrated['performance_score'].notna()
].copy()

print(f'Employees with performance data: {len(complete_data):,}')
print(f'Unique employees: {complete_data["employee_id_hash"].nunique():,}')
print()

# Check target variable distribution
print('Target variable (has_promotion) distribution:')
print(complete_data['has_promotion'].value_counts())
print()
print(f'Promotion rate: {(complete_data["has_promotion"].sum() / len(complete_data)) * 100:.2f}%')
print()

# ============================================================================
# ANALYSIS 6: FEATURE AVAILABILITY
# ============================================================================

print('='*80)
print('ANALYSIS 6: FEATURE AVAILABILITY FOR MPCIM')
print('='*80)
print()

features = {
    'Performance Score': complete_data['performance_score'].notna().sum(),
    'Competency Score': complete_data['competency_score'].notna().sum(),
    'Talent Score': complete_data['talent_score'].notna().sum(),
    'Tenure': complete_data['tenure_years'].notna().sum(),
    'Gender': complete_data['gender'].notna().sum(),
    'Marital Status': complete_data['marital_status'].notna().sum(),
}

print('Feature availability:')
for feature, count in features.items():
    pct = (count / len(complete_data)) * 100
    print(f'  {feature}: {count:,} ({pct:.1f}%)')
print()

# ============================================================================
# ANALYSIS 7: CORRELATION ANALYSIS
# ============================================================================

print('='*80)
print('ANALYSIS 7: CORRELATION ANALYSIS')
print('='*80)
print()

# Select numeric columns for correlation
numeric_cols = ['tenure_years', 'performance_score', 'competency_score', 
                'talent_score', 'has_promotion']
corr_data = complete_data[numeric_cols].dropna()

if len(corr_data) > 0:
    print(f'Correlation analysis on {len(corr_data):,} records with complete data')
    print()
    correlation = corr_data.corr()
    print('Correlation Matrix:')
    print(correlation)
    print()
    
    print('Correlation with Promotion:')
    promo_corr = correlation['has_promotion'].sort_values(ascending=False)
    print(promo_corr)
else:
    print('⚠️  Not enough complete data for correlation analysis')
print()

# ============================================================================
# ANALYSIS 8: RECOMMENDATIONS
# ============================================================================

print('='*80)
print('ANALYSIS 8: RECOMMENDATIONS FOR MPCIM THESIS')
print('='*80)
print()

print('✅ STRENGTHS:')
print(f'  1. Large sample size: {complete_data["employee_id_hash"].nunique():,} unique employees')
print(f'  2. Rich performance data: {len(df_performance):,} assessments')
print(f'  3. Detailed KPI data: {len(df_kpi):,} KPI items')
print(f'  4. Clear target variable: {df_promotion["employee_id_hash"].nunique():,} promotions')
print()

print('⚠️  LIMITATIONS:')
print(f'  1. Limited competency data: {df_competency["employee_id_hash"].nunique()} employees')
print(f'  2. Limited talent data: {df_talent["employee_id_hash"].nunique()} employees')
print(f'  3. Class imbalance: {promotion_rate:.2f}% promotion rate')
print()

print('🎯 RECOMMENDED APPROACH:')
print()
print('Option 1: PERFORMANCE-BASED PREDICTION (RECOMMENDED)')
print('  - Focus: Performance dimension only')
print('  - Sample: ~1,700 employees with performance data')
print('  - Target: Promotion prediction')
print('  - Features: Performance scores, tenure, demographics')
print('  - Combine with behavioral data from Excel')
print()

print('Option 2: MULTI-DIMENSIONAL (LIMITED)')
print('  - Focus: Performance + Competency + Talent')
print('  - Sample: Very limited (~11 employees with all dimensions)')
print('  - Not recommended due to small sample size')
print()

print('Option 3: TEMPORAL ANALYSIS')
print('  - Focus: Performance trends over time')
print('  - Use multiple performance periods per employee')
print('  - Predict future performance or promotion')
print()

print('='*80)
print('ANALYSIS COMPLETE')
print('='*80)
