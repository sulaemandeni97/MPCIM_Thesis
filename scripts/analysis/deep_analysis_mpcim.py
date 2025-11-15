import pandas as pd
import numpy as np

print('='*80)
print('MPCIM DATASET - DEEP ANALYSIS')
print('='*80)

# Load all sheets
xl_file = pd.ExcelFile('/Users/denisulaeman/Downloads/MPCIM Dataset.xlsx')

# Load each sheet
df_kpi = pd.read_excel(xl_file, sheet_name='KPI atau OKR')
df_behavior = pd.read_excel(xl_file, sheet_name='Behavior atau Prilaku')
df_functional = pd.read_excel(xl_file, sheet_name='FunctionalCompetency')

print('\n### 1. EMPLOYEE COVERAGE ANALYSIS ###\n')

# Convert employee_id to string for consistency
df_kpi['employee_id'] = df_kpi['employee_id'].astype(str)
df_behavior['employee_id'] = df_behavior['employee_id'].astype(str)
df_functional['employee_id'] = df_functional['employee_id'].astype(str)

# Get unique employees per dimension
emp_kpi = set(df_kpi['employee_id'].unique())
emp_behavior = set(df_behavior['employee_id'].unique())
emp_functional = set(df_functional['employee_id'].unique())

print(f"Employees with KPI data: {len(emp_kpi)}")
print(f"Employees with Behavior data: {len(emp_behavior)}")
print(f"Employees with Functional data: {len(emp_functional)}")

# Check overlaps
overlap_all = emp_kpi & emp_behavior & emp_functional
overlap_kpi_behavior = emp_kpi & emp_behavior
overlap_kpi_functional = emp_kpi & emp_functional
overlap_behavior_functional = emp_behavior & emp_functional

print(f"\n✅ Employees with ALL 3 dimensions: {len(overlap_all)}")
print(f"   Employees with KPI + Behavior: {len(overlap_kpi_behavior)}")
print(f"   Employees with KPI + Functional: {len(overlap_kpi_functional)}")
print(f"   Employees with Behavior + Functional: {len(overlap_behavior_functional)}")

print(f"\nTotal unique employees across all sheets: {len(emp_kpi | emp_behavior | emp_functional)}")

# Calculate coverage percentage
total_employees = len(emp_kpi | emp_behavior | emp_functional)
coverage_pct = (len(overlap_all) / total_employees) * 100
print(f"Coverage (employees with complete data): {coverage_pct:.1f}%")

print('\n### 2. KPI/OKR DIMENSION ANALYSIS ###\n')

# Aggregate KPI scores per employee
kpi_agg = df_kpi.groupby('employee_id').agg({
    'current_value': 'mean',
    'final_result': 'sum',
    'weight': 'sum',
    'kpi': 'count'
}).rename(columns={'kpi': 'num_kpis'})

kpi_agg['weighted_score'] = (kpi_agg['final_result'] / kpi_agg['weight']) * 100

print(f"Average KPIs per employee: {kpi_agg['num_kpis'].mean():.1f}")
print(f"Min KPIs: {kpi_agg['num_kpis'].min()}, Max KPIs: {kpi_agg['num_kpis'].max()}")
print(f"\nWeighted Score Statistics:")
print(kpi_agg['weighted_score'].describe())

print('\n### 3. BEHAVIOR DIMENSION ANALYSIS ###\n')

# Pivot behavior data
behavior_pivot = df_behavior.pivot_table(
    index='employee_id',
    columns='category',
    values='value',
    aggfunc='mean'
)

# Calculate average behavior score
behavior_pivot['behavior_avg'] = behavior_pivot.mean(axis=1)

print(f"Behavior categories: {list(behavior_pivot.columns[:-1])}")
print(f"\nBehavior Score Statistics:")
print(behavior_pivot['behavior_avg'].describe())

print('\n### 4. FUNCTIONAL COMPETENCY ANALYSIS ###\n')

print(f"Functional Competency Statistics:")
print(df_functional['total_average'].describe())

print(f"\nMissing values in competency domains:")
missing_comp = df_functional[['marketing', 'finance', 'operation', 'hr', 'bd']].isnull().sum()
print(missing_comp)

print('\n### 5. INTEGRATED DATASET CREATION ###\n')

# Create integrated dataset for employees with all 3 dimensions
integrated_data = []

for emp_id in overlap_all:
    # Get KPI score
    kpi_score = kpi_agg.loc[emp_id, 'weighted_score'] if emp_id in kpi_agg.index else None
    
    # Get behavior score
    behavior_score = behavior_pivot.loc[emp_id, 'behavior_avg'] if emp_id in behavior_pivot.index else None
    
    # Get functional score
    func_data = df_functional[df_functional['employee_id'] == emp_id]
    func_score = func_data['total_average'].values[0] if len(func_data) > 0 else None
    
    integrated_data.append({
        'employee_id': emp_id,
        'kpi_score': kpi_score,
        'behavior_score': behavior_score,
        'functional_score': func_score
    })

df_integrated = pd.DataFrame(integrated_data)

print(f"Integrated dataset shape: {df_integrated.shape}")
print(f"\nIntegrated dataset preview:")
print(df_integrated.head(10))

print(f"\nIntegrated dataset statistics:")
print(df_integrated[['kpi_score', 'behavior_score', 'functional_score']].describe())

# Check for missing values in integrated dataset
print(f"\nMissing values in integrated dataset:")
print(df_integrated.isnull().sum())

print('\n### 6. CORRELATION ANALYSIS ###\n')

correlation = df_integrated[['kpi_score', 'behavior_score', 'functional_score']].corr()
print("Correlation between dimensions:")
print(correlation)

print('\n### 7. DATA QUALITY SUMMARY ###\n')

print(f"✅ Usable employees for multi-dimensional analysis: {len(overlap_all)}")
print(f"✅ Complete data coverage: {(df_integrated.notna().all(axis=1).sum() / len(df_integrated)) * 100:.1f}%")

if len(overlap_all) < 100:
    print(f"\n⚠️  WARNING: Only {len(overlap_all)} employees have complete data!")
    print(f"   This might be too small for robust ML model training.")
    print(f"   Recommendation: Consider using 2-dimension combinations or imputation.")

print('\n### 8. RECOMMENDATIONS ###\n')

print("📊 Dataset Options:")
print(f"   Option 1: Use {len(overlap_all)} employees with ALL 3 dimensions (most complete)")
print(f"   Option 2: Use {len(overlap_kpi_behavior)} employees with KPI + Behavior")
print(f"   Option 3: Use {len(overlap_kpi_functional)} employees with KPI + Functional")
print(f"   Option 4: Use imputation to fill missing dimensions")

print("\n🎯 Missing Components for Predictive Model:")
print("   ❌ Target variable (promotion, career move, performance rating)")
print("   ❌ Leadership assessment dimension")
print("   ❌ Career aspiration dimension")
print("   ❌ Demographic data (position, department, tenure)")
print("   ❌ Historical/temporal data (changes over time)")

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
