"""
MPCIM Thesis - Baseline Model Development
Author: Deni Sulaeman
Date: October 21, 2025

This script develops baseline models to compare:
1. Performance-only model (Single dimension)
2. Behavioral-only model (Single dimension)
3. Dual-dimensional model (Performance + Behavioral)

Using Logistic Regression for interpretability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory
import os
output_dir = '/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/results/baseline_models'
os.makedirs(output_dir, exist_ok=True)

print('='*80)
print('MPCIM THESIS - BASELINE MODEL DEVELOPMENT')
print('='*80)
print()

# ============================================================================
# 1. LOAD PROCESSED DATA
# ============================================================================

print('1. LOADING PROCESSED DATA')
print('-'*80)

data_dir = '/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/processed'

# Load balanced training data
X_train = pd.read_csv(f'{data_dir}/X_train_balanced.csv')
y_train = pd.read_csv(f'{data_dir}/y_train_balanced.csv')['has_promotion']

# Load test data (original distribution)
X_test = pd.read_csv(f'{data_dir}/X_test.csv')
y_test = pd.read_csv(f'{data_dir}/y_test.csv')['has_promotion']

print(f'✅ Data loaded successfully')
print(f'   Training set: {len(X_train):,} samples (balanced)')
print(f'   Test set: {len(X_test):,} samples (original distribution)')
print()

print(f'Training distribution:')
print(f'  Class 0: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)')
print(f'  Class 1: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)')
print()

print(f'Test distribution:')
print(f'  Class 0: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%)')
print(f'  Class 1: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)')
print()

# ============================================================================
# 2. DEFINE FEATURE SETS
# ============================================================================

print('2. DEFINING FEATURE SETS')
print('-'*80)

# Performance-only features
performance_features = [
    'performance_score',
    'performance_level_encoded',
    'performance_rating_encoded'
]

# Behavioral-only features
behavioral_features = [
    'behavior_avg',
    'behavioral_level_encoded'
]

# Dual-dimensional features (all features)
dual_features = X_train.columns.tolist()

print(f'Performance-only features: {len(performance_features)}')
for f in performance_features:
    print(f'  - {f}')
print()

print(f'Behavioral-only features: {len(behavioral_features)}')
for f in behavioral_features:
    print(f'  - {f}')
print()

print(f'Dual-dimensional features: {len(dual_features)}')
print(f'  (All {len(dual_features)} features)')
print()

# ============================================================================
# 3. TRAIN BASELINE MODELS
# ============================================================================

print('3. TRAINING BASELINE MODELS')
print('-'*80)

# Initialize models
models = {}
predictions = {}
probabilities = {}

# Model 1: Performance-only
print('Training Model 1: Performance-only...')
model_perf = LogisticRegression(random_state=42, max_iter=1000)
model_perf.fit(X_train[performance_features], y_train)
models['Performance-only'] = model_perf
predictions['Performance-only'] = model_perf.predict(X_test[performance_features])
probabilities['Performance-only'] = model_perf.predict_proba(X_test[performance_features])[:, 1]
print('✅ Performance-only model trained')

# Model 2: Behavioral-only
print('Training Model 2: Behavioral-only...')
model_beh = LogisticRegression(random_state=42, max_iter=1000)
model_beh.fit(X_train[behavioral_features], y_train)
models['Behavioral-only'] = model_beh
predictions['Behavioral-only'] = model_beh.predict(X_test[behavioral_features])
probabilities['Behavioral-only'] = model_beh.predict_proba(X_test[behavioral_features])[:, 1]
print('✅ Behavioral-only model trained')

# Model 3: Dual-dimensional
print('Training Model 3: Dual-dimensional...')
model_dual = LogisticRegression(random_state=42, max_iter=1000)
model_dual.fit(X_train[dual_features], y_train)
models['Dual-dimensional'] = model_dual
predictions['Dual-dimensional'] = model_dual.predict(X_test[dual_features])
probabilities['Dual-dimensional'] = model_dual.predict_proba(X_test[dual_features])[:, 1]
print('✅ Dual-dimensional model trained')

print()

# ============================================================================
# 4. EVALUATE MODELS
# ============================================================================

print('4. MODEL EVALUATION')
print('-'*80)

results = []

for model_name in ['Performance-only', 'Behavioral-only', 'Dual-dimensional']:
    y_pred = predictions[model_name]
    y_prob = probabilities[model_name]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    print(f'{model_name}:')
    print(f'  Accuracy:  {accuracy:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall:    {recall:.4f}')
    print(f'  F1-Score:  {f1:.4f}')
    print(f'  ROC-AUC:   {roc_auc:.4f}')
    print()

# Create results dataframe
results_df = pd.DataFrame(results)
print('Summary Table:')
print(results_df.to_string(index=False))
print()

# ============================================================================
# 5. CONFUSION MATRICES
# ============================================================================

print('5. CONFUSION MATRICES')
print('-'*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_name in enumerate(['Performance-only', 'Behavioral-only', 'Dual-dimensional']):
    cm = confusion_matrix(y_test, predictions[model_name])
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Not Promoted', 'Promoted'],
                yticklabels=['Not Promoted', 'Promoted'],
                cbar=False)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
    
    # Add accuracy in subtitle
    acc = accuracy_score(y_test, predictions[model_name])
    axes[idx].text(0.5, -0.15, f'Accuracy: {acc:.4f}', 
                   ha='center', transform=axes[idx].transAxes)

plt.tight_layout()
plt.savefig(f'{output_dir}/01_confusion_matrices.png', dpi=300, bbox_inches='tight')
print('✅ Plot saved: 01_confusion_matrices.png')
plt.close()

# ============================================================================
# 6. ROC CURVES
# ============================================================================

print()
print('6. ROC CURVES')
print('-'*80)

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#e74c3c', '#3498db', '#2ecc71']

for idx, model_name in enumerate(['Performance-only', 'Behavioral-only', 'Dual-dimensional']):
    y_prob = probabilities[model_name]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
            label=f'{model_name} (AUC = {roc_auc:.4f})')

# Plot diagonal
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/02_roc_curves.png', dpi=300, bbox_inches='tight')
print('✅ Plot saved: 02_roc_curves.png')
plt.close()

# ============================================================================
# 7. PRECISION-RECALL CURVES
# ============================================================================

print()
print('7. PRECISION-RECALL CURVES')
print('-'*80)

fig, ax = plt.subplots(figsize=(10, 8))

for idx, model_name in enumerate(['Performance-only', 'Behavioral-only', 'Dual-dimensional']):
    y_prob = probabilities[model_name]
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    
    ax.plot(recall_curve, precision_curve, color=colors[idx], linewidth=2,
            label=f'{model_name}')

# Baseline (random)
baseline = (y_test == 1).sum() / len(y_test)
ax.axhline(baseline, color='k', linestyle='--', linewidth=1, 
           label=f'Baseline ({baseline:.4f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/03_precision_recall_curves.png', dpi=300, bbox_inches='tight')
print('✅ Plot saved: 03_precision_recall_curves.png')
plt.close()

# ============================================================================
# 8. METRICS COMPARISON
# ============================================================================

print()
print('8. METRICS COMPARISON')
print('-'*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors_bar = ['#e74c3c', '#3498db', '#2ecc71']

for idx, metric in enumerate(metrics):
    values = results_df[metric].values
    bars = axes[idx].bar(range(len(results_df)), values, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[idx].set_xticks(range(len(results_df)))
    axes[idx].set_xticklabels(results_df['Model'], rotation=15, ha='right')
    axes[idx].set_ylabel(metric)
    axes[idx].set_title(f'{metric} Comparison', fontweight='bold')
    axes[idx].set_ylim([0, 1])
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        axes[idx].text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                      f'{val:.4f}', ha='center', fontweight='bold', fontsize=9)

# Overall comparison in last subplot
axes[5].axis('off')
table_data = []
for _, row in results_df.iterrows():
    table_data.append([
        row['Model'],
        f"{row['Accuracy']:.4f}",
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['F1-Score']:.4f}",
        f"{row['ROC-AUC']:.4f}"
    ])

table = axes[5].table(cellText=table_data,
                     colLabels=['Model', 'Acc', 'Prec', 'Rec', 'F1', 'AUC'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header
for i in range(6):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color rows
for i in range(1, 4):
    for j in range(6):
        table[(i, j)].set_facecolor(colors_bar[i-1])
        table[(i, j)].set_alpha(0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/04_metrics_comparison.png', dpi=300, bbox_inches='tight')
print('✅ Plot saved: 04_metrics_comparison.png')
plt.close()

# ============================================================================
# 9. FEATURE IMPORTANCE (DUAL MODEL)
# ============================================================================

print()
print('9. FEATURE IMPORTANCE (DUAL-DIMENSIONAL MODEL)')
print('-'*80)

# Get coefficients from dual model
coefficients = model_dual.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': dual_features,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
})
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print('Top 10 Most Important Features:')
print(feature_importance.head(10)[['Feature', 'Coefficient']].to_string(index=False))
print()

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
colors_feat = ['#2ecc71' if x > 0 else '#e74c3c' for x in feature_importance['Coefficient']]
ax.barh(range(len(feature_importance)), feature_importance['Coefficient'], 
        color=colors_feat, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['Feature'])
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Feature Importance - Dual-Dimensional Model', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/05_feature_importance.png', dpi=300, bbox_inches='tight')
print('✅ Plot saved: 05_feature_importance.png')
plt.close()

# ============================================================================
# 10. CLASSIFICATION REPORTS
# ============================================================================

print()
print('10. DETAILED CLASSIFICATION REPORTS')
print('-'*80)

for model_name in ['Performance-only', 'Behavioral-only', 'Dual-dimensional']:
    print(f'\n{model_name}:')
    print('-' * 60)
    print(classification_report(y_test, predictions[model_name], 
                                target_names=['Not Promoted', 'Promoted'],
                                digits=4))

# ============================================================================
# 11. SAVE RESULTS
# ============================================================================

print()
print('11. SAVING RESULTS')
print('-'*80)

# Save results dataframe
results_df.to_csv(f'{output_dir}/baseline_results.csv', index=False)
print('✅ Saved: baseline_results.csv')

# Save models
import joblib
for model_name, model in models.items():
    filename = model_name.lower().replace('-', '_').replace(' ', '_')
    joblib.dump(model, f'{output_dir}/{filename}_model.pkl')
    print(f'✅ Saved: {filename}_model.pkl')

print()

# ============================================================================
# 12. SUMMARY REPORT
# ============================================================================

print('12. GENERATING SUMMARY REPORT')
print('-'*80)

report_path = '/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/results/Baseline_Models_Report.txt'

with open(report_path, 'w') as f:
    f.write('='*80 + '\n')
    f.write('MPCIM THESIS - BASELINE MODELS SUMMARY\n')
    f.write('='*80 + '\n\n')
    
    f.write('1. MODELS TRAINED:\n')
    f.write('   a) Performance-only (Single Dimension)\n')
    f.write(f'      Features: {len(performance_features)}\n')
    f.write('   b) Behavioral-only (Single Dimension)\n')
    f.write(f'      Features: {len(behavioral_features)}\n')
    f.write('   c) Dual-dimensional (MPCIM Approach)\n')
    f.write(f'      Features: {len(dual_features)}\n\n')
    
    f.write('2. DATASET:\n')
    f.write(f'   Training: {len(X_train)} samples (balanced with SMOTE)\n')
    f.write(f'   Test: {len(X_test)} samples (original distribution)\n\n')
    
    f.write('3. RESULTS COMPARISON:\n\n')
    f.write(results_df.to_string(index=False))
    f.write('\n\n')
    
    f.write('4. KEY FINDINGS:\n')
    
    # Find best model for each metric
    best_acc = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_prec = results_df.loc[results_df['Precision'].idxmax(), 'Model']
    best_rec = results_df.loc[results_df['Recall'].idxmax(), 'Model']
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    best_auc = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
    
    f.write(f'   Best Accuracy:  {best_acc}\n')
    f.write(f'   Best Precision: {best_prec}\n')
    f.write(f'   Best Recall:    {best_rec}\n')
    f.write(f'   Best F1-Score:  {best_f1}\n')
    f.write(f'   Best ROC-AUC:   {best_auc}\n\n')
    
    # Calculate improvement
    dual_acc = results_df[results_df['Model'] == 'Dual-dimensional']['Accuracy'].values[0]
    perf_acc = results_df[results_df['Model'] == 'Performance-only']['Accuracy'].values[0]
    beh_acc = results_df[results_df['Model'] == 'Behavioral-only']['Accuracy'].values[0]
    
    f.write('5. IMPROVEMENT ANALYSIS:\n')
    f.write(f'   Dual vs Performance-only: {((dual_acc - perf_acc) / perf_acc * 100):.2f}%\n')
    f.write(f'   Dual vs Behavioral-only:  {((dual_acc - beh_acc) / beh_acc * 100):.2f}%\n\n')
    
    f.write('6. TOP 5 IMPORTANT FEATURES (Dual Model):\n')
    for idx, row in feature_importance.head(5).iterrows():
        f.write(f'   {row["Feature"]}: {row["Coefficient"]:.4f}\n')
    f.write('\n')
    
    f.write('7. CONCLUSION:\n')
    if dual_acc > max(perf_acc, beh_acc):
        f.write('   ✅ Dual-dimensional approach OUTPERFORMS single-dimension models!\n')
        f.write('   ✅ This validates the MPCIM framework hypothesis.\n')
    else:
        f.write('   ⚠️  Single-dimension models perform comparably.\n')
        f.write('   → Further optimization needed for dual-dimensional model.\n')

print('✅ Report saved: Baseline_Models_Report.txt')
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print('='*80)
print('BASELINE MODELS COMPLETE!')
print('='*80)
print()

print('📊 Results Summary:')
print(results_df.to_string(index=False))
print()

print(f'🏆 Best Model:')
best_model = results_df.loc[results_df['F1-Score'].idxmax()]
print(f'   Model: {best_model["Model"]}')
print(f'   F1-Score: {best_model["F1-Score"]:.4f}')
print(f'   ROC-AUC: {best_model["ROC-AUC"]:.4f}')
print()

print(f'📁 Results saved in: results/baseline_models/')
print(f'📊 Plots: 5 visualization files')
print(f'💾 Models: 3 trained models saved')
print(f'📄 Report: Baseline_Models_Report.txt')
print()

# Check if dual is best
dual_row = results_df[results_df['Model'] == 'Dual-dimensional'].iloc[0]
if dual_row['F1-Score'] == results_df['F1-Score'].max():
    print('✅ DUAL-DIMENSIONAL MODEL IS THE BEST!')
    print('   This validates the MPCIM framework!')
else:
    print('⚠️  Single-dimension model performed better')
    print('   Consider advanced algorithms (Random Forest, XGBoost)')

print()
print('🚀 Next steps:')
print('   1. Try advanced algorithms (Random Forest, XGBoost, Neural Networks)')
print('   2. Hyperparameter tuning')
print('   3. SHAP analysis for explainability')
print()
print('='*80)
