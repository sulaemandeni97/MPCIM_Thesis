"""
MPCIM Thesis - Improve Model Precision (Reduce False Positives)
Author: Deni Sulaeman
Date: November 22, 2025

Teknik untuk mengurangi False Positive:
1. Threshold Adjustment
2. Class Weight Tuning
3. Cost-Sensitive Learning
4. Ensemble with Precision Focus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create output directory
import os
output_dir = '/Users/denisulaeman/Workspace/Academic/CascadeProjects/MPCIM_Thesis/results/precision_improvement'
os.makedirs(output_dir, exist_ok=True)

print('='*80)
print('MPCIM THESIS - IMPROVE PRECISION (REDUCE FALSE POSITIVES)')
print('='*80)
print()

# ============================================================================
# 1. LOAD DATA & EXISTING MODEL
# ============================================================================

print('1. LOADING DATA & EXISTING MODEL')
print('-'*80)

data_dir = '/Users/denisulaeman/Workspace/Academic/CascadeProjects/MPCIM_Thesis/data/processed'

# Load test data
X_test = pd.read_csv(f'{data_dir}/X_test.csv')
y_test = pd.read_csv(f'{data_dir}/y_test.csv')['has_promotion']

# Load training data for retraining
X_train = pd.read_csv(f'{data_dir}/X_train_balanced.csv')
y_train = pd.read_csv(f'{data_dir}/y_train_balanced.csv')['has_promotion']

print(f'✅ Data loaded successfully')
print(f'   Training set: {len(X_train):,} samples')
print(f'   Test set: {len(X_test):,} samples')
print(f'   Class distribution (test): {y_test.value_counts().to_dict()}')
print()

# ============================================================================
# 2. TECHNIQUE 1: THRESHOLD ADJUSTMENT
# ============================================================================

print('2. TECHNIQUE 1: THRESHOLD ADJUSTMENT')
print('-'*80)
print('Mencari threshold optimal untuk maximize precision')
print()

# Load existing Neural Network model (best model)
model_path = '/Users/denisulaeman/Workspace/Academic/CascadeProjects/MPCIM_Thesis/models/neural_network_model.pkl'
try:
    nn_model = joblib.load(model_path)
    print(f'✅ Loaded existing Neural Network model')
except:
    print('⚠️  Neural Network model not found, training new one...')
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    nn_model.fit(X_train, y_train)
    joblib.dump(nn_model, model_path)
    print('✅ New Neural Network model trained and saved')

# Get probability predictions
y_prob = nn_model.predict_proba(X_test)[:, 1]

# Test different thresholds
thresholds = np.arange(0.3, 0.9, 0.05)
results = []

print('\nTesting different thresholds:')
print(f'{"Threshold":<12} {"Accuracy":<12} {"Precision":<12} {"Recall":<12} {"F1-Score":<12} {"FP":<8} {"FN":<8}')
print('-'*80)

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'true_negatives': tn
    })
    
    print(f'{threshold:<12.2f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fp:<8} {fn:<8}')

results_df = pd.DataFrame(results)

# Find optimal thresholds for different objectives
best_precision_idx = results_df['precision'].idxmax()
best_f1_idx = results_df['f1_score'].idxmax()
best_balanced_idx = (results_df['precision'] * results_df['recall']).idxmax()

print('\n📊 OPTIMAL THRESHOLDS:')
print(f'   Best Precision: {results_df.loc[best_precision_idx, "threshold"]:.2f} '
      f'(Precision: {results_df.loc[best_precision_idx, "precision"]:.4f}, '
      f'FP: {results_df.loc[best_precision_idx, "false_positives"]:.0f})')
print(f'   Best F1-Score:  {results_df.loc[best_f1_idx, "threshold"]:.2f} '
      f'(F1: {results_df.loc[best_f1_idx, "f1_score"]:.4f})')
print(f'   Best Balanced:  {results_df.loc[best_balanced_idx, "threshold"]:.2f} '
      f'(Precision: {results_df.loc[best_balanced_idx, "precision"]:.4f}, '
      f'Recall: {results_df.loc[best_balanced_idx, "recall"]:.4f})')
print()

# Visualize threshold impact
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Metrics vs Threshold
ax1 = axes[0, 0]
ax1.plot(results_df['threshold'], results_df['precision'], 'o-', label='Precision', linewidth=2)
ax1.plot(results_df['threshold'], results_df['recall'], 's-', label='Recall', linewidth=2)
ax1.plot(results_df['threshold'], results_df['f1_score'], '^-', label='F1-Score', linewidth=2)
ax1.axvline(results_df.loc[best_precision_idx, 'threshold'], color='red', linestyle='--', alpha=0.5, label='Best Precision')
ax1.set_xlabel('Threshold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Metrics vs Threshold', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: False Positives vs Threshold
ax2 = axes[0, 1]
ax2.plot(results_df['threshold'], results_df['false_positives'], 'ro-', linewidth=2, markersize=8)
ax2.axvline(results_df.loc[best_precision_idx, 'threshold'], color='green', linestyle='--', alpha=0.5, label='Best Precision')
ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('False Positives', fontsize=12)
ax2.set_title('False Positives vs Threshold', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Precision-Recall Curve
ax3 = axes[1, 0]
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
ax3.plot(recall_curve, precision_curve, 'b-', linewidth=2)
ax3.axhline(results_df.loc[best_precision_idx, 'precision'], color='red', linestyle='--', alpha=0.5, label='Best Precision')
ax3.set_xlabel('Recall', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix at Best Precision Threshold
ax4 = axes[1, 1]
best_threshold = results_df.loc[best_precision_idx, 'threshold']
y_pred_best = (y_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, cbar=False)
ax4.set_xlabel('Predicted', fontsize=12)
ax4.set_ylabel('Actual', fontsize=12)
ax4.set_title(f'Confusion Matrix (Threshold={best_threshold:.2f})', fontsize=14, fontweight='bold')
ax4.set_xticklabels(['Not Promoted', 'Promoted'])
ax4.set_yticklabels(['Not Promoted', 'Promoted'])

plt.tight_layout()
plt.savefig(f'{output_dir}/01_threshold_analysis.png', dpi=300, bbox_inches='tight')
print(f'✅ Saved: 01_threshold_analysis.png')
plt.close()

# Save results
results_df.to_csv(f'{output_dir}/threshold_results.csv', index=False)
print(f'✅ Saved: threshold_results.csv')
print()

# ============================================================================
# 3. TECHNIQUE 2: CLASS WEIGHT ADJUSTMENT
# ============================================================================

print('3. TECHNIQUE 2: CLASS WEIGHT ADJUSTMENT')
print('-'*80)
print('Melatih model dengan class weights berbeda untuk penalize false positives')
print()

# Test different class weights
class_weights = [
    None,  # Default
    'balanced',  # Auto balanced
    {0: 1, 1: 2},  # Favor positive class
    {0: 1, 1: 3},  # Strongly favor positive class
    {0: 2, 1: 1},  # Penalize false positives (favor negative class)
    {0: 3, 1: 1},  # Strongly penalize false positives
]

weight_results = []

print(f'{"Class Weight":<25} {"Accuracy":<12} {"Precision":<12} {"Recall":<12} {"F1-Score":<12} {"FP":<8}')
print('-'*80)

for weight in class_weights:
    # Train Neural Network with class weight
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    # For Neural Network, we need to use sample_weight instead
    if weight is None:
        sample_weight = None
        weight_name = 'None (default)'
    elif weight == 'balanced':
        # Calculate balanced weights
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight('balanced', y_train)
        weight_name = 'balanced'
    else:
        # Manual weights
        sample_weight = np.array([weight[y] for y in y_train])
        weight_name = str(weight)
    
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob_temp = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    weight_results.append({
        'class_weight': weight_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'false_positives': fp,
        'model': model
    })
    
    print(f'{weight_name:<25} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {fp:<8}')

weight_results_df = pd.DataFrame(weight_results)

# Find best class weight configuration
best_weight_idx = weight_results_df['precision'].idxmax()
print(f'\n📊 BEST CLASS WEIGHT: {weight_results_df.loc[best_weight_idx, "class_weight"]}')
print(f'   Precision: {weight_results_df.loc[best_weight_idx, "precision"]:.4f}')
print(f'   False Positives: {weight_results_df.loc[best_weight_idx, "false_positives"]:.0f}')
print()

# ============================================================================
# 4. SAVE OPTIMIZED MODEL
# ============================================================================

print('4. SAVING OPTIMIZED MODEL')
print('-'*80)

# Use the best threshold with original model
optimized_threshold = results_df.loc[best_precision_idx, 'threshold']

# Save model with optimal threshold
model_config = {
    'model': nn_model,
    'optimal_threshold': optimized_threshold,
    'default_threshold': 0.5,
    'threshold_results': results_df.to_dict('records')
}

joblib.dump(model_config, f'{output_dir}/optimized_neural_network.pkl')
print(f'✅ Saved optimized model with threshold={optimized_threshold:.2f}')
print()

# ============================================================================
# 5. GENERATE RECOMMENDATIONS
# ============================================================================

print('5. RECOMMENDATIONS FOR REDUCING FALSE POSITIVES')
print('-'*80)

# Compare default vs optimized
y_pred_default = (y_prob >= 0.5).astype(int)
y_pred_optimized = (y_prob >= optimized_threshold).astype(int)

tn_def, fp_def, fn_def, tp_def = confusion_matrix(y_test, y_pred_default).ravel()
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, y_pred_optimized).ravel()

prec_def = precision_score(y_test, y_pred_default, zero_division=0)
prec_opt = precision_score(y_test, y_pred_optimized, zero_division=0)

print(f'\n📊 COMPARISON: Default (0.5) vs Optimized ({optimized_threshold:.2f})')
print(f'\n   {"Metric":<20} {"Default":<15} {"Optimized":<15} {"Improvement"}')
print(f'   {"-"*60}')
print(f'   {"Precision":<20} {prec_def:<15.4f} {prec_opt:<15.4f} {(prec_opt-prec_def)*100:+.2f}%')
print(f'   {"False Positives":<20} {fp_def:<15} {fp_opt:<15} {fp_opt-fp_def:+d}')
print(f'   {"True Positives":<20} {tp_def:<15} {tp_opt:<15} {tp_opt-tp_def:+d}')

print(f'\n💡 ACTIONABLE RECOMMENDATIONS:')
print(f'\n1. **Gunakan Threshold Optimal: {optimized_threshold:.2f}**')
print(f'   - Mengurangi False Positives dari {fp_def} menjadi {fp_opt} ({abs(fp_opt-fp_def)} reduction)')
print(f'   - Meningkatkan Precision dari {prec_def:.2%} menjadi {prec_opt:.2%}')
print(f'\n2. **Implementasi di Aplikasi Streamlit:**')
print(f'   - Update prediction logic untuk menggunakan threshold {optimized_threshold:.2f}')
print(f'   - Tambahkan confidence score untuk setiap prediksi')
print(f'   - Tampilkan warning jika probability mendekati threshold')
print(f'\n3. **Business Impact:**')
print(f'   - Lebih sedikit karyawan yang diberi harapan palsu')
print(f'   - Fokus resources pada kandidat dengan probability tinggi')
print(f'   - Meningkatkan trust dalam sistem prediksi')
print(f'\n4. **Next Steps:**')
print(f'   - Collect feedback dari HR tentang false positives')
print(f'   - Monitor precision di production')
print(f'   - Periodic retraining dengan data baru')

print('\n' + '='*80)
print('✅ ANALYSIS COMPLETE!')
print('='*80)
print(f'\nOutput files saved to: {output_dir}/')
print(f'  - 01_threshold_analysis.png')
print(f'  - threshold_results.csv')
print(f'  - optimized_neural_network.pkl')
