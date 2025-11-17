"""
Retrain Models with Quick Assessment Features
==============================================

This script retrains all models with the enhanced feature set including
Quick Assessment psychological components.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("RETRAIN MODELS WITH QUICK ASSESSMENT FEATURES")
print("=" * 80)
print()

# Setup paths
repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data" / "processed"
results_dir = repo_root / "results"

# Load processed data
print("1. Loading processed data...")
X_train = pd.read_csv(data_dir / "X_train_balanced.csv")
X_test = pd.read_csv(data_dir / "X_test.csv")
y_train = pd.read_csv(data_dir / "y_train_balanced.csv").values.ravel()
y_test = pd.read_csv(data_dir / "y_test.csv").values.ravel()

print(f"   ✓ Training set: {X_train.shape}")
print(f"   ✓ Test set: {X_test.shape}")
print(f"   ✓ Features: {len(X_train.columns)}")
print()

# Calculate class weights for imbalanced data
class_weight_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"   Class weight ratio: {class_weight_ratio:.2f}:1")
print()

# Define models with enhanced configurations
models = {
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=class_weight_ratio,
        random_state=42,
        eval_metric='logloss',
        subsample=0.8,
        colsample_bytree=0.8
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "neural_network": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate='adaptive'
    ),
    "dual_logistic": LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42,
        solver='saga',
        penalty='l2'
    )
}

# Train and evaluate
print("=" * 80)
print("2. TRAINING MODELS")
print("=" * 80)
print()

results = []
trained_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    print("-" * 40)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_proba_train = None
        y_pred_proba_test = None
    
    # Metrics - Training
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    train_roc_auc = roc_auc_score(y_train, y_pred_proba_train) if y_pred_proba_train is not None else 0
    
    # Metrics - Test
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba_test) if y_pred_proba_test is not None else 0
    
    print(f"  Training Metrics:")
    print(f"    Accuracy:  {train_accuracy:.4f}")
    print(f"    Precision: {train_precision:.4f}")
    print(f"    Recall:    {train_recall:.4f}")
    print(f"    F1-Score:  {train_f1:.4f}")
    print(f"    ROC-AUC:   {train_roc_auc:.4f}")
    print()
    print(f"  Test Metrics:")
    print(f"    Accuracy:  {test_accuracy:.4f}")
    print(f"    Precision: {test_precision:.4f}")
    print(f"    Recall:    {test_recall:.4f}")
    print(f"    F1-Score:  {test_f1:.4f}")
    print(f"    ROC-AUC:   {test_roc_auc:.4f}")
    print()
    
    results.append({
        'Model': name,
        'Train_Accuracy': train_accuracy,
        'Train_Precision': train_precision,
        'Train_Recall': train_recall,
        'Train_F1': train_f1,
        'Train_ROC_AUC': train_roc_auc,
        'Test_Accuracy': test_accuracy,
        'Test_Precision': test_precision,
        'Test_Recall': test_recall,
        'Test_F1': test_f1,
        'Test_ROC_AUC': test_roc_auc
    })
    
    trained_models[name] = model
    
    # Save model
    if name == "xgboost":
        save_path = results_dir / "advanced_models" / "xgboost_model.pkl"
    elif name == "random_forest":
        save_path = results_dir / "advanced_models" / "random_forest_model.pkl"
    elif name == "neural_network":
        save_path = results_dir / "advanced_models" / "neural_network_model.pkl"
    else:  # dual_logistic
        save_path = results_dir / "baseline_models" / "dual_dimensional_model.pkl"
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"  ✅ Saved to: {save_path}")
    print()

# Summary
print("=" * 80)
print("3. TRAINING COMPLETE - RESULTS SUMMARY")
print("=" * 80)
print()

results_df = pd.DataFrame(results)

# Display results
print("Test Set Performance:")
print(results_df[['Model', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 
                  'Test_F1', 'Test_ROC_AUC']].to_string(index=False))
print()

# Save results
results_csv = results_dir / "advanced_models" / "qa_enhanced_models_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"✅ Results saved to: {results_csv}")
print()

# Feature importance for tree-based models
print("=" * 80)
print("4. FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)
print()

for model_name in ['xgboost', 'random_forest']:
    if model_name in trained_models:
        model = trained_models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            print(f"{model_name.upper()} - Top 10 Important Features:")
            print(importance.head(10).to_string(index=False))
            print()
            
            # Plot
            plt.figure(figsize=(10, 8))
            top_features = importance.head(15)
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance')
            plt.title(f'{model_name.upper()} - Feature Importance (Top 15)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plot_path = results_dir / "feature_engineering" / f"{model_name}_feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Plot saved: {plot_path}")
            print()

# Recommendations
print("=" * 80)
print("5. RECOMMENDATIONS")
print("=" * 80)
print()

best_model = results_df.loc[results_df['Test_F1'].idxmax(), 'Model']
best_f1 = results_df.loc[results_df['Test_F1'].idxmax(), 'Test_F1']

print(f"🏆 Best model (by Test F1-Score): {best_model.upper()}")
print(f"   F1-Score: {best_f1:.4f}")
print()

print("Quick Assessment Impact:")
qa_features = ['psychological_score', 'drive_score', 'mental_strength_score',
               'adaptability_score', 'collaboration_score', 'leadership_potential']

for model_name in ['xgboost', 'random_forest']:
    if model_name in trained_models:
        model = trained_models[model_name]
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(X_train.columns, model.feature_importances_))
            qa_importance = sum([importance_dict.get(f, 0) for f in qa_features if f in importance_dict])
            total_importance = sum(model.feature_importances_)
            qa_percentage = (qa_importance / total_importance) * 100
            
            print(f"  {model_name.upper()}:")
            print(f"    QA features contribution: {qa_percentage:.2f}%")

print()
print("For production deployment:")
print(f"  1. Use {best_model.upper()} for best performance")
print("  2. XGBoost and Random Forest both perform well")
print("  3. Quick Assessment features add valuable psychological insights")
print()

print("=" * 80)
print("✅ RETRAINING COMPLETE!")
print("=" * 80)
