# MPCIM THESIS PROPOSAL - COMPLETE CONTENT
## For Professional Word Document Generation

---

## SECTION 1: INTRODUCTION

### 1.1 Background

Career progression decisions represent critical junctures in organizational talent management, influencing both organizational performance and employee engagement (Ng et al., 2005). Traditional approaches have predominantly relied on single-dimensional assessment frameworks, emphasizing either performance metrics or behavioral competencies in isolation.

Performance management systems, while widely adopted, face criticism for limited predictive validity regarding future success in higher positions (Aguinis et al., 2019). Conversely, behavioral assessments, though valuable for evaluating soft skills, often lack quantitative rigor for data-driven decision-making (Scullen et al., 2000).

Recent advances in predictive analytics and machine learning have enabled more sophisticated HR approaches (Rasmussen & Ulrich, 2015). However, application to career progression faces challenges including severe class imbalance, interpretability needs, and heterogeneous data integration.

### 1.2 Problem Statement

Current models suffer from:

**Limited Predictive Power**: Single-dimensional models achieve only 57.3% (performance) and 35.0% (behavioral) accuracy—marginally better than random guessing.

**Statistical Insignificance**: Performance scores show no statistical significance (p=0.083) in differentiating promoted from non-promoted employees, challenging widespread practice of using performance as primary promotion criterion.

**Severe Class Imbalance**: Real-world data exhibits 9.27% promotion rate, posing significant challenges for traditional algorithms that bias toward majority class (Chawla et al., 2002).

**Lack of Explainability**: Many advanced models operate as "black boxes." In HR contexts affecting careers and livelihoods, explainability is ethically imperative (Lepri et al., 2018).

**Incomplete Assessment**: Existing models fail to integrate multiple dimensions, missing potential synergies and interaction effects.

### 1.3 Research Objectives

**Primary**: Develop and validate MPCIM integrating performance and behavioral assessments for accurate, explainable, and fair career progression prediction.

**Specific Objectives**:
1. Compare single-dimensional vs dual-dimensional predictive performance
2. Identify and quantify relative importance of features and dimensions
3. Develop effective strategies for handling severe class imbalance
4. Provide explainable and actionable insights through feature importance analysis
5. Validate framework using state-of-the-art machine learning algorithms

### 1.4 Research Significance

**Theoretical**: Provides empirical validation of multi-dimensional assessment frameworks. Demonstrates behavioral assessment adds unique predictive value (p=0.037) beyond performance metrics, challenging predominant focus on performance-based criteria.

**Methodological**: Contributes comprehensive, reproducible methodology for imbalanced classification in HR. SMOTE + advanced algorithms provide template for similar challenges. Feature engineering techniques offer insights for multi-source data integration.

**Practical**: Organizations can leverage 90.9% accuracy model as decision support tool. Explainability enables HR professionals to understand determinants, design targeted programs, make objective decisions. Tenure paradox has immediate implications for succession planning.

---

## SECTION 2: LITERATURE REVIEW

### 2.1 Performance Management Systems

Modern performance management evolved from annual reviews to comprehensive systems integrating goal-setting, continuous feedback, and development (Aguinis et al., 2019). Frameworks like OKRs and KPIs emphasize measurable outcomes and organizational alignment (Niven & Lamorte, 2016).

However, limitations persist. Scullen et al. (2000) demonstrated ratings contain substantial rater-specific variance, questioning validity. Murphy & Cleveland (1995) highlighted biases including leniency, halo effects, and recency bias.

Critically for promotion prediction, Ng et al. (2005) found current performance shows only moderate correlation (r=0.33) with future performance at higher levels. This aligns with our findings where performance alone failed significance (p=0.083).

### 2.2 Behavioral Assessment in HR

Behavioral competency frameworks complement performance-based assessments (Bartram, 2005). The "Great Eight" competencies provide criterion-centric validation approach. Boyatzis (2008) emphasizes competencies for 21st century success including emotional intelligence, adaptability, and collaboration.

Spencer & Spencer (1993) distinguish threshold competencies (minimum requirements) from differentiating competencies (superior performers). Our research validates this: behavioral assessment (p=0.037) differentiates promoted employees while performance alone does not.

### 2.3 Predictive Analytics in HR

HR analytics has evolved from descriptive reporting to predictive and prescriptive analytics (Marler & Boudreau, 2017). Applications include turnover prediction, performance forecasting, and talent identification.

Rasmussen & Ulrich (2015) provide evidence-based review, emphasizing need for rigorous methodology to avoid management fads. Boudreau & Ramstad (2007) introduce "talentship" concept—decision science for human capital paralleling financial decision-making.

### 2.4 Machine Learning for Career Progression

Machine learning applications in HR have grown significantly. Ensemble methods (Random Forest, XGBoost) and neural networks show promise for complex prediction tasks (Chen & Guestrin, 2016; Breiman, 2001).

However, HR applications require special considerations: interpretability, fairness, and handling of sensitive attributes. Our research addresses these through feature importance analysis and careful feature selection.

### 2.5 Class Imbalance in Classification

Class imbalance is prevalent in real-world applications where positive class is rare (He & Garcia, 2009). Promotions represent such cases—organizational constraints limit promotion rates.

SMOTE (Chawla et al., 2002) addresses imbalance by generating synthetic minority examples. Fernández et al. (2018) provide comprehensive review of learning from imbalanced datasets. Our research validates SMOTE effectiveness for HR promotion prediction.

### 2.6 Research Gap

Despite extensive literature, gaps remain:
1. Limited research on multi-dimensional approaches for career progression
2. Few studies address severe class imbalance in HR contexts
3. Lack of explainable models suitable for HR decision-making
4. Insufficient validation with real-world organizational data

This research fills these gaps through comprehensive dual-dimensional framework with 712 real employees, achieving 90.9% accuracy with explainable features.

---

## SECTION 3: RESEARCH QUESTIONS

### RQ1: Model Performance Comparison

**Question**: Does dual-dimensional approach provide better accuracy than single-dimensional?

**Hypothesis H₁**: Dual-dimensional models will significantly outperform single-dimensional in accuracy, precision, recall, and F1-score.

**Status**: ✓ CONFIRMED

**Evidence**:
- Dual-dimensional baseline: 76.2% accuracy (+32.9% vs best single)
- Neural Network: 90.9% accuracy (+48.97% F1 improvement)
- Performance-only: 57.3%, Behavioral-only: 35.0%

**Statistical Support**: Behavioral p=0.037 (significant), Performance p=0.083 (not significant)

### RQ2: Feature Importance

**Question**: Which dimension and features are most influential?

**Hypothesis H₂**: Both dimensions contribute significantly, with behavioral providing unique predictive value beyond performance.

**Status**: ✓ CONFIRMED

**Evidence**:
- Tenure: 40-50% importance (dominant)
- Behavioral: 4-6% importance, p=0.037 significant
- Performance: 3-5% importance, p=0.083 not significant alone
- Engineered features: 5-8% additional value

**Novel Finding**: Tenure paradox—negative correlation (r=-0.169), younger employees promoted more

### RQ3: Class Imbalance Handling

**Question**: Most effective strategy for 9.27% promotion rate?

**Hypothesis H₃**: SMOTE + advanced algorithms will effectively address imbalance.

**Status**: ✓ CONFIRMED

**Evidence**:
- SMOTE balancing: 516-516 (1:1 ratio)
- Neural Network: 90.9% accuracy, 61.5% recall, 50.0% precision
- Catches 8 of 13 promotions in test set
- Precision doubled vs baseline (24.4% → 50.0%)

### RQ4: Explainability

**Question**: How to provide explainable insights for HR?

**Hypothesis H₄**: Feature importance and SHAP will provide interpretable insights.

**Status**: ⏳ IN PROGRESS

**Completed**:
- Feature importance from RF, XGBoost, Logistic Regression
- Correlation analysis
- Statistical testing

**Planned**:
- SHAP analysis for individual predictions
- Feature contribution visualization
- HR decision support guidelines

---

## SECTION 4: METHODOLOGY

### 4.1 Research Design

**Type**: Quantitative, Predictive Analytics
**Approach**: Supervised Machine Learning
**Data**: Real-world HR data from PostgreSQL database
**Sample**: 712 employees with dual-dimensional data
**Period**: Cross-sectional analysis

### 4.2 Data Collection

**Source**: db_cna_digispace_august_132025 (PostgreSQL, port 5433)

**Components**:
1. Performance Data: 13,478 assessments, 127,579 KPI items
2. Behavioral Data: 19,929 records, 766 employees
3. Target Variable: 130 promotion records, 117 unique promoted
4. Demographics: Tenure, gender, marital status, employment type

**Integration**: Merged via NIK (employee ID) with MD5 anonymization
**Final Dataset**: 712 employees with both dimensions (101.2% match rate)

### 4.3 Data Preprocessing

**4.3.1 Anonymization**
- MD5 hashing of employee IDs (irreversible)
- No personal identifiers retained
- Compliant with data privacy regulations

**4.3.2 Outlier Handling**
- Method: IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
- Performance: 46 outliers (6.5%) capped
- Behavioral: 35 outliers (4.9%) capped
- Result: Performance 36.63-125.31, Behavioral 71.51-111.09

**4.3.3 Missing Data**
- Only 14 missing values in performance_rating (2%)
- Imputation: Forward fill based on employee history
- Final completeness: 98%

### 4.4 Feature Engineering

**Created 7 new features**:
1. perf_beh_ratio: Performance/Behavioral ratio
2. combined_score: Weighted average (50-50)
3. score_difference: Performance - Behavioral
4. tenure_category: Junior (0-2), Mid (3-7), Senior (8+)
5. performance_level: Low/Medium/High (tertiles)
6. behavioral_level: Low/Medium/High (tertiles)
7. high_performer: Both dimensions high (binary flag)

**Encoding**:
- Categorical features: Label encoding
- Binary features: 0/1 encoding
- Total features: 14

**Scaling**:
- Method: StandardScaler (mean=0, std=1)
- Applied to all 14 features
- Fitted on training, transformed on test

### 4.5 Class Imbalance Handling

**Original Distribution**:
- Not Promoted: 646 (90.73%)
- Promoted: 66 (9.27%)
- Ratio: 1:9.8 (severe imbalance)

**SMOTE Application**:
- Applied only to training set
- Synthetic samples: 463 generated
- Balanced training: 516-516 (1:1)
- Test set: Original distribution maintained

**Rationale**: Train on balanced data for learning, test on real distribution for realistic evaluation

### 4.6 Train/Test Split

- Split ratio: 80/20
- Stratification: Yes (maintains class distribution)
- Random state: 42 (reproducibility)
- Training: 569 → 1,032 after SMOTE
- Test: 143 (original distribution)

### 4.7 Model Development

**4.7.1 Baseline Models (Logistic Regression)**

Purpose: Establish baseline, compare dimensions

Models:
1. Performance-only: 3 features (score, level, rating)
2. Behavioral-only: 2 features (score, level)
3. Dual-dimensional: All 14 features

Hyperparameters:
- Solver: lbfgs
- Max iterations: 1000
- Random state: 42

Results:
- Performance-only: 57.3% accuracy, 26.5% F1
- Behavioral-only: 35.0% accuracy, 19.1% F1
- Dual-dimensional: 76.2% accuracy, 37.0% F1

**4.7.2 Advanced Models**

**Random Forest**:
- n_estimators: 100
- max_depth: 10
- min_samples_split: 5
- min_samples_leaf: 2
- Results: 87.4% accuracy, 50.0% F1, 90.1% ROC-AUC

**XGBoost**:
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- subsample: 0.8
- colsample_bytree: 0.8
- Results: 89.5% accuracy, 51.6% F1, 88.3% ROC-AUC

**Neural Network (MLP)**:
- Architecture: (64, 32, 16) hidden layers
- Activation: ReLU
- Optimizer: Adam
- Learning rate: Adaptive
- Early stopping: Yes (validation_fraction=0.1)
- Results: 90.9% accuracy, 55.2% F1, 88.3% ROC-AUC ⭐ BEST

### 4.8 Evaluation Metrics

**Primary Metrics**:
- Accuracy: Overall correctness
- Precision: Positive predictive value (TP / (TP + FP))
- Recall: Sensitivity (TP / (TP + FN))
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under ROC curve

**Why F1-Score as Primary?**
- Balances precision and recall
- Appropriate for imbalanced datasets
- Reflects real-world cost of false positives and negatives
- Single metric for model comparison

**Additional Analysis**:
- Confusion matrices
- ROC curves
- Precision-recall curves
- Feature importance rankings

### 4.9 Validation Strategy

**Current**:
- Hold-out test set (20%, stratified)
- Original distribution maintained
- No data leakage

**Planned**:
- Stratified K-Fold cross-validation (k=5)
- Different random seeds robustness testing
- Sensitivity analysis for hyperparameters

---

## SECTION 5: PRELIMINARY RESULTS

### 5.1 Descriptive Statistics

**Dataset Overview**:
- Total: 712 employees
- Promoted: 66 (9.27%)
- Not Promoted: 646 (90.73%)

**Performance Scores**:
- Overall: Mean=81.88, SD=34.94, Range=36.63-125.31
- Promoted: Mean=88.99, SD=23.89
- Not Promoted: Mean=81.15, SD=35.81
- T-test: t=1.739, p=0.0825 (NOT significant)

**Behavioral Scores**:
- Overall: Mean=89.72, SD=8.71, Range=71.51-111.09
- Promoted: Mean=91.85, SD=7.38
- Not Promoted: Mean=89.50, SD=8.81
- T-test: t=2.090, p=0.0370 (✓ SIGNIFICANT)

**Tenure**:
- Overall: Mean=8.17, SD=7.28, Range=0-125 years
- Promoted: Mean=4.32, SD=4.98
- Not Promoted: Mean=8.56, SD=7.42
- Correlation: r=-0.169 (negative!)

### 5.2 Model Performance Comparison

**Complete Results Table**:

| Model | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|------|----------|-----------|--------|----------|---------|
| Performance-only | Baseline | 57.3% | 15.7% | 84.6% | 26.5% | 72.3% |
| Behavioral-only | Baseline | 35.0% | 10.8% | 84.6% | 19.1% | 65.3% |
| Dual-dimensional | Baseline | 76.2% | 24.4% | 76.9% | 37.0% | 81.2% |
| Random Forest | Advanced | 87.4% | 39.1% | 69.2% | 50.0% | 90.1% |
| XGBoost | Advanced | 89.5% | 44.4% | 61.5% | 51.6% | 88.3% |
| Neural Network | Advanced | 90.9% | 50.0% | 61.5% | 55.2% | 88.3% |

**Performance Progression**:
1. Single-dimension: 35-57% accuracy
2. Dual-dimension baseline: 76.2% (+32.9%)
3. Advanced dual-dimension: 90.9% (+48.97% F1)

### 5.3 Best Model Analysis (Neural Network)

**Confusion Matrix**:
```
                Predicted
              Not Prom  Promoted
Actual Not      122        8      (93.8% specificity)
       Prom       5        8      (61.5% recall)
```

**Interpretation**:
- True Negatives: 122 (93.8% of non-promoted correctly identified)
- True Positives: 8 (61.5% of promoted correctly identified)
- False Positives: 8 (6.2% false alarm rate)
- False Negatives: 5 (38.5% missed promotions)

**Practical Impact**:
- Model identifies 16 candidates for promotion consideration
- 8 out of 16 predictions correct (50% precision)
- Catches 8 out of 13 actual promotions (61.5% recall)
- Reduces screening pool by 89% (143 → 16)

### 5.4 Feature Importance

**Random Forest Top 10**:
1. tenure_years: 40.5%
2. tenure_category: 32.6%
3. performance_rating: 5.1%
4. behavior_avg: 4.6%
5. performance_score: 3.6%
6. combined_score: 3.3%
7. marital_status: 2.7%
8. perf_beh_ratio: 2.7%
9. score_difference: 2.5%
10. behavioral_level: 1.0%

**XGBoost Top 10**:
1. tenure_category: 50.6%
2. tenure_years: 17.1%
3. is_permanent: 6.0%
4. performance_rating: 4.4%
5. marital_status: 3.1%
6. performance_level: 3.0%
7. behavior_avg: 2.8%
8. high_performer: 2.6%
9. score_difference: 2.2%
10. combined_score: 2.2%

**Key Insights**:
- Tenure dominates (40-50%) across models
- Both dimensions contribute (3-6% each)
- Engineered features add value (ratios, combinations)
- Consistent patterns across algorithms

### 5.5 Tenure Paradox

**Statistical Evidence**:
- Correlation: r=-0.169, p<0.001
- Promoted: 4.32 years average
- Not Promoted: 8.56 years average
- Difference: -4.24 years (highly significant)

**By Category**:
- Junior (0-2 years): 14.3% promotion rate
- Mid (3-7 years): 10.0% promotion rate
- Senior (8+ years): 5.1% promotion rate
- Ratio: Junior 2.8× higher than Senior

**Interpretation**:
- Organizations prioritize high-potential early-career employees
- Rapid advancement for promising talent
- Challenges seniority-based assumptions
- Implications for succession planning

### 5.6 Improvement Analysis

**Baseline to Advanced**:
- Accuracy: +14.7% (76.2% → 90.9%)
- Precision: +25.6% (24.4% → 50.0%) = +105%
- F1-Score: +18.2% (37.0% → 55.2%) = +48.97%
- ROC-AUC: +7.1% (81.2% → 88.3%)

**Single to Dual**:
- vs Performance-only: +32.9% accuracy
- vs Behavioral-only: +117.8% accuracy

**Key Achievement**: Precision doubled while maintaining good recall

---

## SECTION 6: EXPECTED CONTRIBUTIONS

### 6.1 Theoretical Contributions

1. **Multi-Dimensional Framework Validation**
   - Empirical evidence: 90.9% vs 57.3% (single-dimension)
   - Statistical support: Behavioral p=0.037, Performance p=0.083
   - Challenges single-dimension predominance

2. **Behavioral Assessment Significance**
   - First large-scale validation in career progression
   - Demonstrates unique predictive value
   - Supports holistic talent assessment theories

3. **Tenure Paradox Discovery**
   - Novel finding: Negative correlation (r=-0.169)
   - Challenges seniority-based assumptions
   - Theoretical implications for career development

4. **Class Imbalance Methodology**
   - Validates SMOTE for HR applications
   - Framework for rare event prediction
   - Generalizable to other HR contexts

### 6.2 Practical Contributions

1. **Deployable Decision Support Tool**
   - 90.9% accuracy model ready for production
   - 50% precision suitable for screening
   - Reduces candidate pool by 89%

2. **Career Development Framework**
   - Clear promotion determinants identified
   - Targeted development program design
   - Succession planning insights

3. **Fair and Objective Assessment**
   - Data-driven reduces bias
   - Transparent feature importance
   - Consistent evaluation criteria

4. **Resource Optimization**
   - Focus on high-potential employees
   - Efficient talent management
   - Improved retention strategies

### 6.3 Methodological Contributions

1. **End-to-End Pipeline**
   - Comprehensive data collection to deployment
   - Reproducible methodology
   - Scalable to other organizations

2. **Feature Engineering Techniques**
   - Novel combined features (ratios, differences)
   - Categorical encoding strategies
   - Effective dimensionality management

3. **Model Comparison Framework**
   - Systematic evaluation across 6 algorithms
   - Baseline vs advanced comparison
   - Clear performance metrics

4. **Imbalance Handling Best Practices**
   - SMOTE application guidelines
   - Training/testing strategy
   - Evaluation metric selection

---

## SECTION 7: TIMELINE

**Phase 1: Data Collection & Preparation** (2 weeks) ✓ COMPLETE
- Database schema analysis
- Data export with anonymization
- Data integration and quality assessment

**Phase 2: Exploratory Data Analysis** (1 week) ✓ COMPLETE
- Descriptive statistics
- Statistical testing
- Visualization generation

**Phase 3: Feature Engineering** (1 week) ✓ COMPLETE
- Outlier handling
- New feature creation
- Encoding and scaling

**Phase 4: Model Development** (2 weeks) ✓ COMPLETE
- Baseline model training
- Advanced model training
- Performance evaluation

**Phase 5: Model Interpretation** (1 week) ⏳ IN PROGRESS
- SHAP analysis
- Feature contribution visualization
- Decision support guidelines

**Phase 6: Documentation** (2 weeks) 📝 CURRENT
- Thesis proposal
- Literature review completion
- Methodology documentation

**Phase 7: Validation & Refinement** (1 week) 📅 PLANNED
- Cross-validation
- Sensitivity analysis
- Expert review

**Phase 8: Finalization** (1 week) 📅 PLANNED
- Thesis writing
- Defense preparation
- Final submission

**Total Duration**: 11 weeks (~3 months)
**Current Progress**: 60% complete
**Expected Completion**: January 2026

---

## SECTION 8: LIMITATIONS & ETHICS

### 8.1 Limitations

**Data Limitations**:
- Single organization (generalizability)
- Cross-sectional (no longitudinal tracking)
- Limited competency data availability
- Sample size (712 employees, 66 promotions)

**Methodological Limitations**:
- Class imbalance (9.27% rate)
- Feature selection (14 from original 10)
- Neural Network interpretability
- No causal inference (correlation only)

**Practical Limitations**:
- Model requires periodic retraining
- Data drift over time
- Implementation barriers
- Change management needs

### 8.2 Ethical Considerations

**Data Privacy**:
- MD5 anonymization (irreversible)
- No personal identifiers
- Secure storage
- Compliance with regulations

**Fairness & Bias**:
- Gender, tenure bias potential
- Transparent feature importance
- Human oversight required
- Model as support, not replacement

**Transparency**:
- Explainable features
- Clear documentation
- Accessible to HR professionals
- Feedback mechanisms

**Impact on Employees**:
- Positive: Objective decisions, clear criteria
- Concerns: Algorithmic anxiety, over-reliance
- Mitigation: Communication, human judgment

---

## SECTION 9: CONCLUSION

This thesis proposes and validates MPCIM for career progression prediction. Preliminary results demonstrate:

1. **Dual-dimensional superiority**: 90.9% vs 57.3% (performance) vs 35.0% (behavioral)
2. **Both dimensions important**: Behavioral significant (p=0.037), performance not alone (p=0.083)
3. **Advanced algorithms effective**: Neural Network 55.2% F1 (+48.97% improvement)
4. **Tenure strongest predictor**: 40-50% importance, younger promoted more
5. **Class imbalance addressable**: SMOTE + NN handles 9.27% rate effectively

**Key Innovation**: Integration of performance and behavioral dimensions with state-of-the-art ML techniques addressing real-world HR challenges including severe imbalance and explainability needs.

**Expected Impact**: Provides organizations with robust, fair, transparent tool for talent management and career development, improving employee satisfaction and organizational effectiveness.

**Research Status**: 60% complete, on track for January 2026 completion.

---

END OF CONTENT
