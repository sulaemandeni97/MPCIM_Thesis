# MPCIM Framework - Database Analysis & Recommendations

## Executive Summary

**Database**: PostgreSQL with 195 tables
**Relevant Tables**: 60 tables identified for MPCIM
**Data Richness**: EXCELLENT - Far beyond initial Excel data!

---

## Available Dimensions for MPCIM Framework

### ✅ 1. PERFORMANCE/OKR DIMENSION (Complete!)

**Primary Tables**:
- `performance_contract` - Main performance contract per employee
- `performance_contract_item` - Individual KPIs/OKRs with weights
- `pa_okr_objective` - OKR objectives
- `pa_okr_key_result` - Key results with current/target values
- `pa_transaction` - Performance assessment transactions
- `performance_summary` - Aggregated performance scores

**Key Fields**:
- `final_result` - Overall performance score
- `weight` - KPI weights
- `current_value`, `target_value` - Achievement metrics
- `status_value` - Performance status/rating

### ✅ 2. COMPETENCY DIMENSION (Complete!)

**Primary Tables**:
- `employee_competency_assesment` - Main competency assessment
- `employee_competency_assesment_items` - Individual competency scores
- `competency_assignments` - Competency requirements per position
- `competency_assignment_items` - Detailed competency items

**Key Fields**:
- `final_result` - Overall competency score
- `skill` - Competency/skill name
- `level` - Required/achieved level
- `score` - Individual competency scores

### ✅ 3. BEHAVIORAL DIMENSION (Available!)

**From Excel**: Already have behavioral assessment data
**Database**: Can be linked via `employee_id`

### ✅ 4. TALENT/POTENTIAL DIMENSION (NEW!)

**Primary Tables**:
- `talent_scorecard_transaction` - Talent assessment
- `talent_scorecard_final` - Final talent scores
- `talent_scorecard_assesment_summary` - Multi-rater assessment

**Key Fields**:
- `final_score` - Overall talent score
- `group` - Talent category/box (9-box grid?)
- Multi-level approval scores

### ✅ 5. FEEDBACK & COACHING DIMENSION (NEW!)

**Primary Tables**:
- `feedback_performance` - Performance feedback
- `feedback_performance_answers` - Detailed feedback responses
- `coaching` - Coaching sessions (GROW model)

**Key Fields**:
- Feedback scores and comments
- Coaching goals, reality, options, will

### ⚠️ 6. CAREER PROGRESSION (TARGET VARIABLE!)

**Available Data**:
- `employee_promotion_request` - Promotion history
- `employee_transfer_request` - Transfer/role change history
- `grade` & `grade_class` - Job grades/bands
- `job_level` - Job level hierarchy
- `job_position` - Position information

**This is your TARGET VARIABLE for prediction!**

### ✅ 7. EMPLOYEE MASTER DATA (Complete!)

**Primary Table**: `employee`
**Demographic Data**:
- Position, department, tenure
- Job level, grade
- Employment status
- Approval line (manager hierarchy)

---

## Critical Findings

### 🎉 EXCELLENT NEWS!

1. **Complete Multi-Dimensional Data Available**:
   - Performance/OKR ✅
   - Competency ✅
   - Behavioral ✅ (from Excel)
   - Talent Assessment ✅
   - Feedback/Coaching ✅

2. **TARGET VARIABLE EXISTS**:
   - `employee_promotion_request` - Promotion history
   - `employee_transfer_request` - Career moves
   - `grade` changes - Grade progression

3. **Rich Historical Data**:
   - `pa_periode` - Multiple performance periods
   - Time-series data available
   - Can track progression over time

4. **Multi-Rater Assessment**:
   - `talent_scorecard_assesment_summary` has scores from multiple approval lines
   - 360-degree feedback capability

### 🎯 Recommended MPCIM Dimensions

Based on database analysis:

1. **Performance Dimension**: 
   - Source: `performance_contract`, `pa_okr_objective`
   - Metrics: OKR achievement, KPI scores

2. **Competency Dimension**:
   - Source: `employee_competency_assesment`
   - Metrics: Technical/functional competency scores

3. **Behavioral Dimension**:
   - Source: Excel data + `feedback_performance`
   - Metrics: Behavioral assessment scores

4. **Talent/Potential Dimension**:
   - Source: `talent_scorecard_transaction`
   - Metrics: Talent scores, potential ratings

5. **Feedback Dimension**:
   - Source: `feedback_performance`, `coaching`
   - Metrics: Feedback scores, coaching engagement

6. **Career Aspiration** (if available in forms):
   - Source: `forms`, `form_answers`
   - Check if there are career aspiration surveys

---

## Data Coverage Estimation

### Expected Coverage (Need to Query):

```sql
-- Employees with performance data
SELECT COUNT(DISTINCT employee_id) FROM performance_contract;

-- Employees with competency data
SELECT COUNT(DISTINCT employee_id) FROM employee_competency_assesment;

-- Employees with talent data
SELECT COUNT(DISTINCT employee_id) FROM talent_scorecard_transaction;

-- Employees with promotion history (TARGET)
SELECT COUNT(DISTINCT employee_id) FROM employee_promotion_request;

-- Complete coverage
SELECT 
  COUNT(DISTINCT e.id) as total_employees,
  COUNT(DISTINCT pc.employee_id) as with_performance,
  COUNT(DISTINCT eca.employee_id) as with_competency,
  COUNT(DISTINCT tst.employee_id) as with_talent,
  COUNT(DISTINCT epr.employee_id) as with_promotion
FROM employee e
LEFT JOIN performance_contract pc ON e.id = pc.employee_id
LEFT JOIN employee_competency_assesment eca ON e.id = eca.employee_id
LEFT JOIN talent_scorecard_transaction tst ON e.id = tst.employee_id
LEFT JOIN employee_promotion_request epr ON e.id = epr.employee_id;
```

---

## Predictive Analytics Targets

### Option 1: Promotion Prediction (RECOMMENDED)
**Target**: Will employee be promoted in next period?
**Label Source**: `employee_promotion_request`
**Features**: All 5+ dimensions

### Option 2: Performance Rating Prediction
**Target**: Next period performance rating
**Label Source**: `performance_contract.status_value`
**Features**: Historical performance + other dimensions

### Option 3: Talent Box Prediction
**Target**: Talent category (9-box grid)
**Label Source**: `talent_scorecard_transaction.group`
**Features**: All dimensions

### Option 4: Career Path Recommendation
**Target**: Next suitable position
**Label Source**: `employee_transfer_request`, `job_position`
**Features**: All dimensions + job requirements

---

## Revised Research Framework

### Research Questions

**RQ1**: How do multi-dimensional assessments (performance, competency, behavioral, talent, feedback) correlate with career progression?

**RQ2**: Can we accurately predict employee promotion/career advancement using integrated multi-dimensional data?

**RQ3**: Which dimensions contribute most significantly to career progression prediction?

**RQ4**: How can we provide explainable career recommendations based on multi-dimensional assessment?

### Methodology

1. **Data Integration**:
   - Extract data from 5+ dimensions
   - Join on employee_id
   - Handle temporal aspects (multiple periods)

2. **Feature Engineering**:
   - Aggregate scores per dimension
   - Calculate trends (improvement over time)
   - Derive composite metrics

3. **Target Variable Creation**:
   - Binary: Promoted (Yes/No) in next period
   - Multi-class: Career move type (promotion/lateral/no change)
   - Regression: Time to next promotion

4. **Model Development**:
   - Baseline: Single dimension (performance only)
   - MPCIM: All dimensions integrated
   - Compare accuracy improvement

5. **Explainability**:
   - SHAP values for feature importance
   - Identify key drivers per dimension
   - Generate actionable recommendations

---

## Implementation Roadmap

### Phase 1: Data Extraction & Integration (Week 1-2)
- [ ] Connect to database
- [ ] Extract all relevant tables
- [ ] Perform data quality analysis
- [ ] Create integrated dataset
- [ ] Handle missing values

### Phase 2: Exploratory Analysis (Week 2-3)
- [ ] Analyze coverage per dimension
- [ ] Check correlations between dimensions
- [ ] Identify data quality issues
- [ ] Validate target variable distribution

### Phase 3: Feature Engineering (Week 3-4)
- [ ] Aggregate dimension scores
- [ ] Create temporal features
- [ ] Normalize/standardize scores
- [ ] Handle categorical variables

### Phase 4: Model Development (Week 4-8)
- [ ] Split train/test sets
- [ ] Train baseline models
- [ ] Train MPCIM models
- [ ] Hyperparameter tuning
- [ ] Cross-validation

### Phase 5: Evaluation & Explainability (Week 8-10)
- [ ] Compare model performance
- [ ] SHAP analysis
- [ ] Generate insights
- [ ] Create visualizations

### Phase 6: Prototype Development (Week 10-16)
- [ ] Design UI/UX
- [ ] Implement backend API
- [ ] Integrate ML model
- [ ] Build dashboard
- [ ] User testing

### Phase 7: Validation & Writing (Week 16-24)
- [ ] Expert validation
- [ ] Case studies
- [ ] Write thesis
- [ ] Prepare presentation

---

## Next Steps - URGENT

### 1. Database Access
**Need**: Connection credentials to query actual data
**Purpose**: 
- Verify data coverage
- Check data quality
- Estimate sample size

### 2. Data Extraction
**Tables to Export**:
- `employee` (master data)
- `performance_contract` + `performance_contract_item`
- `employee_competency_assesment` + items
- `talent_scorecard_transaction`
- `feedback_performance`
- `employee_promotion_request` (TARGET!)
- `employee_transfer_request`
- `grade`, `job_level`, `job_position`

### 3. Preliminary Analysis
- Count employees per dimension
- Check overlap/coverage
- Identify data gaps
- Estimate feasibility

---

## Advantages Over Initial Excel Data

| Aspect | Excel Data | Database |
|--------|------------|----------|
| **Dimensions** | 3 (KPI, Behavior, Functional) | 5+ (+ Talent, Feedback, Coaching) |
| **Coverage** | 32 employees with complete data | Potentially 100s-1000s |
| **Target Variable** | ❌ None | ✅ Promotion, Transfer, Grade |
| **Historical Data** | ❌ Single point | ✅ Multiple periods |
| **Demographic** | ❌ Limited | ✅ Complete (position, level, tenure) |
| **Multi-Rater** | ❌ No | ✅ Yes (talent scorecard) |

---

## Conclusion

**The database contains EVERYTHING you need for a robust MPCIM thesis!**

✅ Multi-dimensional assessment data
✅ Target variables for prediction
✅ Historical/temporal data
✅ Rich demographic information
✅ Potentially large sample size

**This is FAR BETTER than the initial Excel data.**

**Critical Next Step**: Get database access to verify actual data coverage and quality.
