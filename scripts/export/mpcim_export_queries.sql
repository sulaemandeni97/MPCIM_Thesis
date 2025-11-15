-- ============================================================================
-- MPCIM DATA EXPORT QUERIES (WITH ANONYMIZATION)
-- ============================================================================
-- 
-- IMPORTANT: These queries anonymize employee_id and remove sensitive data
-- Safe for research purposes
--
-- ============================================================================

-- ----------------------------------------------------------------------------
-- QUERY 1: DATA COVERAGE ANALYSIS
-- Check how many employees have data in each dimension
-- ----------------------------------------------------------------------------

SELECT 
  COUNT(DISTINCT e.id) as total_employees,
  COUNT(DISTINCT pc.employee_id) as with_performance,
  COUNT(DISTINCT eca.employee_id) as with_competency,
  COUNT(DISTINCT tst.employee_id) as with_talent,
  COUNT(DISTINCT epr.employee_id) as with_promotion_history,
  COUNT(DISTINCT etr.employee_id) as with_transfer_history
FROM employee e
LEFT JOIN performance_contract pc ON e.id = pc.employee_id
LEFT JOIN employee_competency_assesment eca ON e.id = eca.employee_id
LEFT JOIN talent_scorecard_transaction tst ON e.id = tst.employee_id
LEFT JOIN employee_promotion_request epr ON e.id = epr.employee_id
LEFT JOIN employee_transfer_request etr ON e.id = etr.employee_id;

-- ----------------------------------------------------------------------------
-- QUERY 2: EMPLOYEE MASTER DATA (ANONYMIZED)
-- Export employee demographics without sensitive info
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    MD5(e.id::text) as employee_id_hash,
    e.job_position_id,
    e.job_level_id,
    e.grade_id,
    e.employment_status_id,
    e.company_id,
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.join_date)) as tenure_years,
    CASE 
      WHEN e.gender = 'male' THEN 'M'
      WHEN e.gender = 'female' THEN 'F'
      ELSE 'O'
    END as gender,
    e.approval_line_id
  FROM employee e
  WHERE e.id IS NOT NULL
) TO '/tmp/mpcim_employee_master.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 3: PERFORMANCE DATA (ANONYMIZED)
-- Export performance contract and scores
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    MD5(pc.employee_id::text) as employee_id_hash,
    pc.pa_periode_id,
    pc.final_result as performance_score,
    pc.status_value as performance_rating,
    pc.created_at::date as assessment_date,
    COUNT(pci.id) as num_kpis,
    AVG(pci.current_value) as avg_kpi_achievement,
    SUM(pci.weight) as total_weight
  FROM performance_contract pc
  LEFT JOIN performance_contract_item pci ON pc.id = pci.performance_contract_id
  WHERE pc.employee_id IS NOT NULL
  GROUP BY pc.id, pc.employee_id, pc.pa_periode_id, pc.final_result, 
           pc.status_value, pc.created_at
) TO '/tmp/mpcim_performance.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 4: COMPETENCY DATA (ANONYMIZED)
-- Export competency assessment scores
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    MD5(eca.employee_id::text) as employee_id_hash,
    eca.job_position_id,
    eca.final_result as competency_score,
    eca.status,
    eca.created_at::date as assessment_date,
    COUNT(ecai.id) as num_competencies,
    AVG(ecai.score) as avg_competency_score
  FROM employee_competency_assesment eca
  LEFT JOIN employee_competency_assesment_items ecai 
    ON eca.id = ecai.employee_competency_assesment_id
  WHERE eca.employee_id IS NOT NULL
  GROUP BY eca.id, eca.employee_id, eca.job_position_id, 
           eca.final_result, eca.status, eca.created_at
) TO '/tmp/mpcim_competency.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 5: TALENT SCORECARD DATA (ANONYMIZED)
-- Export talent assessment scores
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    MD5(tst.employee_id::text) as employee_id_hash,
    tst.year,
    tst.final_score as talent_score,
    tst."group" as talent_category,
    tst.created_at::date as assessment_date
  FROM talent_scorecard_transaction tst
  WHERE tst.employee_id IS NOT NULL
) TO '/tmp/mpcim_talent.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 6: PROMOTION HISTORY (ANONYMIZED) - TARGET VARIABLE!
-- Export promotion history for prediction target
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    MD5(epr.employee_id::text) as employee_id_hash,
    epr.created_at::date as promotion_date,
    epr.status,
    EXTRACT(YEAR FROM epr.created_at) as promotion_year
  FROM employee_promotion_request epr
  WHERE epr.employee_id IS NOT NULL
) TO '/tmp/mpcim_promotion_history.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 7: TRANSFER HISTORY (ANONYMIZED)
-- Export transfer/career move history
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    MD5(etr.employee_id::text) as employee_id_hash,
    etr.created_at::date as transfer_date,
    etr.status,
    EXTRACT(YEAR FROM etr.created_at) as transfer_year
  FROM employee_transfer_request etr
  WHERE etr.employee_id IS NOT NULL
) TO '/tmp/mpcim_transfer_history.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 8: INTEGRATED DATASET (ANONYMIZED)
-- Export integrated multi-dimensional data
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    MD5(e.id::text) as employee_id_hash,
    
    -- Demographics (anonymized)
    e.job_position_id,
    e.job_level_id,
    e.grade_id,
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.join_date)) as tenure_years,
    CASE 
      WHEN e.gender = 'male' THEN 'M'
      WHEN e.gender = 'female' THEN 'F'
      ELSE 'O'
    END as gender,
    
    -- Performance dimension
    pc.final_result as performance_score,
    pc.status_value as performance_rating,
    
    -- Competency dimension
    eca.final_result as competency_score,
    
    -- Talent dimension
    tst.final_score as talent_score,
    tst."group" as talent_category,
    
    -- Target variable: Has promotion history?
    CASE 
      WHEN epr.id IS NOT NULL THEN 1 
      ELSE 0 
    END as has_promotion,
    
    -- Assessment dates
    pc.created_at::date as performance_date,
    eca.created_at::date as competency_date,
    tst.created_at::date as talent_date
    
  FROM employee e
  LEFT JOIN performance_contract pc ON e.id = pc.employee_id
  LEFT JOIN employee_competency_assesment eca ON e.id = eca.employee_id
  LEFT JOIN talent_scorecard_transaction tst ON e.id = tst.employee_id
  LEFT JOIN employee_promotion_request epr ON e.id = epr.employee_id
  WHERE e.id IS NOT NULL
) TO '/tmp/mpcim_integrated_dataset.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 9: JOB POSITION REFERENCE
-- Export job position mapping (for context)
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    id as job_position_id,
    name as position_name,
    job_family_id,
    parent_id
  FROM job_position
) TO '/tmp/mpcim_job_positions.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 10: JOB LEVEL REFERENCE
-- Export job level mapping
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    id as job_level_id,
    name as level_name,
    group_job_level
  FROM job_level
) TO '/tmp/mpcim_job_levels.csv' WITH CSV HEADER;

-- ----------------------------------------------------------------------------
-- QUERY 11: GRADE REFERENCE
-- Export grade mapping
-- ----------------------------------------------------------------------------

\copy (
  SELECT 
    id as grade_id,
    name as grade_name,
    band_name,
    job_level_id
  FROM grade
) TO '/tmp/mpcim_grades.csv' WITH CSV HEADER;

-- ============================================================================
-- END OF EXPORT QUERIES
-- ============================================================================
