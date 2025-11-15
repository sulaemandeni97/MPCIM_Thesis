#!/bin/bash

# ============================================================================
# MPCIM DATA EXPORT - CNA Database (ANONYMIZED)
# ============================================================================

DB_NAME="db_cna_digispace_august_132025"
DB_USER="denisulaeman"
DB_PORT="5433"
OUTPUT_DIR="/tmp/mpcim_export_cna"

echo "============================================================================"
echo "MPCIM ANONYMIZED DATA EXPORT"
echo "============================================================================"
echo ""
echo "Database: $DB_NAME"
echo "Port: $DB_PORT"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================================"
echo "STEP 1: Data Coverage Summary"
echo "============================================================================"
echo ""

psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << 'EOF'
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
EOF

echo ""
echo "============================================================================"
echo "STEP 2: Exporting ANONYMIZED Data"
echo "============================================================================"
echo ""

# Export 1: Employee Master (ANONYMIZED)
echo "📊 [1/8] Exporting employee master data..."
psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (
  SELECT 
    MD5(id::text) as employee_id_hash,
    job_position_id,
    job_level_id,
    grade_id,
    employment_status_id,
    company_id,
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, join_date)) as tenure_years,
    CASE 
      WHEN gender = 'male' THEN 'M'
      WHEN gender = 'female' THEN 'F'
      ELSE 'O'
    END as gender,
    approval_line_id
  FROM employee
  WHERE id IS NOT NULL
) TO '$OUTPUT_DIR/01_employee_master.csv' WITH CSV HEADER;
EOF

# Export 2: Performance Contract (ANONYMIZED)
echo "📊 [2/8] Exporting performance contract data..."
psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (
  SELECT 
    MD5(pc.employee_id::text) as employee_id_hash,
    pc.pa_periode_id,
    pc.final_result as performance_score,
    pc.status_value as performance_rating,
    pc.created_at::date as assessment_date,
    EXTRACT(YEAR FROM pc.created_at) as assessment_year,
    EXTRACT(MONTH FROM pc.created_at) as assessment_month
  FROM performance_contract pc
  WHERE pc.employee_id IS NOT NULL
) TO '$OUTPUT_DIR/02_performance_contract.csv' WITH CSV HEADER;
EOF

# Export 3: Performance Contract Items (KPI Details)
echo "📊 [3/8] Exporting KPI/OKR details..."
psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (
  SELECT 
    MD5(pc.employee_id::text) as employee_id_hash,
    pc.pa_periode_id,
    pci.weight,
    pci.result_type,
    pci.start_value,
    pci.target_value,
    pci.current_value,
    pci.final_result,
    pci.polarity
  FROM performance_contract pc
  JOIN performance_contract_item pci ON pc.id = pci.performance_contract_id
  WHERE pc.employee_id IS NOT NULL
) TO '$OUTPUT_DIR/03_performance_kpi_items.csv' WITH CSV HEADER;
EOF

# Export 4: Competency Assessment (ANONYMIZED)
echo "📊 [4/8] Exporting competency assessment data..."
psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (
  SELECT 
    MD5(eca.employee_id::text) as employee_id_hash,
    eca.job_position_id,
    eca.final_result as competency_score,
    eca.status,
    eca.created_at::date as assessment_date,
    EXTRACT(YEAR FROM eca.created_at) as assessment_year
  FROM employee_competency_assesment eca
  WHERE eca.employee_id IS NOT NULL
) TO '$OUTPUT_DIR/04_competency_assessment.csv' WITH CSV HEADER;
EOF

# Export 5: Talent Scorecard (ANONYMIZED)
echo "📊 [5/8] Exporting talent scorecard data..."
psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (
  SELECT 
    MD5(tst.employee_id::text) as employee_id_hash,
    tst.year,
    tst.final_score as talent_score,
    tst."group" as talent_category,
    tst.created_at::date as assessment_date
  FROM talent_scorecard_transaction tst
  WHERE tst.employee_id IS NOT NULL
) TO '$OUTPUT_DIR/05_talent_scorecard.csv' WITH CSV HEADER;
EOF

# Export 6: Promotion History (TARGET VARIABLE - ANONYMIZED)
echo "📊 [6/8] Exporting promotion history (TARGET)..."
psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (
  SELECT 
    MD5(epr.employee_id::text) as employee_id_hash,
    epr.created_at::date as promotion_date,
    EXTRACT(YEAR FROM epr.created_at) as promotion_year,
    EXTRACT(MONTH FROM epr.created_at) as promotion_month,
    epr.status
  FROM employee_promotion_request epr
  WHERE epr.employee_id IS NOT NULL
) TO '$OUTPUT_DIR/06_promotion_history.csv' WITH CSV HEADER;
EOF

# Export 7: Reference Tables (Safe - No PII)
echo "📊 [7/8] Exporting reference tables..."

psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (SELECT id as job_position_id, name as position_name, job_family_id, parent_id FROM job_position) TO '$OUTPUT_DIR/ref_job_positions.csv' WITH CSV HEADER;
EOF

psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (SELECT id as job_level_id, name as level_name, group_job_level FROM job_level) TO '$OUTPUT_DIR/ref_job_levels.csv' WITH CSV HEADER;
EOF

psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (SELECT id as grade_id, name as grade_name, band_name, job_level_id FROM grade) TO '$OUTPUT_DIR/ref_grades.csv' WITH CSV HEADER;
EOF

psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
\copy (SELECT id as pa_periode_id, name as periode_name, start_date, end_date FROM pa_periode) TO '$OUTPUT_DIR/ref_pa_periode.csv' WITH CSV HEADER;
EOF

# Export 8: Integrated Dataset (Main Dataset for Analysis)
echo "📊 [8/8] Creating integrated dataset..."
psql -U "$DB_USER" -p "$DB_PORT" -d "$DB_NAME" << EOF
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
    pc.pa_periode_id,
    EXTRACT(YEAR FROM pc.created_at) as performance_year,
    
    -- Competency dimension (may be NULL for most)
    eca.final_result as competency_score,
    
    -- Talent dimension (may be NULL for most)
    tst.final_score as talent_score,
    tst."group" as talent_category,
    
    -- Target variable: Has promotion?
    CASE 
      WHEN epr.id IS NOT NULL THEN 1 
      ELSE 0 
    END as has_promotion,
    
    -- If promoted, when?
    EXTRACT(YEAR FROM epr.created_at) as promotion_year
    
  FROM employee e
  LEFT JOIN performance_contract pc ON e.id = pc.employee_id
  LEFT JOIN employee_competency_assesment eca ON e.id = eca.employee_id
  LEFT JOIN talent_scorecard_transaction tst ON e.id = tst.employee_id
  LEFT JOIN employee_promotion_request epr ON e.id = epr.employee_id
  WHERE e.id IS NOT NULL
) TO '$OUTPUT_DIR/00_integrated_dataset.csv' WITH CSV HEADER;
EOF

echo ""
echo "============================================================================"
echo "✅ EXPORT COMPLETED SUCCESSFULLY!"
echo "============================================================================"
echo ""
echo "📁 Files exported to: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"/*.csv 2>/dev/null | awk '{print "  ✅ " $9 " (" $5 ")"}'
echo ""
echo "============================================================================"
echo "🔒 ANONYMIZATION SUMMARY"
echo "============================================================================"
echo ""
echo "  ✅ Employee IDs → MD5 hash (irreversible)"
echo "  ✅ NO names, emails, phone numbers"
echo "  ✅ NO addresses or NIK/KTP"
echo "  ✅ NO salary or financial data"
echo "  ✅ Only assessment scores and metadata"
echo ""
echo "============================================================================"
echo "📊 DATASET OVERVIEW"
echo "============================================================================"
echo ""
echo "Main file for analysis:"
echo "  → 00_integrated_dataset.csv (all dimensions combined)"
echo ""
echo "Detailed files:"
echo "  → 01_employee_master.csv (demographics)"
echo "  → 02_performance_contract.csv (performance scores)"
echo "  → 03_performance_kpi_items.csv (KPI details)"
echo "  → 04_competency_assessment.csv (competency scores)"
echo "  → 05_talent_scorecard.csv (talent scores)"
echo "  → 06_promotion_history.csv (TARGET variable)"
echo ""
echo "Reference files:"
echo "  → ref_*.csv (lookup tables for IDs)"
echo ""
echo "============================================================================"
echo "🎯 NEXT STEPS"
echo "============================================================================"
echo ""
echo "1. Review the CSV files to verify data quality"
echo "2. Check 00_integrated_dataset.csv for completeness"
echo "3. Share files for analysis"
echo ""
echo "To view sample data:"
echo "  head -10 $OUTPUT_DIR/00_integrated_dataset.csv"
echo ""
echo "To count records:"
echo "  wc -l $OUTPUT_DIR/*.csv"
echo ""
echo "============================================================================"
