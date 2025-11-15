-- ============================================================================
-- CONTOH ANONYMIZATION TECHNIQUES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- TECHNIQUE 1: MD5 HASHING (Paling Umum)
-- Mengubah employee_id menjadi hash yang tidak bisa di-reverse
-- ----------------------------------------------------------------------------

SELECT 
  -- Original ID (JANGAN EXPORT INI!)
  -- id as employee_id,
  
  -- Anonymized ID (EXPORT INI!)
  MD5(id::text) as employee_id_hash,
  
  -- Data yang AMAN untuk di-export (tidak ada PII)
  job_position_id,
  job_level_id,
  grade_id,
  EXTRACT(YEAR FROM AGE(CURRENT_DATE, join_date)) as tenure_years,
  
  -- Gender di-generalize
  CASE 
    WHEN gender = 'male' THEN 'M'
    WHEN gender = 'female' THEN 'F'
    ELSE 'O'
  END as gender
  
  -- JANGAN EXPORT FIELD INI:
  -- name, email, phone, address, nik, ktp, etc.
  
FROM employee
LIMIT 5;

-- ----------------------------------------------------------------------------
-- TECHNIQUE 2: RANDOM ID (Alternatif MD5)
-- Menggunakan UUID random
-- ----------------------------------------------------------------------------

SELECT 
  gen_random_uuid() as employee_id_random,
  job_position_id,
  grade_id
FROM employee
LIMIT 5;

-- ----------------------------------------------------------------------------
-- TECHNIQUE 3: SEQUENTIAL NUMBERING
-- Ganti dengan nomor urut sederhana
-- ----------------------------------------------------------------------------

SELECT 
  ROW_NUMBER() OVER (ORDER BY id) as employee_number,
  job_position_id,
  grade_id
FROM employee
LIMIT 5;

-- ----------------------------------------------------------------------------
-- TECHNIQUE 4: GENERALIZATION (Untuk Data Numerik)
-- Bulatkan atau kelompokkan data sensitif
-- ----------------------------------------------------------------------------

SELECT 
  MD5(id::text) as employee_id_hash,
  
  -- Gaji di-generalize ke range (jika ada)
  CASE 
    WHEN salary < 5000000 THEN 'Low'
    WHEN salary BETWEEN 5000000 AND 10000000 THEN 'Medium'
    WHEN salary > 10000000 THEN 'High'
  END as salary_range,
  
  -- Umur di-generalize ke kelompok
  CASE 
    WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) < 25 THEN '< 25'
    WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) BETWEEN 25 AND 35 THEN '25-35'
    WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, birth_date)) BETWEEN 36 AND 45 THEN '36-45'
    ELSE '> 45'
  END as age_group
  
FROM employee
LIMIT 5;

-- ----------------------------------------------------------------------------
-- TECHNIQUE 5: SUPPRESSION (Hapus Data Sensitif)
-- Jangan export kolom yang sensitif sama sekali
-- ----------------------------------------------------------------------------

-- JANGAN EXPORT:
-- - name, first_name, last_name
-- - email, phone, mobile
-- - address, city, postal_code
-- - nik, ktp, passport
-- - bank_account, npwp
-- - salary, bonus (kecuali di-generalize)
-- - birth_date (gunakan age atau age_group)
-- - photo, signature

-- BOLEH EXPORT:
-- - Scores (performance, competency, talent)
-- - Job level, grade, position
-- - Tenure (tahun kerja)
-- - Gender (M/F/O)
-- - Department, division (jika tidak terlalu spesifik)

-- ============================================================================
-- COMPLETE ANONYMIZED EXPORT EXAMPLE
-- ============================================================================

\copy (
  SELECT 
    -- Anonymized ID
    MD5(e.id::text) as employee_id_hash,
    
    -- Safe demographic data
    e.job_position_id,
    e.job_level_id,
    e.grade_id,
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.join_date)) as tenure_years,
    CASE 
      WHEN e.gender = 'male' THEN 'M'
      WHEN e.gender = 'female' THEN 'F'
      ELSE 'O'
    END as gender,
    
    -- Performance data (safe - just scores)
    pc.final_result as performance_score,
    pc.status_value as performance_rating,
    
    -- Competency data (safe - just scores)
    eca.final_result as competency_score,
    
    -- Talent data (safe - just scores)
    tst.final_score as talent_score,
    tst."group" as talent_category,
    
    -- Target variable (safe - just boolean)
    CASE 
      WHEN epr.id IS NOT NULL THEN 1 
      ELSE 0 
    END as has_promotion,
    
    -- Dates (safe - just year or aggregated)
    EXTRACT(YEAR FROM pc.created_at) as performance_year,
    EXTRACT(YEAR FROM eca.created_at) as competency_year
    
  FROM employee e
  LEFT JOIN performance_contract pc ON e.id = pc.employee_id
  LEFT JOIN employee_competency_assesment eca ON e.id = eca.employee_id
  LEFT JOIN talent_scorecard_transaction tst ON e.id = tst.employee_id
  LEFT JOIN employee_promotion_request epr ON e.id = epr.employee_id
  WHERE e.id IS NOT NULL
) TO '/tmp/mpcim_export/anonymized_data.csv' WITH CSV HEADER;

-- ============================================================================
-- VERIFICATION: Check Anonymization
-- ============================================================================

-- Setelah export, verifikasi tidak ada data sensitif:
-- 1. Buka CSV file
-- 2. Pastikan TIDAK ADA kolom: name, email, phone, address, nik, salary
-- 3. Pastikan employee_id sudah dalam bentuk hash
-- 4. Pastikan hanya ada scores dan categorical data

-- ============================================================================
