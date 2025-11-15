# Panduan Export Data MPCIM - Step by Step

## Persiapan

### Yang Anda Butuhkan:
1. ✅ Akses ke database PostgreSQL (sudah ada)
2. ✅ PostgreSQL client `psql` (sudah terinstall)
3. ✅ Username dan password database
4. ✅ Nama database

---

## Cara 1: Menggunakan Script Otomatis (RECOMMENDED)

### Step 1: Edit Konfigurasi Database

Buka file script:
```bash
nano /tmp/export_mpcim_data.sh
```

Edit baris berikut sesuai database Anda:
```bash
DB_HOST="localhost"           # Ganti jika database di server lain
DB_PORT="5432"                # Port PostgreSQL (default 5432)
DB_NAME="your_database_name"  # GANTI dengan nama database Anda
DB_USER="your_username"       # GANTI dengan username Anda
```

Simpan dengan: `Ctrl+O`, `Enter`, `Ctrl+X`

### Step 2: Buat Script Executable

```bash
chmod +x /tmp/export_mpcim_data.sh
```

### Step 3: Jalankan Script

```bash
/tmp/export_mpcim_data.sh
```

Script akan meminta password database Anda (untuk setiap query).

### Step 4: Hasil Export

File akan tersimpan di: `/tmp/mpcim_export/`

File yang dihasilkan:
- `employee_master.csv` - Data demografis karyawan (anonymized)
- `performance.csv` - Data performance/OKR
- `competency.csv` - Data competency assessment
- `talent.csv` - Data talent scorecard
- `promotion_history.csv` - Riwayat promosi (TARGET!)
- `job_positions.csv` - Referensi posisi
- `job_levels.csv` - Referensi level
- `grades.csv` - Referensi grade

---

## Cara 2: Manual Export (Jika Script Bermasalah)

### Step 1: Connect ke Database

```bash
psql -h localhost -p 5432 -U your_username -d your_database_name
```

Masukkan password saat diminta.

### Step 2: Cek Data Coverage Dulu

Copy-paste query ini di psql:

```sql
SELECT 
  COUNT(DISTINCT e.id) as total_employees,
  COUNT(DISTINCT pc.employee_id) as with_performance,
  COUNT(DISTINCT eca.employee_id) as with_competency,
  COUNT(DISTINCT tst.employee_id) as with_talent,
  COUNT(DISTINCT epr.employee_id) as with_promotion_history
FROM employee e
LEFT JOIN performance_contract pc ON e.id = pc.employee_id
LEFT JOIN employee_competency_assesment eca ON e.id = eca.employee_id
LEFT JOIN talent_scorecard_transaction tst ON e.id = tst.employee_id
LEFT JOIN employee_promotion_request epr ON e.id = epr.employee_id;
```

Catat hasilnya! Ini penting untuk tahu berapa banyak data yang tersedia.

### Step 3: Export Data Satu Per Satu

Buat folder dulu:
```bash
mkdir -p /tmp/mpcim_export
```

Lalu di psql, jalankan query export:

#### Export 1: Employee Master
```sql
\copy (
  SELECT 
    MD5(e.id::text) as employee_id_hash,
    e.job_position_id,
    e.job_level_id,
    e.grade_id,
    EXTRACT(YEAR FROM AGE(CURRENT_DATE, e.join_date)) as tenure_years,
    CASE 
      WHEN e.gender = 'male' THEN 'M'
      WHEN e.gender = 'female' THEN 'F'
      ELSE 'O'
    END as gender
  FROM employee e
  WHERE e.id IS NOT NULL
) TO '/tmp/mpcim_export/employee_master.csv' WITH CSV HEADER;
```

#### Export 2: Performance Data
```sql
\copy (
  SELECT 
    MD5(pc.employee_id::text) as employee_id_hash,
    pc.final_result as performance_score,
    pc.status_value as performance_rating,
    pc.created_at::date as assessment_date
  FROM performance_contract pc
  WHERE pc.employee_id IS NOT NULL
) TO '/tmp/mpcim_export/performance.csv' WITH CSV HEADER;
```

#### Export 3: Competency Data
```sql
\copy (
  SELECT 
    MD5(eca.employee_id::text) as employee_id_hash,
    eca.final_result as competency_score,
    eca.created_at::date as assessment_date
  FROM employee_competency_assesment eca
  WHERE eca.employee_id IS NOT NULL
) TO '/tmp/mpcim_export/competency.csv' WITH CSV HEADER;
```

#### Export 4: Talent Data
```sql
\copy (
  SELECT 
    MD5(tst.employee_id::text) as employee_id_hash,
    tst.final_score as talent_score,
    tst."group" as talent_category,
    tst.created_at::date as assessment_date
  FROM talent_scorecard_transaction tst
  WHERE tst.employee_id IS NOT NULL
) TO '/tmp/mpcim_export/talent.csv' WITH CSV HEADER;
```

#### Export 5: Promotion History (PENTING!)
```sql
\copy (
  SELECT 
    MD5(epr.employee_id::text) as employee_id_hash,
    epr.created_at::date as promotion_date,
    epr.status,
    EXTRACT(YEAR FROM epr.created_at) as promotion_year
  FROM employee_promotion_request epr
  WHERE epr.employee_id IS NOT NULL
) TO '/tmp/mpcim_export/promotion_history.csv' WITH CSV HEADER;
```

### Step 4: Keluar dari psql
```sql
\q
```

---

## Cara 3: Menggunakan Database GUI Tool

Jika Anda lebih nyaman dengan GUI:

### Menggunakan DBeaver / pgAdmin / TablePlus:

1. **Connect ke database**
2. **Buka SQL Editor**
3. **Copy-paste query** dari file `/tmp/mpcim_export_queries.sql`
4. **Run query** satu per satu
5. **Export hasil** ke CSV

---

## Verifikasi Hasil Export

Setelah export selesai, cek file:

```bash
ls -lh /tmp/mpcim_export/
```

Cek isi file (5 baris pertama):
```bash
head -5 /tmp/mpcim_export/employee_master.csv
head -5 /tmp/mpcim_export/performance.csv
head -5 /tmp/mpcim_export/promotion_history.csv
```

---

## Keamanan Data

### ✅ Yang Sudah Di-Anonymize:
- `employee_id` → Diganti dengan MD5 hash
- Tidak ada nama, email, NIK
- Tidak ada data gaji/salary
- Hanya data assessment scores

### ⚠️ Pastikan:
- File CSV tidak di-commit ke Git public
- Simpan di folder secure
- Hapus setelah selesai analisis

---

## Troubleshooting

### Error: "permission denied"
**Solusi**: Pastikan folder `/tmp/mpcim_export` bisa ditulis:
```bash
chmod 777 /tmp/mpcim_export
```

### Error: "relation does not exist"
**Solusi**: Nama tabel mungkin berbeda. Cek dengan:
```sql
\dt
```

### Error: "password authentication failed"
**Solusi**: Cek username dan password Anda

### Error: "column does not exist"
**Solusi**: Beberapa kolom mungkin tidak ada di database Anda. Skip query tersebut.

---

## Next Steps Setelah Export

1. **Zip file** untuk kemudahan transfer:
   ```bash
   cd /tmp
   zip -r mpcim_export.zip mpcim_export/
   ```

2. **Share file** ke saya untuk analisis

3. **Saya akan analisis**:
   - Data coverage
   - Data quality
   - Missing values
   - Correlation analysis
   - Feasibility untuk ML model

---

## Kontak

Jika ada masalah saat export, share:
1. Error message yang muncul
2. Query yang bermasalah
3. Screenshot jika perlu

Saya akan bantu troubleshoot!
