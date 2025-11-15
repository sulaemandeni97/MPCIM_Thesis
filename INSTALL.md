# INSTALL & RUN (Ringkas)

Panduan singkat untuk menyiapkan environment dan menjalankan aplikasi Streamlit serta skrip analisis di repo ini (direkomendasikan untuk macOS / Apple Silicon).

## Prasyarat
- Homebrew (opsional, tapi digunakan untuk memasang Miniforge pada panduan ini)
- Internet untuk mengunduh paket

Catatan penting: Python sistem (mis. Python 3.13) dapat menyebabkan pip mencoba membangun paket berat seperti pandas dari source — ini bisa gagal di macOS/arm64. Rekomendasi: gunakan Miniforge/conda untuk mendapatkan binary prebuilt (conda-forge).

---

## 1) Pasang Miniforge (jika belum)
Jika Anda belum punya Miniforge/conda, pasang via Homebrew:

```bash
brew install --cask miniforge
```

Kemudian jalankan (di terminal zsh):

```bash
conda init zsh
# tutup dan buka ulang terminal atau jalankan:
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
```

## 2) Buat conda environment rekomendasi

```bash
conda create -n mpcim python=3.11 -y
conda activate mpcim
```

## 3) Install dependensi utama (conda-forge)

Ini akan memasang Streamlit, pandas, numpy, plotly, scipy, scikit-learn, xgboost, dan lainnya dari channel conda-forge.

```bash
conda install -n mpcim -c conda-forge streamlit pandas numpy plotly scipy scikit-learn xgboost -y
```

Jika Anda ingin menjalankan skrip EDA yang menghasilkan plot, juga pasang matplotlib & seaborn:

```bash
conda install -n mpcim -c conda-forge matplotlib seaborn -y
```

Catatan: Paket yang saya install saat setup dapat berbeda versi kecil dari file `app/requirements.txt` (yang adalah daftar pip). Menggunakan conda-forge memberi binary yang lebih stabil di macOS/arm64.

## 4) Menjalankan aplikasi Streamlit

1. Aktifkan environment:

```bash
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate mpcim
```

2. Masuk ke folder `app` dan jalankan Streamlit:

```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/app
streamlit run Home.py --server.port 8501
```

3. Buka browser di: http://localhost:8501

Untuk menghentikan server: tekan Ctrl+C di terminal tempat server berjalan.

## 5) Menjalankan skrip EDA & Analisis

Contoh menjalankan skrip EDA yang sudah ada:

```bash
conda activate mpcim
python /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/scripts/analysis/01_exploratory_data_analysis.py
```

Hasil EDA disimpan otomatis di `results/eda_plots/` dan ringkasan di `results/EDA_Summary_Report.txt`.

## 6) Lokasi data penting
- Dataset utama (yang digunakan app & skrip): `data/final/integrated_performance_behavioral.csv`

Jika data berada di path lain, update variabel `data_path` di `app/Home.py` atau di skrip yang relevan.

## 7) Troubleshooting cepat
- Jika pip install gagal pada Python 3.13 karena pandas build error, gunakan conda/miniforge (langkah di atas).
- Jika Streamlit menunjukkan peringatan CORS/XSRF, lihat `~/.streamlit/config.toml` atau jalankan Streamlit tanpa menonaktifkan XSRF.
- Jika ada modul hilang saat menjalankan skrip (mis. `ModuleNotFoundError`), pasang modul tersebut di env `mpcim` (pip atau conda).

## 8) Opsional: export environment

Setelah environment siap, rekomendasi untuk reproduksibilitas:

```bash
conda activate mpcim
conda env export --name mpcim > environment-mpcim.yml
```

File `environment-mpcim.yml` dapat dibagikan agar orang lain bisa membuat environment yang sama.

---

Jika Anda mau, saya bisa:
- membuat file `environment-mpcim.yml` sekarang, atau
- menambahkan instruksi Docker untuk containerizing app, atau
- commit file ini ke repo (saya sudah menambahkannya di root).

Teruskan instruksi apa yang Anda inginkan selanjutnya.

## 9) Menggunakan file environment minimal (direkomendasikan)

Saya juga menyertakan file `environment-mpcim-minimal.yml` yang berisi daftar paket utama tanpa build-string spesifik — ini lebih portable lintas mesin.

Untuk membuat environment dari file minimal tersebut jalankan:

```bash
conda env create -f environment-mpcim-minimal.yml
conda activate mpcim
```

Catatan:
- File minimal membiarkan conda memecahkan build yang cocok untuk platform Anda (mis. osx-arm64 vs linux-64).
- Jika Anda perlu versi yang identik dengan yang saya gunakan, gunakan `environment-mpcim.yml` (yang menyertakan build strings).

