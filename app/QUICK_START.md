# 🚀 MPCIM Dashboard - Quick Start Guide

## Cara Cepat Menjalankan Aplikasi

### Opsi 1: Menggunakan Terminal (Recommended)

1. **Buka Terminal** dan navigasi ke folder app:
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/app
```

2. **Install dependencies** (hanya sekali):
```bash
pip3 install -r requirements.txt
```

3. **Jalankan aplikasi**:
```bash
streamlit run Home.py
```

4. **Buka browser** di `http://localhost:8501`

### Opsi 2: Menggunakan Script

1. **Buka Terminal**:
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/app
chmod +x run_app.sh
./run_app.sh
```

### Opsi 3: Dari IDE (VS Code/PyCharm)

1. Buka terminal di IDE
2. Pastikan virtual environment aktif (jika ada)
3. Jalankan: `streamlit run Home.py`

---

## 📋 Checklist Sebelum Menjalankan

- [ ] Python 3.8+ terinstall (`python3 --version`)
- [ ] pip terinstall (`pip3 --version`)
- [ ] Data CSV tersedia di `/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/integrated_performance_behavioral.csv`
- [ ] Dependencies terinstall (`pip3 list | grep streamlit`)

---

## 🎯 Fitur Utama yang Bisa Dicoba

### 1. Data Explorer
- Upload atau lihat data default
- Filter berdasarkan promotion status, gender, score ranges
- Search dalam kolom tertentu
- Download filtered data

### 2. EDA Results
- Lihat statistical tests (T-test, Cohen's d)
- Correlation analysis
- Distribution comparisons
- 3D scatter plots

### 3. Model Performance
- Bandingkan 4 model ML (Logistic Regression, Random Forest, XGBoost, Neural Network)
- Lihat confusion matrix
- ROC curves
- Feature importance

### 4. Prediction Tool
- Input data karyawan
- Dapatkan prediksi promosi
- Lihat probability dan confidence
- Feature contribution analysis
- Batch prediction dengan upload CSV

---

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'streamlit'"
**Solusi:**
```bash
pip3 install streamlit
```

### Error: "Address already in use"
**Solusi:** Port 8501 sudah digunakan. Gunakan port lain:
```bash
streamlit run Home.py --server.port 8502
```

### Error: "Data tidak ditemukan"
**Solusi:** Pastikan file CSV ada di lokasi yang benar atau update path di file Python:
```python
data_path = Path("YOUR_PATH/integrated_performance_behavioral.csv")
```

### Aplikasi lambat
**Solusi:**
- Gunakan dataset yang lebih kecil untuk testing
- Clear cache: Klik menu (☰) → "Clear cache"
- Restart aplikasi

---

## 📱 Akses dari Device Lain

Jika ingin akses dari smartphone/tablet di network yang sama:

1. Cari IP address komputer:
```bash
ifconfig | grep "inet "
```

2. Jalankan aplikasi dengan:
```bash
streamlit run Home.py --server.address 0.0.0.0
```

3. Akses dari device lain: `http://YOUR_IP:8501`

---

## 💾 Export & Share

### Export Visualisasi
- Setiap chart Plotly memiliki tombol camera (📷) untuk download PNG
- Klik kanan pada chart → "Save image as"

### Export Data
- Gunakan tombol "Download" di Data Explorer
- Batch prediction results bisa di-download sebagai CSV

### Share Dashboard
- Deploy ke Streamlit Cloud (gratis)
- Deploy ke Heroku
- Deploy ke AWS/GCP

---

## 🎨 Customization

### Ubah Tema
Buat file `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Ubah Port Default
```bash
streamlit run Home.py --server.port 8080
```

### Auto-reload saat development
Streamlit otomatis reload saat file berubah. Tidak perlu restart manual!

---

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python/
- **Pandas Docs**: https://pandas.pydata.org/docs/

---

## 🆘 Butuh Bantuan?

Jika mengalami masalah:

1. Check error message di terminal
2. Lihat Streamlit logs
3. Restart aplikasi
4. Clear browser cache
5. Reinstall dependencies

---

**Happy Analyzing! 🎉**
