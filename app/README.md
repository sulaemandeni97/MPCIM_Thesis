# MPCIM Dashboard - Interactive Web Application

## 📊 Overview

Aplikasi web interaktif untuk visualisasi dan analisis hasil penelitian **Multi-Dimensional Performance-Career Integration Model (MPCIM)**. Dashboard ini memungkinkan eksplorasi data, analisis model ML, dan prediksi promosi karyawan.

## 🚀 Features

### 1. 🏠 Home
- Overview penelitian MPCIM
- Quick statistics
- Panduan penggunaan aplikasi

### 2. 📊 Data Explorer
- Upload dan eksplorasi dataset
- Filter interaktif (promotion status, gender, score ranges)
- Search functionality
- Statistik deskriptif
- Visualisasi distribusi
- Correlation analysis
- Export filtered data

### 3. 📈 EDA Results
- Analisis statistik komprehensif
- T-tests dan significance testing
- Cohen's d effect size
- Correlation heatmaps
- Distribution comparisons (histogram, violin plots)
- 3D scatter plots
- Key insights dan recommendations

### 4. 🤖 Model Performance
- Perbandingan performa model ML
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix
- ROC curves
- Feature importance analysis
- Radar charts
- Model recommendations

### 5. 🔮 Prediction
- Prediksi promosi individual
- Input interaktif untuk data karyawan
- Probability gauge visualization
- Feature contribution analysis
- Recommendations berdasarkan hasil
- Batch prediction (upload CSV)
- Comparison with benchmarks

## 🛠️ Installation

### Prerequisites
- Python 3.8 atau lebih tinggi
- pip package manager

### Setup

1. **Clone atau navigate ke project directory:**
```bash
cd /Users/denisulaeman/CascadeProjects/MPCIM_Thesis/app
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Pastikan data tersedia:**
   - Data harus ada di: `/Users/denisulaeman/CascadeProjects/MPCIM_Thesis/data/final/integrated_performance_behavioral.csv`
   - Atau sesuaikan path di masing-masing file

## 🎯 Usage

### Running the Application

```bash
streamlit run Home.py
```

Aplikasi akan terbuka di browser default Anda di `http://localhost:8501`

### Navigation

Gunakan sidebar untuk navigasi antar halaman:
- **Home**: Halaman utama
- **Data Explorer**: Eksplorasi dataset
- **EDA Results**: Hasil analisis eksploratori
- **Model Performance**: Performa model ML
- **Prediction**: Tool prediksi promosi

## 📁 Project Structure

```
app/
├── Home.py                          # Main application file
├── pages/
│   ├── 1_📊_Data_Explorer.py       # Data exploration page
│   ├── 2_📈_EDA_Results.py         # EDA results page
│   ├── 3_🤖_Model_Performance.py   # Model performance page
│   └── 4_🔮_Prediction.py          # Prediction tool page
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🔧 Configuration

### Data Path
Jika data Anda berada di lokasi berbeda, update path di setiap file:

```python
data_path = Path("YOUR_PATH_HERE/integrated_performance_behavioral.csv")
```

### Model Integration
Untuk mengintegrasikan model ML yang sudah dilatih:

1. Load model di `Prediction.py`:
```python
import joblib
model = joblib.load('path/to/your/model.pkl')
```

2. Update fungsi `predict_promotion()`:
```python
def predict_promotion(features):
    # Prepare features
    X = prepare_features(features)
    
    # Make prediction
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]
    
    return prediction, probability
```

## 📊 Data Format

Dataset harus memiliki kolom berikut:

| Column | Type | Description |
|--------|------|-------------|
| `employee_id` | string | Unique identifier |
| `performance_score` | float | Performance score (0-100) |
| `behavior_avg` | float | Average behavioral score (0-100) |
| `tenure_years` | float | Years of service |
| `gender` | string | M/F |
| `marital_status` | string | Single/Married/Divorced/Widowed |
| `is_permanent` | string | t/f (true/false) |
| `has_promotion` | int | 0/1 (target variable) |

## 🎨 Customization

### Colors
Update color schemes di masing-masing file:
```python
color_discrete_map={0: '#e74c3c', 1: '#2ecc71'}  # Red for 0, Green for 1
```

### Styling
Custom CSS dapat ditambahkan di `Home.py`:
```python
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
```

## 🐛 Troubleshooting

### Port Already in Use
```bash
streamlit run Home.py --server.port 8502
```

### Module Not Found
```bash
pip install -r requirements.txt --upgrade
```

### Data Not Loading
- Periksa path file CSV
- Pastikan file memiliki format yang benar
- Check file permissions

## 📝 Notes

- **Mock Data**: Beberapa visualisasi menggunakan mock data (ROC curves, confusion matrix). Update dengan data aktual dari model Anda.
- **Model Integration**: Fungsi prediksi saat ini menggunakan rule-based logic. Integrate dengan trained model untuk production.
- **Performance**: Untuk dataset besar (>100K rows), pertimbangkan untuk menggunakan caching dan sampling.

## 🔒 Security

- **Jangan hardcode** credentials atau API keys
- Gunakan environment variables untuk sensitive data
- Implement authentication jika deploy ke production

## 📈 Future Enhancements

- [ ] User authentication
- [ ] Database integration
- [ ] Real-time model training
- [ ] Export reports to PDF
- [ ] Email notifications
- [ ] Multi-language support
- [ ] Dark mode theme

## 👨‍💻 Author

**Deni Sulaeman**  
Master Program in Information Systems  
MPCIM Thesis Research

## 📄 License

This application is part of academic research. Please contact the author for usage permissions.

---

**Last Updated**: October 22, 2025
