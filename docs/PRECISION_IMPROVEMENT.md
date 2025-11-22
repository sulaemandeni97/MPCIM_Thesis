# 🎯 Precision Improvement - Mengurangi False Positives

## 📊 Problem Statement

Berdasarkan analisis Gemini AI, model Neural Network menunjukkan:
- ✅ **Accuracy tinggi**: 90.91%
- ❌ **Precision rendah**: 50.00%
- ⚠️ **False Positives tinggi**: Model cenderung memprediksi promosi yang salah

### Apa itu False Positive?

**False Positive** = Model memprediksi karyawan akan **DIPROMOSIKAN**, padahal kenyataannya **TIDAK DIPROMOSIKAN**

### Dampak Bisnis False Positive

1. **Harapan Palsu**: Karyawan diberi ekspektasi yang salah
2. **Salah Alokasi Resources**: Training/development ke kandidat yang salah
3. **Perencanaan Suksesi Buruk**: Kandidat yang salah dipersiapkan untuk posisi senior
4. **Demotivasi**: Karyawan kecewa ketika ekspektasi tidak terpenuhi
5. **Kehilangan Trust**: HR kehilangan kepercayaan pada sistem prediksi

---

## 🔧 Solusi yang Diimplementasikan

### 1. Threshold Adjustment (Teknik Utama)

**Konsep:**
- Default threshold klasifikasi = **0.50** (50%)
- Jika `probability >= 0.50` → Prediksi PROMOTED
- Jika `probability < 0.50` → Prediksi NOT PROMOTED

**Masalah dengan threshold 0.50:**
- Terlalu "optimistic"
- Menghasilkan banyak false positives (12 kasus)
- Precision hanya 36.84%

**Solusi:**
- Naikkan threshold ke **0.70** (70%)
- Jika `probability >= 0.70` → Prediksi PROMOTED
- Jika `probability < 0.70` → Prediksi NOT PROMOTED

**Hasil:**
```
Metric                  Default (0.5)    Optimized (0.7)    Improvement
─────────────────────────────────────────────────────────────────────
Precision               36.84%           43.75%             +6.91%
False Positives         12               9                  -3 (-25%)
True Positives          7                7                  0
Accuracy                87.41%           89.51%             +2.10%
```

### 2. Implementasi di Kode

#### A. Update `prediction_service.py`

```python
@staticmethod
def _predict(model, scaled_features: np.ndarray, threshold: float = 0.70):
    """
    Make prediction with optimized threshold.
    
    Default threshold = 0.70 (optimized to reduce false positives)
    Previous default = 0.50 (resulted in 12 false positives)
    New threshold = 0.70 (reduces to 9 false positives, +6.91% precision)
    """
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(scaled_features)[0][1])
    else:
        logits = float(model.decision_function(scaled_features)[0])
        probability = 1 / (1 + np.exp(-logits))
    
    # Use optimized threshold to reduce false positives
    prediction = int(probability >= threshold)
    return prediction, probability
```

#### B. Update UI di `Prediction.py`

**Fitur baru:**
1. **Threshold Information Box**: Menjelaskan mengapa threshold 70%
2. **Confidence Indicator**: Berdasarkan jarak dari threshold
3. **Threshold Gap Metric**: Menunjukkan seberapa jauh dari threshold
4. **Borderline Warning**: Alert jika probability dekat threshold
5. **Moderate Confidence Info**: Rekomendasi untuk kasus moderate

**Contoh Warning:**
```
⚠️ Borderline Case - Probability (72%) sangat dekat dengan threshold (70%)

Rekomendasi:
- Review manual oleh HR diperlukan
- Pertimbangkan faktor kualitatif lainnya
- Monitor performa karyawan lebih ketat
```

#### C. Update Gauge Visualization

- Threshold line berubah dari **50%** → **70%**
- Color zones:
  - 🔴 Red (0-50%): Low probability
  - 🟡 Yellow (50-70%): Medium (below threshold)
  - 🟢 Green (70-100%): High (above threshold)

---

## 📈 Analisis Threshold

### Threshold Testing Results

| Threshold | Accuracy | Precision | Recall | F1-Score | False Positives |
|-----------|----------|-----------|--------|----------|-----------------|
| 0.30      | 86.01%   | 33.33%    | 53.85% | 0.4118   | 14              |
| 0.40      | 86.71%   | 35.00%    | 53.85% | 0.4242   | 13              |
| 0.50      | 87.41%   | 36.84%    | 53.85% | 0.4375   | 12              |
| 0.60      | 88.81%   | 41.18%    | 53.85% | 0.4667   | 10              |
| **0.70**  | **89.51%** | **43.75%** | **53.85%** | **0.4828** | **9** |
| 0.80      | 89.51%   | 43.75%    | 53.85% | 0.4828   | 9               |

**Mengapa 0.70?**
- ✅ Precision tertinggi (43.75%)
- ✅ F1-Score terbaik (0.4828)
- ✅ Balance optimal antara precision dan recall
- ✅ False positives minimal (9)

### Precision-Recall Trade-off

```
Threshold ↑ → Precision ↑, Recall ↓
Threshold ↓ → Precision ↓, Recall ↑
```

**Threshold 0.70 memberikan:**
- Precision yang baik (43.75%)
- Recall yang masih acceptable (53.85%)
- Trade-off optimal untuk use case HR

---

## 💡 Rekomendasi Penggunaan

### 1. Untuk HR Team

**Interpretasi Hasil:**

| Probability Range | Interpretation | Action |
|-------------------|----------------|--------|
| **90-100%** | Very High Confidence | Strong candidate, proceed with promotion planning |
| **70-90%** | High Confidence | Good candidate, consider for promotion |
| **60-70%** | Borderline | Requires manual review and additional evaluation |
| **50-60%** | Low Confidence | Not recommended, focus on development |
| **0-50%** | Very Low | Not ready for promotion |

**Best Practices:**
1. ✅ Trust predictions dengan probability > 80%
2. ⚠️ Manual review untuk probability 60-80%
3. ❌ Jangan promosikan jika probability < 60%

### 2. Untuk Data Scientists

**Monitoring:**
```python
# Track precision in production
from sklearn.metrics import precision_score

# Collect actual outcomes
y_true = [actual_promotion_outcomes]
y_pred = [model_predictions]

current_precision = precision_score(y_true, y_pred)

# Alert if precision drops below threshold
if current_precision < 0.40:
    send_alert("Model precision degraded!")
```

**Periodic Retraining:**
- Retrain model setiap 6 bulan dengan data baru
- Re-evaluate optimal threshold
- Update threshold jika distribution berubah

### 3. Untuk Developers

**API Integration:**
```python
# Example API response
{
    "prediction": 1,
    "probability": 0.75,
    "confidence": "High",
    "threshold_gap": 0.05,
    "recommendation": "Good candidate for promotion",
    "threshold_used": 0.70
}
```

---

## 📊 Visualisasi

### Threshold Analysis Chart

File: `results/precision_improvement/01_threshold_analysis.png`

**Charts included:**
1. **Metrics vs Threshold**: Shows how precision, recall, F1 change
2. **False Positives vs Threshold**: Shows FP reduction
3. **Precision-Recall Curve**: Trade-off visualization
4. **Confusion Matrix**: At optimal threshold

---

## 🔄 Alternative Approaches (Tested)

### 1. Class Weight Adjustment

**Tested:**
- `None` (default)
- `balanced`
- `{0: 1, 1: 2}` (favor positive)
- `{0: 2, 1: 1}` (penalize FP)

**Result:**
- ❌ No significant improvement
- All configurations gave similar precision (~36.84%)
- **Conclusion**: Threshold adjustment is more effective

### 2. Cost-Sensitive Learning

**Concept:**
- Assign different costs to FP vs FN
- Train model to minimize total cost

**Status:**
- Not implemented yet
- Could be future improvement

### 3. Ensemble Methods

**Concept:**
- Combine multiple models
- Use voting with precision focus

**Status:**
- Not implemented yet
- Could be future improvement

---

## 📝 Files Modified

1. **`app/services/prediction_service.py`**
   - Updated `_predict()` method
   - Changed default threshold from 0.5 → 0.70
   - Added documentation

2. **`app/pages/4_🔮_Prediction.py`**
   - Added threshold information box
   - Updated confidence calculation
   - Added borderline warnings
   - Updated gauge visualization

3. **`scripts/modeling/05_improve_precision.py`** (New)
   - Threshold analysis script
   - Generates optimization report
   - Creates visualizations

4. **`results/precision_improvement/`** (New)
   - `01_threshold_analysis.png`: Visualization
   - `threshold_results.csv`: Detailed results
   - `optimized_neural_network.pkl`: Model with optimal threshold

---

## 🎯 Impact Summary

### Before Optimization
- Threshold: 0.50
- Precision: 36.84%
- False Positives: 12
- **Problem**: Too many wrong promotion predictions

### After Optimization
- Threshold: 0.70
- Precision: 43.75%
- False Positives: 9
- **Benefit**: More accurate, fewer false hopes

### Business Value
- 💰 **Cost Savings**: Reduced wasted training/development resources
- 😊 **Employee Satisfaction**: Fewer disappointed employees
- 📈 **Better Planning**: More accurate succession planning
- 🎯 **Increased Trust**: HR has more confidence in predictions

---

## 🚀 Next Steps

### Short Term
1. ✅ Monitor precision in production
2. ✅ Collect feedback from HR on predictions
3. ✅ Track false positive rate

### Medium Term
1. 🔄 Implement A/B testing (0.70 vs 0.75 threshold)
2. 🔄 Add confidence intervals to predictions
3. 🔄 Create dashboard for model monitoring

### Long Term
1. 📊 Implement cost-sensitive learning
2. 📊 Explore ensemble methods
3. 📊 Add explainability features (SHAP values)

---

## 📚 References

1. **Precision-Recall Trade-off**: [Scikit-learn Documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
2. **Threshold Optimization**: [Towards Data Science Article](https://towardsdatascience.com/optimal-threshold-for-imbalanced-classification-5884e870c293)
3. **Cost-Sensitive Learning**: [Research Paper](https://www.sciencedirect.com/science/article/pii/S0957417416302056)

---

**Last Updated**: November 22, 2025  
**Author**: Deni Sulaeman  
**Version**: 1.0
