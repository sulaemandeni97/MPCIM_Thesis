"""
Page Analysis Service
Provides AI-powered analysis for different pages using Gemini AI
"""

import os
from typing import Dict, Any, Optional
import pandas as pd


class PageAnalysisService:
    """
    Service for analyzing page data using Gemini AI.
    Provides interpretations for Data Explorer, EDA Results, and Model Performance.
    """
    
    def __init__(self):
        """Initialize Gemini AI service."""
        self.api_key = os.getenv('GEMINI_API_KEY')
        self.enabled = False
        
        if self.api_key and self.api_key != 'your_gemini_api_key_here':
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.enabled = True
            except Exception as e:
                print(f"⚠️ Failed to initialize Gemini: {e}")
                self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if AI analysis is enabled."""
        return self.enabled
    
    def analyze_data_explorer(self, df: pd.DataFrame, stats: Dict[str, Any]) -> str:
        """
        Analyze Data Explorer statistics and provide insights.
        
        Args:
            df: Dataset DataFrame
            stats: Dictionary with dataset statistics
        
        Returns:
            str: AI-generated analysis
        """
        if not self.enabled:
            return self._fallback_data_explorer_analysis(stats)
        
        prompt = f"""Sebagai HR Analytics Expert, analisis dataset karyawan berikut:

**Dataset Overview:**
- Total Karyawan: {stats.get('total_rows', 0):,}
- Jumlah Fitur: {stats.get('total_columns', 0)}
- Tingkat Promosi: {stats.get('promotion_rate', 0):.1f}%
- Promoted: {stats.get('promoted_count', 0)} karyawan
- Not Promoted: {stats.get('not_promoted_count', 0)} karyawan

**Performance Metrics:**
- Rata-rata Performance Score: {stats.get('avg_performance', 0):.2f}
- Rata-rata Behavioral Score: {stats.get('avg_behavioral', 0):.2f}
- Rata-rata Tenure: {stats.get('avg_tenure', 0):.1f} tahun

**Quick Assessment Coverage:**
- Coverage: {stats.get('qa_coverage', 0):.1f}%
- Karyawan dengan QA: {stats.get('qa_count', 0)}

Berikan analisis dalam format berikut:

## 📊 Ringkasan Dataset
[Ringkasan singkat tentang karakteristik dataset]

## 🎯 Temuan Utama
- [3-4 insight penting dari data]

## ⚠️ Perhatian Khusus
- [Area yang perlu diperhatikan]

## 💡 Rekomendasi
- [2-3 rekomendasi untuk HR]

Gunakan bahasa Indonesia yang profesional dan mudah dipahami."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini analysis error: {e}")
            return self._fallback_data_explorer_analysis(stats)
    
    def analyze_eda_results(self, stats: Dict[str, Any]) -> str:
        """
        Analyze EDA results and provide statistical insights.
        
        Args:
            stats: Dictionary with EDA statistics
        
        Returns:
            str: AI-generated analysis
        """
        if not self.enabled:
            return self._fallback_eda_analysis(stats)
        
        prompt = f"""Sebagai Data Scientist, analisis hasil Exploratory Data Analysis (EDA) berikut:

**Distribusi Promosi:**
- Promoted: {stats.get('promoted_pct', 0):.1f}%
- Not Promoted: {stats.get('not_promoted_pct', 0):.1f}%
- Imbalance Ratio: {stats.get('imbalance_ratio', 0):.1f}:1

**Perbandingan Promoted vs Not Promoted:**

Performance Scores:
- Promoted: {stats.get('promoted_perf_mean', 0):.2f} (±{stats.get('promoted_perf_std', 0):.2f})
- Not Promoted: {stats.get('not_promoted_perf_mean', 0):.2f} (±{stats.get('not_promoted_perf_std', 0):.2f})

Behavioral Scores:
- Promoted: {stats.get('promoted_behav_mean', 0):.2f} (±{stats.get('promoted_behav_std', 0):.2f})
- Not Promoted: {stats.get('not_promoted_behav_mean', 0):.2f} (±{stats.get('not_promoted_behav_std', 0):.2f})

Psychological Scores (QA):
- Promoted: {stats.get('promoted_psych_mean', 0):.2f} (±{stats.get('promoted_psych_std', 0):.2f})
- Not Promoted: {stats.get('not_promoted_psych_mean', 0):.2f} (±{stats.get('not_promoted_psych_std', 0):.2f})

**Korelasi dengan Promosi:**
- Performance Score: {stats.get('corr_performance', 0):.3f}
- Behavioral Score: {stats.get('corr_behavioral', 0):.3f}
- Psychological Score: {stats.get('corr_psychological', 0):.3f}

Berikan analisis dalam format:

## 📈 Analisis Distribusi
[Interpretasi distribusi dan imbalance]

## 🔍 Perbedaan Kelompok
[Analisis perbedaan antara promoted dan not promoted]

## 🔗 Analisis Korelasi
[Interpretasi korelasi dan feature importance]

## 🎯 Insight Statistik
- [3-4 temuan statistik penting]

## 💡 Implikasi untuk Model
- [Rekomendasi untuk modeling]

Gunakan bahasa Indonesia yang profesional."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini analysis error: {e}")
            return self._fallback_eda_analysis(stats)
    
    def analyze_model_performance(self, model_results: Dict[str, Any]) -> str:
        """
        Analyze model performance metrics and provide insights.
        
        Args:
            model_results: Dictionary with model performance metrics
        
        Returns:
            str: AI-generated analysis
        """
        if not self.enabled:
            return self._fallback_model_analysis(model_results)
        
        # Get best model
        best_model = model_results.get('best_model', {})
        
        prompt = f"""Sebagai Machine Learning Expert, analisis performa model berikut:

**Model Terbaik: {best_model.get('name', 'Unknown')}**

Metrics:
- Accuracy: {best_model.get('accuracy', 0):.4f} ({best_model.get('accuracy', 0)*100:.2f}%)
- Precision: {best_model.get('precision', 0):.4f}
- Recall: {best_model.get('recall', 0):.4f}
- F1-Score: {best_model.get('f1_score', 0):.4f}
- ROC-AUC: {best_model.get('roc_auc', 0):.4f}

**Perbandingan Model:**
{self._format_model_comparison(model_results.get('all_models', []))}

**Feature Importance (Top 5):**
{self._format_feature_importance(model_results.get('feature_importance', []))}

**Quick Assessment Contribution:**
- QA Features dalam Top 10: {model_results.get('qa_in_top10', 0)}
- Total Kontribusi QA: {model_results.get('qa_contribution', 0):.1f}%

Berikan analisis dalam format:

## 🏆 Performa Model Terbaik
[Evaluasi performa model terbaik]

## 📊 Analisis Metrics
[Interpretasi accuracy, precision, recall, F1, ROC-AUC]

## 🔍 Perbandingan Model
[Analisis perbandingan antar model]

## 🎯 Feature Importance
[Interpretasi fitur-fitur penting]

## 🧠 Kontribusi Quick Assessment
[Analisis dampak QA features]

## ⚠️ Limitasi & Perhatian
- [Potensi limitasi model]

## 💡 Rekomendasi
- [Saran untuk improvement]

Gunakan bahasa Indonesia yang profesional."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini analysis error: {e}")
            return self._fallback_model_analysis(model_results)
    
    def _format_model_comparison(self, models: list) -> str:
        """Format model comparison for prompt."""
        if not models:
            return "Data tidak tersedia"
        
        lines = []
        for model in models[:5]:  # Top 5 models
            name = model.get('name', 'Unknown')
            acc = model.get('accuracy', 0)
            f1 = model.get('f1_score', 0)
            lines.append(f"- {name}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        return "\n".join(lines)
    
    def _format_feature_importance(self, features: list) -> str:
        """Format feature importance for prompt."""
        if not features:
            return "Data tidak tersedia"
        
        lines = []
        for i, feat in enumerate(features[:5], 1):
            name = feat.get('feature', 'Unknown')
            importance = feat.get('importance', 0)
            lines.append(f"{i}. {name}: {importance:.4f}")
        
        return "\n".join(lines)
    
    def _fallback_data_explorer_analysis(self, stats: Dict[str, Any]) -> str:
        """Fallback analysis when Gemini is not available."""
        return f"""## 📊 Ringkasan Dataset

Dataset ini berisi **{stats.get('total_rows', 0):,} karyawan** dengan **{stats.get('total_columns', 0)} fitur**. Tingkat promosi adalah **{stats.get('promotion_rate', 0):.1f}%**, menunjukkan {"distribusi yang seimbang" if 30 <= stats.get('promotion_rate', 0) <= 70 else "class imbalance"}.

## 🎯 Temuan Utama

- **Performance**: Rata-rata score {stats.get('avg_performance', 0):.2f} menunjukkan performa {"baik" if stats.get('avg_performance', 0) >= 75 else "perlu ditingkatkan"}
- **Behavioral**: Score rata-rata {stats.get('avg_behavioral', 0):.2f} mengindikasikan kompetensi perilaku yang {"solid" if stats.get('avg_behavioral', 0) >= 75 else "perlu pengembangan"}
- **Quick Assessment**: Coverage {stats.get('qa_coverage', 0):.1f}% memberikan insight psikologis untuk {"mayoritas" if stats.get('qa_coverage', 0) >= 80 else "sebagian"} karyawan

## 💡 Rekomendasi

- Fokus pada pengembangan karyawan dengan performance score di bawah rata-rata
- Tingkatkan coverage Quick Assessment untuk analisis yang lebih komprehensif
- Monitor tenure vs promotion rate untuk mengidentifikasi career progression patterns

*💡 Aktifkan Gemini AI untuk analisis yang lebih mendalam*"""
    
    def _fallback_eda_analysis(self, stats: Dict[str, Any]) -> str:
        """Fallback EDA analysis when Gemini is not available."""
        return f"""## 📈 Analisis Distribusi

Dataset menunjukkan distribusi **{stats.get('promoted_pct', 0):.1f}% promoted** vs **{stats.get('not_promoted_pct', 0):.1f}% not promoted** dengan imbalance ratio **{stats.get('imbalance_ratio', 0):.1f}:1**.

## 🔍 Perbedaan Kelompok

Karyawan yang dipromosikan memiliki:
- Performance score **{stats.get('promoted_perf_mean', 0) - stats.get('not_promoted_perf_mean', 0):.2f} poin lebih tinggi**
- Behavioral score **{stats.get('promoted_behav_mean', 0) - stats.get('not_promoted_behav_mean', 0):.2f} poin lebih tinggi**
- Psychological score **{stats.get('promoted_psych_mean', 0) - stats.get('not_promoted_psych_mean', 0):.2f} poin lebih tinggi**

## 🎯 Insight Statistik

- Korelasi terkuat dengan promosi: {"Performance" if abs(stats.get('corr_performance', 0)) > abs(stats.get('corr_behavioral', 0)) else "Behavioral"} Score
- Semua dimensi (Performance, Behavioral, Psychological) menunjukkan perbedaan signifikan
- Model ML berpotensi memberikan prediksi yang akurat

*💡 Aktifkan Gemini AI untuk analisis statistik yang lebih komprehensif*"""
    
    def _fallback_model_analysis(self, model_results: Dict[str, Any]) -> str:
        """Fallback model analysis when Gemini is not available."""
        best_model = model_results.get('best_model', {})
        
        return f"""## 🏆 Performa Model Terbaik

**{best_model.get('name', 'Unknown')}** mencapai accuracy **{best_model.get('accuracy', 0)*100:.2f}%** dengan F1-Score **{best_model.get('f1_score', 0):.4f}**.

## 📊 Analisis Metrics

- **Precision {best_model.get('precision', 0):.4f}**: {"Baik" if best_model.get('precision', 0) >= 0.7 else "Perlu ditingkatkan"} dalam memprediksi promosi
- **Recall {best_model.get('recall', 0):.4f}**: {"Efektif" if best_model.get('recall', 0) >= 0.7 else "Perlu improvement"} dalam menangkap kandidat promosi
- **ROC-AUC {best_model.get('roc_auc', 0):.4f}**: {"Excellent" if best_model.get('roc_auc', 0) >= 0.8 else "Good" if best_model.get('roc_auc', 0) >= 0.7 else "Fair"} discriminative ability

## 🧠 Kontribusi Quick Assessment

QA features berkontribusi **{model_results.get('qa_contribution', 0):.1f}%** terhadap prediksi, menunjukkan pentingnya faktor psikologis dalam keputusan promosi.

## 💡 Rekomendasi

- Model siap untuk deployment dengan monitoring berkala
- Pertimbangkan ensemble methods untuk meningkatkan robustness
- Lakukan regular retraining dengan data terbaru

*💡 Aktifkan Gemini AI untuk analisis model yang lebih detail*"""


def create_page_analysis_service() -> PageAnalysisService:
    """Factory function to create PageAnalysisService instance."""
    return PageAnalysisService()
