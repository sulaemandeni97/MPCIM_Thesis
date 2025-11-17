"""
Gemini Pro Service untuk menganalisis dan mendeskripsikan hasil prediksi ML
Alternative ke OpenAI dengan Google Gemini Pro (FREE!)

Author: Deni Sulaeman
Date: November 17, 2025
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    import google.generativeai as genai
except ImportError:
    genai = None


@dataclass
class AnalysisResult:
    """Hasil analisis dari Gemini"""
    summary: str
    detailed_analysis: str
    recommendations: list[str]
    risk_factors: list[str]
    strengths: list[str]
    raw_response: str


class GeminiAnalysisService:
    """Service untuk menganalisis hasil prediksi menggunakan Google Gemini Pro"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini service
        
        Args:
            api_key: Google API key. Jika None, akan mengambil dari environment variable GEMINI_API_KEY
        """
        if genai is None:
            raise ImportError(
                "Google Generative AI library tidak terinstall. "
                "Install dengan: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key tidak ditemukan. "
                "Berikan api_key saat inisialisasi atau set environment variable GEMINI_API_KEY"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Safety settings - permissive for business analytics
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        # Use Gemini 2.5 Flash model (latest stable, fast and free)
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            safety_settings=self.safety_settings
        )
        
        # Generation config
        self.generation_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
    
    def _create_analysis_prompt(
        self,
        prediction_result: Dict[str, Any],
        employee_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Membuat prompt untuk analisis Gemini
        
        Args:
            prediction_result: Hasil prediksi dari model ML
            employee_data: Data karyawan yang diprediksi
            benchmark_data: Data benchmark untuk perbandingan
        
        Returns:
            Prompt string untuk Gemini
        """
        prompt = f"""Anda adalah seorang HR Analytics Expert yang bertugas menganalisis hasil prediksi promosi karyawan dari model Machine Learning MPCIM (Multi-Dimensional Performance-Career Integration Model).

**HASIL PREDIKSI:**
- Prediksi: {prediction_result.get('label', 'N/A')}
- Probabilitas: {prediction_result.get('probability', 0) * 100:.1f}%
- Model: {prediction_result.get('model_key', 'N/A')}

**DATA KARYAWAN:**
- Performance Score: {employee_data.get('performance_score', 'N/A')}
- Behavioral Score: {employee_data.get('behavior_avg', 'N/A')}
- Tenure (Masa Kerja): {employee_data.get('tenure_years', 'N/A')} tahun
- Gender: {employee_data.get('gender', 'N/A')}
- Status Pernikahan: {employee_data.get('marital_status', 'N/A')}
- Status Kepegawaian: {employee_data.get('is_permanent', 'N/A')}
- Performance Rating: {employee_data.get('performance_rating', 'N/A')}

**QUICK ASSESSMENT (PSYCHOLOGICAL COMPONENTS):**
- Psychological Score: {employee_data.get('psychological_score', 'N/A'):.1f}
- Drive Score: {employee_data.get('drive_score', 'N/A'):.1f}
- Mental Strength: {employee_data.get('mental_strength_score', 'N/A'):.1f}
- Adaptability: {employee_data.get('adaptability_score', 'N/A'):.1f}
- Collaboration: {employee_data.get('collaboration_score', 'N/A'):.1f}
- Has QA Data: {'Yes' if employee_data.get('has_quick_assessment', 0) == 1 else 'No (using averages)'}

**DERIVED SIGNALS:**
- Tenure Category: {prediction_result.get('derived_signals', {}).get('tenure_category', 'N/A')}
- Performance Level: {prediction_result.get('derived_signals', {}).get('performance_level', 'N/A')}
- Behavioral Level: {prediction_result.get('derived_signals', {}).get('behavioral_level', 'N/A')}
- Combined Score: {prediction_result.get('derived_signals', {}).get('combined_score', 'N/A'):.1f}
- Score Difference: {prediction_result.get('derived_signals', {}).get('score_difference', 'N/A'):.1f}
- Holistic Score: {prediction_result.get('derived_signals', {}).get('holistic_score', 'N/A'):.1f} (40% perf + 30% beh + 30% psych)
- Leadership Potential: {prediction_result.get('derived_signals', {}).get('leadership_potential', 'N/A'):.1f}
- Score Alignment: {prediction_result.get('derived_signals', {}).get('score_alignment', 'N/A'):.2f} (consistency across dimensions)
"""

        if benchmark_data:
            prompt += f"""
**BENCHMARK DATA:**
- Rata-rata Performance (Promoted): {benchmark_data.get('promoted', {}).get('performance_score', 'N/A'):.1f}
- Rata-rata Behavioral (Promoted): {benchmark_data.get('promoted', {}).get('behavior_avg', 'N/A'):.1f}
- Rata-rata Tenure (Promoted): {benchmark_data.get('promoted', {}).get('tenure_years', 'N/A'):.1f} tahun
- Rata-rata Performance (Not Promoted): {benchmark_data.get('not_promoted', {}).get('performance_score', 'N/A'):.1f}
- Rata-rata Behavioral (Not Promoted): {benchmark_data.get('not_promoted', {}).get('behavior_avg', 'N/A'):.1f}
- Rata-rata Tenure (Not Promoted): {benchmark_data.get('not_promoted', {}).get('tenure_years', 'N/A'):.1f} tahun
"""

        prompt += """
**TUGAS ANDA:**
Berikan analisis komprehensif dalam format berikut:

1. **RINGKASAN EKSEKUTIF** (2-3 kalimat)
   - Kesimpulan utama tentang prediksi promosi
   - Tingkat kepercayaan prediksi

2. **ANALISIS DETAIL**
   - Analisis performance score dan perbandingan dengan benchmark
   - Analisis behavioral score dan perbandingan dengan benchmark
   - Analisis tenure dan implikasinya (ingat: tenure paradox - karyawan junior lebih sering dipromosikan)
   - **Analisis Quick Assessment (Psychological Components)**:
     * Drive & Ambition: Motivasi dan semangat untuk berkembang
     * Mental Strength: Ketahanan mental dalam menghadapi tantangan
     * Collaboration: Kemampuan bekerja sama dalam tim
     * Adaptability: Kemampuan beradaptasi dengan perubahan
     * Leadership Potential: Potensi kepemimpinan berdasarkan kombinasi faktor psikologis
   - Analisis Holistic Score (kombinasi performance, behavioral, dan psychological)
   - Analisis Score Alignment (konsistensi across 3 dimensi)
   - Analisis kombinasi semua faktor

3. **KEKUATAN (STRENGTHS)**
   - 3-5 poin kekuatan karyawan yang mendukung promosi
   - Pertimbangkan aspek psychological jika QA data tersedia
   - Berdasarkan data aktual dan perbandingan benchmark

4. **FAKTOR RISIKO (RISK FACTORS)**
   - 2-4 poin yang mungkin menghambat promosi
   - Area yang perlu improvement

5. **REKOMENDASI ACTIONABLE**
   - 3-5 rekomendasi konkret untuk HR atau karyawan
   - Fokus pada improvement yang bisa dilakukan
   - Pertimbangkan timeline dan prioritas

**CATATAN PENTING:**
- Gunakan bahasa Indonesia yang profesional namun mudah dipahami
- Berikan insight yang actionable, bukan hanya deskripsi data
- Pertimbangkan konteks bisnis dan HR best practices
- Jika probabilitas borderline (40-60%), jelaskan ketidakpastian
- Hindari bias gender, status pernikahan, atau faktor demografis lainnya
- Fokus pada merit-based factors (performance, behavior, tenure)

Berikan analisis Anda sekarang:"""

        return prompt
    
    def analyze_prediction(
        self,
        prediction_result: Dict[str, Any],
        employee_data: Dict[str, Any],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        Menganalisis hasil prediksi menggunakan Gemini Pro
        
        Args:
            prediction_result: Hasil prediksi dari model ML
            employee_data: Data karyawan yang diprediksi
            benchmark_data: Data benchmark untuk perbandingan
        
        Returns:
            AnalysisResult object dengan analisis lengkap
        """
        try:
            prompt = self._create_analysis_prompt(
                prediction_result, employee_data, benchmark_data
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            
            # Try to get response text safely
            raw_response = None
            try:
                raw_response = response.text
            except (ValueError, AttributeError) as e:
                # Response blocked or invalid
                print(f"⚠️  Gemini response blocked: {e}")
                pass
            
            # Check if we got valid response
            if not raw_response or len(raw_response.strip()) == 0:
                # Generate fallback analysis
                label = prediction_result.get('label', 'N/A')
                prob = prediction_result.get('probability', 0) * 100
                perf = employee_data.get('performance_score', 0)
                beh = employee_data.get('behavior_avg', 0)
                tenure = employee_data.get('tenure_years', 0)
                
                print(f"🔄 Using fallback analysis for {label} ({prob:.1f}%)")
                return self._generate_fallback_analysis(label, prob, perf, beh, tenure)
            
            # Parse response untuk extract sections
            summary, detailed, recommendations, risks, strengths = self._parse_response(raw_response)
            
            return AnalysisResult(
                summary=summary,
                detailed_analysis=detailed,
                recommendations=recommendations,
                risk_factors=risks,
                strengths=strengths,
                raw_response=raw_response
            )
            
        except Exception as e:
            # Fallback jika ada error - use rule-based analysis
            print(f"❌ Gemini analysis error: {e}")
            
            # Extract data for fallback
            label = prediction_result.get('label', 'N/A')
            prob = prediction_result.get('probability', 0) * 100
            perf = employee_data.get('performance_score', 0)
            beh = employee_data.get('behavior_avg', 0)
            tenure = employee_data.get('tenure_years', 0)
            
            print(f"🔄 Using fallback analysis due to error")
            return self._generate_fallback_analysis(label, prob, perf, beh, tenure)
    
    def _parse_response(self, response: str) -> tuple[str, str, list[str], list[str], list[str]]:
        """
        Parse response dari Gemini untuk extract sections
        
        Args:
            response: Raw response dari Gemini
        
        Returns:
            Tuple of (summary, detailed_analysis, recommendations, risk_factors, strengths)
        """
        lines = response.split('\n')
        
        summary = ""
        detailed = ""
        recommendations = []
        risks = []
        strengths = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Detect sections
            line_upper = line.upper()
            if 'RINGKASAN' in line_upper or 'EXECUTIVE' in line_upper or 'EKSEKUTIF' in line_upper:
                current_section = 'summary'
                continue
            elif 'ANALISIS DETAIL' in line_upper or 'DETAILED' in line_upper:
                current_section = 'detailed'
                continue
            elif 'KEKUATAN' in line_upper or 'STRENGTH' in line_upper:
                current_section = 'strengths'
                continue
            elif 'RISIKO' in line_upper or 'RISK' in line_upper or 'FAKTOR RISIKO' in line_upper:
                current_section = 'risks'
                continue
            elif 'REKOMENDASI' in line_upper or 'RECOMMENDATION' in line_upper:
                current_section = 'recommendations'
                continue
            
            # Add content to appropriate section
            if line and not line.startswith('#') and not line.startswith('**'):
                if current_section == 'summary':
                    summary += line + " "
                elif current_section == 'detailed':
                    detailed += line + "\n"
                elif current_section == 'strengths':
                    if line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit()):
                        cleaned = line.lstrip('-•*0123456789. ')
                        if cleaned:
                            strengths.append(cleaned)
                elif current_section == 'risks':
                    if line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit()):
                        cleaned = line.lstrip('-•*0123456789. ')
                        if cleaned:
                            risks.append(cleaned)
                elif current_section == 'recommendations':
                    if line.startswith('-') or line.startswith('•') or line.startswith('*') or (line and line[0].isdigit()):
                        cleaned = line.lstrip('-•*0123456789. ')
                        if cleaned:
                            recommendations.append(cleaned)
        
        # Fallback jika parsing gagal
        if not summary:
            summary = "Analisis telah dihasilkan. Lihat detail lengkap di bawah."
        if not detailed:
            detailed = response
        if not recommendations:
            recommendations = ["Lihat analisis lengkap untuk rekomendasi detail"]
        if not risks:
            risks = ["Lihat analisis lengkap untuk faktor risiko"]
        if not strengths:
            strengths = ["Lihat analisis lengkap untuk kekuatan karyawan"]
        
        return summary.strip(), detailed.strip(), recommendations, risks, strengths
    
    def _generate_fallback_analysis(
        self, 
        label: str, 
        prob: float, 
        perf: float, 
        beh: float, 
        tenure: float
    ) -> 'AnalysisResult':
        """Generate fallback analysis when API fails"""
        
        # Determine promotion likelihood
        if prob >= 70:
            likelihood = "tinggi"
            tone = "sangat positif"
        elif prob >= 50:
            likelihood = "sedang"
            tone = "cukup baik"
        else:
            likelihood = "rendah"
            tone = "perlu peningkatan"
        
        # Generate summary
        summary = f"""Karyawan ini memiliki probabilitas promosi {likelihood} ({prob:.1f}%). 
Berdasarkan analisis data, profil karyawan menunjukkan outlook yang {tone} untuk promosi."""
        
        # Generate strengths
        strengths = []
        if perf >= 80:
            strengths.append(f"Performance Score yang kuat ({perf}) menunjukkan kinerja di atas rata-rata")
        if beh >= 80:
            strengths.append(f"Behavioral Score yang excellent ({beh}) mencerminkan sikap kerja yang positif")
        if 3 <= tenure <= 7:
            strengths.append(f"Masa kerja {tenure} tahun berada di sweet spot untuk promosi")
        if not strengths:
            strengths.append("Karyawan menunjukkan potensi untuk berkembang")
        
        # Generate risks
        risks = []
        if perf < 70:
            risks.append(f"Performance Score ({perf}) masih di bawah threshold optimal untuk promosi")
        if beh < 70:
            risks.append(f"Behavioral Score ({beh}) perlu ditingkatkan untuk meningkatkan peluang promosi")
        if tenure > 10:
            risks.append(f"Masa kerja yang panjang ({tenure} tahun) tanpa promosi dapat menjadi concern")
        if not risks:
            risks.append("Tidak ada faktor risiko signifikan yang teridentifikasi")
        
        # Generate recommendations
        recommendations = []
        if perf < 80:
            recommendations.append("Fokus pada peningkatan performance metrics melalui training dan mentoring")
        if beh < 80:
            recommendations.append("Tingkatkan soft skills dan behavioral competencies")
        recommendations.append("Maintain consistency dalam kinerja dan sikap kerja")
        recommendations.append("Diskusikan career development plan dengan supervisor")
        
        detailed = f"""
**Analisis Prediksi Promosi**

Prediksi: {label} dengan probabilitas {prob:.1f}%

**Metrik Karyawan:**
- Performance Score: {perf}
- Behavioral Score: {beh}
- Tenure: {tenure} tahun

**Interpretasi:**
Berdasarkan data historis, karyawan dengan profil serupa memiliki peluang promosi {likelihood}. 
Kombinasi performance dan behavioral scores menunjukkan {tone} untuk advancement.
"""
        
        return AnalysisResult(
            summary=summary,
            detailed_analysis=detailed,
            recommendations=recommendations,
            risk_factors=risks,
            strengths=strengths,
            raw_response="[Fallback Analysis - Generated without AI]"
        )
    
    def quick_summary(
        self,
        prediction_result: Dict[str, Any],
        employee_data: Dict[str, Any]
    ) -> str:
        """
        Generate quick summary (lebih cepat)
        
        Args:
            prediction_result: Hasil prediksi dari model ML
            employee_data: Data karyawan
        
        Returns:
            Quick summary string
        """
        try:
            prompt = f"""Berikan ringkasan singkat (maksimal 3 kalimat) tentang prediksi promosi ini:

Prediksi: {prediction_result.get('label', 'N/A')} ({prediction_result.get('probability', 0) * 100:.1f}%)
Performance: {employee_data.get('performance_score', 'N/A')}
Behavioral: {employee_data.get('behavior_avg', 'N/A')}
Tenure: {employee_data.get('tenure_years', 'N/A')} tahun

Gunakan bahasa Indonesia yang profesional dan mudah dipahami."""

            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.5,
                    'max_output_tokens': 200,
                },
                safety_settings=self.safety_settings
            )
            
            # Try to get response text safely
            response_text = None
            try:
                response_text = response.text
            except (ValueError, AttributeError):
                pass
            
            # Check if we got valid response
            if response_text and len(response_text.strip()) > 0:
                return response_text.strip()
            else:
                # Fallback if blocked
                label = prediction_result.get('label', 'N/A')
                prob = prediction_result.get('probability', 0) * 100
                perf = employee_data.get('performance_score', 'N/A')
                beh = employee_data.get('behavior_avg', 'N/A')
                return f"Prediksi: {label} ({prob:.1f}%). Performance {perf} dan Behavioral {beh} menunjukkan profil yang {'kuat' if prob > 70 else 'perlu ditingkatkan'}."
            
        except Exception as e:
            # Fallback summary
            label = prediction_result.get('label', 'N/A')
            prob = prediction_result.get('probability', 0) * 100
            return f"Prediksi: {label} dengan probabilitas {prob:.1f}%."


def create_gemini_service(api_key: Optional[str] = None) -> GeminiAnalysisService:
    """
    Factory function untuk membuat Gemini service
    
    Args:
        api_key: Gemini API key (optional)
    
    Returns:
        GeminiAnalysisService instance
    """
    return GeminiAnalysisService(api_key=api_key)
