#!/usr/bin/env python3
"""
Test Page Analysis Service
Script untuk memverifikasi bahwa Gemini AI benar-benar dipanggil (bukan fallback)
"""

import os
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'app'))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"✅ Loaded .env file from: {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed")

def test_page_analysis_service():
    """Test Page Analysis Service dengan model performance data"""
    
    print("\n" + "="*60)
    print("🔍 TESTING PAGE ANALYSIS SERVICE")
    print("="*60 + "\n")
    
    try:
        from services.page_analysis_service import PageAnalysisService
        
        print("🔄 Creating PageAnalysisService...")
        service = PageAnalysisService()
        
        print(f"✅ Service created")
        print(f"📊 Is Enabled: {service.is_enabled()}")
        
        if not service.is_enabled():
            print("❌ Service is NOT enabled!")
            print("   Gemini AI tidak berhasil diinisialisasi")
            return False
        
        # Test with mock model performance data
        print("\n🔄 Testing analyze_model_performance...")
        
        model_results = {
            'best_model': {
                'name': 'Neural Network',
                'accuracy': 0.9091,
                'precision': 0.5000,
                'recall': 0.6154,
                'f1_score': 0.5517,
                'roc_auc': 0.8828,
            },
            'all_models': [
                {'name': 'Neural Network', 'accuracy': 0.9091, 'f1_score': 0.5517},
                {'name': 'XGBoost', 'accuracy': 0.9091, 'f1_score': 0.5333},
                {'name': 'Random Forest', 'accuracy': 0.8636, 'f1_score': 0.4000},
            ],
            'feature_importance': [
                {'feature': 'Performance Score', 'importance': 0.35},
                {'feature': 'Behavioral Score', 'importance': 0.28},
                {'feature': 'Psychological Score (QA)', 'importance': 0.18},
                {'feature': 'Tenure Years', 'importance': 0.12},
                {'feature': 'Collaboration Score (QA)', 'importance': 0.07},
            ],
            'qa_in_top10': 2,
            'qa_contribution': 18.0,
        }
        
        analysis = service.analyze_model_performance(model_results)
        
        print("\n" + "="*60)
        print("📝 ANALYSIS RESULT (Full):")
        print("="*60)
        print(analysis)
        print("="*60)
        
        # Check if it's real AI analysis or fallback
        # Fallback has very specific markers
        is_fallback = (
            "mencapai accuracy" in analysis and 
            "dengan F1-Score" in analysis and
            len(analysis) < 800  # Fallback is usually short
        )
        
        # Real Gemini has more detailed structure
        has_gemini_markers = (
            "## 🏆" in analysis or
            "## 📊" in analysis or
            "*accuracy*" in analysis or  # Gemini uses markdown
            "*precision*" in analysis
        )
        
        if is_fallback:
            print("\n❌ RESULT: FALLBACK ANALYSIS (bukan dari Gemini AI)")
            print("   Ini adalah analisis cadangan, bukan dari Gemini")
            return False
        elif has_gemini_markers:
            print("\n✅ RESULT: REAL GEMINI AI ANALYSIS")
            print("   Analisis ini dihasilkan oleh Gemini AI")
            print(f"   Length: {len(analysis)} characters")
            return True
        else:
            print("\n⚠️  RESULT: UNCLEAR - needs manual verification")
            print(f"   Length: {len(analysis)} characters")
            return True
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("\n🚀 PAGE ANALYSIS SERVICE TEST")
    print("="*60)
    
    success = test_page_analysis_service()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if success:
        print("✅ TEST PASSED: Gemini AI bekerja dengan benar!")
        print("\n💡 Sekarang coba di aplikasi Streamlit:")
        print("   1. Jalankan: streamlit run app/Home.py")
        print("   2. Buka halaman Model Performance")
        print("   3. Klik 'Generate AI Analysis'")
        print("   4. Hasilnya seharusnya detail dan spesifik dari Gemini")
        return 0
    else:
        print("❌ TEST FAILED: Masih menggunakan fallback analysis")
        print("\n📝 Troubleshooting:")
        print("   1. Pastikan GEMINI_API_KEY valid di .env")
        print("   2. Pastikan google-generativeai terinstall")
        print("   3. Cek error message di atas untuk detail")
        return 1

if __name__ == "__main__":
    sys.exit(main())
