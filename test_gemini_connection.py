#!/usr/bin/env python3
"""
Test Gemini AI Connection
Script untuk memverifikasi koneksi ke Google Gemini API
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
    else:
        print(f"⚠️  .env file not found at: {env_path}")
except ImportError:
    print("⚠️  python-dotenv not installed")

def test_gemini_connection():
    """Test Gemini API connection"""
    
    print("\n" + "="*60)
    print("🔍 TESTING GEMINI AI CONNECTION")
    print("="*60 + "\n")
    
    # Check API key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ GEMINI_API_KEY tidak ditemukan di environment variables")
        print("\n📝 Cara memperbaiki:")
        print("   1. Buat file .env di root project")
        print("   2. Tambahkan: GEMINI_API_KEY=your_actual_api_key")
        print("   3. Get API key dari: https://makersuite.google.com/app/apikey")
        return False
    
    if api_key == 'your_gemini_api_key_here':
        print("❌ GEMINI_API_KEY masih menggunakan placeholder")
        print("\n📝 Cara memperbaiki:")
        print("   1. Edit file .env")
        print("   2. Ganti 'your_gemini_api_key_here' dengan API key asli")
        print("   3. Get API key dari: https://makersuite.google.com/app/apikey")
        return False
    
    print(f"✅ API Key ditemukan: {api_key[:10]}...{api_key[-4:]}")
    
    # Try to import google.generativeai
    try:
        import google.generativeai as genai
        print("✅ google-generativeai library terinstall")
    except ImportError:
        print("❌ google-generativeai library tidak terinstall")
        print("\n📝 Cara memperbaiki:")
        print("   pip install google-generativeai")
        return False
    
    # Try to configure and test
    try:
        print("\n🔄 Mencoba koneksi ke Gemini API...")
        genai.configure(api_key=api_key)
        
        # Test with a simple prompt
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello, are you working?")
        
        if response and response.text:
            print("✅ Koneksi berhasil!")
            print(f"📝 Response dari Gemini: {response.text[:100]}...")
            return True
        else:
            print("⚠️  Koneksi berhasil tapi response kosong")
            return False
            
    except Exception as e:
        print(f"❌ Error saat koneksi ke Gemini: {str(e)}")
        print("\n📝 Kemungkinan penyebab:")
        print("   1. API key tidak valid")
        print("   2. Tidak ada koneksi internet")
        print("   3. API quota habis")
        print("   4. Region tidak didukung")
        return False

def test_ai_service():
    """Test AI service dari aplikasi"""
    
    print("\n" + "="*60)
    print("🔍 TESTING AI SERVICE")
    print("="*60 + "\n")
    
    try:
        from services.ai_service import get_ai_status, create_ai_service
        
        # Get status
        status = get_ai_status()
        print(f"Status: {status['message']}")
        print(f"Has Gemini: {status['has_gemini']}")
        print(f"Has OpenAI: {status['has_openai']}")
        print(f"Has Any AI: {status['has_any']}")
        
        if status['has_any']:
            print("\n🔄 Mencoba membuat AI service...")
            service, service_name, is_enabled = create_ai_service()
            
            if is_enabled:
                print(f"✅ AI Service berhasil dibuat: {service_name}")
                return True
            else:
                print("❌ AI Service tidak bisa dibuat")
                return False
        else:
            print("❌ Tidak ada AI service yang tersedia")
            return False
            
    except Exception as e:
        print(f"❌ Error saat test AI service: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    
    print("\n🚀 GEMINI AI CONNECTION TEST")
    print("="*60)
    
    # Test 1: Direct Gemini connection
    gemini_ok = test_gemini_connection()
    
    # Test 2: AI service
    service_ok = test_ai_service()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Gemini Connection: {'✅ PASSED' if gemini_ok else '❌ FAILED'}")
    print(f"AI Service: {'✅ PASSED' if service_ok else '❌ FAILED'}")
    
    if gemini_ok and service_ok:
        print("\n🎉 Semua test PASSED! Gemini AI siap digunakan.")
        return 0
    else:
        print("\n⚠️  Ada test yang FAILED. Periksa error di atas.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
