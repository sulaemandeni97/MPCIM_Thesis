"""
AI Service Factory
Creates appropriate AI service based on available API keys
"""

import os
from typing import Optional
from pathlib import Path

# Load environment variables from .env file (local development)
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).resolve().parents[2] / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Load from Streamlit secrets (production deployment)
try:
    import streamlit as st
    # Check if running in Streamlit and secrets are available
    if hasattr(st, 'secrets') and st.secrets:
        # Inject Streamlit secrets into environment variables
        for key in st.secrets:
            if key not in os.environ:
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # Not running in Streamlit or secrets not available


def create_ai_service():
    """
    Create AI service based on available API keys.
    Prioritizes Gemini (free) over OpenAI (paid).
    
    Returns:
        tuple: (service_instance, service_name, is_enabled)
    """
    # Check for Gemini API key first (recommended - free!)
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    # Try Gemini first
    if gemini_key and gemini_key != 'your_gemini_api_key_here':
        try:
            from services.gemini_service import GeminiAnalysisService
            service = GeminiAnalysisService()
            return service, "Gemini Pro", True
        except Exception as e:
            print(f"Failed to initialize Gemini service: {e}")
    
    # Fallback to OpenAI
    if openai_key and openai_key != 'your_openai_api_key_here':
        try:
            from services.openai_service import OpenAIAnalysisService
            service = OpenAIAnalysisService()
            return service, "OpenAI GPT-4", True
        except Exception as e:
            print(f"Failed to initialize OpenAI service: {e}")
    
    # No AI service available
    return None, "None", False


def get_ai_status():
    """
    Get current AI service status.
    
    Returns:
        dict: Status information
    """
    gemini_key = os.getenv('GEMINI_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    has_gemini = gemini_key and gemini_key != 'your_gemini_api_key_here'
    has_openai = openai_key and openai_key != 'your_openai_api_key_here'
    
    status = {
        'has_gemini': has_gemini,
        'has_openai': has_openai,
        'has_any': has_gemini or has_openai,
        'recommended': 'Gemini' if has_gemini else 'OpenAI' if has_openai else None,
        'message': ''
    }
    
    if has_gemini:
        status['message'] = "✅ Gemini AI Ready (FREE)"
    elif has_openai:
        status['message'] = "✅ OpenAI Ready (Paid)"
    else:
        status['message'] = "⚠️ No AI service configured. Set GEMINI_API_KEY in .env file."
    
    return status
