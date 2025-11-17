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
    if hasattr(st, 'secrets'):
        try:
            # Try to access secrets
            if 'GEMINI_API_KEY' in st.secrets:
                os.environ['GEMINI_API_KEY'] = str(st.secrets['GEMINI_API_KEY'])
                print(f"✅ Loaded GEMINI_API_KEY from Streamlit secrets")
            if 'OPENAI_API_KEY' in st.secrets:
                os.environ['OPENAI_API_KEY'] = str(st.secrets['OPENAI_API_KEY'])
                print(f"✅ Loaded OPENAI_API_KEY from Streamlit secrets")
        except Exception as e:
            print(f"⚠️ Could not load Streamlit secrets: {e}")
except Exception as e:
    print(f"⚠️ Streamlit not available or secrets not configured: {e}")


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
    
    # Detect if running in Streamlit Cloud
    is_streamlit_cloud = False
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            is_streamlit_cloud = True
    except:
        pass
    
    status = {
        'has_gemini': has_gemini,
        'has_openai': has_openai,
        'has_any': has_gemini or has_openai,
        'recommended': 'Gemini' if has_gemini else 'OpenAI' if has_openai else None,
        'message': '',
        'is_streamlit_cloud': is_streamlit_cloud
    }
    
    if has_gemini:
        status['message'] = "✅ Gemini AI Ready (FREE)"
    elif has_openai:
        status['message'] = "✅ OpenAI Ready (Paid)"
    else:
        # Different message for local vs Streamlit Cloud
        if is_streamlit_cloud:
            status['message'] = "⚠️ AI Not Configured\n\nAdd GEMINI_API_KEY in App Settings → Secrets"
        else:
            status['message'] = "⚠️ AI Not Configured\n\nSet GEMINI_API_KEY in .env file"
    
    return status
