# 🚀 Streamlit Cloud Deployment Guide

## 🎯 Cara Deploy dengan Gemini AI (Tanpa .env di Git)

### ⚠️ PENTING: .env TIDAK BOLEH di-commit ke Git!

File `.env` mengandung API key yang rahasia. Jangan pernah commit ke Git!

**Solusi**: Gunakan **Streamlit Secrets** untuk production deployment.

---

## 📋 Step-by-Step Deployment

### Step 1: Push Code ke GitHub (Tanpa .env)

```bash
# Pastikan .env ada di .gitignore
echo ".env" >> .gitignore

# Commit dan push (tanpa .env)
git add .
git commit -m "Ready for deployment"
git push origin qa-integration-complete
```

✅ `.env` **TIDAK** akan ter-commit (sudah di .gitignore)

---

### Step 2: Deploy ke Streamlit Cloud

1. **Buka Streamlit Cloud**:
   ```
   https://share.streamlit.io/
   ```

2. **Login dengan GitHub**

3. **Create New App**:
   - Click "New app"
   - Repository: `sulaemandeni97/MPCIM_Thesis`
   - Branch: `qa-integration-complete`
   - Main file path: `app/Home.py`
   - Click "Deploy"

4. **Wait for deployment** (2-3 menit)

---

### Step 3: Add Secrets (API Keys)

**PENTING**: Ini cara menambahkan API key tanpa commit ke Git!

1. **Buka App Settings**:
   - Go to your deployed app
   - Click "⚙️" (Settings) di kanan atas
   - Go to "Secrets" tab

2. **Add Secrets** (format TOML):
   ```toml
   # Secrets for MPCIM Thesis App
   
   # Google Gemini Pro API Key
   GEMINI_API_KEY = "AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU"
   
   # OpenAI API Key (optional)
   # OPENAI_API_KEY = "your_openai_key_here"
   ```

3. **Save**:
   - Click "Save"
   - App akan restart otomatis
   - Gemini AI akan aktif! ✅

---

## 🔍 Cara Kerja Streamlit Secrets

### Local Development (.env):
```python
# File: .env (local only, not in git)
GEMINI_API_KEY=AIzaSy...your_key...
```

```python
# Code loads from .env
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('GEMINI_API_KEY')
```

### Production (Streamlit Secrets):
```toml
# Streamlit Cloud → App Settings → Secrets
GEMINI_API_KEY = "AIzaSy...your_key..."
```

```python
# Code automatically reads from Streamlit secrets
import streamlit as st
key = st.secrets.get("GEMINI_API_KEY")
# OR
key = os.getenv('GEMINI_API_KEY')  # Also works!
```

**Streamlit automatically injects secrets as environment variables!** ✅

---

## 🔧 Update Code untuk Support Both

Mari kita update `ai_service.py` untuk support both local (.env) dan production (Streamlit secrets):

```python
import os
from pathlib import Path

# Try to load from .env (local development)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parents[2] / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

# Try to load from Streamlit secrets (production)
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        # Streamlit secrets available
        for key in st.secrets:
            if key not in os.environ:
                os.environ[key] = st.secrets[key]
except Exception:
    pass

# Now get API key (works for both local and production)
gemini_key = os.getenv('GEMINI_API_KEY')
```

---

## ✅ Verification

### Check if Secrets are Loaded:

1. **In Streamlit Cloud**:
   - Open your deployed app
   - Go to Prediction page
   - Check sidebar → Should show "✅ Gemini Pro Ready"

2. **Test Prediction**:
   - Enter employee data
   - Enable Gemini AI analysis
   - Click Predict
   - Should see AI analysis! ✅

---

## 🔒 Security Best Practices

### ✅ DO:
- ✅ Use `.env` for local development
- ✅ Add `.env` to `.gitignore`
- ✅ Use Streamlit Secrets for production
- ✅ Rotate API keys regularly
- ✅ Use different keys for dev and production

### ❌ DON'T:
- ❌ Commit `.env` to Git
- ❌ Share API keys publicly
- ❌ Hardcode API keys in code
- ❌ Use same key for multiple projects
- ❌ Store keys in plain text (except .env locally)

---

## 📊 Environment Variables Flow

```
┌─────────────────────────────────────────┐
│         LOCAL DEVELOPMENT               │
├─────────────────────────────────────────┤
│ 1. Create .env file                     │
│ 2. Add GEMINI_API_KEY=...              │
│ 3. python-dotenv loads it              │
│ 4. os.getenv('GEMINI_API_KEY') works   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│         GIT REPOSITORY                  │
├─────────────────────────────────────────┤
│ ✅ .gitignore includes .env             │
│ ✅ .env.example (with placeholders)     │
│ ❌ .env NOT committed                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      STREAMLIT CLOUD DEPLOYMENT         │
├─────────────────────────────────────────┤
│ 1. Deploy from GitHub                   │
│ 2. Add secrets in App Settings          │
│ 3. Streamlit injects as env vars        │
│ 4. os.getenv('GEMINI_API_KEY') works   │
└─────────────────────────────────────────┘
```

---

## 🎓 For Thesis Defense

### Demo Scenario:

1. **Show Local Version**:
   ```bash
   streamlit run app/Home.py
   ```
   - Gemini AI works (from .env)

2. **Show Deployed Version**:
   ```
   https://your-app.streamlit.app
   ```
   - Gemini AI works (from Streamlit Secrets)

3. **Explain Security**:
   - ".env tidak di-commit ke Git"
   - "API key aman di Streamlit Secrets"
   - "Best practice untuk production"

---

## 🔧 Troubleshooting

### Issue 1: "GEMINI_API_KEY not found" in Streamlit Cloud

**Solution**:
1. Go to App Settings → Secrets
2. Add:
   ```toml
   GEMINI_API_KEY = "your_key_here"
   ```
3. Save and restart

### Issue 2: "GEMINI_API_KEY not found" locally

**Solution**:
```bash
# Check if .env exists
ls -la .env

# Check content
cat .env | grep GEMINI_API_KEY

# Recreate if needed
cp .env.example .env
nano .env  # Add your real key
```

### Issue 3: Secrets not loading in code

**Solution**: Update `ai_service.py` to support Streamlit secrets:
```python
import streamlit as st

# Try Streamlit secrets first
try:
    if 'GEMINI_API_KEY' in st.secrets:
        os.environ['GEMINI_API_KEY'] = st.secrets['GEMINI_API_KEY']
except Exception:
    pass
```

---

## 📝 Quick Reference

### Local Development:
```bash
# 1. Create .env
cp .env.example .env

# 2. Add your key
echo 'GEMINI_API_KEY=AIzaSy...your_key...' > .env

# 3. Run
streamlit run app/Home.py
```

### Streamlit Cloud:
```
1. Deploy app from GitHub
2. Go to Settings → Secrets
3. Add: GEMINI_API_KEY = "your_key"
4. Save → App restarts → AI works!
```

---

## 🎉 Summary

### Local (.env):
- ✅ Create `.env` file
- ✅ Add `GEMINI_API_KEY=...`
- ✅ `.env` in `.gitignore`
- ✅ Never commit to Git

### Production (Streamlit Secrets):
- ✅ Deploy from GitHub (without .env)
- ✅ Add secrets in App Settings
- ✅ Streamlit injects as env vars
- ✅ API key secure and working

### Both Work:
```python
# This works in both environments
gemini_key = os.getenv('GEMINI_API_KEY')
```

**Your API key is safe and Gemini AI works in production!** 🔒✅

---

**Last Updated**: November 17, 2025, 11:30 PM  
**Status**: Production Ready  
**Security**: ✅ Best Practices  
**Deployment**: 🚀 Ready for Streamlit Cloud
