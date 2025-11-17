# 🔑 Gemini AI API Setup Guide

## 🎯 Overview

Aplikasi ini menggunakan **Google Gemini AI** untuk analisis prediksi yang lebih mendalam. Gemini API **GRATIS** dan mudah disetup!

---

## 📋 Prerequisites

- Google Account
- Internet connection
- 5 menit waktu setup

---

## 🚀 Step-by-Step Setup

### Step 1: Dapatkan API Key Gemini

1. **Buka Google AI Studio:**
   ```
   https://makersuite.google.com/app/apikey
   ```

2. **Login dengan Google Account Anda**

3. **Klik "Create API Key"**
   - Pilih project (atau buat baru)
   - Copy API key yang dihasilkan

4. **Simpan API Key** (jangan share ke siapapun!)

---

### Step 2: Setup di Local Development

#### Option A: Menggunakan File .env (Recommended)

1. **Copy file .env.example:**
   ```bash
   cp .env.example .env
   ```

2. **Edit file .env:**
   ```bash
   nano .env
   # atau
   code .env
   ```

3. **Paste API key Anda:**
   ```env
   # Google Gemini Pro (RECOMMENDED - FREE!)
   GEMINI_API_KEY=AIzaSy...your_actual_key_here...
   
   # OpenAI (Optional - leave as is if not using)
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Save file** (Ctrl+O, Enter, Ctrl+X untuk nano)

5. **Verify .env tidak ter-commit:**
   ```bash
   git status
   # .env should NOT appear (it's in .gitignore)
   ```

#### Option B: Menggunakan Environment Variables

```bash
# Temporary (hanya untuk session ini)
export GEMINI_API_KEY="AIzaSy...your_key..."

# Permanent (tambahkan ke ~/.zshrc atau ~/.bashrc)
echo 'export GEMINI_API_KEY="AIzaSy...your_key..."' >> ~/.zshrc
source ~/.zshrc
```

---

### Step 3: Setup untuk Production/Deployment

#### Streamlit Cloud:

1. **Deploy app ke Streamlit Cloud**

2. **Buka App Settings:**
   - Go to app dashboard
   - Click "⚙️ Settings"
   - Go to "Secrets" tab

3. **Add secrets:**
   ```toml
   # Secrets
   GEMINI_API_KEY = "AIzaSy...your_key..."
   ```

4. **Save** - App akan restart otomatis

#### Heroku:

```bash
heroku config:set GEMINI_API_KEY="AIzaSy...your_key..."
```

#### Docker:

```bash
docker run -e GEMINI_API_KEY="AIzaSy...your_key..." your-image
```

#### Railway/Render:

1. Go to Environment Variables
2. Add: `GEMINI_API_KEY` = `AIzaSy...your_key...`

---

## ✅ Verify Setup

### Test 1: Check Environment Variable

```bash
python3 -c "import os; print('✅ API Key loaded!' if os.getenv('GEMINI_API_KEY') else '❌ API Key not found')"
```

### Test 2: Test API Connection

```bash
python3 << 'EOF'
import os
import google.generativeai as genai

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    print("❌ GEMINI_API_KEY not found in environment")
    exit(1)

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Say 'API is working!' in one sentence")
    print(f"✅ API Test Success: {response.text}")
except Exception as e:
    print(f"❌ API Test Failed: {e}")
EOF
```

### Test 3: Run Streamlit App

```bash
streamlit run app/Home.py
```

Buka Prediction page dan test dengan Gemini AI analysis.

---

## 🔒 Security Best Practices

### ✅ DO:
- ✅ Use `.env` file for local development
- ✅ Add `.env` to `.gitignore`
- ✅ Use environment variables for production
- ✅ Use secrets management (Streamlit Secrets, etc.)
- ✅ Rotate API keys regularly
- ✅ Use `.env.example` with placeholder values

### ❌ DON'T:
- ❌ Commit API keys to git
- ❌ Share API keys publicly
- ❌ Hardcode API keys in code
- ❌ Use same key for dev and production
- ❌ Store keys in plain text files (except .env)

---

## 📊 API Limits & Pricing

### Gemini API (FREE Tier):

```
Rate Limits:
- 60 requests per minute
- 1,500 requests per day
- 1 million tokens per month

Cost: FREE! 🎉
```

**Perfect for:**
- Development
- Testing
- Small to medium apps
- Academic projects (like this thesis!)

### Need More?

Upgrade to paid tier:
- Higher rate limits
- More tokens
- Priority support

Visit: https://ai.google.dev/pricing

---

## 🐛 Troubleshooting

### Issue 1: "GEMINI_API_KEY not found"

**Solution:**
```bash
# Check if .env exists
ls -la .env

# Check if key is set
cat .env | grep GEMINI_API_KEY

# Reload environment
source .env  # or restart terminal
```

### Issue 2: "API key invalid"

**Solution:**
1. Verify key is correct (no extra spaces)
2. Regenerate key from Google AI Studio
3. Check key hasn't expired

### Issue 3: "Rate limit exceeded"

**Solution:**
1. Wait 1 minute
2. Reduce request frequency
3. Upgrade to paid tier if needed

### Issue 4: "Module 'google.generativeai' not found"

**Solution:**
```bash
pip install google-generativeai
# or
pip install -r app/requirements.txt
```

---

## 📝 Example .env File

```env
# AI Provider Configuration

# Google Gemini Pro (RECOMMENDED - FREE!)
GEMINI_API_KEY=AIzaSyAnoAuDTGaVY1nQOsgRVVv_ddz2Lo9CrS8

# OpenAI (Optional)
OPENAI_API_KEY=your_openai_api_key_here
```

**⚠️ IMPORTANT:** Never commit this file to git!

---

## 🔄 Updating API Keys

### Rotate Keys Regularly:

1. **Generate new key** in Google AI Studio
2. **Update .env** with new key
3. **Update production secrets**
4. **Delete old key** from Google AI Studio
5. **Test** to ensure everything works

### When to Rotate:

- Every 3-6 months (security best practice)
- If key is exposed/leaked
- When team members change
- Before production deployment

---

## 📚 Additional Resources

### Documentation:
- Gemini API Docs: https://ai.google.dev/docs
- Python SDK: https://ai.google.dev/tutorials/python_quickstart
- Pricing: https://ai.google.dev/pricing

### Support:
- Google AI Forum: https://discuss.ai.google.dev/
- GitHub Issues: https://github.com/google/generative-ai-python/issues

---

## ✅ Quick Checklist

Before deploying:
- [ ] API key obtained from Google AI Studio
- [ ] `.env` file created locally
- [ ] API key added to `.env`
- [ ] `.env` in `.gitignore`
- [ ] API connection tested
- [ ] Production secrets configured
- [ ] App tested with Gemini AI
- [ ] Security review completed

---

## 🎉 You're All Set!

Gemini AI is now configured and ready to use!

**Test it:**
1. Run: `streamlit run app/Home.py`
2. Go to Prediction page
3. Enter employee data
4. Enable Gemini AI analysis
5. See comprehensive insights!

**Need help?** Check troubleshooting section or create an issue.

---

**Last Updated**: November 17, 2025  
**Version**: 1.0  
**Status**: Production Ready  
**Cost**: FREE! 🎉
