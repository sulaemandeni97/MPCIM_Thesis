# 🔧 AI Analysis Troubleshooting Guide

## 🎯 Common Issues & Solutions

### Issue 1: "⚠️ Gemini AI tidak tersedia. Pastikan GEMINI_API_KEY sudah dikonfigurasi"

**Symptoms**:
- Warning message appears when clicking "Generate AI Analysis"
- No AI analysis generated
- Falls back to basic analysis

**Causes**:
1. API key not set in `.env` file
2. `.env` file not in correct location
3. API key invalid or expired
4. Environment variables not loaded

**Solutions**:

#### Solution 1A: Check .env File Exists
```bash
# Check if .env exists in project root
ls -la /Users/denisulaeman/Workspace/Academic/CascadeProjects/MPCIM_Thesis/.env

# If not found, create it
cp .env.example .env
```

#### Solution 1B: Verify API Key in .env
```bash
# Check .env content
cat .env

# Should show:
# GEMINI_API_KEY=AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU
```

#### Solution 1C: Verify API Key Format
- Length: 39 characters
- Starts with: `AIzaSy`
- Example: `AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU`

#### Solution 1D: Test API Key Loading
```bash
cd /Users/denisulaeman/Workspace/Academic/CascadeProjects/MPCIM_Thesis

python3 -c "
import sys
sys.path.insert(0, 'app')
from services.page_analysis_service import create_page_analysis_service

service = create_page_analysis_service()
print(f'Service enabled: {service.is_enabled()}')
print(f'API key present: {service.api_key is not None}')
"
```

**Expected Output**:
```
✅ Loaded .env from /path/to/.env
🔍 API Key check: Found
🔍 API Key length: 39
✅ Gemini AI initialized successfully
Service enabled: True
API key present: True
```

---

### Issue 2: AI Analysis Contains Hallucinations or Incorrect Data

**Symptoms**:
- AI mentions numbers not in the actual data
- Analysis includes made-up statistics
- Interpretations don't match visualizations

**Causes**:
1. Old version of `page_analysis_service.py`
2. Gemini AI not following strict instructions
3. Prompt not specific enough

**Solutions**:

#### Solution 2A: Update to Latest Version
```bash
git pull origin qa-integration-complete
```

#### Solution 2B: Verify Anti-Hallucination Prompts
Check that prompts include:
- "BERDASARKAN DATA YANG DIBERIKAN SAJA"
- "JANGAN menambahkan informasi yang tidak ada"
- "HANYA gunakan angka yang tertera"
- Specific numbers embedded in format instructions

#### Solution 2C: Regenerate Analysis
- Click "Generate AI Analysis" again
- Gemini may give different (more accurate) response
- Compare with actual data in visualizations

#### Solution 2D: Use Fallback Analysis
If AI continues to hallucinate:
- Fallback analysis is always accurate
- Based on actual data only
- No AI interpretation, just facts

---

### Issue 3: "Error: 'tuple' object has no attribute 'get_provider'"

**Symptoms**:
- Error in Prediction page
- AI service initialization fails

**Cause**:
- Old code trying to call method on tuple

**Solution**:
```bash
# Pull latest code
git pull origin qa-integration-complete

# Restart Streamlit
streamlit run app/Home.py
```

---

### Issue 4: Analysis Takes Too Long (>30 seconds)

**Symptoms**:
- Spinner keeps spinning
- No response from Gemini
- Eventually times out

**Causes**:
1. Gemini API slow or overloaded
2. Network issues
3. API quota exceeded

**Solutions**:

#### Solution 4A: Wait Longer
- Gemini can take 10-30 seconds
- Be patient, especially for complex analysis

#### Solution 4B: Check Internet Connection
```bash
# Test connectivity
ping google.com

# Test Gemini API
curl -H "Content-Type: application/json" \
     -d '{"contents":[{"parts":[{"text":"test"}]}]}' \
     "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
```

#### Solution 4C: Check API Quota
- Go to https://makersuite.google.com/app/apikey
- Check if quota exceeded
- Wait or upgrade plan

#### Solution 4D: Use Fallback
- If timeout, fallback analysis will be shown
- Fallback is instant and always works

---

### Issue 5: Different Analysis Each Time

**Symptoms**:
- Clicking "Generate AI Analysis" multiple times gives different results
- Numbers are consistent but wording varies

**Explanation**:
- This is NORMAL behavior
- Gemini AI is generative (not deterministic)
- Each generation uses different wording
- Core insights should be similar
- Numbers should be identical (from data)

**Verification**:
- Check that numbers match actual data
- Verify insights are reasonable
- Compare with visualizations
- Use the version you prefer

---

### Issue 6: "Invalid operation: response blocked"

**Symptoms**:
- Error message about blocked response
- Fallback analysis shown instead

**Cause**:
- Gemini safety filters triggered
- Rare occurrence

**Solution**:
- This is automatic
- Fallback analysis will be used
- No action needed
- Try regenerating if desired

---

## 🔍 Debugging Steps

### Step 1: Check API Key
```bash
# In project root
cat .env | grep GEMINI_API_KEY

# Should output:
# GEMINI_API_KEY=AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU
```

### Step 2: Test Service Initialization
```bash
cd /Users/denisulaeman/Workspace/Academic/CascadeProjects/MPCIM_Thesis

python3 << 'EOF'
import sys
sys.path.insert(0, 'app')

from services.page_analysis_service import PageAnalysisService

print("Creating service...")
service = PageAnalysisService()

print(f"\nResults:")
print(f"- Enabled: {service.is_enabled()}")
print(f"- API Key: {'Present' if service.api_key else 'Missing'}")
print(f"- Length: {len(service.api_key) if service.api_key else 0}")

if service.is_enabled():
    print("\n✅ Service working correctly!")
else:
    print("\n❌ Service not working - check API key")
EOF
```

### Step 3: Test AI Generation
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, 'app')

from services.page_analysis_service import create_page_analysis_service

service = create_page_analysis_service()

if service.is_enabled():
    # Test with sample data
    stats = {
        'total_rows': 1500,
        'total_columns': 25,
        'promotion_rate': 35.2,
        'promoted_count': 528,
        'not_promoted_count': 972,
        'avg_performance': 78.5,
        'avg_behavioral': 82.3,
        'avg_tenure': 5.2,
        'qa_coverage': 85.0,
        'qa_count': 1275,
    }
    
    print("Generating analysis...")
    analysis = service.analyze_data_explorer(None, stats)
    
    print("\nAnalysis generated:")
    print(analysis[:200] + "...")
    print("\n✅ AI generation working!")
else:
    print("❌ Service not enabled")
EOF
```

### Step 4: Check Console Logs
When running Streamlit, check terminal for:
```
✅ Loaded .env from /path/to/.env
🔍 API Key check: Found
🔍 API Key length: 39
✅ Gemini AI initialized successfully
```

If you see:
```
⚠️ GEMINI_API_KEY not found or invalid
```
Then API key is not loaded correctly.

---

## 📊 Verification Checklist

### Before Generating Analysis:

- [ ] `.env` file exists in project root
- [ ] `GEMINI_API_KEY` is set in `.env`
- [ ] API key is 39 characters long
- [ ] API key starts with `AIzaSy`
- [ ] No spaces or quotes around API key value
- [ ] `python-dotenv` is installed
- [ ] `google-generativeai` is installed

### After Generating Analysis:

- [ ] Analysis appears (not just warning)
- [ ] Numbers in analysis match data
- [ ] No made-up statistics
- [ ] Insights are reasonable
- [ ] Format is correct (markdown headers)
- [ ] Language is professional Indonesian

---

## 🎯 Expected Behavior

### Data Explorer Analysis:
```markdown
## 📊 Ringkasan Dataset

Dataset ini berisi 1,500 karyawan dengan 25 fitur...
[Uses actual numbers from stats]

## 🎯 Temuan Utama

- Performance Score rata-rata 78.5...
[References actual avg_performance value]
```

### EDA Results Analysis:
```markdown
## 📈 Analisis Distribusi

Dataset menunjukkan distribusi 35.2% promoted vs 64.8% not promoted...
[Uses actual promoted_pct and not_promoted_pct]

## 🔍 Perbedaan Kelompok

Karyawan yang dipromosikan memiliki:
- Performance score 8.5 poin lebih tinggi...
[Uses actual difference: promoted_perf_mean - not_promoted_perf_mean]
```

### Model Performance Analysis:
```markdown
## 🏆 Performa Model Terbaik

XGBoost Classifier mencapai accuracy 87.5%...
[Uses actual best_model.accuracy value]

## 📊 Analisis Metrics

- Accuracy 87.5%: Model sangat akurat...
[References exact accuracy percentage]
```

---

## 🚀 Quick Fixes

### Quick Fix 1: Reset Everything
```bash
cd /Users/denisulaeman/Workspace/Academic/CascadeProjects/MPCIM_Thesis

# Pull latest code
git pull origin qa-integration-complete

# Verify .env
cat .env | grep GEMINI_API_KEY

# Reinstall dependencies
pip install -r app/requirements.txt

# Restart Streamlit
streamlit run app/Home.py
```

### Quick Fix 2: Regenerate API Key
1. Go to https://makersuite.google.com/app/apikey
2. Delete old key
3. Create new key
4. Update `.env`:
   ```bash
   GEMINI_API_KEY=NEW_KEY_HERE
   ```
5. Restart Streamlit

### Quick Fix 3: Use Fallback
If AI not working:
- Fallback analysis is automatic
- Always accurate
- Based on actual data
- No AI needed

---

## 📞 Still Having Issues?

### Check These Files:
1. `.env` - API key configuration
2. `app/services/page_analysis_service.py` - Service implementation
3. `app/pages/2_📈_EDA_Results.py` - EDA analysis integration

### Verify Installation:
```bash
pip list | grep -E "google-generativeai|python-dotenv|streamlit"
```

Should show:
```
google-generativeai    0.3.0+
python-dotenv          1.0.0+
streamlit              1.29.0+
```

### Test Gemini API Directly:
```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

response = model.generate_content("Test")
print(response.text)
```

---

## ✅ Success Indicators

### You know it's working when:

1. **Console shows**:
   ```
   ✅ Loaded .env from /path/to/.env
   ✅ Gemini AI initialized successfully
   ```

2. **No warning appears** when clicking "Generate AI Analysis"

3. **Analysis contains**:
   - Actual numbers from your data
   - Professional Indonesian language
   - Markdown formatted sections
   - Reasonable insights

4. **Numbers match** visualizations and tables

5. **Analysis completes** in 5-30 seconds

---

**Last Updated**: November 18, 2025, 9:35 AM  
**Version**: 2.0 (Anti-Hallucination Update)  
**Status**: Production Ready ✅
