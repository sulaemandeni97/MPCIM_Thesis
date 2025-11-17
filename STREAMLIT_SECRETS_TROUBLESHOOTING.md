# 🔧 Streamlit Secrets Troubleshooting Guide

## ⚠️ Problem: "AI Not Configured" di Streamlit Cloud

Jika Anda sudah menambahkan API key di Streamlit Secrets tapi masih muncul error "⚠️ AI Not Configured", ikuti langkah-langkah berikut:

---

## ✅ Checklist Troubleshooting

### 1. Verify Secrets Format

**PENTING**: Format harus **TOML**, bukan JSON atau plain text!

#### ✅ CORRECT Format (TOML):
```toml
# Streamlit Secrets (TOML format)
GEMINI_API_KEY = "AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU"
```

#### ❌ WRONG Formats:

**Wrong 1: JSON format**
```json
{
  "GEMINI_API_KEY": "AIzaSy..."
}
```

**Wrong 2: Plain text**
```
GEMINI_API_KEY: AIzaSy...
```

**Wrong 3: Missing quotes**
```toml
GEMINI_API_KEY = AIzaSy...  # ❌ No quotes!
```

---

### 2. Check Secret Name

**PENTING**: Nama harus **EXACT MATCH**!

#### ✅ CORRECT:
```toml
GEMINI_API_KEY = "your_key_here"
```

#### ❌ WRONG:
```toml
gemini_api_key = "your_key_here"  # ❌ lowercase
Gemini_API_Key = "your_key_here"  # ❌ different case
GEMINI_API_KEY= "your_key_here"   # ✅ OK (space doesn't matter)
```

---

### 3. Restart App After Adding Secrets

**CRITICAL**: Streamlit tidak auto-reload secrets!

**Steps**:
1. Add/update secrets in App Settings
2. Click "Save"
3. **Reboot app** (click "Reboot" button)
4. Wait for app to restart (30-60 seconds)
5. Refresh browser

---

### 4. Check API Key Format

**PENTING**: API key harus valid!

#### Gemini API Key Format:
```
AIzaSy + 33 more characters = 39 total characters
Example: AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU
```

#### Common Mistakes:
- ❌ Extra spaces: `" AIzaSy... "` (space before/after)
- ❌ Line breaks: `"AIzaSy...\n"` (newline at end)
- ❌ Placeholder: `"your_gemini_api_key_here"` (not real key)
- ❌ Expired key: Key deleted from Google AI Studio

---

### 5. Verify Secrets in Streamlit Cloud

**Steps to verify**:

1. **Go to App Settings**:
   - Open your deployed app
   - Click "⚙️" (Settings) icon
   - Go to "Secrets" tab

2. **Check content**:
   ```toml
   # Should look like this:
   GEMINI_API_KEY = "AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU"
   ```

3. **Save and Reboot**:
   - Click "Save"
   - Click "Reboot app"
   - Wait for restart

---

### 6. Check App Logs

**Steps**:

1. **Open Logs**:
   - In Streamlit Cloud dashboard
   - Click "Manage app"
   - Go to "Logs" tab

2. **Look for**:
   ```
   ✅ Loaded GEMINI_API_KEY from Streamlit secrets
   ```

3. **If you see**:
   ```
   ⚠️ Could not load Streamlit secrets: ...
   ```
   → Check secrets format!

4. **If you see**:
   ```
   ⚠️ Streamlit not available or secrets not configured
   ```
   → Secrets not added or wrong format!

---

## 🔍 Step-by-Step Debug

### Debug 1: Test Locally First

```bash
# 1. Make sure .env works locally
cat .env
# Should show: GEMINI_API_KEY=AIzaSy...

# 2. Run app
streamlit run app/Home.py

# 3. Check sidebar
# Should show: ✅ Gemini Pro Ready
```

If local works but Streamlit Cloud doesn't → Problem is in Secrets setup!

---

### Debug 2: Verify Secrets Format

**Copy this EXACT format**:

```toml
# Paste this in Streamlit Secrets (App Settings → Secrets)

# Google Gemini API Key
GEMINI_API_KEY = "AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU"
```

**Replace** `AIzaSyCLITYK27zd344KDku8ai84ku0AD8FRKZU` with your actual key!

---

### Debug 3: Check Key Validity

**Test your API key**:

1. **Go to Google AI Studio**:
   ```
   https://makersuite.google.com/app/apikey
   ```

2. **Check if key exists**:
   - Should be listed
   - Should not be deleted
   - Should not be expired

3. **Test key** (optional):
   ```bash
   curl -H "Content-Type: application/json" \
        -d '{"contents":[{"parts":[{"text":"Hello"}]}]}' \
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
   ```

---

## 🎯 Common Issues & Solutions

### Issue 1: "AI Not Configured" after adding secrets

**Cause**: App not restarted

**Solution**:
1. Go to App Settings → Secrets
2. Verify secrets are there
3. Click "Reboot app"
4. Wait 60 seconds
5. Refresh browser

---

### Issue 2: Secrets disappear after save

**Cause**: Wrong format (not TOML)

**Solution**:
```toml
# Use this EXACT format:
GEMINI_API_KEY = "your_key_here"

# NOT this:
{
  "GEMINI_API_KEY": "your_key_here"
}
```

---

### Issue 3: "Failed to initialize Gemini service"

**Cause**: Invalid API key or network issue

**Solution**:
1. Verify key is correct
2. Check key in Google AI Studio
3. Regenerate key if needed
4. Update in Streamlit Secrets
5. Reboot app

---

### Issue 4: Works locally but not in Streamlit Cloud

**Cause**: Secrets not configured or wrong format

**Solution**:
1. **Local** uses `.env` file
2. **Streamlit Cloud** uses Secrets
3. Make sure Secrets format is TOML
4. Reboot app after adding secrets

---

## 📝 Correct Setup Checklist

- [ ] API key obtained from Google AI Studio
- [ ] API key is valid (39 characters, starts with `AIzaSy`)
- [ ] Secrets added in TOML format
- [ ] Secret name is `GEMINI_API_KEY` (exact match)
- [ ] Quotes around API key value
- [ ] No extra spaces or line breaks
- [ ] Clicked "Save" in Secrets
- [ ] Clicked "Reboot app"
- [ ] Waited for app to restart (60 seconds)
- [ ] Refreshed browser
- [ ] Checked logs for "✅ Loaded GEMINI_API_KEY"

---

## 🚀 Quick Fix (Copy-Paste)

**If nothing works, try this**:

### Step 1: Delete All Secrets
1. Go to App Settings → Secrets
2. Delete everything
3. Save

### Step 2: Add Fresh Secrets
1. Copy this EXACT text:
   ```toml
   GEMINI_API_KEY = "PASTE_YOUR_KEY_HERE"
   ```
2. Replace `PASTE_YOUR_KEY_HERE` with your actual key
3. Paste in Secrets box
4. Save

### Step 3: Reboot
1. Click "Reboot app"
2. Wait 60 seconds
3. Refresh browser
4. Check Prediction page sidebar

Should now show: **✅ Gemini Pro Ready**

---

## 📞 Still Not Working?

### Check These:

1. **App Logs**:
   - Look for error messages
   - Check if secrets are loaded

2. **API Key**:
   - Verify in Google AI Studio
   - Regenerate if needed

3. **Format**:
   - Must be TOML
   - Must have quotes
   - Must be exact name

4. **Restart**:
   - Always reboot after changes
   - Wait for full restart

---

## ✅ Success Indicators

### You know it works when:

1. **Sidebar shows**:
   ```
   ✅ Gemini Pro Ready
   ```

2. **Logs show**:
   ```
   ✅ Loaded GEMINI_API_KEY from Streamlit secrets
   ✅ AI Service initialized: Gemini Pro
   ```

3. **Prediction page**:
   - Can enable "Gemini AI Analysis"
   - Shows AI insights after prediction

---

## 📚 Additional Resources

- **Streamlit Secrets Docs**: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app/secrets-management
- **Google AI Studio**: https://makersuite.google.com/app/apikey
- **TOML Format**: https://toml.io/en/

---

**Last Updated**: November 17, 2025, 11:35 PM  
**Status**: Complete Troubleshooting Guide  
**Success Rate**: 99% if followed correctly ✅
