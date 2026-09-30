# KO Trader v7 - Keyfiyyət Nəzarəti Hesabatı

**Tarix:** 2026-09-30  
**Versiya:** 8.1  
**Status:** ⚠️ Təhsil Amaçlı | Production-Ready Deyil

---

## 📋 İcmal

| Kateqoriya | Vəziyyət | Ciddilik |
|-----------|---------|----------|
| Kodu Dublikaları | ❌ Tapıldı | Aşağı |
| Path/Directory Mühüm | ❌ Təhlükəli | Orta |
| Error Handling | ❌ Zəif | Orta-Yüksək |
| API Key Management | ⚠️ Riskli | Yüksək |
| Versioning | ❌ Yoxdur | Orta |
| Logging | ❌ Yoxdur | Aşağı-Orta |
| Testing | ❌ Yoxdur | Yüksək |
| Documentation | ⚠️ Natiq | Orta |

---

## 🐛 TAPıLAN PROBLEMLƏR (TEFƏRRÜATLı)

### **1. DUPLICATE IMPORTS (robot.py, sətir 12-15)**

```python
# ❌ PROBLEM
import os, json, pickle, warnings, traceback, csv
from datetime import datetime    
import os, json, pickle, warnings, traceback, csv  # ← DUBLIKAT!
from datetime import datetime
```

**Təsiri:** 
- Kodun oxunaqlılığı aşağı düşür
- Kompüter əlavə işçilik görür (minimal)
- Proqram çalışır, amma "lazy code" işarəsi

**Həlli:** Birini sil (sətir 14-15 silin)

---

### **2. GLOBAL PATH DEPENDENCY (Hər yerə `data/` yazılıb)**

**Yerlər:**
- robot.py: `DATA_DIR = "data"`
- macro_data.py: `DATA_DIR = "data"`
- news_sentiment.py: `DATA_DIR = "data"`

**Təsiri:**
- Proqram `/home/user/project` içində işlədikdə OK
- Amma `/var/app/runner` içində işlədikdə `data/` tapılmaz
- Server/Docker-də çalışmaz
- Qovluq yetkinliyi sorunu ola bilər

**Problem Ssenari:**
```bash
cd /home/projects/KO-Trader-v7
python robot.py  # ✅ Işləyir

cd /tmp
python /home/projects/KO-Trader-v7/robot.py  # ❌ FAIL - data/ tapılmaz
```

**Həlli:** Absolute path istifadə et
```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
```

---

### **3. API KEY / TOKEN SEÇKİSİ (Açıq Rahatsızlıq)**

**Yerlər:**
- robot.py, sətir 60: `token = os.getenv("TELEGRAM_BOT_TOKEN")`
- news_sentiment.py, sətir 47: `key = os.getenv("FINNHUB_API_KEY")`
- macro_data.py, sətir 94: `key = os.getenv("FINNHUB_API_KEY")`
- macro_data.py, sətir 170: `key = os.getenv("FRED_API_KEY")`

**Təsiri:**
- Token-lər environment-də olmalıdır
- `.env` faylı yoxdur (instructions yoxdur)
- Səhvsə, kodda "silent fail" olur
- Production-da keçə bilər

**Təhlükəsizlik Riskləri:**
- GitHub-a `.env` upload olunsa, token-lər açıq kalır
- `.gitignore` yoxdur (əgər dərsə)
- Bəzən hardcoded key-lər ola bilər

**Həlli:** 
1. `.env.example` şablonu yarat
2. `.env` faylını `.gitignore`-a əlavə et
3. `python-dotenv` istifadə et

---

### **4. WEAK ERROR HANDLING (Çoxlu Bare Except)**

**Problem Yerlər:**

robot.py:
- sətir 80: `except:` → silib, `except Exception as e: print(f"cache error: {e}")`
- sətir 107: `except:` → hiçbir log
- sətir 193: `except Exception as e: print(...)` → OK, ama sadəcə print
- sətir 221: `except: model = build_model(input_shape)` → Silent fail

**Təsiri:**
- Hata olsa, sən nə olduğunu bilmirsən
- Debug etmək çətinləşir
- Production-da "niyə işlətmir?" sorusu cəvabsız qalır

**Bəd Nümunə:**
```python
# ❌ PROBLEM
try:
    load_model(path)
except:  # Ne oldu? Bilmirsən.
    pass

# ✅ DÜZƏLT
try:
    load_model(path)
except Exception as e:
    logger.error(f"Model yüklənə bilmədi {path}: {e}")
    # Və ya fallback: yeni model yarat
```

**Həlli:** Logging sistemi əlavə et

---

### **5. NO VERSIONING (requirements.txt)**

```python
# ❌ Versyon yoxdur
yfinance
pandas
numpy
tensorflow
```

**Ssenari:**
```bash
# Kompüter A (2026-09-30)
pip install yfinance  # 0.2.32 qurulur

# Kompüter B (2026-12-01)
pip install yfinance  # 0.3.0 qurulur → API DƏYIŞDI!

# robot.py fərqli davranır! Səhvlər başlanır.
```

**Həlli:** Versiyon pin et
```
yfinance==0.2.32
pandas==2.1.0
numpy==1.24.0
tensorflow==2.13.0
scikit-learn==1.3.0
vaderSentiment==3.3.2
pytz==2023.3
requests==2.31.0
python-dotenv==1.0.0
```

---

### **6. NO LOGGING SYSTEM**

**Problemlər:**
- `print()` istifadə edilir (sadəcə ekrana)
- Hata logları saxlanılmır
- Proqram çalışdıqdan sonra əvvəlki messajlar itir
- Server-də (cron job) print-lər harada?

**Ssenari:**
```bash
# Cron job olarak çalışan robot
0 9 * * * cd /var/app && python robot.py

# print() nereye gidiyor? Bilinmiyor.
# Hata olsa, kim biliyor?
```

**Həlli:** Python logging module istifadə et

---

### **7. MODEL RETRAIN HER GÜN (Inefficient)**

robot.py, sətir 223-226:
```python
if TF_AVAILABLE:
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=30, batch_size=16, verbose=0, validation_split=0.2, callbacks=[early_stop])
    model.save(brain_path)
```

**Problemlər:**
- Model HER GÜN yenidən tətim olunur (30 epoch!)
- Data əvvəlki kəsintidə əlavə edilir (concat)
- Əgər data-da drift olarsa, model "üşüyür"
- Retraining strategy yoxdur

**Təsiri:**
- Yavaş işləmə
- Model stability sorunu
- "Overfitting" riski artır

**Həlli:** 
- Əgər model artıq varsa, test et, yenidən tətim etmə
- Həftədə bir və ya "performance düşdü" olunca tətim et

---

### **8. PAPER TRADING LOGIC (Sadə, Riskli)**

robot.py, sətir 389-409:
```python
# AL - confidence >= 60%
if signal == "AL" and conf >= 60 and wallet["shares"] == 0:
    wallet["shares"] = wallet["balance"] / last_price
    
# SAT - stop loss -3%, take profit +4%, və ya signal
if is_sl or is_tp or is_sig:
    wallet["balance"] = wallet["shares"] * last_price
```

**Problemlər:**
1. Stop Loss / Take Profit hard-coded (-3%, +4%)
   - Hansı basis? Niyə bu rəqəmlər?
   - Volatilite öl qeyd olunmur

2. Confidence Threshold (60%)
   - Necə hesablanır? (model.predict() output)
   - 55% vs 60% arasında nə fərq?

3. Real World Farq:
   - Paper trading ≠ Real trading
   - Commission, slippage, liquidity yoxdur
   - Market gaps olur, limit order-lar fail ola bilər

**Həlli:** 
- Parametrləri konfigurəsiya faylında qoy
- "Why this value?" documentation yaz

---

### **9. NO DATA VALIDATION**

prepare_xy(), sətir 120-195:
```python
# Data gəldikdə, sən çox az check edirsən
close = df['Close']
if isinstance(close, pd.DataFrame): close = close.iloc[:,0]

# Amma:
# - Close NaN olsa? (checked: fillna() var)
# - Close < 0 olsa? (check yoxdur!)
# - Volume = 0 olsa? (check yoxdur)
# - Futures date-i past-da olsa? (check yoxdur)
```

**Təsiri:**
- Bad data → bad model
- Silent fail → qayd edilməmiş xəta

**Həlli:** Data validation əlavə et

---

### **10. NO UNIT TESTS**

Cəbbə bir test yoxdur.

**Ssenari:**
```python
# Tu sənə bir patch göndərdi
# Sən `robot.py` dəyişdin
# 2 saat sonra qəbul etdin
# Amma üç ay sonra... "Niyə paper wallet hesabı səhvdir?" 🤔

# Test olsa idi:
# pytest → OK/FAIL → Sənə dəqiqcə söylərdi "bu xəta buradan"
```

**Həlli:** Basic tests yaz

---

## 🔧 OXŞARLıQ / RISK MATRİSİ

```
RISK        ÇƏTINLIK  MÜHÜMLÜK
────────────────────────────────
Dublikat    Çox Aşağı  Aşağı      → Dərhal düzəlt
Path Bug    Orta       Orta       → Dərhal düzəlt
API Keys    Aşağı      Yüksək     → Dərhal düzəlt
Error Log   Orta       Orta       → Dərhal düzəlt
Versioning Orta       Orta       → Dərhal düzəlt
Model Cache Yüksək    Aşağı-Orta → Sonra
Testing    Yüksək     Yüksək     → Long-term
```

---

## ✅ TAMAMLAMA PLAN

### **AŞAMA 1: KRITIK (Bu Həftə)**
- [ ] Dublikat imports sil
- [ ] Path handling düzəlt (BASE_DIR)
- [ ] .env.example yarat
- [ ] .gitignore yarat
- [ ] requirements.txt versioning əlavə et
- [ ] Error handling iyilişdir (logging)
- [ ] README update et

### **AŞAMA 2: ƏHƏMIYYƏTLI (Gələn Həftə)**
- [ ] Model retraining logic refactor et
- [ ] Config file (parameters) əlavə et
- [ ] Basic data validation

### **AŞAMA 3: LONG-TERM**
- [ ] Unit tests yaz
- [ ] Model versioning
- [ ] Backtesting framework

---

## 📝 QEYDLƏR

Bu layihə **təhsil/research amaçlı** təyin olunmuşdur. Production trading bot deyil.

Amma səhvlərdən qaçınmaq üçün, ən az "başlanğıc hijjiyası" tələb olunur:

1. Kod oxunaqlı olmalıdır
2. Xətalar qeyd olunmalıdır
3. Versiya tutulmalıdır
4. Confidential data güvənli olmalıdır

---

**Hazırlayan:** GitHub Copilot  
**Status:** Ready for Patching
