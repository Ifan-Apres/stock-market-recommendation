# 📈 Stock Market Recommendation: AI & Quant Platform

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FIfan-Apres%2Fstock-market-recommendation)
[![Live Web Application](https://img.shields.io/badge/Vercel_Live_App-Online-black?logo=vercel&logoColor=white)](https://stock-market-recommendation-silk.vercel.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Ifan--Apres%2Fstock--market--recommendation-blue?logo=github)](https://github.com/Ifan-Apres/stock-market-recommendation)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Platform Riset Ekuitas Institusional & Rekomendasi Saham Kuantitatif Berbasis Multi-Engine Machine Learning (GBDT + Macro-Aware PyTorch LSTM + ARIMA + GARCH), Scraper Resmi Arus Modal Asing BEI (IDX Foreign Flow), Terminal Teknikal Multi-Pane (Lightweight Charts), Matriks Eksekusi Taktis & Position Sizer, Analisis Fundamental Laporan Keuangan Multi-Tahun + AI Archetypes (Gemini 3.8 Flash), dan Dynamic Universe Manager untuk Bursa Efek Indonesia (BEI / IDX).**  
> **Dikembangkan dan Dikelola oleh TIM New York: Ifan Apres & Sekar Widhastri.**

---

## 🌟 Ikhtisar Proyek (Project Overview)

**Stock Market Recommendation** adalah platform investasi saham dan riset kuantitatif tingkat institusional yang dirancang dengan standar terminal Wall Street: antarmuka gelap berorientasi data (*dense high-contrast dark terminal*), responsif, dan berlandaskan komputasi matematis presisi tanpa estimasi sembarangan (*zero-guessing doctrine*). Platform ini mengintegrasikan:

1. **Scraper Resmi Arus Modal Asing BEI (*IDX Foreign Flow Engine*)**: Mengambil ringkasan perdagangan harian (*Trading Summary*) langsung dari bursa resmi dengan impersonasi browser anti-Cloudflare, menghitung akumulasi/distribusi dana asing per saham hingga ke nominal Rupiah terakhir secara deterministik.
2. **Terminal Analisis Teknikal & Bandarmologi (Pro Studio Multi-Pane)**: Terminal grafik canggih berbasis TradingView Lightweight Charts dengan tumpukan multi-pane tersinkronisasi: harga candlestick/garis, overlay dinamis (EMA 10/20/50/200, Bollinger Bands 20,2, level target TP/SL), sub-pane MACD (12,26,9), RSI (14), Volume SMA-20, serta grafik batang *Net Foreign Flow* 65 hari bursa.
3. **Tactical Execution Matrix & Integrated Position Sizer**: Matriks eksekusi taktis ATR-14 dengan konvergensi *Single Source of Truth* harga pasar. Menghitung otomatis area akumulasi (*Buy on Weakness*), level konfirmasi (*Buy on Breakout*), target take profit berjenjang (TP 1, TP 2, Target Utama), batas risiko (*Cut Loss*), serta kalkulator alokasi lot dan proyeksi nominal Rupiah terukur.
4. **Machine Learning Diperkaya Arus Asing (GBDT + PyTorch LSTM 14-Dimensi)**: Memprediksi *Target Excess Alpha Relatif* ($R_{\text{saham}, 5D} - R_{\text{IHSG}, 5D} > 0$) menggunakan perpaduan indikator teknikal, variabel makro lintas aset (TNX, USD/IDR, Brent, S&P 500), dan laju perputaran dana asing.
5. **Deep Dive Fundamental Multi-Tahun & AI Archetypes (Gemini 3.8 Flash)**: Bedah tuntas laporan keuangan historis 4–5 tahun dengan standardisasi format mata uang ganda (USD vs IDR), evaluasi risiko kontraksi omzet berturut-turut (*top-line contraction*), kartu DNA emiten (*Blue Chip, Swing Trading Pick, High Quality Business, Foreign Flow Magnet*), serta ulasan naratif 4 pilar bisnis.
6. **Cross-Sectional Top Decile & High-Conviction Thresholding**: Menyaring *noise* pasar dengan hanya mengeksekusi rekomendasi BUY pada ambang probabilitas keyakinan $\ge 0.55$ atau saham-saham peringkat 10% teratas bursa.
7. **Institutional Morning Brief AI**: Riset pembuka sesi terkurasi dengan scraping berita makro semalam, ringkasan eksekutif 10-detik, dan doktrin riset **AlphaTech**.
8. **Ekonometrika Risiko Lanjutan**: Pemodelan volatilitas kondisional Student-t GARCH(1,1), Value at Risk (VaR 95% & 99%), Expected Shortfall (ES), dan alokasi portofolio optimal dengan **20% Kas Siaga Wajib**.

---

## 📊 Hasil Uji Validasi & Benchmark Performa Machine Learning

Evaluasi model dilakukan secara ketat menggunakan *Out-of-Sample Test Split* (~14.600 observasi pasar historis) pada horizon prediksi alpha 5 hari bursa ke depan:

| Metrik Evaluasi Kuantitatif | Baseline (Model Lama) | Model Baru (+ Foreign Flow BEI) | Peningkatan / Delta | Dampak Praktis bagi Investor |
| :--- | :---: | :---: | :---: | :--- |
| **High-Conviction Precision (Prob $\ge$ 55%)** | 51.23% | **55.78%** | <span style="color:green">**+4.55% 🚀**</span> | **Lompatan signifikan!** Memfilter sinyal palsu (*retail trap*) dengan konfirmasi akumulasi institusi asing. |
| **Top Decile Precision (Top 10% Paling Kuat)** | 53.81% | **56.00%** | <span style="color:green">**+2.19% 🚀**</span> | Saham-saham dengan peringkat keyakinan tertinggi menghasilkan *win-rate outperformance* paling konsisten. |
| **GBDT ROC-AUC Score** | 0.5401 | **0.5433** | <span style="color:green">**+0.0032**</span> | Pemisahan probabilitas antara saham *outperformer* dan saham *underperformer* semakin tajam. |
| **GBDT Overall Precision** | 53.79% | **54.48%** | <span style="color:green">**+0.69%**</span> | Kualitas sinyal beli secara keseluruhan meningkat di seluruh semesta saham. |
| **GBDT Overall Accuracy** | 53.19% | **53.45%** | <span style="color:green">**+0.26%**</span> | Akurasi arah pergerakan alpha saham terhadap indeks IHSG semakin solid. |
| **PyTorch LSTM ROC-AUC** | 0.5259 | **0.5298** | <span style="color:green">**+0.0039**</span> | Model sekuensial deep learning lebih tajam membaca tren akumulasi bertahap. |
| **PyTorch LSTM Loss** | 0.6920 | **0.6919** | <span style="color:green">**-0.0001**</span> | Konvergensi pelatihan neural network lebih stabil. |

---

## 🚀 Fitur Unggulan (Core Architecture)

### 1. 🌐 Direct IDX Foreign Flow Scraper & Graceful Fallback
* **Bypass Cloudflare WAF BEI**: Menggunakan [src/idx_scraper.py](file:///d:/Portofolio/stock-market-recommendation/src/idx_scraper.py) dengan library `curl_cffi` impersonasi TLS `safari18_0` dan `chrome124`. Menembus proteksi situs resmi BEI tanpa blokir IP atau Captcha.
* **Presisi Nominal Transaksi**: Mengambil data resmi:
  $$\text{VWAP} = \frac{\text{Value}}{\text{Volume}}, \quad \text{Net Foreign IDR} = (\text{ForeignBuy} - \text{ForeignSell}) \times \text{VWAP}$$
* **Disk Caching Berkecepatan Tinggi**: Data harian 963 saham disimpan dalam cache ringkas di `data/raw/idx_daily/` (~50 KB/hari). Pipeline hanya mengambil hari baru setiap sore.
* **Graceful Fallback Safeguard**: Jika bursa libur atau pipeline berjalan saat sesi perdagangan berlangsung, sistem otomatis beralih ke *Institutional Order Flow Directional Pressure Model*, menjamin **100% ketersediaan data dan zero-downtime**.

### 2. 🕯️ Terminal Analisis Teknikal & Bandarmologi Multi-Pane (Lightweight Charts Pro)
* **High-Performance Multi-Pane Canvas**: Menggunakan TradingView Lightweight Charts dengan sinkronisasi *crosshair* dan *time scale* lintas panel secara real-time.
* **Overlays Interaktif Mandiri**: Pengguna dapat mengaktifkan/menonaktifkan EMA 10, EMA 20, EMA 50, EMA 200, Bollinger Bands (20,2), serta garis horizontal TP/SL langsung di atas grafik harga.
* **Sub-Panes Terdedikasi**:
  * **MACD (12, 26, 9)**: Histogram momentum dengan pewarnaan dinamis bull/bear.
  * **RSI (14)**: Dilengkapi garis acuan batas oversold (30), netral (50), dan overbought (70).
  * **Volume Transaksi**: Dilengkapi kurva rata-rata bergerak SMA 20 hari.
  * **Net Foreign Flow (3 Bulan / 65 Hari)**: Histogram arus modal asing riil harian (akumulasi hijau vs distribusi merah).
* **Live OHLC Legend Dinamis**: Membaca otomatis posisi kursor kueri Open, High, Low, Close, serta persentase perubahan harian relatif terhadap *previous close*.

### 3. 🎯 Tactical Execution Matrix & Integrated Position Sizer
* **Single Source of Truth Price Engine**: Mengeliminasi disparitas data antara kartu profil, legenda terminal teknikal, dan matriks order. Seluruh modul mengonsumsi harga penutupan candle pasar yang sama.
* **6 Precision Levels Eksekusi Taktis**:
  * *Buy on Weakness (BOW)*: Rentang akumulasi di area support dinamis.
  * *Buy on Breakout (BOB)*: Level picu konfirmasi momentum di atas resistance kunci.
  * *Take Profit 1 (Taktis)* & *Take Profit 2 (Optimal)*: Target penguncian profit parsial.
  * *Target Utama*: Target ekspansi tren harga berdasarkan proyeksi volatilitas ATR-14.
  * *Cut Loss*: Batas toleransi risiko ketat di bawah support psikologis.
* **Long Position Logical Guard Rail**: Memvalidasi secara otomatis bahwa seluruh level Take Profit posisi Long berada di atas harga beli (Entry), mengoreksi anomali data usang atau korup secara real-time.
* **Position Sizer & Risk Calculator**: Kalkulator alokasi lot otomatis berdasarkan modal nominal (IDR) dengan fraksi 100 lembar/lot. Menampilkan nominal beli, sisa kas siaga, serta potensi untung/rugi nominal (Rp) dan persentase (%) yang terkalibrasi persis dengan target taktis.

### 4. 🧠 Macro-Aware PyTorch LSTM & GBDT dengan Fitur Arus Asing (14-Dimensi)
Fitur input kuantitatif diperkaya dengan 4 dimensi arus asing:
* `Foreign_Flow_Norm_1D`: Net foreign flow IDR dinormalisasi terhadap rata-rata perputaran transaksi 20 hari ($\text{Net Foreign} / \text{Value SMA 20}$).
* `Foreign_Flow_5D_Accum`: Akumulasi arus modal asing 5 hari beruntun untuk mendeteksi akumulasi bertahap (*stealth loading*).
* `Foreign_Participation`: Rasio keterlibatan volume asing terhadap total likuiditas saham.
* `Foreign_Flow_Momentum`: Kecepatan akselerasi modal asing masuk/keluar dalam 3 hari bursa.
* **Fitur Makro & Sektoral**: Yield US 10Y, USD/IDR, Minyak Brent, `Oil_Energy_Tailwind`, `Rate_Bank_Sensitivity`, `FX_Consumer_Headwind`.

### 5. 📑 Deep Dive Laporan Keuangan Multi-Tahun & AI Archetypes (Gemini 3.8 Flash)
* **Standarisasi Mata Uang Ganda (Dual-Currency Standard)**:
  * **USD (Dolar AS)**: Menggunakan format internasional seragam `$B` (Miliar), `$M` (Juta), dan `$T` (Triliun) untuk emiten berbasis pembukuan Dolar (misal: PTBA, ADRO, MEDC, HEXA).
  * **IDR (Rupiah)**: Menggunakan format lokal institusional `Rp Jt`, `Rp M`, dan `Rp T`.
* **Presisi Desimal EPS Seragam**: Mengikuti standar Wall Street (US GAAP / SEC 10-K), baris Laba Per Saham (EPS) untuk denominasi USD diformat secara seragam dengan **2 angka di belakang koma (`$X.XX`)** di seluruh kolom tahun historis.
* **Deteksi Kontraksi Omzet (Top-Line Contraction Alert)**: Sistem otomatis menurunkan status kesehatan finansial emiten yang mengalami kontraksi pendapatan YoY berturut-turut, menyuntikkan peringatan risiko objektif pada sintesis riset AI Gemini 3.8 Flash.
* **Kartu DNA & Archetype Emiten**:
  * 💎 **Blue Chip LQ45**: Saham pilar indeks berkapitalisasi raksasa dan likuiditas tinggi.
  * ⚡ **Prime Swing Trading**: Saham dengan setup momentum kuantitatif berpeluang *breakout* tinggi.
  * ⭐ **Fundamental Kokoh**: Saham dengan ROE prima dan solvabilitas utang terkendali (DER < 1.0x).
  * 💰 **Foreign Flow Magnet**: Saham yang menjadi target akumulasi bersih institusi asing global.
* **Research Switcher 3-in-1**: Tab interaktif untuk berpindah antara 6 Level Presisi Eksekusi, Ulasan Naratif Tren & Price Action, dan 4 Pilar Fundamental Bisnis (Laba Bersih & EPS, Pendapatan & Ops, EBITDA & Margin, Struktur Neraca).

### 6. 📰 Institutional Morning Brief AI (AlphaTech Doctrine)
* **Scraping Makro Semalam**: Mengambil katalis dari Wall Street (S&P 500), geopolitik minyak Brent, dan indeks Dolar AS.
* **Executive Key Takeaways 10-Detik**: Ringkasan Arah Indeks, Katalis Global, Risiko Makro, dan Panduan Taktis Alokasi Kas.
* **Fallback Otomatis**: Generator kuantitatif deterministik yang siap menggantikan narasi AI jika terjadi kuota limit/rate limit.

### 7. 📉 Ekonometrika Risiko Lanjutan & Standardisasi VaR (GARCH-t)
* **Student-t GARCH(1,1)**: Memodelkan volatilitas kondisional untuk mengantisipasi risiko ekor tebal (*fat-tail risk*) pada 66 emiten aktif.
* **Value at Risk (VaR 95% & 99%)** & **Expected Shortfall (ES)**: Estimasi ilmiah batas potensi penurunan maksimum harian dalam format desimal terstandarisasi (`[0.005, 0.25]` / 0.5% - 25%), dilengkapi *dual-guard frontend & API formatter* untuk mengeliminasi anomali scaling (misal: `-3.86%` bukan `-386.00%`).
* **20% Kas Siaga Wajib**: Menjamin ketersediaan likuiditas cadangan pada optimasi alokasi portofolio kuantitatif.
* **IDX Tick Size Rounding**: Level Entry, Target Price (TP), dan Stop Loss (SL) otomatis dibulatkan sesuai fraksi harga resmi Bursa Efek Indonesia.

### 8. 🛡️ Multi-Model Consensus Shield & No-Divergence Guard
* **Hard High-Conviction Floor ($\ge 0.55$)**: Menghapus ambang longgar top-decile 0.52. Setiap sinyal BUY wajib memiliki probabilitas gabungan minimal 55%.
* **Majority Agreement Rule**: Minimal 2 dari 3 model (GBDT, PyTorch LSTM, ARIMA) harus sepakat dalam zona *bullish* ($\ge 0.50$).
* **No-Divergence Guard**: Jika ada salah satu model memprediksi *bearish* ($\min(\text{GBDT}, \text{LSTM}, \text{ARIMA}) < 0.48$), saham otomatis berstatus **`HOLD`** (menunggu konfirmasi), melindungi modal dari sinyal BUY palsu saat model bertentangan.
* **Regularisasi PyTorch LSTM Ditingkatkan**: Peningkatan *dropout* menjadi `0.35` dan Adam *weight decay* menjadi `5e-4` guna meredam *overfitting* terhadap *noise* intraday komoditas dan saham siklikal.

---

## 🏛️ Cakupan 11 Sektor Resmi Bursa Efek Indonesia (IDX-IC)

1. **Financials (Keuangan)**: BBCA, BBRI, BMRI, BBNI, BRIS, BBTN
2. **Energy (Energi)**: ADRO, PTBA, ITMG, PGAS, MEDC, AKRA, HRUM, INDY, AADI, ADMR
3. **Basic Materials (Barang Baku)**: ANTM, MDKA, INCO, TPIA, BRPT, INKP, AMMN, MBMA
4. **Consumer Non-Cyclicals (Konsumer Primer)**: ICBP, INDF, UNVR, AMRT, MYOR, CPIN, SIDO
5. **Consumer Cyclicals (Konsumer Non-Primer)**: ASII, ACES, MAPI, ERAA
6. **Healthcare (Kesehatan)**: KLBF, MIKA, HEAL, SILO
7. **Technology (Teknologi)**: GOTO, EMTK, BUKA
8. **Infrastructures (Infrastruktur & Telekomunikasi)**: TLKM, ISAT, EXCL, TOWR, TBIG, PGEO, BREN
9. **Properties & Real Estate (Properti)**: BSDE, CTRA, PWON, SMRA
10. **Industrials (Perindustrian)**: UNTR, HEXA, AUTO, SMSM
11. **Transportation & Logistics (Transportasi & Logistik)**: BIRD, SMDR, ASSA

---

## 📂 Struktur Direktori & File (Directory Architecture)

```text
stock-market-recommendation/
├── .agents/rules/alphatech.md         # Kaidah doktrin sistem AI AlphaTech
├── .github/
│   └── workflows/
│       └── daily_pipeline.yml         # Otomasi GitHub Actions harian (17:00 & 05:00 WIB)
├── api/
│   ├── index.py                       # Serverless handler untuk Vercel
│   └── main.py                        # FastAPI REST API, security middleware, routing
├── data/
│   ├── raw/                           # Ingestion data pasar mentah
│   │   ├── raw_market_data.csv        # Data OHLCV 66 saham aktif (2020 - sekarang)
│   │   ├── benchmark_market_data.csv  # Data historis IHSG (^JKSE)
│   │   ├── global_macro_data.csv      # Snapshot penutupan makro harian
│   │   ├── historical_macro_data.csv  # Time-series harian TNX, USD/IDR, Brent, SP500
│   │   ├── fundamental_financial_data.csv # Snapshot rasio keuangan fundamental
│   │   ├── idx_daily/                 # Cache harian ringkasan saham & arus asing resmi BEI
│   │   └── financial_statements/      # Laporan keuangan per emiten
│   └── processed/                     # Hasil komputasi kuantitatif
│       ├── processed_market_features.csv      # Matriks fitur teknikal, makro & foreign flow
│       ├── advanced_quant_metrics.csv         # Metrik Sharpe, Beta, VaR, Pivots, Foreign Flow
│       ├── active_universe.json               # State semesta dinamis N=66 emiten
│       ├── foreign_flow_summary.json          # Ringkasan arus asing harian macro & 66 emiten
│       ├── financial_statements_summary.json  # Laporan keuangan 5 thn + AI Archetypes + Price History
│       ├── watchlist_analysis.json            # Level taktis 6 level & narasi 4 pilar seluruh emiten
│       ├── price_history_30d.json             # Factual OHLCV 30 hari untuk grafik deep dive
│       ├── alpha_model.joblib                 # Bobot model GBDT terlatih baru
│       ├── lstm_model.pth                     # Bobot PyTorch LSTM 14-dimensi baru
│       ├── latest_morning_brief.json          # Editorial Morning Brief AI + Macro Foreign Flow
│       ├── latest_portfolio_allocation.csv    # Rekomendasi bobot alokasi modal & kas
│       ├── latest_alpha_recommendations_swing.csv     # Rekomendasi Swing Trader (LQ45)
│       ├── latest_alpha_recommendations_dividend.csv  # Rekomendasi Dividend & Value
│       └── latest_alpha_recommendations_favorites.csv # Rekomendasi Portofolio Pilihan
├── src/
│   ├── __init__.py
│   ├── auth.py                        # Sistem autentikasi PBKDF2, HMAC JWT, & Rate Limiter
│   ├── config.py                      # Konfigurasi semesta saham, sektor, & parameter
│   ├── universe_manager.py            # Dynamic Universe Manager & Circuit Breaker
│   ├── idx_scraper.py                 # Scraper resmi BEI anti-Cloudflare (curl_cffi)
│   ├── foreign_flow.py                # Engine arus modal asing hybrid & fallback
│   ├── stock_analyzer.py              # Generator riset taktis 6 level & 4 pilar fundamental
│   ├── 01_data_ingestion.py           # Engine penarikan OHLCV, macro & fundamental
│   ├── 02_feature_eng.py              # Ekstraksi fitur, GARCH, Excess Alpha, Foreign Flow
│   ├── 03_model_inference.py          # Ensemble GBDT+LSTM 14D+ARIMA & Top Decile
│   └── morning_brief.py               # Generator Morning Brief Gemini 3.8 Flash
├── scripts/
│   ├── generate_financials_summary.py # Generator laporan keuangan multi-tahun & sinkronisasi
│   └── sync_model_storage.py          # Enkripsi AES-256 model weights & sync vault
├── index.html                         # Dashboard web modern responsif (TradingView, Deep Dive, FF)
├── login.html                         # Halaman login antarmuka pengguna
├── main.py                            # Master runner pipeline quant
├── requirements.txt                   # Dependensi pustaka Python dasar
├── requirements-pipeline.txt          # Dependensi lengkap pipeline GitHub Actions
├── vercel.json                        # Konfigurasi deployment serverless Vercel
└── README.md                          # Dokumentasi resmi proyek
```

---

## 🛠️ Instalasi & Menjalankan Lokal (Quickstart Guide)

### 1. Prasyarat Sistem
* **Python 3.10 – 3.13** (diuji di Windows, Linux, dan macOS).
* Koneksi internet untuk penarikan data bursa terkini.

### 2. Kloning & Pemasangan Dependensi
```bash
# Kloning repositori
git clone https://github.com/Ifan-Apres/stock-market-recommendation.git
cd stock-market-recommendation

# Buat virtual environment
python -m venv venv

# Aktivasi virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Pasang dependensi pustaka
pip install -r requirements.txt
```

### 3. Konfigurasi Kunci API (`.env`)
Buat file `.env` di root direktori proyek:
```ini
GEMINI_API_KEY=AIzaSy... (Kunci API Google Gemini Anda)
JWT_SECRET_KEY=kunci_rahasia_jwt_minimal_32_karakter_bebas
ADMIN_PIPELINE_SECRET=kunci_rahasia_admin_pipeline_automation
```

### 4. Menjalankan Master Pipeline Kuantitatif
Untuk menjalankan seluruh tahapan komputasi (*Ingestion $\rightarrow$ Feature Engineering $\rightarrow$ Model Retraining & Inference $\rightarrow$ Foreign Flow $\rightarrow$ Morning Brief AI $\rightarrow$ Financials Deep Dive $\rightarrow$ Watchlist Analyzer*):

```bash
python main.py
```

### 5. Menjalankan Server Dashboard & REST API
```bash
uvicorn api.main:app --reload --port 8080
```
Buka peramban (*browser*) Anda di:
* **Dashboard Web Interaktif**: [http://localhost:8080/](http://localhost:8080/)
* **Dokumentasi Interaktif API**: Tersedia di lingkungan lokal pengembang (`ENABLE_DOCS=true`) dan dinonaktifkan di *production* untuk mencegah *reconnaissance* publik.

---

## 🛡️ Arsitektur Keamanan & Pertahanan Aset Kuantitatif (Enterprise Defense Perimeter)

Platform ini mengimplementasikan perimeter pertahanan berlapis (*Defense-in-Depth*) untuk melindungi kekayaan intelektual model machine learning, formula kuantitatif, dan dataset dari praktik *unauthorized scraping*, *data crawling*, dan eksfiltrasi kredensial:

```mermaid
graph TD
    A["Request Masuk"] --> B{"1. Anti-Bot WAF"}
    B -- "Library Scraper (requests/curl/scrapy)" --> X["403 Forbidden"]
    B -- "Browser Sah" --> C{"2. Anti-Crawling & Burst Limiter"}
    C -- "Burst >10 req / 2s" --> Y["429 Too Many Requests"]
    C -- "Normal (<45 req / min)" --> D{"3. Origin & Sec-Fetch Validator"}
    D -- "External / Untrusted Domain" --> Z["403 Unauthorized Origin"]
    D -- "Domain Resmi / Whitelist" --> E{"4. Bearer Token Gatekeeper (HMAC-SHA256)"}
    E -- "Tanpa Token / Tamu" --> F["Public Teaser Layer (/morning-brief)"]
    E -- "Token Sah Terverifikasi" --> G["Full Institutional Access + Digital Canary Trace (X-Audit-Trace-ID)"]
```

1. **Anti-Bot Web Application Firewall (WAF)**:
   - Memindai signature `User-Agent` untuk memblokir otomatis library scraping terprogram (`python-requests`, `aiohttp`, `curl`, `wget`, `scrapy`, `postmanruntime`, `go-http-client`, `httpx`).
2. **Cryptographic Bearer Token Authentication (HMAC-SHA256)**:
   - Seluruh endpoint data rekomendasi institusional, bedah emiten, riwayat harga, dan laporan keuangan wajib menyertakan token kriptografis berwaktu kedaluwarsa. Kata sandi pengguna diamankan dengan algoritma **PBKDF2-HMAC-SHA256 (100.000 iterasi)** dengan salt acak 128-bit.
3. **Adaptive Behavioral Rate Limiting (Anti-Burst & Anti-Crawling)**:
   - **Burst Protection**: Membatasi maksimal 10 request per 2 detik guna menggagalkan script *looping* otomatis.
   - **Window Protection**: Membatasi maksimal 45 request per 60 detik (sangat lega bagi penjelajahan manusia, namun mematikan bagi bot *crawling*).
4. **Browser-Origin & Cross-Site Protection**:
   - Memvalidasi header `Origin` dan `Referer` untuk mencegah token yang disalin digunakan di luar domain resmi peramban.
5. **Digital Canary Watermarking (`X-Audit-Trace-ID`)**:
   - Menyisipkan tanda pengenal jejak audit unik berbasis user hash dan time-window pada setiap respon data terautentikasi untuk melacak dan mendeteksi sumber kebocoran data.
6. **Zero-Trust Git Hygiene**:
   - Dataset mentah (`data/raw/`) serta bobot biner model machine learning (`.joblib`, `.pth`) dikecualikan sepenuhnya dari pelacakan repositori publik (`.gitignore`) demi perlindungan hak cipta model kuantitatif.
7. **Multi-Layer Anti-Bot Registration Armor**:
   - **Invisible Honeypot Trap**: Memasang jebakan input tersembunyi (`company_fax`) yang otomatis memblokir script scraper/bot jika terisi.
   - **Submission Speed Trap**: Menolak submit formulir yang lebih cepat dari 1,5 detik (menangkap eksekusi headless script).
   - **IP Registration Quota**: Membatasi pendaftaran maksimal 3 akun per jam per alamat IP guna menggagalkan mass-account generation.
   - **Disposable Email Blacklist**: Memblokir registrasi dari penyedia email sementara/throwaway (misal: mailinator, guerrillamail, tempmail).
8. **Permanent Git History Purge & Encrypted Model Storage Vault**:
   - **Git Commit Graph Scrubbing**: Seluruh riwayat commit lawas telah dibersihkan secara permanen menggunakan `git-filter-repo` (ukuran pack repositori terpangkas >90% dari 8.04 MB menjadi ~790 KB), mengeliminasi total kemungkinan eksfiltrasi data mentah dan bobot model historis melalui `git clone`.
   - **Encrypted Storage Vault (`scripts/sync_model_storage.py`)**: Dilengkapi utilitas enkripsi AES-256 (PBKDF2-HMAC-SHA256) dan integrasi private repository gratis ke Hugging Face Hub untuk pencadangan model weights (`.joblib`, `.pth`) dan raw datasets secara aman.

---

## 🌐 Dokumentasi Endpoint REST API Terproteksi

| Method | Endpoint | Akses / Autentikasi | Deskripsi Data |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | **Public** | Menampilkan antarmuka Dashboard Web interaktif. |
| `GET` | `/morning-brief` | **Public (Teaser)** | Narasi editorial Morning Brief IHSG terkini, berita makro, dan arus modal asing makro. |
| `GET` | `/status` | **Public** | Informasi status kesehatan sistem dan waktu pembaruan pipeline terakhir. |
| `POST`| `/api/v1/auth/login` | **Public (Rate-Limited)** | Autentikasi pengguna & pembuatan Bearer Token bertanda tangan kriptografis. |
| `POST`| `/api/v1/auth/register` | **Public (Rate-Limited)** | Pendaftaran pengguna baru ke basis data terenkripsi PBKDF2. |
| `GET` | `/api/v1/auth/me` | **Bearer Token Wajib** | Verifikasi sesi profil pengguna yang sedang aktif. |
| `GET` | `/recommendations` | **Bearer Token Wajib** | Daftar rekomendasi kuantitatif 26+ saham (`all`, `swing`, `dividend`, `favorites`). |
| `GET` | `/watchlist-analysis` | **Bearer Token Wajib** | Analisis lengkap teknikal 6 level & fundamental 4 pilar seluruh konstituen BEI. |
| `GET` | `/stocks/analyze/{ticker}` | **Bearer Token Wajib** | Laporan riset taktis instan (BOW, BOB, TP 1, TP 2, Target Utama, Cut Loss) per emiten. |
| `GET` | `/portfolio/allocate` | **Bearer Token Wajib** | Perhitungan alokasi modal optimal berdasarkan nominal modal (`?capital=50000000`). |
| `GET` | `/models/compare/{ticker}` | **Bearer Token Wajib** | Konsensus perbandingan probabilitas multi-model (GBDT vs LSTM vs ARIMA). |
| `GET` | `/api/foreign-flow` | **Bearer Token Wajib** | Ringkasan arus modal asing makro IHSG dan seluruh 66 konstituen. |
| `GET` | `/api/foreign-flow/{ticker}`| **Bearer Token Wajib** | Deret data net foreign flow 30 hari dan status akumulasi per emiten. |
| `GET` | `/api/financials` | **Bearer Token Wajib** | Ringkasan laporan keuangan multi-tahun dan status kesehatan seluruh emiten. |
| `GET` | `/api/financials/{ticker}` | **Bearer Token Wajib** | Laporan keuangan multi-tahun, rasio, dan kartu AI Archetype per emiten. |
| `GET` | `/api/history/{ticker}` | **Bearer Token Wajib** | Riwayat harga faktual OHLCV 30 hari langsung dari database BEI. |
| `GET` | `/api/analysis/{ticker}` | **Bearer Token Wajib** | Bedah emiten mendalam bertenaga Google Gemini 3.8 Flash (Health, Fit, Verdict). |
| `POST`| `/pipeline/run` | **Admin Secret Wajib** | Memicu eksekusi ulang seluruh pipeline kuantitatif di background. |

---

## 👥 Tim Riset Kuantitatif & Rekayasa Sistem (TIM New York)

Platform ini dikembangkan dan dikelola secara kolaboratif oleh **TIM New York**:

| Nama Kontributor | Peran & Tanggung Jawab Utama | Profil GitHub |
| :--- | :--- | :--- |
| **Ifan Apres** | *Lead Quantitative Engineer & Fullstack Systems Architect* — Bertanggung jawab atas arsitektur komputasi, pipeline data kuantitatif, model PyTorch LSTM multi-dimensi, scraper arus kas asing BEI anti-Cloudflare, integrasi target excess alpha, sistem deployment Vercel & CI/CD automation. | [@Ifan-Apres](https://github.com/Ifan-Apres) |
| **Sekar Widhastri** | *Senior Market & Research Analyst* — Bertanggung jawab atas formulasi strategi analisis pasar ekonometrika, metodologi risk-parity alokasi portofolio, evaluasi sinyal teknikal BEI, dan kurasi editorial riset pasar. | [@sekarwidhastri](https://github.com/sekarwidhastri) |

---

## ⚖️ Disclaimer & Batasan Tanggung Jawab

*Aplikasi ini dikembangkan untuk tujuan riset kuantitatif, analisis data, dan edukasi finansial. Seluruh rekomendasi yang dihasilkan oleh model machine learning dan kecerdasan buatan merupakan indikator probabilitas statistik pasar dan bukan merupakan anjuran mutlak untuk membeli atau menjual efek tertentu. Keputusan investasi dan manajemen risiko sepenuhnya berada di tangan investor masing-masing.*
