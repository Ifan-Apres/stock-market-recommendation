# 📘 PANDUAN LENGKAP METODOLOGI KUANTITATIF, FORMULA PERHITUNGAN, ALGORITMA MACHINE LEARNING, DAN GLOSARIUM ISTILAH

> **Dokumen Resmi Arsitektur & Fundamental Sistem**  
> **Platform Rekomendasi Saham & Riset Ekuitas Kuantitatif Institusional**  
> *AlphaTech Quantitative Research • Bursa Efek Indonesia (BEI / IDX)*  
> *Versi Rilis: 4.5 (Tree Blend Ensemble • Multi-Engine Quant Suite)*

---

## DAFTAR ISI
1. [Pendahuluan & Filosofi Sistem](#1-pendahuluan--filosofi-sistem)
2. [Arsitektur Ingesti Data & Sumber Pasar](#2-arsitektur-ingesti-data--sumber-pasar)
   - 2.1 Yahoo Finance (Data Pasar Historis & Makro Lintas Aset)
   - 2.2 Scraper Resmi BEI & Impersonasi TLS (Anti-Cloudflare WAF)
   - 2.3 Pelacakan Arus Modal Asing (Foreign Flow / Bandarmologi)
   - 2.4 Data Laporan Keuangan Fundamental (RTI & BEI)
   - 2.5 Manajemen Semesta Emiten Dinamis (66 Saham & 11 Sektor BEI)
3. [Rekayasa Fitur Kuantitatif (32 Fitur Alpha)](#3-rekayasa-fitur-kuantitatif-32-fitur-alpha)
   - 3.1 Indikator Teknikal & Osilator
   - 3.2 Indikator Arus Modal Cerdas (Foreign Flow & Divergensi Akumulasi)
   - 3.3 Indikator Siklus Musiman Dividen (Seasonality)
   - 3.4 Interaksi Sensitivitas Sektor-Makro
   - 3.5 Level Lantai Pivot (Support & Resistance)
   - 3.6 Label Target Pembelajaran: Forward Excess Return 5-Hari
4. [Rangkaian Algoritma Machine Learning & Statistika](#4-rangkaian-algoritma-machine-learning--statistika)
   - 4.1 Tree Blend Ensemble (CatBoost Classifier + HistGradientBoosting)
   - 4.2 PyTorch Deep Learning LSTM Sequence Model
   - 4.3 ARIMA Time-Series Predictor
   - 4.4 Model Volatilitas Kondisional GARCH(1,1) Student-t & Fallback EWMA
   - 4.5 Bobot Konsensus Blending Multi-Engine
   - 4.6 Perisai Konsensus (Consensus Shield) & Ambang Batas Eksekusi
5. [Logika Rekomendasi Teknikal & Manajemen Level Harga](#5-logika-rekomendasi-teknikal--manajemen-level-harga)
   - 5.1 Aturan Klasifikasi Sinyal Eksekusi
   - 5.2 Rumus Target Price (TP) & Stop Loss (SL) Berbasis ATR-14
   - 5.3 Fraksi Harga Resmi BEI (IDX Tick Size Rounding)
   - 5.4 Risk-to-Reward Ratio (RRR)
6. [Algoritma Alokasi Portofolio Modal](#6-algoritma-alokasi-portofolio-modal)
   - 6.1 Formula Pembobotan Sharpe-to-Volatility
   - 6.2 Algoritma Capped Weights & Proteksi Kas Siaga
   - 6.3 Pembulatan Lot Saham Resmi (1 Lot = 100 Lembar)
   - 6.4 Rekonsiliasi Kembalian Sisa Lot ke Kas Siaga
   - 6.5 Simulasi Arus Kas Dividen Tahunan
7. [Metrik Risiko & Volatilitas Institusional](#7-metrik-risiko--volatilitas-institusional)
   - 7.1 Sharpe Ratio & Return Disetahunkan (Annualized Return)
   - 7.2 Beta IHSG (Sensitivitas Pasar)
   - 7.3 Maximum Drawdown 1 Tahun (Max DD)
   - 7.4 Value at Risk (VaR 95% & 99% 1-Day) & Expected Shortfall (ES)
8. [Formula Kalkulator Lot & Manajemen Risiko Frontend](#8-formula-kalkulator-lot--manajemen-risiko-frontend)
9. [Kamus Istilah Lengkap (Glosarium Finansial & Kuantitatif A–Z)](#9-kamus-istilah-lengkap-glosarium-finansial--kuantitatif-a-z)

---

## 1. PENDAHULUAN & FILOSOFI SISTEM

Platform ini dibangun di atas paradigma **Institutional Quantitative Investing**. Alih-alih mengandalkan intuisi atau spekulasi subjektif, seluruh keputusan seleksi emiten, alokasi modal, dan manajemen risiko didasarkan pada perpaduan:
1. **Statistika Ekonometrika Lanjutan**: Mengukur volatilitas harga tak-konstan dan risiko ekor (*fat-tail risk*) pasar negara berkembang (Emerging Market) seperti BEI.
2. **Machine Learning Ensemble**: Mengombinasikan model *gradient boosted decision trees* tabular dengan deep learning sekuensial dan model runtun waktu parametrik.
3. **Mikrostruktur Pasar Bursa Efek Indonesia**: Memperhitungkan aturan fraksi harga resmi (tick size), batasan lot (100 lembar), siklus dividen historis, serta dinamika bandarmologi / arus modal asing (*foreign flow*).

### Arsitektur Pipeline Dua Fase (Two-Phase Pipeline)
Sistem beroperasi setiap hari bursa (Senin – Jumat) melalui jadwal otomasi GitHub Actions:
- **Fase 1: Opening Pulse (10:00 WIB / 03:00 UTC)**:
  Berjalan 1 jam setelah bel pembukaan BEI. Mengunduh harga pembukaan (*Open*), harga intraday berjalan, dan memvalidasi apakah emiten dalam daftar rekomendasi menyentuh area beli (*Buy Zone*), target profit (*TP*), atau batas rugi (*SL*).
- **Fase 2: EOD Settlement & Retraining (17:15 WIB / 10:15 UTC)**:
  Berjalan 1 jam setelah bel penutupan BEI. Mengunduh data resmi penutupan pasar harian, laporan *Trading Summary* BEI, melatih ulang model Machine Learning, menghitung metrik volatilitas GARCH(1,1), dan menghasilkan rekomendasi penuh untuk sesi esok hari.

---

## 2. ARSITEKTUR INGESTI DATA & SUMBER PASAR

```
                      +-------------------------------+
                      |      SUMBER DATA PASAR        |
                      +---------------+---------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
+-------------------------------+                         +-------------------------------+
|       Yahoo Finance           |                         |   Portal Resmi BEI (idx.co.id)|
|  - OHLCV 66 Saham (3 Tahun)   |                         |  - Ringkasan Saham Harian     |
|  - Benchmark IHSG (^JKSE)     |                         |  - Net Foreign Flow Riil      |
|  - Makro: US10Y, USD/IDR, Oil |                         |  - TLS Safari/Chrome Imperson.|
+---------------+---------------+                         +---------------+---------------+
                |                                                         |
                +----------------------------+----------------------------+
                                             |
                                             v
                              +-------------------------------+
                              |    RTI / Financial Reports    |
                              |  - Laporan Keuangan (4-5 Thn) |
                              |  - Valuasi: PER, PBV, ROE, DER|
                              +---------------+---------------+
                                             |
                                             v
                              +-------------------------------+
                              | data/raw/raw_market_data.csv  |
                              | data/raw/idx_daily/*.json     |
                              +-------------------------------+
```

### 2.1 Yahoo Finance
Mengunduh data deret waktu harian mencakup:
- **OHLCV Saham**: Harga *Open*, *High*, *Low*, *Close*, *Adj Close*, dan *Volume* untuk seluruh 66 saham konstituen.
- **Benchmark Pasar**: Indeks Harga Saham Gabungan (IHSG / `^JKSE`).
- **Variabel Makro Global & Lintas Aset**:
  - Imbal Hasil Obligasi AS 10-Tahun (`^TNX`): Indikator *risk-free rate* global dan arah aliran dana asing.
  - Nilai Tukar USD/IDR (`IDR=X`): Indikator depresiasi/apresiasi Rupiah yang memengaruhi biaya impor emiten.
  - Minyak Mentah Brent (`BZ=F`): Katalis harga komoditas global untuk sektor energi BEI.
  - Indeks S&P 500 (`^GSPC`): Sentimen pasar ekuitas global Wall Street.

### 2.2 Scraper Resmi BEI & Impersonasi TLS
Bursa Efek Indonesia memproteksi portal resminya (`www.idx.co.id`) menggunakan Cloudflare Web Application Firewall (WAF) tingkat lanjut yang memblokir pustaka HTTP Python standar (seperti `urllib` atau `requests`).
- **Implementasi**: Modul `src/idx_scraper.py` memanfaatkan `curl_cffi` dengan teknik **TLS Fingerprint Impersonation** (`safari18_0` dan `chrome124`).
- **Hasil**: Sistem dapat mengekstrak data *Trading Summary* resmi BEI setiap penutupan bursa tanpa terkena blokir IP atau Captcha.

### 2.3 Pelacakan Arus Modal Asing (Foreign Flow / Bandarmologi)
Dana investor asing menguasai proporsi kepemilikan saham beredar (*free-float*) yang signifikan pada saham-saham likuid BEI (terutama perbankan *Big 4*).
- **Data Primer**: Nilai beli bersih asing (*Net Foreign Buy/Sell IDR*) yang diambil langsung dari tabel ringkasan BEI.
- **Foreign Ownership Weighting ($w_{foreign}$)**:
  Setiap emiten dikalibrasi dengan bobot partisipasi asing historis:

  $$
  \text{BBCA} = 0.55, \quad \text{BBRI} = 0.48, \quad \text{BMRI} = 0.45, \quad \text{TLKM} = 0.42, \quad \text{ADRO} = 0.38
  $$

- **Proxy Model Cadangan**: Jika data resmi BEI belum dirilis pada akhir pekan/libur bursa, sistem menggunakan formulasi proxy:

  $$
  \text{Net Foreign Flow} = \text{Value}_t \times w_{foreign} \times \text{clip}\left(\text{Return}_{1D} \times 7.5, -0.65, 0.65\right)
  $$

### 2.4 Data Laporan Keuangan Fundamental
Disinkronisasikan dari ringkasan laporan keuangan BEI & RTI:
- **Rentang Historis**: 4 hingga 5 tahun buku (2021 – 2025).
- **Variabel Finansial**: Pendapatan (*Revenue*), Laba Bersih (*Net Income*), *Earnings Per Share* (EPS), *Price to Earnings Ratio* (PER), *Price to Book Value* (PBV), *Return on Equity* (ROE), *Debt to Equity Ratio* (DER), *Current Ratio* (CR), *Dividend Yield*, dan *Dividend Payout Ratio* (DPR).

### 2.5 Manajemen Semesta Emiten Dinamis (66 Saham & 11 Sektor BEI)
Portofolio mencakup 66 emiten terlikuid yang dikelompokkan ke dalam 11 sektor resmi BEI:
1. **Financials (Keuangan)**: BBCA, BBRI, BMRI, BBNI, BBTN, BRIS, BDMN
2. **Energy (Energi)**: ADRO, AADI, PTBA, ITMG, PGAS, MEDC, AKRA
3. **Basic Materials (Barang Baku)**: ANTM, MDKA, INCO, AMMN, MBMA, INKP, TKIM, SMGR, INTP
4. **Consumer Non-Cyclicals (Konsumen Primer)**: ICBP, INDF, UNVR, AMRT, MYOR, CPIN, JPFA, CMRY
5. **Consumer Cyclicals (Konsumen Non-Primer)**: ASII, ACES, MAPI, AUTO, ERAA
6. **Healthcare (Kesehatan)**: KLBF, MIKA, HEAL, SIDO
7. **Industrials (Perindustrian)**: UNTR, ASII, HEXA, ARNA
8. **Infrastructure (Infrastruktur)**: TLKM, ISAT, EXCL, TOWR, TBIG, JSMR, POWR, PGEO, BREN
9. **Technology (Teknologi)**: GOTO, EMTK, BUKA, WIRG
10. **Properties & Real Estate (Properti)**: BSDE, CTRA, SMRA, PWON
11. **Transportation & Logistics (Transportasi)**: BIRD, SMDR, TMAS, MPMX

---

## 3. REKAYASA FITUR KUANTITATIF (32 FITUR ALPHA)

Diimplementasikan dalam modul `src/02_feature_eng.py`, setiap data bar harian ditransformasikan menjadi 32 dimensi fitur prediktif.

### 3.1 Indikator Teknikal & Osilator

#### 1. Log Return (Pengembalian Logaritmik)
Mengukur perubahan harga kontinu:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

#### 2. Jarak Rata-rata Bergerak (Distance to SMA)
Mengukur deviasi persentase harga relatif terhadap tren jangka pendek (20 hari) dan jangka panjang (200 hari):

$$
\text{Dist\_SMA\_20}_t = \frac{P_t - \text{SMA}_{20}(P)_t}{\text{SMA}_{20}(P)_t}
$$

$$
\text{Dist\_SMA\_200}_t = \frac{P_t - \text{SMA}_{200}(P)_t}{\text{SMA}_{200}(P)_t}
$$

#### 3. Moving Average Convergence Divergence (MACD)
Mengukur pergeseran momentum tren:

$$
\text{EMA}_{12}(P)_t = \alpha_{12} P_t + (1 - \alpha_{12}) \text{EMA}_{12}(P)_{t-1}, \quad \text{di mana } \alpha = \frac{2}{N + 1}
$$

$$
\text{MACD}_t = \text{EMA}_{12}(P)_t - \text{EMA}_{26}(P)_t
$$

$$
\text{MACD\_Signal}_t = \text{EMA}_9(\text{MACD})_t
$$

$$
\text{MACD\_Hist}_t = \text{MACD}_t - \text{MACD\_Signal}_t
$$

#### 4. Relative Strength Index (RSI 14-Hari)
Mengukur kecepatan dan besaran perubahan harga untuk mendeteksi *overbought* ($>70$) atau *oversold* ($<30$):

$$
\Delta P_t = P_t - P_{t-1}
$$

$$
\text{Gain}_t = \max(\Delta P_t, 0), \quad \text{Loss}_t = \max(-\Delta P_t, 0)
$$

$$
\text{RS} = \frac{\text{RollingMean}(\text{Gain}, 14)}{\text{RollingMean}(\text{Loss}, 14) + 10^{-9}}
$$

$$
\text{RSI}_{14} = 100 - \left(\frac{100}{1 + \text{RS}}\right)
$$

#### 5. Chaikin Money Flow (CMF 20-Hari)
Mengukur akumulasi/distribusi volume institusional:

$$
\text{Money Flow Multiplier}_t = \frac{(Close_t - Low_t) - (High_t - Close_t)}{(High_t - Low_t) + 10^{-9}}
$$

$$
\text{Money Flow Volume}_t = \text{Multiplier}_t \times Volume_t
$$

$$
\text{CMF}_{20} = \frac{\sum_{i=0}^{19} \text{Money Flow Volume}_{t-i}}{\sum_{i=0}^{19} Volume_{t-i} + 10^{-9}}
$$

#### 6. Money Flow Index (MFI 14-Hari)
Osilator harga berbobot volume:

$$
\text{Typical Price}_t = \frac{High_t + Low_t + Close_t}{3}
$$

$$
\text{Raw Money Flow}_t = \text{Typical Price}_t \times Volume_t
$$

$$
\text{MFI}_{14} = 100 - \left(\frac{100}{1 + \frac{\text{Pos Flow}_{14}}{\text{Neg Flow}_{14} + 10^{-9}}}\right)
$$

---

### 3.2 Indikator Arus Modal Cerdas (Foreign Flow & Divergensi)

#### 1. Foreign Flow Normalized 1D
Menstandarisasi nilai beli bersih asing terhadap rata-rata transaksi 20 hari emiten:

$$
\text{Foreign\_Flow\_Norm\_1D}_t = \text{clip}\left(\frac{\text{Net Foreign IDR}_t}{\text{SMA}_{20}(\text{Value})_t + 10^{-9}}, -3.0, 3.0\right)
$$

#### 2. Akumulasi Asing 5-Hari (Foreign Flow 5D Accumulation)
Mengukur ketahanan akumulasi modal asing selama 1 pekan perdagangan:

$$
\text{Foreign\_Flow\_5D\_Accum}_t = \text{clip}\left(\frac{\sum_{i=0}^4 \text{Net Foreign IDR}_{t-i}}{5 \times \text{SMA}_{20}(\text{Value})_t + 10^{-9}}, -3.0, 3.0\right)
$$

#### 3. Foreign Accumulation Divergence (Divergensi Arus Asing)
Mendeteksi anomali di mana investor asing melakukan akumulasi masif saat harga saham sedang turun/konsolidasi (sinyal *Smart Money Accumulation*):

$$
\text{Foreign\_Accum\_Divergence}_t = \text{clip}\left(\text{Foreign\_Flow\_5D\_Accum}_t - \text{Return\_5D}_t, -3.0, 3.0\right)
$$

#### 4. Foreign Flow Intensity

$$
\text{Foreign\_Flow\_Intensity}_t = \text{clip}\left(\text{Foreign\_Flow\_Norm\_1D}_t \times \text{Foreign\_Participation}, -3.0, 3.0\right)
$$

---

### 3.3 Indikator Siklus Musiman Dividen (Seasonality)
Bursa Efek Indonesia memiliki siklus musiman pembagian dividen yang sangat teratur:
- **Kalender Historis**:
  - BBCA: Maret, April, Desember (Interim & Final)
  - BBRI: Januari, Maret
  - ADRO: Januari, Mei, Juni, Desember
  - ASII: Mei, Oktober
- **Fitur Boolean & Momentum**:

  $$
  \text{Is\_Dividend\_Season}_t = \begin{cases} 1.0 & \text{jika } \text{Bulan}_t \in \text{Musim Dividen Emiten} \\ 0.0 & \text{lainnya} \end{cases}
  $$

  $$
  \text{Dividend\_Season\_Momentum}_t = \text{Is\_Dividend\_Season}_t \times \text{Return\_20D}_t
  $$

---

### 3.4 Interaksi Sensitivitas Sektor-Makro
Mengkorelasikan variabel makroekonomi global secara spesifik dengan sektor yang relevan di BEI:
1. **Oil Energy Tailwind**:

   $$
   \text{Oil\_Energy\_Tailwind}_t = \begin{cases} \text{Brent Oil Return}_{1D} & \text{jika Sektor} = \text{Energy} \\ 0.0 & \text{lainnya} \end{cases}
   $$

2. **Rate Bank Sensitivity**:

   $$
   \text{Rate\_Bank\_Sensitivity}_t = \begin{cases} \Delta\text{US 10Y Yield} & \text{jika Sektor} = \text{Financials} \\ 0.0 & \text{lainnya} \end{cases}
   $$

3. **FX Consumer Headwind**:

   $$
   \text{FX\_Consumer\_Headwind}_t = \begin{cases} \text{USD/IDR Return}_{1D} & \text{jika Sektor} \in \{\text{Consumer}, \text{Healthcare}\} \\ 0.0 & \text{lainnya} \end{cases}
   $$

---

### 3.5 Level Lantai Pivot (Support & Resistance)
Dihitung dari harga sesi harian sebelumnya untuk menentukan level batas beli (*Breakout/Weakness*) dan target harga:

$$
\text{Pivot Point (PP)} = \frac{High + Low + Close}{3}
$$

$$
\text{Support 1 (S1)} = (2 \times PP) - High
$$

$$
\text{Support 2 (S2)} = PP - (High - Low)
$$

$$
\text{Resistance 1 (R1)} = (2 \times PP) - Low
$$

$$
\text{Resistance 2 (R2)} = PP + (High - Low)
$$

---

### 3.6 Label Target Pembelajaran: Forward Excess Return 5-Hari
Target prediksi Machine Learning **bukan** sekadar kenaikan harga nominal, melainkan **Excess Alpha** (kemampuan saham mengalahkan IHSG):

$$
\text{Target Return 5D}_t = \ln\left(\frac{P_{t+5}}{P_t}\right)
$$

$$
\text{Target Excess Return 5D}_t = \text{Target Return 5D}_t - \text{Benchmark Return 5D}_t
$$

$$
\text{Target Class 5D}_t = \begin{cases} 1 & \text{jika } \text{Target Excess Return 5D}_t > 0 \\ 0 & \text{lainnya} \end{cases}
$$

---

## 4. RANGKAIAN ALGORITMA MACHINE LEARNING & STATISTIKA

Sistem menggunakan **Multi-Engine Consensus Blending** yang menggabungkan 4 algoritma berbeda untuk meminimalkan bias model individual:

```
                            +---------------------------------------+
                            |          INPUT 32 FITUR ALPHA         |
                            +-------------------+-------------------+
                                                |
        +-----------------------+---------------+-----------------------+
        |                       |                                       |
        v                       v                                       v
+----------------+      +----------------+                      +----------------+
| HistGB (Tab)   |      | CatBoost (Tab) |                      | PyTorch BiLSTM |
|  - Depth: 4    |      |  - Depth: 5    |                      |  - Lookback: 20|
|  - lr: 0.02    |      |  - lr: 0.03    |                      |  - 2 Layer     |
+-------+--------+      +-------+--------+                      +-------+--------+
        |                       |                                       |
        +-----------+-----------+                                       |
                    |                                                   |
                    v                                                   v
        +-----------------------+                               +----------------+
        |   Tree Blend (50/50)  |                               | LSTM Prob      |
        +-----------+-----------+                               +-------+--------+
                    |                                                   |
                    | [Bobot: 60%]                              [Bobot: 25%]
                    +-------------------+       +-----------------------+
                                        |       |
                                        v       v
                            +-------------------------------+
                            |   ARIMA (1,1,1) Time-Series   | <--- [Bobot: 15%]
                            +---------------+---------------+
                                            |
                                            v
                            +-------------------------------+
                            |   KONSENSUS PROBABILITAS      |
                            |   BULLISH COMPOSITE ALPHA     |
                            +---------------+---------------+
                                            |
                                            v
                            +-------------------------------+
                            |    MULTI-MODEL SHIELD GATE    |
                            | - Prob >= 0.55                |
                            | - Minimal 2 dari 3 Bullish    |
                            | - Tidak ada model < 0.48      |
                            +-------------------------------+
```

### 4.1 Tree Blend Ensemble (CatBoost + HistGradientBoosting)
Model pohon tabular terbukti paling unggul dalam memproses data tabular finansial non-linear:
1. **HistGradientBoostingClassifier**:
   - Algoritma pemisah berbasis *histogram binning*.
   - Hyperparameter: `learning_rate = 0.02`, `max_iter = 150`, `max_depth = 4`, `min_samples_leaf = 40`, `l2_regularization = 3.0`.
2. **CatBoostClassifier**:
   - Keunggulan penanganan fitur kategorikal sektor dan regularisasi simetris (*oblivious trees*) yang sangat tahan terhadap *overfitting*.
   - Hyperparameter: `iterations = 250`, `learning_rate = 0.03`, `depth = 5`, `l2_leaf_reg = 5.0`.
3. **Kombinasi Tree Blend**:

   $$
   P_{\text{Tree\_Blend}} = 0.50 \times P_{\text{HistGB}} + 0.50 \times P_{\text{CatBoost}}
   $$

### 4.2 PyTorch Deep Learning LSTM Sequence Model
Menangkap dinamika temporal dan ketergantungan urutan (*sequence dependency*) pergerakan harga selama 20 hari perdagangan bursa:
- **Arsitektur Jaringan**:
  - `input_dim`: 16 fitur sekuensial (Return, RSI, Dist SMA, CMF, Volume Ratio, Makro & Arus Asing)
  - `hidden_dim`: 32 neuron
  - `num_layers`: 2 layer LSTM
  - `dropout`: 0.35 (mencegah menghafal noise pasar)
  - `output layer`: Linear ke 1 neuron dengan aktivasi `Sigmoid`
- **Fungsi Loss & Optimasi**: Binary Cross-Entropy Loss (`BCELoss`), optimizer `AdamW` (`learning_rate = 0.003`, `weight_decay = 1e-4`), `batch_size = 256`, 6 epoch.

### 4.3 ARIMA Statistical Time-Series Predictor
Model ekonometrika autoregresif murni untuk memprediksi arah tren harga tanpa fitur eksternal:
- **Spesifikasi**: $\text{ARIMA}(p=1, d=1, q=1)$ dengan fallback ke $\text{ARIMA}(1, 0, 0)$:

  $$
  \Delta P_t = c + \phi_1 \Delta P_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t
  $$

- **Transformasi Probabilitas**: Pengembalian ekspektasi 5-hari diubah menjadi probabilitas kontinu via kurva logistik curam:

  $$
  P_{\text{ARIMA}} = \text{clip}\left(\frac{1}{1 + e^{-25 \times \text{Expected\_Return}}}, 0.10, 0.90\right)
  $$

### 4.4 Model Volatilitas Kondisional GARCH(1,1) Student-t & Fallback EWMA
Volatilitas pasar saham tidak pernah konstan melainkan berkelompok (*volatility clustering*).
- **Formulasi GARCH(1,1)**:

  $$
  r_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \text{Student-}t(\nu)
  $$

  $$
  \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
  $$

  - Persyaratan Stabilitas (*Stationarity*): $\alpha > 0, \beta > 0, \alpha + \beta < 0.999$.
  - Derajat Kebebasan ($\nu$): Menangkap fenomena ekor tebal (*fat tails / leptokurtic*).
- **Penyaring Anomali & Fallback RiskMetrics EWMA**:
  Jika model GARCH gagal konvergen atau nilai $\nu$ tidak wajar ($<2.5$ atau $>60$), sistem otomatis mengalihkan ke model RiskMetrics Exponentially Weighted Moving Average (EWMA, $\lambda = 0.94$):

  $$
  \sigma_t^2 = 0.94 \sigma_{t-1}^2 + 0.06 r_{t-1}^2
  $$

### 4.5 Bobot Konsensus Blending Multi-Engine
Probabilitas komposit akhir (*Composite Bullish Probability*) dihitung melalui bobot teruji:

$$
P_{\text{Bullish}} = 0.60 \times P_{\text{Tree\_Blend}} + 0.25 \times P_{\text{LSTM}} + 0.15 \times P_{\text{ARIMA}}
$$

### 4.6 Perisai Konsensus (Consensus Shield) & Ambang Batas Eksekusi
Untuk memitigasi sinyal palsu (*false positives*), emiten hanya berhak menerima sinyal **BUY** jika lolos 3 lapis proteksi:
1. **Ambang Keras (Hard Threshold)**: $P_{\text{Bullish}} \ge 0.55$ (55%).
2. **Suara Mayoritas Model**: Minimal 2 dari 3 model (Tree Blend, LSTM, ARIMA) harus menghasilkan probabilitas bullish ($\ge 0.50$).
3. **Penyaring Divergensi Model (Divergence Guard)**: Tidak boleh ada satu pun model yang memberikan probabilitas bearish tajam ($P_{\text{model}} < 0.48$).

---

## 5. LOGIKA REKOMENDASI TEKNIKAL & MANAJEMEN LEVEL HARGA

### 5.1 Aturan Klasifikasi Sinyal Eksekusi
Jika emiten lolos *Perisai Konsensus*, jenis aksi teknikal diklasifikasikan berdasarkan kondisi osilator dan level lantai:

| Sinyal Rekomendasi | Kondisi Pemicu Kuantitatif | Rasional Eksekusi |
| :--- | :--- | :--- |
| **BUY ON WEAKNESS** | Lolos Konsensus **DAN** ($\text{RSI}_{14} \le 45.0$ **ATAU** $Close \le Support_1 \times 1.01$) | Membeli saat harga terkoreksi ke area lantai pantulan teknikal. |
| **BUY ON BREAKOUT** | Lolos Konsensus **DAN** ($\text{Volume\_Ratio} \ge 1.25$ **DAN** $Close \ge Resistance_1 \times 0.99$) | Membeli saat harga menembus atap resistensi didukung lonjakan volume transaksi. |
| **TRADING BUY** | Lolos Konsensus, namun berada di area netral di antara Support dan Resistance. | Pembelian bertahap mengikuti momentum tren positif yang sedang berjalan. |
| **SELL ON STRENGTH** | $P_{\text{Bullish}} \le 0.45$ **ATAU** $\text{RSI}_{14} \ge 70.0$ | Mengamankan profit karena aset sudah jenuh beli (*overbought*) atau probabilitas melemah. |
| **HOLD** | Tidak memenuhi kriteria di atas ($0.45 < P_{\text{Bullish}} < 0.55$). | Menahan posisi dan mengamati konfirmasi tren pasar berikutnya. |

---

### 5.2 Rumus Target Price (TP) & Stop Loss (SL) Berbasis ATR-14
Average True Range 14-Hari ($\text{ATR}_{14}$) digunakan sebagai pengukur volatilitas riil pergerakan harga saham untuk menentukan jarak batas rugi dan target profit yang dinamis:

$$
\text{TR}_t = \max\left(High_t - Low_t, \, |High_t - Close_{t-1}|, \, |Low_t - Close_{t-1}|\right)
$$

$$
\text{ATR}_{14} = \text{RollingMean}(\text{TR}, 14)
$$

Untuk sinyal **BUY (BoW, BoB, Trading Buy)**:
1. **Stop Loss (Batas Pengaman Kerugian)**:

   $$
   \text{Raw\_SL} = \max\left(Support_1, \, Close - 1.5 \times \text{ATR}_{14}\right)
   $$

   $$
   \text{Raw\_SL} = \min\left(\text{Raw\_SL}, \, Close - 0.8 \times \text{ATR}_{14}\right)
   $$

   $$
   \text{Stop Loss} = \text{RoundToTick}(\text{Raw\_SL})
   $$

2. **Target Price (Target Ambil Untung)**:

   $$
   \text{Raw\_TP} = \min\left(Resistance_1, \, Close + 2.0 \times \text{ATR}_{14}\right)
   $$

   $$
   \text{Raw\_TP} = \max\left(\text{Raw\_TP}, \, Close + 1.2 \times \text{ATR}_{14}\right)
   $$

   $$
   \text{Target Price} = \text{RoundToTick}(\text{Raw\_TP})
   $$

---

### 5.3 Fraksi Harga Resmi BEI (IDX Tick Size Rounding)
Seluruh level harga (*Entry*, *TP*, *SL*) wajib dibulatkan ke fraksi harga resmi Bursa Efek Indonesia sesuai SK Direksi PT Bursa Efek Indonesia:

$$
\text{Fraksi Harga (Tick)} = \begin{cases} 
\text{Rp 1} & \text{jika } P < \text{Rp 200} \\ 
\text{Rp 2} & \text{jika } \text{Rp 200} \le P < \text{Rp 500} \\ 
\text{Rp 5} & \text{jika } \text{Rp 500} \le P < \text{Rp 2.000} \\ 
\text{Rp 10} & \text{jika } \text{Rp 2.000} \le P < \text{Rp 5.000} \\ 
\text{Rp 25} & \text{jika } P \ge \text{Rp 5.000} 
\end{cases}
$$

Implementasi kode:

$$
\text{Harga Bulat} = \max\left(50, \, \text{round}\left(\frac{P}{\text{Tick}}\right) \times \text{Tick}\right)
$$

---

### 5.4 Risk-to-Reward Ratio (RRR)
Rasio yang membandingkan potensi keuntungan terhadap risiko kerugian:

$$
\text{Potential Gain} = \text{Target Price} - \text{Entry Price}
$$

$$
\text{Potential Risk} = |\text{Entry Price} - \text{Stop Loss}|
$$

$$
\text{RRR} = \frac{\text{Potential Gain}}{\text{Potential Risk}}
$$
*Standar Institusional: Sistem menargetkan sinyal dengan RRR $\ge 1.5$ (keuntungan minimal 1,5x lipat dari risiko).*

---

## 6. ALGORITMA ALOKASI PORTOFOLIO MODAL

Diimplementasikan dalam modul `src/03_model_inference.py` (`optimize_portfolio_allocation`), sistem mengalokasikan modal investor secara otomatis menggunakan prinsip efisiensi variansi-rataan (*mean-variance efficiency*).

### 6.1 Formula Pembobotan Sharpe-to-Volatility
Kandidat saham dipilih dari maksimal 5 emiten bersinyal BUY dengan Sharpe Ratio tertinggi. Skor masing-masing aset dihitung dengan membagi rasio Sharpe terhadap estimasi volatilitas tahunan GARCH(1,1):

$$
Score_i = \frac{\max(Sharpe_i, \, 0.05)}{\text{GARCH\_Vol}_i + 0.05}
$$

### 6.2 Algoritma Capped Weights & Proteksi Kas Siaga
Untuk mencegah konsentrasi modal berlebih pada satu saham, bobot dialokasikan menggunakan algoritma konveks *cap-and-redistribute*:
1. **Pagu Tunggal (Single-Stock Cap)**: Maksimal **25%** dari total modal per emiten ($w_i \le 0.25$).
2. **Pagu Ekuitas Total (Equity Budget)**: Maksimal **80%** dari total modal dialokasikan ke saham.
3. **Mandat Kas Siaga (Cash Reserve Guarantee)**: Minimal **20%** dari total modal wajib dialokasikan ke Kas Siaga / RDN / RDPU sebagai penyangga likuiditas dari gejolak pasar (*stress test*).
4. **Redistribusi Eksedens**: Jika ada bobot emiten melebihi 25%, kelebihannya didistribusikan secara proporsional ke emiten lain yang bobotnya masih di bawah pagu.

### 6.3 Pembulatan Lot Saham Resmi (1 Lot = 100 Lembar)
Modal target untuk tiap emiten dihitung dalam Rupiah, kemudian dikonversikan ke dalam satuan lot resmi BEI menggunakan pembulatan ke bawah (*floor*):

$$
\text{Nominal Target}_i = \text{Modal Total} \times w_i
$$

$$
\text{Harga Per Lot}_i = Close_i \times 100
$$

$$
\text{Jumlah Lot}_i = \left\lfloor \frac{\text{Nominal Target}_i}{\text{Harga Per Lot}_i} \right\rfloor
$$

$$
\text{Nilai Pembelian Riil}_i = \text{Jumlah Lot}_i \times \text{Harga Per Lot}_i
$$

### 6.4 Rekonsiliasi Kembalian Sisa Lot ke Kas Siaga
Karena lot saham tidak bisa dibeli dalam angka desimal, selisih sisa uang yang tidak terpakai dari pembulatan lot (*residual cash*) **100% dialihkan kembali ke Kas Siaga**:

$$
\text{Kas Siaga Riil} = \text{Modal Total} - \sum_{i=1}^N \text{Nilai Pembelian Riil}_i
$$

Dengan demikian:

$$
\text{Total Ekuitas Saham Riil} + \text{Total Kas Siaga Riil} \equiv \text{Modal Total (100.0\% Sempurna)}
$$

### 6.5 Simulasi Arus Kas Dividen Tahunan
Mengestimasi pemasukan kas pasif tahunan kotor (*Gross Estimated Annual Dividend*) yang dihasilkan oleh emiten konstituen portofolio:

$$
\text{Estimasi Dividen Kas (IDR)} = \sum_{i=1}^N \left( \text{Nilai Pembelian Riil}_i \times \frac{\text{Dividend Yield}_i}{100} \right)
$$

$$
\text{Rata-rata Yield Portofolio (\% p.a.)} = \frac{\text{Estimasi Dividen Kas}}{\sum \text{Nilai Pembelian Riil}} \times 100\%
$$

---

## 7. METRIK RISIKO & VOLATILITAS INSTITUSIONAL

### 7.1 Sharpe Ratio & Return Disetahunkan
Mengukur efisiensi imbal hasil portofolio per unit risiko total terhadap suku bunga acuan bebas risiko Bank Indonesia (BI-Rate = 6.0% p.a.):

$$
\text{Annualized Return (1Y)} = \frac{P_t - P_{t-252}}{P_{t-252}}
$$

$$
\text{Annualized Volatility (1Y)} = \text{StdDev}(r_t, 252) \times \sqrt{252}
$$

$$
\text{Sharpe Ratio} = \frac{\text{Annualized Return} - R_f}{\text{Annualized Volatility}}, \quad \text{di mana } R_f = 0.06
$$

### 7.2 Beta IHSG (Sensitivitas Risiko Sistemik)
Mengukur kovariansi pergerakan saham terhadap pergerakan Indeks Harga Saham Gabungan (IHSG):

$$
\beta_i = \frac{\text{Cov}(R_i, \, R_{\text{IHSG}})}{\text{Var}(R_{\text{IHSG}})}
$$

- $\beta = 1.0$: Volatilitas saham bergerak seirama dengan IHSG.
- $\beta > 1.0$: Saham agresif (lebih bergejolak dibanding IHSG).
- $\beta < 1.0$: Saham defensif (lebih tahan banting saat pasar turun).

### 7.3 Maximum Drawdown 1 Tahun (Max DD)
Mengukur penurunan persentase terbesar dari titik puncak historis (*peak*) ke lembah terendah (*trough*) selama 252 hari bursa:

$$
\text{Peak}_t = \max_{s \le t} (P_s)
$$

$$
\text{Drawdown}_t = \frac{P_t - \text{Peak}_t}{\text{Peak}_t}
$$

$$
\text{Max Drawdown (1Y)} = \min_{t \in [0, 252]} (\text{Drawdown}_t)
$$

### 7.4 Value at Risk (VaR 95% & 99% 1-Day) & Expected Shortfall (ES)
Mengestimasi batas potensi kerugian maksimal dalam 1 hari perdagangan pada tingkat keyakinan (*confidence level*) 95% dan 99% menggunakan parameter Student-t GARCH:

$$
\text{VaR}_{95, 1D} = -\left(\mu + \sigma_{t+1} \sqrt{\frac{\nu - 2}{\nu}} \, t_{\nu}(0.05)\right)
$$

$$
\text{Expected Shortfall (ES)}_{95} = E\left[R \mid R \le -\text{VaR}_{95}\right]
$$

*Arti Praktis: Jika portofolio memiliki VaR 95% sebesar -1.42%, artinya dalam kondisi pasar normal, terdapat probabilitas 95% bahwa kerugian harian tidak akan melebihi 1.42% dari modal.*

---

## 8. FORMULA KALKULATOR LOT & MANAJEMEN RISIKO FRONTEND

Fitur kalkulator interaktif pada menu antarmuka web menerapkan formula manajemen risiko institusional untuk menentukan ukuran posisi (*position sizing*):

```
[Total Modal (IDR)] x [Persen Risiko Modal (1% - 3%)] = [Maksimal Toleransi Rugi (IDR)]
                                                                    |
                                   +--------------------------------+
                                   |
                                   v
             [Maksimal Toleransi Rugi] / [Resiko Per Lembar (Entry - SL)]
                                   |
                                   v
             [Jumlah Lembar Saham] / 100 lembar
                                   |
                                   v
             [Math.floor] = [JUMLAH LOT BEI MAKSIMAL]
```

1. **Maksimal Toleransi Kerugian (IDR)**:

   $$
   \text{Max Risk IDR} = \text{Modal} \times \left(\frac{\text{Risk Pct}}{100}\right)
   $$

2. **Risiko Riil Per Lembar Saham**:

   $$
   \text{Risk Per Share} = \text{Entry Price} - \text{Stop Loss}
   $$

3. **Jumlah Lembar Saham Maksimal**:

   $$
   \text{Max Shares} = \frac{\text{Max Risk IDR}}{\text{Risk Per Share}}
   $$

4. **Jumlah Lot Rekomendasi (Pembulatan ke Bawah)**:

   $$
   \text{Recommended Lots} = \left\lfloor \frac{\text{Max Shares}}{100} \right\rfloor
   $$

5. **Modal Investasi yang Dibutuhkan**:

   $$
   \text{Invested Capital} = \text{Recommended Lots} \times 100 \times \text{Entry Price}
   $$

---

## 9. KAMUS ISTILAH LENGKAP (GLOSARIUM FINANSIAL & KUANTITATIF A–Z)

### A
- **Aksi Korporat (Corporate Action)**: Tindakan emiten yang berdampak material terhadap pemegang saham (contoh: dividen, *rights issue*, *stock split*, *warrant*, merger/akuisisi).
- **Alokasi Modal (Capital Allocation)**: Strategi pembagian modal uang tunai ke berbagai instrumen atau saham untuk memaksimalkan imbal hasil dan menekan risiko ke level terendah.
- **Alpha ($\alpha$)**: Nilai lebih atau imbal hasil ekstra yang dihasilkan oleh strategi investasi di atas tolok ukur pasar (*benchmark* IHSG).
- **Annualized Return**: Tingkat keuntungan rata-rata yang disetahunkan (diasumsikan 252 hari perdagangan bursa).
- **ARIMA (Autoregressive Integrated Moving Average)**: Algoritma ekonometrika untuk memproyeksikan deret waktu harga berbasis korelasi masa lalu dan galat acak.
- **Average True Range (ATR)**: Indikator pengukur volatilitas absolut yang memperhitungkan jarak selisih harga tertinggi, terendah, dan celah (*gap*) harga penutupan kemarin.

### B
- **Bandarmologi**: Analisis aliran transaksi untuk mengamati akumulasi atau distribusi saham oleh pihak dengan modal raksasa (*smart money* / investor institusi).
- **Beta ($\beta$)**: Pengukur volatilitas relatif suatu saham terhadap indeks pasar (IHSG).
- **Blue Chip**: Saham perusahaan berkapitalisasi pasar besar, fundamental kokoh, likuiditas tinggi, dan rekam jejak dividen konsisten (contoh: BBCA, BBRI, BMRI, ASII).
- **Bollinger Bands**: Pita volatilitas yang dibentuk dari rata-rata bergerak 20 hari $\pm$ 2 standar deviasi.
- **Buy on Breakout (BoB)**: Strategi membeli saat harga menembus level resistensi ke atas dengan konfirmasi volume tinggi.
- **Buy on Weakness (BoW)**: Strategi membeli saat harga saham mengalami penurunan sementara mendekati level *support* kuat.

### C
- **CatBoost**: Algoritma Machine Learning *gradient boosting* spesialis pengolahan data tabular yang unggul dalam mencegah kebocoran data (*overfitting*) pada data pasar keuangan.
- **Chaikin Money Flow (CMF)**: Indikator arus uang berbasis posisi harga penutupan dalam rentang harian yang dibobotkan oleh volume perdagangan.
- **Consensus Blending**: Metode penggabungan prediksi beberapa model independen dengan bobot tertentu untuk mencapai hasil yang lebih stabil dan objektif.
- **Cum Date (Cumulative Date)**: Hari terakhir bagi investor untuk membeli saham agar berhak menerima dividen yang telah diumumkan.
- **Current Ratio (CR)**: Rasio aset lancar dibagi liabilitas jangka pendek untuk mengukur likuiditas keuangan jangka pendek emiten.

### D
- **Debt to Equity Ratio (DER)**: Rasio total liabilitas dibagi ekuitas untuk mengukur tingkat leverage utang suatu perusahaan.
- **Dividend Payout Ratio (DPR)**: Persentase laba bersih perusahaan yang dibagikan kepada pemegang saham sebagai dividen tunai.
- **Dividend Seasonality**: Siklus musiman historis pembagian dividen emiten BEI (Interim pada Nov–Jan dan Final pada April–Juli).
- **Dividend Yield**: Persentase nilai dividen per lembar saham tahunan dibandingkan harga pasar saham saat ini.
- **Drawdown**: Persentase penurunan nilai harga atau portofolio dari titik puncak tertinggi ke titik terendah sebelum membuat rekor tertinggi baru.

### E
- **Earnings Per Share (EPS)**: Laba bersih perusahaan dibagi dengan jumlah total lembar saham yang beredar.
- **Excess Return**: Selisih imbal hasil saham di atas pengembalian tolok ukur indeks IHSG.
- **Ex Date (Expired Date)**: Hari perdagangan pertama di mana pembeli saham baru sudah tidak lagi berhak mendapatkan dividen periode tersebut.
- **Expected Shortfall (ES)**: Rata-rata kerugian ekspektasi yang terjadi pada saat kondisi ekstrem melebihi batas Value at Risk (VaR).
- **Exponential Moving Average (EMA)**: Rata-rata harga yang memberikan bobot pembobotan eksponensial lebih besar pada harga terbaru.

### F
- **Fat Tails (Leptokurtic)**: Karakteristik distribusi probabilitas di pasar keuangan di mana peristiwa kerugian ekstrem (*black swan*) memiliki peluang kemunculan lebih tinggi dibanding distribusi normal Gaussian.
- **Foreign Flow**: Aliran dana bersih investor asing (beli bersih atau jual bersih) di bursa efek.
- **Fraksi Harga (Tick Size)**: Kelipatan kenaikan atau penurunan harga saham minimum yang disahkan oleh peraturan BEI.
- **Free Float**: Jumlah lembar saham yang beredar bebas di publik dan dimiliki oleh investor non-pengendali (di bawah 5%).

### G
- **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**: Model statistik ekonometrika untuk memodelkan dan meramalkan volatilitas harga saham yang berubah-ubah seiring waktu.

### H
- **HistGradientBoosting**: Implementasi pohon *gradient boosting* berbasis histogram yang cepat dan efisien dalam memproses dataset besar.
- **Holding Period**: Durasi lamanya seorang investor atau trader menyimpan suatu posisi saham.

### I
- **IHSG (Indeks Harga Saham Gabungan / IDX Composite)**: Indeks tolok ukur utama yang mengukur kinerja seluruh saham yang tercatat di Bursa Efek Indonesia.
- **Interim Dividend**: Dividen sementara yang dibagikan emiten di tengah tahun buku berjalan sebelum Rapat Umum Pemegang Saham (RUPS) tahunan.

### K
- **Kas Siaga (Cash Reserve)**: Cadangan kas likuid (tersimpan di RDN atau RDPU) yang sengaja dipertahankan sebagai benteng pertahanan modal saat pasar mengalami kejatuhan.
- **Kompas100 / LQ45**: Indeks saham pilihan BEI yang menyaring emiten dengan likuiditas tinggi dan fundamental baik.

### L
- **Long Short-Term Memory (LSTM)**: Arsitektur *Recurrent Neural Network* (RNN) dalam Deep Learning yang mampu mengingat pola data masa lalu dalam jangka panjang tanpa mengalami masalah *vanishing gradient*.
- **Lot**: Satuan standar resmi perdagangan saham di Bursa Efek Indonesia, di mana 1 lot setara dengan 100 lembar saham.

### M
- **Market Capitalization (Kapitalisasi Pasar)**: Total nilai pasar perusahaan, dihitung dari jumlah seluruh saham beredar dikalikan harga pasar saat ini.
- **Maximum Drawdown (Max DD)**: Penurunan terbesar yang pernah dialami suatu aset dalam kurun waktu tertentu.
- **Money Flow Index (MFI)**: Indikator osilator yang mengukur intensitas aliran uang masuk dan keluar pada suatu saham.

### N
- **Net Foreign Flow (Net Asing)**: Selisih antara nilai total beli asing dengan nilai total jual asing. Bernilai positif jika terjadi *Net Buy* dan negatif jika *Net Sell*.

### P
- **Price to Book Value (PBV)**: Rasio perbandingan antara harga pasar saham dengan nilai buku ekuitas per lembar saham (*Book Value per Share*).
- **Price to Earnings Ratio (PER)**: Rasio perbandingan antara harga pasar saham dengan laba per lembar saham (EPS).
- **Pivot Point**: Level harga acuan tengah yang dihitung dari rata-rata harga High, Low, dan Close kemarin.

### R
- **RDN (Rekening Dana Nasabah)**: Rekening perbankan khusus atas nama investor yang digunakan untuk menampung dana transaksi jual-beli saham di pasar modal.
- **RDPU (Reksa Dana Pasar Uang)**: Instrumen investasi pasar uang berisiko sangat rendah yang likuid, ideal sebagai tempat penempatan Kas Siaga.
- **Relative Strength Index (RSI)**: Osilator momentum yang mengukur kecepatan pergerakan harga pada skala 0 hingga 100.
- **Return on Equity (ROE)**: Rasio profitabilitas yang mengukur seberapa efisien perusahaan menghasilkan laba bersih dari setiap modal ekuitas pemegang saham.
- **Risk-Free Rate ($R_f$)**: Suku bunga investasi tanpa risiko gagal bayar (di Indonesia merujuk pada BI 7-Day Reverse Repo Rate atau SBN).
- **Risk-Reward Ratio (RRR)**: Rasio matematis antara potensi target keuntungan dibandingkan toleransi batas risiko kerugian.

### S
- **Sell on Strength (SoS)**: Strategi menjual saham untuk mengamankan keuntungan saat harga mengalami kenaikan tajam mendekati level atap resistensi.
- **Sharpe Ratio**: Ukuran kinerja investasi yang menghitung kelebihan imbal hasil terhadap suku bunga bebas risiko per unit volatilitas.
- **Stop Loss (SL)**: Perintah batas harga otomatis untuk membatasi kerugian jika arah pergerakan pasar berlawanan dengan rencana trading.
- **Support & Resistance**: Level harga psikologis pasar di mana minat beli diperkirakan menahan penurunan harga (*Support*), atau minat jual diperkirakan menahan kenaikan harga (*Resistance*).

### T
- **Target Price (TP)**: Estimasi level harga wajar target di mana investor merencanakan untuk merealisasikan keuntungan (*take profit*).
- **Tick Size**: Lihat *Fraksi Harga*.
- **Trading Buy**: Keputusan pembelian taktis jangka pendek berdasarkan momentum teknikal positif.

### V
- **Value at Risk (VaR)**: Metrik statistik yang mengukur besaran potensi kerugian maksimal portofolio pada tingkat probabilitas dan rentang waktu tertentu.
- **Volume Ratio**: Perbandingan volume perdagangan hari ini terhadap rata-rata volume perdagangan 20 hari sebelumnya.

---

## DOKUMEN PENDUKUNG & TAUTAN KODE SUMBER

- Modul Ingesti Pasar: [`src/01_data_ingestion.py`](file:///d:/Portofolio/stock-market-recommendation/src/01_data_ingestion.py)
- Modul Rekayasa Fitur & GARCH: [`src/02_feature_eng.py`](file:///d:/Portofolio/stock-market-recommendation/src/02_feature_eng.py)
- Modul Inferensi AI & Optimasi Portofolio: [`src/03_model_inference.py`](file:///d:/Portofolio/stock-market-recommendation/src/03_model_inference.py)
- Modul Arus Modal Asing: [`src/foreign_flow.py`](file:///d:/Portofolio/stock-market-recommendation/src/foreign_flow.py)
- Modul Opening Pulse (10:00 WIB): [`src/opening_pulse.py`](file:///d:/Portofolio/stock-market-recommendation/src/opening_pulse.py)
- Laporan Riset Eksperimen ML: [`docs/RESEARCH_REPORT_RND_ML_UPGRADE.md`](file:///d:/Portofolio/stock-market-recommendation/docs/RESEARCH_REPORT_RND_ML_UPGRADE.md)

---
*© 2026 AlphaTech Quantitative Research Team • Stock Market Recommendation Platform • All Rights Reserved.*
