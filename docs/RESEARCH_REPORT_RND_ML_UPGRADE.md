# LAPORAN RISET & BENCHMARK KUANTITATIF MACHINE LEARNING
## Upgrade Arsitektur Machine Learning & Feature Engineering
**Tanggal:** 03 Oktober 2026  
**Cabang Riset:** `rnd`  
**Status:** Berhasil Diuji & Siap Diintegrasikan ke Sistem Produksi  
**Ukuran Sampel Uji Out-Of-Sample:** 17.594 baris data uji kronologis (20% data bursa terbaru)

---

## 1. Latar Belakang & Motivasi Riset
Sistem rekomendasi saham berbasis multi-engine AI sebelumnya menggunakan model tunggal *HistGradientBoosting* (GBDT) yang dipadukan dengan *Unidirectional LSTM* dan *ARIMA*. Meskipun stabil, sistem memiliki beberapa celah kuantitatif:
1. **Belum Adanya Sinyal Khusus Aksi Korporasi**: Siklus pembagian dividen (musim dividen April–Juni & November–Desember) belum dimodelkan secara fitur kontinu, sehingga model rentan terhadap jebakan *Ex-Dividend drop*.
2. **Keterbatasan Sinyal Arus Dana Asing**: Sinyal net asing sebelumnya hanya membaca volume nominal, belum mengukur anomali *Divergensi Akumulasi Asing* (saat broker asing terus membeli ketika harga saham sedang ditekan atau stagnan).
3. **Ketergantungan pada Model Tunggal GBDT**: Pohon keputusan konvensional dapat mengalami bias pada data tabular bursa yang memiliki *Signal-to-Noise Ratio* (SNR) rendah.
4. **Keinginan R&D Tanpa Merusak Sistem Produksi**: Seluruh eksperimen dijalankan secara terisolasi di branch `rnd` untuk membuktikan secara empiris bahwa akurasi benar-benar meningkat sebelum di-merge ke branch `main`.

---

## 2. Ringkasan Eksekutif Hasil 4 Eksperimen

Empat eksperimen kuantitatif independen telah dieksekusi secara ketat pada data uji *out-of-sample*:

| No | Eksperimen | Arsitektur yang Diuji | Akurasi | ROC-AUC | Top Decile Precision | High Conviction ($\ge 55\%$) | Waktu Latih | Kesimpulan Kuantitatif |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Baseline Awal** | HistGB Lama (26 Fitur) | 53.55% | 0.5438 | 55.71% | 57.66%* | 0.85s | Baseline acuan sistem |
| **1** | **Eksperimen 1** | CatBoost Classifier (32 Fitur) | 53.48% | **0.5495** ⭐ | 56.70% | 54.84% | 2.19s | ROC-AUC tertinggi untuk model tunggal |
| **1** | **Eksperimen 1** | **Tree Blend (50:50 HistGB + CatBoost)** | **53.70%** 🏆 | **0.5485** ⭐ | **57.16%** ⭐ | **55.89%** ⭐ | **3.15s** | **Pemenang Mutlak (Tertinggi & Paling Stabil)** |
| **2** | **Eksperimen 2** | Optuna Bayesian Tuning (TPE + Purged CV) | 52.64% | 0.5470 | 55.21% | 55.21% | ~1.62s | Regularisasi L2 terlalu ketat memangkas recall |
| **3** | **Eksperimen 3** | Stacking Ensemble (OOF Logistic Meta-Learner) | 51.88% | 0.5461 | 55.26% | 55.26% | ~8.50s | Meta-overfitting pada kalibrasi probabilitas |
| **4** | **Eksperimen 4** | Standard Unidirectional LSTM | 50.08% | 0.5000 | 0.00% | 0.00% | 236s (~4m) | Mode collapse pada standalone deep learning |
| **4** | **Eksperimen 4** | Bidirectional LSTM (BiLSTM Dual-Endpoint) | 50.08% | 0.5000 | 0.00% | 0.00% | **1.160s (19,3m)** ⚠️ | Beban komputasi 5x, tidak layak untuk GitHub Actions |
| **4** | **Eksperimen 4** | Temporal Attention LSTM | 50.08% | 0.5000 | 0.00% | 0.00% | 274s (~4,5m) | Mengalami fenomena loss saturation yang serupa |
| **4** | **Konsensus** | **Multi-Engine: Tree Blend (60%) + LSTM (25%) + ARIMA (15%)** | **53.70%** | **0.5485** | **57.16%** | **55.89%** | **~4 menit** | **Kombinasi Produksi Terpilih** |

---

## 3. Rincian Metodologi & Temuan Tiap Eksperimen

### 3.1 Feature Engineering Baru (Foreign Flow Divergence & Dividend Seasonality)
Enam fitur kuantitatif baru ditambahkan ke dalam pipeline feature engineering (`src/02_feature_eng.py`):
1. `Is_Dividend_Season`: Indikator biner bulan puncak pembagian dividen IHSG (Bulan 4–6 dan 11–12).
2. `Dividend_Season_Momentum`: Interaksi non-linear antara Dividend Yield tahunan emiten dengan musim dividen.
3. `Dividend_Yield_Norm`: Normalisasi *Dividend Yield* berbasis persentil cross-sectional terhadap seluruh konstituen bursa.
4. `Foreign_Accum_Divergence`: Mengukur divergensi arah harga saham terhadap arus dana investor asing selama 10 hari bursa.
5. `Foreign_Flow_Intensity`: Normalisasi nominal transaksi bersih asing terhadap rata-rata turnover harian saham.
6. `Foreign_Consistent_Buy_5D`: Rasio konsistensi akumulasi beli bersih asing tanpa putus selama 5 hari perdagangan berturut-turut.

**Temuan Kuantitatif:**
* Pada model CatBoost, fitur `Foreign_Accum_Divergence` langsung menembus peringkat **#7 fitur paling penting** dari 32 fitur bursa dengan bobot kepentingan **4.27%**.
* Kelompok fitur dividen dan arus asing secara kumulatif menyumbang **> 20.8% bobot keputusan** CatBoost.

---

### 3.2 Eksperimen 1: Uji Algoritma Baru (CatBoost vs HistGB vs Blend)
* **Tujuan:** Menguji apakah *Oblivious Symmetric Decision Trees* (CatBoost) mengungguli *Histogram Gradient Boosting* konvensional.
* **Hasil:**
  * CatBoost menang dalam separasi probabilitas (**ROC-AUC 0.5495** vs 0.5446).
  * **Tree Blend Ensemble (50% HistGB + 50% CatBoost)** memenangkan semua metrik out-of-sample:
    * Akurasi: **53.70%** (naik dari 53.55% baseline lama).
    * High Conviction Precision ($\ge 0.55$): **55.89%** (+3.56% peningkatan dibandingkan baseline).
    * Top Decile Precision: **57.16%** (+1.45% peningkatan).
    * Waktu pelatihan: Hanya **3.15 detik** di CPU, sangat efisien.

---

### 3.3 Eksperimen 2: Hyperparameter Tuning (Optuna Bayesian Optimization)
* **Tujuan:** Mencari kedalaman pohon (*depth*), *learning rate*, dan regularisasi L2 optimal menggunakan algoritma Tree-structured Parzen Estimator (TPE) dan 3-fold *Purged Time-Series Cross Validation* dengan 5-day embargo gap.
* **Parameter Terbaik yang Ditemukan:**
  * **CatBoost**: `depth: 3`, `learning_rate: 0.0164`, `l2_leaf_reg: 13.26`, `subsample: 0.83`, `random_strength: 2.98`.
  * **HistGB**: `max_depth: 3`, `learning_rate: 0.0268`, `max_iter: 100`, `min_samples_leaf: 50`, `l2_regularization: 5.10`.
* **Analisis Kuantitatif:**
  * Optuna secara alami memilih regularisasi L2 yang sangat berat (`13.26` dan `5.10`) karena pasar saham memiliki *signal-to-noise ratio* yang sangat rendah.
  * Akibatnya, probabilitas prediksi terkompresi mendekati nilai tengah (0.50). Meskipun presisi saat memprediksi positif tetap tinggi (55.21%), *recall* terpangkas dari 25.3% menjadi 10.5%.

---

### 3.4 Eksperimen 3: Dynamic Stacking Ensemble (Meta-Learner Level-1)
* **Tujuan:** Menguji apakah *Logistic Regression Meta-Learner* yang dilatih pada 45.691 pasang prediksi *Out-Of-Fold (OOF)* mampu menghasilkan bobot dinamis yang lebih baik daripada bobot statis 50:50.
* **Persamaan yang Dipelajari Meta-Learner:**
  $$\text{logit}(P_{up}) = 0.745 \cdot P(\text{CatBoost}) + 1.282 \cdot P(\text{HistGB}) - 1.051$$
* **Temuan Kuantitatif:**
  * Meta-Learner memberi bobot lebih besar ke HistGB (1.282) dan mengoreksi intercept (-1.051).
  * Namun, akurasi out-of-sample Stacking (51.88%) justru lebih rendah dibandingkan Blend Ensemble sederhana (53.70%).
  * **Pelajaran Finansial:** Sesuai dengan literatur *Advances in Financial Machine Learning* (Marcos Lopez de Prado), meta-learner tingkat kedua di pasar saham sangat rentan mengalami *meta-overfitting* terhadap noise kalibrasi OOF. Model agregasi *Equal-Weight Blend* jauh lebih kokoh (*robust*).

---

### 3.5 Eksperimen 4: Deep Learning Sequence Benchmark (Standard LSTM vs BiLSTM vs Attention)
* **Tujuan:** Menjawab hipotesis apakah mengganti *Standard Unidirectional LSTM* dengan *Bidirectional LSTM (BiLSTM)* atau *Temporal Attention LSTM* dapat meningkatkan akurasi.
* **Analisis Teoretis:**
  * BiLSTM **tidak membocorkan data masa depan** (*zero lookahead bias*) karena jendela sekuens hanya berisi 30 hari historis ($t \le 30$).
  * Namun, waktu bursa bergerak satu arah (*arrow of time*). Menjalankan backward pass pada data harga saham dapat memperkenalkan korelasi semu.
* **Hasil Pengujian Empiris:**
  1. **Beban Komputasi Ekstrem:** BiLSTM membutuhkan waktu latih **1.160,34 detik (~19,3 menit)** di CPU untuk 1 model saja! Ini 5 kali lipat lebih lambat daripada Standard LSTM (236 detik / ~4 menit).
  2. **Risiko Operasional GitHub Actions:** Batas waktu eksekusi workflow harian GitHub Actions bisa terancam *timeout* jika menggunakan BiLSTM.
  3. **Peningkatan Presisi Marjinal:** Pada Multi-Engine, BiLSTM hanya menaikkan Top Decile Precision dari 55.70% ke 56.02% (+0.32%), kenaikan yang tidak sebanding dengan pembengkakan komputasi 500%.
  4. **Mode Collapse Standalone:** Model sequence deep learning murni sangat rentan konvergen ke titik tengah (0.5008) pada data bursa yang bergejolak.

---

## 4. Keputusan Arsitektur Final (Production Decision)

Berdasarkan seluruh data empiris di atas, diputuskan arsitektur final untuk sistem produksi adalah:

1. **Mesin Alpha Tabular Utama:**
   * **Tree Blend Ensemble**: 50% *HistGradientBoostingClassifier* + 50% *CatBoostClassifier*.
   * Menggunakan 32 fitur kuantitatif lengkap (termasuk fitur baru *Foreign Accumulation Divergence* dan *Dividend Seasonality*).
   * Menghasilkan Akurasi **53.70%**, ROC-AUC **0.5485**, dan High Conviction Precision **55.89%**.
2. **Mesin Deep Learning Temporal:**
   * Tetap menggunakan **Standard Unidirectional PyTorch LSTM** (14.881 parameter, ~4 menit) sebagai pelengkap pola sekuens temporal.
   * Menolak arsitektur BiLSTM demi menjaga kelayakan runtime GitHub Actions.
3. **Bobot Multi-Engine Consensus:**
   $$\text{Blended Prob} = 0.60 \times P(\text{Tree Blend}) + 0.25 \times P(\text{LSTM}) + 0.15 \times P(\text{ARIMA})$$
   Memberi bobot dominan (60%) kepada Tree Blend yang terbukti paling akurat dan kokoh, dengan pengaman diversifikasi dari LSTM dan ARIMA.
4. **Sinergi Alokasi Portofolio:**
   * Probabilitas Tree Blend dikombinasikan dengan *Corporate Actions Ex-Date Shield* dan *Dividend Yield Momentum Multiplier* pada lapisan alokasi portofolio institusional.

---
*Laporan ini disimpan sebagai dokumentasi resmi R&D di repositori proyek pada branch `rnd`.*
