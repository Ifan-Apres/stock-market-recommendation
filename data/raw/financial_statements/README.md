# 📊 Database Laporan Keuangan Multi-Tahun Emiten (Financial Statements)

Direktori ini menyimpan basis data laporan keuangan terpisah untuk masing-masing emiten Bursa Efek Indonesia (IDX) dalam format Excel Workbook (`.xlsx`) dan CSV (`.csv`).

---

## 📁 Struktur Penyimpanan

Setiap emiten memiliki file tersendiri dengan format penamaan standar:
```text
data/raw/financial_statements/
├── [TICKER]_financial_statements.xlsx   <-- Multi-sheet workbook lengkap (Tahunan & TTM)
└── [TICKER]_income_statement.csv         <-- Ekstraksi Laporan Laba Rugi untuk komputasi cepat
```

Contoh file yang tersedia:
- `BBCA_financial_statements.xlsx` & `BBCA_income_statement.csv` (PT Bank Central Asia Tbk)
- `BBRI_financial_statements.xlsx` & `BBRI_income_statement.csv` (PT Bank Rakyat Indonesia Tbk)
- `BMRI_financial_statements.xlsx` & `BMRI_income_statement.csv` (PT Bank Mandiri Tbk)
- `TLKM_financial_statements.xlsx` & `TLKM_income_statement.csv` (PT Telkom Indonesia Tbk)
- `ASII_financial_statements.xlsx` & `ASII_income_statement.csv` (PT Astra International Tbk)
- ... dan 20 emiten berkapitalisasi besar lainnya.

---

## 📑 Struktur Lembar Kerja Excel (`.xlsx`)

Setiap file `.xlsx` terdiri dari 3 sheet utama yang mencerminkan komponen resmi laporan keuangan standar IFRS / PSAK:

| Sheet Name | Isi Dokumen | Metrik Kunci |
| :--- | :--- | :--- |
| **`Income Statement`** | Laporan Laba Rugi Komprehensif | Pendapatan Usaha (*Total Revenue*), Laba Usaha (*Operating Income*), Laba Bersih (*Net Income*), Beban Bunga, Beban Pajak |
| **`Balance Sheet`** | Laporan Posisi Keuangan (Neraca) | Kas & Setara Kas, Total Aset (*Total Assets*), Total Liabilitas (*Total Debt / Liabilities*), Ekuitas Pemegang Saham (*Stockholders' Equity*) |
| **`Cash Flow`** | Laporan Arus Kas | Arus Kas Operasi (*Operating Cash Flow*), Belanja Modal (*Capital Expenditure*), Arus Kas Bebas (*Free Cash Flow*), Pembayaran Dividen |

---

## ⏳ Periode Data Historis

- **Rentang Waktu**: Tahun buku 2020, 2021, 2022, 2023, 2024, hingga periode *Trailing Twelve Months* (TTM) terkini.
- **Tujuan**:
  1. Melakukan analisis tren pertumbuhan pendapatan (*Revenue CAGR*) dan margin laba (*NPM, OPM*).
  2. Menghitung rasio solvabilitas (*Debt to Equity Ratio*) dan likuiditas (*Current Ratio*) secara historis.
  3. Memvalidasi kemampuan emiten membagikan dividen (*Dividend Payout Ratio & Free Cash Flow yield*).

---

## 🔄 Cara Menambah atau Memperbarui Emiten

Untuk menambahkan laporan keuangan emiten baru ke direktori ini:
1. Simpan file Excel dengan nama `[KODETICKER]_financial_statements.xlsx`.
2. Pastikan sheet berisi minimal kolom: `Year`, `Total_Revenue`, `Operating_Income`, `Net_Income`.
3. Pipeline kuantitatif `src/01_data_ingestion.py` dan `src/02_feature_eng.py` akan secara otomatis mendeteksi dan mengintegrasikan data ke dalam matriks faktor fundamental.
