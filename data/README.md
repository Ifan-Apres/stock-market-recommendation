# 🗄️ Stock Market Recommendation - Data Architecture & Pipeline

Direktori ini menampung seluruh siklus hidup data kuantitatif, mulai dari data mentah (*raw data*) hasil penarikan publik hingga data matang (*processed features & models*) yang siap digunakan oleh mesin rekomendasi dan antarmuka web.

---

## 📁 Struktur Hirarki

```text
data/
├── raw/                                  # Data Mentah (Ingestion Layer)
│   ├── raw_market_data.csv               # Data OHLCV historis harian seluruh semesta saham (2018 - sekarang)
│   ├── benchmark_market_data.csv         # Data historis indeks acuan IHSG (^JKSE)
│   ├── fundamental_financial_data.csv    # Rasio fundamental snapshot (P/E, P/B, ROE, DER, Div Yield)
│   └── financial_statements/             # Folder khusus laporan keuangan multi-tahun per emiten
│       ├── [TICKER]_financial_statements.xlsx  # Multi-sheet Excel (Income, Balance, Cash Flow)
│       └── [TICKER]_income_statement.csv       # Format CSV ringkas per emiten
│
└── processed/                            # Data Terproses & Model Artifacts
    ├── processed_market_features.csv     # Matriks 60+ fitur kuantitatif (Teknikal, GARCH, Momentum)
    ├── advanced_quant_metrics.csv        # Metrik risiko lanjutan (Sharpe, Beta IHSG, VaR 95%, Pivots)
    ├── alpha_model.joblib                # Serialized model Gradient Boosted Decision Tree (GBDT)
    ├── lstm_model.pth                    # Bobot neural network PyTorch LSTM untuk sekuensial return
    ├── latest_alpha_recommendations.csv  # Rekomendasi kuantitatif gabungan terkini
    ├── latest_alpha_recommendations_swing.csv     # Rekomendasi taktis untuk Swing Trader
    ├── latest_alpha_recommendations_dividend.csv  # Rekomendasi defensif untuk Dividend/Value Investor
    ├── latest_alpha_recommendations_favorites.csv # Rekomendasi khusus Top 10 Portfolio Pilihan
    ├── latest_portfolio_allocation.csv   # Hasil optimasi bobot alokasi modal (Risk-Parity & Sharpe)
    └── latest_morning_brief.json         # Narasi editorial Morning Brief harian dari Gemini AI
```

---

## 🔒 Integritas Data & Kepatuhan Hukum (Clean-Room Protocol)

Semua data di dalam direktori ini diperoleh secara legal dari sumber-sumber publik terbuka:
- **OHLCV & Valuasi**: Yahoo Finance Data Feed resmi (`yfinance` API dengan sufiks bursa `.JK`).
- **Suku Bunga Bebas Risiko**: Bank Indonesia BI-Rate (6.00% p.a.).
- **Benchmarking**: Indeks Harga Saham Gabungan (IHSG / `^JKSE`).
- **Tidak ada scraping ilegal**: Tidak menggunakan token rahasia, *reverse-engineering*, maupun data kepemilikan tertutup dari vendor riset mana pun (sesuai kepatuhan PKWTT Infovesta).
