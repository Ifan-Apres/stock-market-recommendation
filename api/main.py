import hashlib
import hmac
import importlib
import json
import logging
import os
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path for absolute imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
from dotenv import load_dotenv  # type: ignore # pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request  # type: ignore # pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles  # type: ignore # pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore # pyrefly: ignore [missing-import]
from fastapi.responses import HTMLResponse, JSONResponse  # type: ignore # pyrefly: ignore [missing-import]
import google.generativeai as genai  # type: ignore # pyrefly: ignore [missing-import]
import numpy as np  # type: ignore # pyrefly: ignore [missing-import]
import pandas as pd  # type: ignore # pyrefly: ignore [missing-import]
from pydantic import BaseModel  # type: ignore # pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from src.auth import (  # type: ignore # pyrefly: ignore [missing-import]
    JWT_SECRET_KEY,
    authenticate_user,
    create_token,
    rate_limiter,
    register_user,
    verify_token,
)
from src.config import (  # type: ignore # pyrefly: ignore [missing-import]
    DEFAULT_TICKERS,
    DIVIDEND_RECOMMENDATION_FILE,
    FAVORITE_TICKERS,
    FAVORITES_RECOMMENDATION_FILE,
    LOG_FORMAT,
    MORNING_BRIEF_FILE,
    PORTFOLIO_ALLOCATION_FILE,
    PROCESSED_DATA_DIR,
    SECTOR_MAP,
    SNAPSHOT_FILE,
    SWING_RECOMMENDATION_FILE,
    WATCHLIST_ANALYSIS_FILE,
)

load_dotenv()
FINANCIALS_SUMMARY_FILE = PROCESSED_DATA_DIR / "financial_statements_summary.json"
PRICE_HISTORY_FILE = PROCESSED_DATA_DIR / "price_history_30d.json"
FOREIGN_FLOW_FILE = PROCESSED_DATA_DIR / "foreign_flow_summary.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Lazy pipeline imports for serverless compatibility
def run_ingestion_pipeline():
    mod = importlib.import_module("src.01_data_ingestion")
    return mod.run_ingestion_pipeline()

def run_feature_engineering_pipeline():
    mod = importlib.import_module("src.02_feature_eng")
    return mod.run_feature_engineering_pipeline()

def run_model_inference_pipeline():
    mod = importlib.import_module("src.03_model_inference")
    return mod.run_model_inference_pipeline()

def run_morning_brief_pipeline():
    mod = importlib.import_module("src.morning_brief")
    return mod.run_morning_brief_pipeline()

def run_watchlist_analyzer_pipeline():
    mod = importlib.import_module("src.stock_analyzer")
    return mod.run_watchlist_analyzer_pipeline()

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("StockMarketRecommendationAPI")

RECOMMENDATION_FILE = SWING_RECOMMENDATION_FILE

ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() in ("true", "1")

app = FastAPI(
    title="Stock Market Recommendation API",
    description="Multi-Engine Algorithmic Recommendation Engine (GBDT + LSTM + ARIMA + GARCH), Institutional Morning Brief & Portfolio Allocator",
    version="4.3.0",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

DATA_DIR = ROOT_DIR / "data"
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# 1. Security Headers & Rate Limiting Middleware
@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "127.0.0.1"
    path = request.url.path
    user_agent = (request.headers.get("user-agent") or "").lower()

    # Anti-Bot & Scraper WAF: Block automated programmatic scrapers from crawling sensitive endpoints
    BLOCKED_BOT_SIGNATURES = [
        "python-requests",
        "aiohttp",
        "curl/",
        "wget/",
        "scrapy",
        "postmanruntime",
        "go-http-client",
        "httpclient",
        "urllib",
        "httpx",
    ]
    if any(sig in user_agent for sig in BLOCKED_BOT_SIGNATURES):
        if path not in ("/", "/status", "/api/status"):
            logger.warning(f"Blocked scraper bot request from {client_ip} targeting {path} [User-Agent: {user_agent}]")
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Akses Ditolak: Akses scraper/bot otomatis terdeteksi (Anti-Scraping Protection Active). Akses data programmatic hanya diizinkan melalui antarmuka web resmi.",
                    "error_code": "BOT_SCRAPING_FORBIDDEN",
                    "client_ip": client_ip,
                },
                headers={"X-Security-Firewall": "Anti-Scraping-Active"},
            )

    # Anti-Brute-Force Rate Limiting on Auth endpoints (max 10 req/min)
    if path.startswith("/api/v1/auth/"):
        allowed, retry_after = rate_limiter.is_allowed(client_ip, "auth", max_requests=10, window_seconds=60)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Terlalu banyak percobaan autentikasi. Demi keamanan, silakan coba lagi dalam {retry_after} detik."},
                headers={"Retry-After": str(retry_after)},
            )
    # Anti-Abuse Rate Limiting on Gemini AI endpoints (max 8 req/min)
    elif "/gemini/" in path or path.startswith("/api/analysis/"):
        allowed, retry_after = rate_limiter.is_allowed(client_ip, "gemini_ai", max_requests=8, window_seconds=60)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Batas kuota akses Gemini AI tercapai. Silakan coba lagi dalam {retry_after} detik."},
                headers={"Retry-After": str(retry_after)},
            )

    # Anti-Crawling & Anti-Bulk-Dumping Rate Limiting on Data Endpoints
    DATA_ROUTES = (
        "/recommendations",
        "/watchlist-analysis",
        "/api/recommendations",
        "/api/watchlist-analysis",
        "/api/stocks/analyze",
        "/stocks/analyze",
        "/api/financials",
        "/api/history",
        "/api/foreign-flow",
        "/portfolio/allocate",
        "/api/portfolio/allocate",
    )
    if any(path.startswith(dr) for dr in DATA_ROUTES):
        # A. Burst protection: max 10 requests per 2 seconds (stops automated loops)
        allowed_burst, _ = rate_limiter.is_allowed(client_ip, "data_burst", max_requests=10, window_seconds=2)
        if not allowed_burst:
            logger.warning(f"Rapid burst crawling detected from {client_ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Aktivitas penjelajahan otomatis terlalu cepat terdeteksi (Anti-Crawling Active). Mohon jeda sejenak sebelum melanjutkan.",
                    "error_code": "BURST_RATE_LIMIT_EXCEEDED",
                },
                headers={"Retry-After": "2", "X-Security-Firewall": "Burst-Limit-Active"},
            )

        # B. Sliding window protection: max 45 requests per 60 seconds (generous for human reading, stops bulk database crawlers)
        allowed_window, retry_after = rate_limiter.is_allowed(client_ip, "data_window", max_requests=45, window_seconds=60)
        if not allowed_window:
            logger.warning(f"Data crawling window rate limit exceeded from {client_ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Batas penjelajahan data per menit tercapai. Demi stabilitas sistem, silakan coba lagi dalam {retry_after} detik.",
                    "error_code": "DATA_RATE_LIMIT_EXCEEDED",
                },
                headers={"Retry-After": str(retry_after), "X-Security-Firewall": "Rate-Limit-Active"},
            )

        # C. Browser Origin Integrity Check
        origin = request.headers.get("origin")
        if origin:
            clean_origin = origin.rstrip("/")
            valid_origins = [o.rstrip("/") for o in ALLOWED_ORIGINS]
            if clean_origin not in valid_origins and not any(clean_origin.startswith(vo) for vo in valid_origins):
                logger.warning(f"Untrusted external origin {origin} attempted to access {path}")
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Akses Ditolak: Permintaan berasal dari Domain/Origin tidak sah (Cross-Origin Protection Active).",
                        "error_code": "UNAUTHORIZED_ORIGIN",
                    },
                    headers={"X-Security-Firewall": "Origin-Validation-Active"},
                )

    response = await call_next(request)

    # Industry-standard HTTP Security & Digital IP Headers
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["X-Security-Policy"] = "AlphaTech-Defense-Suite-v4.5"
    response.headers["X-Intellectual-Property"] = "Copyright 2026 AlphaTech Quantitative Research - Proprietary & Confidential"

    # Digital Canary Watermarking for Authenticated Leak Tracking
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ", 1)[1].strip()
        payload = verify_token(token)
        if payload:
            user_sub = payload.get("sub", "analyst")
            trace_seed = f"{user_sub}:{int(time.time() // 300)}"
            trace_sig = hmac.new(JWT_SECRET_KEY.encode("utf-8"), trace_seed.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
            response.headers["X-Audit-Trace-ID"] = f"trc_{trace_sig}"

    return response

# 2. Controlled CORS Configuration
ALLOWED_ORIGINS = [
    origin.strip() for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8080,http://127.0.0.1:8080,http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000,https://stock-market-recommendation-silk.vercel.app"
    ).split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Dependency to enforce valid Bearer token authentication."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Akses ditolak: Autentikasi diperlukan. Silakan masuk (login) terlebih dahulu untuk membuka akses ke seluruh layer rekomendasi.",
        )
    token = authorization.split("Bearer ", 1)[1].strip()
    payload = verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Sesi login Anda tidak valid atau telah kedaluwarsa. Silakan login kembali.",
        )
    return payload

def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Optional authentication check (returns user payload or None)."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split("Bearer ", 1)[1].strip()
    return verify_token(token)


class RecommendationItem(BaseModel):
    Date: str
    Ticker: str
    Sector: Optional[str] = "General"
    Close: float
    Recommendation: str
    Entry_Price: Optional[float] = None
    Target_Price: Optional[float] = None
    Stop_Loss: Optional[float] = None
    Risk_Reward_Ratio: Optional[float] = None
    Bullish_Probability: float
    GBDT_Prob: Optional[float] = None
    ARIMA_Prob: Optional[float] = None
    LSTM_Prob: Optional[float] = None
    Beta_IHSG: Optional[float] = None
    Sharpe_Ratio: Optional[float] = None
    Max_Drawdown_1Y: Optional[float] = None
    Annualized_Return_1Y: Optional[float] = None
    GARCH_Vol: Optional[float] = None
    VaR_95_1D: Optional[float] = None
    Debt_to_Equity: Optional[float] = None
    Current_Ratio: Optional[float] = None
    RSI_14: Optional[float] = None
    MACD_Hist: Optional[float] = None
    MFI_14: Optional[float] = None
    CMF_20: Optional[float] = None
    PE_Ratio: Optional[float] = None
    PB_Ratio: Optional[float] = None
    ROE: Optional[float] = None
    Dividend_Yield: Optional[float] = None


class AllocationItem(BaseModel):
    Ticker: str
    Sector: str
    Recommendation: str
    Allocation_Pct: float
    Nominal_IDR: float
    Shares_Lot: int
    Close_Price: float
    Target_Price: float
    Stop_Loss: float
    Bullish_Probability: float


class PipelineResponse(BaseModel):
    status: str
    message: str


def execute_full_pipeline():
    logger.info("Executing automated end-to-end quant pipeline...")
    try:
        run_ingestion_pipeline()
        run_feature_engineering_pipeline()
        run_model_inference_pipeline()
        run_morning_brief_pipeline()
        run_watchlist_analyzer_pipeline()
        logger.info("Full quant pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Error during daily pipeline execution: {str(e)}", exc_info=True)


def _clean_record(record: Any) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    if not isinstance(record, dict):
        return cleaned
    for k, v in record.items():
        key = str(k)
        if pd.isna(v) or v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
            cleaned[key] = None
        elif isinstance(v, (np.floating, float)):
            cleaned[key] = round(float(v), 4)
        elif isinstance(v, (np.integer, int)):
            cleaned[key] = int(v)
        else:
            cleaned[key] = v
    return cleaned


@app.get("/status", tags=["Status"])
@app.get("/api/status", tags=["Status"])
def get_pipeline_status() -> Dict[str, Any]:
    last_update = "Not Executed"
    if RECOMMENDATION_FILE.exists():
        mtime = os.path.getmtime(RECOMMENDATION_FILE)
        last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "online",
        "engine": "Hybrid Quant Suite (GBDT + ARIMA + PyTorch LSTM + GARCH)",
        "last_pipeline_update": last_update,
        "sectors_covered": list(SECTOR_MAP.keys()),
        "recommendation_file": str(RECOMMENDATION_FILE),
    }


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def web_dashboard():
    index_file = Path(__file__).resolve().parent.parent / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Stock Market Recommendation Platform is running.</h1>")


@app.get("/login", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/login.html", response_class=HTMLResponse, tags=["Dashboard"])
def web_login():
    login_file = Path(__file__).resolve().parent.parent / "login.html"
    if login_file.exists():
        return HTMLResponse(content=login_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Halaman Login Tidak Ditemukan</h1>")


class AuthRegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    role: Optional[str] = "Market Explorer"
    company_fax: Optional[str] = None  # Anti-Bot Honeypot Trap (Must remain empty)
    client_ts: Optional[float] = None  # Timestamp for human typing speed verification
    turnstile_token: Optional[str] = None  # Optional Cloudflare Turnstile token


class AuthLoginRequest(BaseModel):
    username: str
    password: Optional[str] = None
    remember_me: Optional[bool] = True


@app.post("/api/v1/auth/register", tags=["Auth"])
def auth_register(req: AuthRegisterRequest, request: Request) -> Dict[str, Any]:
    """Registers a new user account with multi-layer anti-bot armor and PBKDF2 hashing."""
    client_ip = request.client.host if request.client else "127.0.0.1"

    # 1. Anti-Bot Honeypot Trap: Reject if hidden field is filled
    if req.company_fax and req.company_fax.strip():
        logger.warning(f"Registration honeypot trap triggered by {client_ip} [Field content: {req.company_fax}]")
        raise HTTPException(
            status_code=400,
            detail="Aktivitas otomasi mencurigakan terdeteksi (Anti-Bot Trap). Pendaftaran dibatalkan.",
        )

    # 2. Registration Quota: Max 3 accounts per hour per IP (Stops mass account generators)
    allowed, retry_after = rate_limiter.is_allowed(client_ip, "register_account", max_requests=3, window_seconds=3600)
    if not allowed:
        logger.warning(f"Registration quota exceeded for IP {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Batas pendaftaran akun baru per jam tercapai untuk jaringan Anda. Silakan coba lagi dalam {max(1, retry_after // 60)} menit atau masuk menggunakan akun yang ada.",
        )

    # 3. Submission Speed Trap: Humans take >= 1.5 seconds to fill registration
    if req.client_ts:
        now_ms = time.time() * 1000
        time_elapsed_ms = now_ms - req.client_ts
        if time_elapsed_ms < 1500:
            logger.warning(f"Registration speed trap triggered by {client_ip} (form filled in {time_elapsed_ms:.0f}ms)")
            raise HTTPException(
                status_code=400,
                detail="Pengisian formulir terlalu cepat (terindikasi automated bot script). Harap isi formulir secara manual.",
            )

    # 4. Optional Cloudflare Turnstile Server Verification
    turnstile_secret = os.getenv("TURNSTILE_SECRET_KEY")
    if turnstile_secret:
        if not req.turnstile_token:
            raise HTTPException(status_code=400, detail="Verifikasi Cloudflare Turnstile diperlukan.")
        try:
            import urllib.parse
            import urllib.request
            verify_payload = urllib.parse.urlencode({
                "secret": turnstile_secret,
                "response": req.turnstile_token,
                "remoteip": client_ip,
            }).encode("utf-8")
            cf_req = urllib.request.Request(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data=verify_payload,
                method="POST",
            )
            with urllib.request.urlopen(cf_req, timeout=5) as cf_resp:
                cf_result = json.loads(cf_resp.read().decode("utf-8"))
                if not cf_result.get("success"):
                    raise HTTPException(status_code=400, detail="Verifikasi Cloudflare Turnstile gagal. Silakan muat ulang halaman.")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Turnstile verification error: {e}")

    # 5. Core Registration with PBKDF2
    success, message, user_data = register_user(
        name=req.name,
        email=req.email,
        password=req.password,
        role=req.role or "Market Explorer",
    )
    if not success or not user_data:
        raise HTTPException(status_code=400, detail=message)

    token = create_token(user_data)
    return {
        "status": "success",
        "message": message,
        "token": token,
        "user": user_data,
    }


@app.post("/api/v1/auth/login", tags=["Auth"])
def auth_login(req: AuthLoginRequest) -> Dict[str, Any]:
    """Authenticates user credentials using cryptographic verification and issues signed bearer token."""
    if not req.password:
        raise HTTPException(status_code=400, detail="Kata sandi wajib diisi.")

    success, message, user_data = authenticate_user(req.username, req.password)
    if not success or not user_data:
        raise HTTPException(status_code=401, detail=message)

    token = create_token(user_data)
    return {
        "status": "success",
        "message": message,
        "token": token,
        "user": user_data,
    }


@app.get("/api/v1/auth/me", tags=["Auth"])
def auth_me(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """Returns the authenticated user profile."""
    return {
        "status": "authenticated",
        "user": current_user,
    }



@app.get("/morning-brief", tags=["Morning Market Brief"])
@app.get("/api/morning-brief", tags=["Morning Market Brief"])
def get_morning_brief() -> Dict[str, Any]:
    """
    Returns the latest institutional Morning Brief IHSG narrative.
    """
    if MORNING_BRIEF_FILE.exists():
        try:
            with open(MORNING_BRIEF_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Generate fresh if not available
    return run_morning_brief_pipeline()


@app.get("/sectors", tags=["Sectors"])
@app.get("/api/sectors", tags=["Sectors"])
def get_sectors() -> Dict[str, List[str]]:
    """
    Returns the 11 official IDX sectors and their constituent tickers.
    """
    return SECTOR_MAP


@app.get("/universe", tags=["Universe Management"])
@app.get("/api/universe", tags=["Universe Management"])
def get_universe_telemetry() -> Dict[str, Any]:
    """
    Returns diagnostic telemetry for the 66-stock dynamic universe,
    including sector distribution, health statuses, and standby replacements.
    """
    try:
        from src.universe_manager import get_universe_manager
        mgr = get_universe_manager()
        return mgr.get_universe_diagnostics()
    except Exception as e:
        return {
            "capacity": 66,
            "active_tickers_count": len(DEFAULT_TICKERS),
            "active_tickers": DEFAULT_TICKERS,
            "sector_breakdown": {s: len(t) for s, t in SECTOR_MAP.items()},
            "error": str(e),
        }


@app.get("/portfolio/allocate", response_model=List[AllocationItem], tags=["Portfolio Optimizer"])
@app.get("/api/portfolio/allocate", response_model=List[AllocationItem], tags=["Portfolio Optimizer"])
def get_portfolio_allocation(
    capital: float = Query(50000000.0, description="Total capital in IDR (Rupiah)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict]:
    """
    Computes optimal portfolio allocation weights and nominal IDR for user's capital.
    """
    if PORTFOLIO_ALLOCATION_FILE.exists():
        try:
            df = pd.read_csv(PORTFOLIO_ALLOCATION_FILE)
            # Re-scale nominal if capital differs from default
            records = []
            for _, row in df.iterrows():
                r = row.to_dict()
                pct = r["Allocation_Pct"] / 100.0
                nominal = round(capital * pct, 0)
                close = r["Close_Price"]
                lot = int((nominal / close) // 100) if close > 1.0 else 0
                r["Nominal_IDR"] = nominal
                r["Shares_Lot"] = lot
                records.append(_clean_record(r))
            return records
        except Exception as e:
            logger.error(f"Error reading portfolio allocation: {str(e)}")

    raise HTTPException(status_code=404, detail="Portfolio allocation data not generated yet. Run pipeline first.")


@app.get("/recommendations", response_model=List[RecommendationItem], tags=["Recommendations"])
@app.get("/api/recommendations", response_model=List[RecommendationItem], tags=["Recommendations"])
def get_latest_recommendations(
    mode: str = Query("all", description="Strategy mode: 'all', 'swing', 'dividend', or 'favorites'"),
    universe: Optional[str] = Query(None, description="Alias for mode parameter"),
    sector: Optional[str] = Query(None, description="Optional sector filter (e.g. 'Financials', 'Energy')"),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict]:
    active_mode = (universe or mode or "all").lower()

    try:
        if active_mode == "dividend":
            df = pd.read_csv(DIVIDEND_RECOMMENDATION_FILE)
        elif active_mode == "favorites":
            df = pd.read_csv(FAVORITES_RECOMMENDATION_FILE)
        elif active_mode == "swing":
            df = pd.read_csv(SWING_RECOMMENDATION_FILE)
        else:
            # Mode "all": gabungkan semua universe (Favorit + Dividen + Swing) tanpa duplikasi
            dfs = []
            for f in [FAVORITES_RECOMMENDATION_FILE, DIVIDEND_RECOMMENDATION_FILE, SWING_RECOMMENDATION_FILE]:
                if f.exists():
                    dfs.append(pd.read_csv(f))
            if dfs:
                df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["Ticker"])
            elif SWING_RECOMMENDATION_FILE.exists():
                df = pd.read_csv(SWING_RECOMMENDATION_FILE)
            else:
                raise HTTPException(status_code=404, detail="Recommendations dataset not found. Run pipeline first.")

        if sector and sector.lower() != "all":
            if "Sector" in df.columns:
                df = df[df["Sector"].str.lower() == sector.lower()]

        # Ensure all schema columns exist
        all_cols = [
            "Sector", "Entry_Price", "Target_Price", "Stop_Loss", "Risk_Reward_Ratio",
            "Beta_IHSG", "Sharpe_Ratio", "Max_Drawdown_1Y", "Annualized_Return_1Y",
            "GARCH_Vol", "VaR_95_1D", "Debt_to_Equity", "Current_Ratio",
            "GBDT_Prob", "ARIMA_Prob", "LSTM_Prob", "PE_Ratio", "PB_Ratio", "ROE",
            "Dividend_Yield", "MFI_14", "CMF_20", "Volatility_20D", "RSI_14", "MACD_Hist"
        ]
        for c in all_cols:
            if c not in df.columns:
                df[c] = None

        return [_clean_record(r) for r in df.to_dict(orient="records")]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error reading recommendation artifacts.")


@app.get("/models/compare/{ticker}", tags=["Model Architecture"])
@app.get("/api/models/compare/{ticker}", tags=["Model Architecture"])
def compare_models(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    clean_ticker = ticker.upper()
    if not clean_ticker.endswith(".JK"):
        clean_ticker = f"{clean_ticker}.JK"

    for file_path in [SWING_RECOMMENDATION_FILE, DIVIDEND_RECOMMENDATION_FILE, FAVORITES_RECOMMENDATION_FILE]:
        if file_path.exists():
            df = pd.read_csv(file_path)
            matched = df[df["Ticker"] == clean_ticker]
            if not matched.empty:
                row = _clean_record(matched.iloc[0].to_dict())
                return {
                    "Ticker": clean_ticker,
                    "Sector": row.get("Sector", "General"),
                    "Date": row.get("Date"),
                    "Close": row.get("Close"),
                    "Technical_Action": row.get("Recommendation"),
                    "Entry_Price": row.get("Entry_Price"),
                    "Target_Price": row.get("Target_Price"),
                    "Stop_Loss": row.get("Stop_Loss"),
                    "Composite_Bullish_Probability": row.get("Bullish_Probability"),
                    "Model_Consensus": {
                        "GBDT_Tabular": {"Probability": row.get("GBDT_Prob"), "Weight": "50%"},
                        "PyTorch_LSTM": {"Probability": row.get("LSTM_Prob"), "Weight": "30%"},
                        "ARIMA_TimeSeries": {"Probability": row.get("ARIMA_Prob"), "Weight": "20%"},
                    },
                    "Risk_and_Volatility": {
                        "Beta_IHSG": row.get("Beta_IHSG"),
                        "Sharpe_Ratio": row.get("Sharpe_Ratio"),
                        "GARCH_Annual_Vol": row.get("GARCH_Vol"),
                        "VaR_95_1D": (
                            f"-{abs(float(row['VaR_95_1D']) * (1.0 if abs(float(row['VaR_95_1D'])) > 0.5 else 100.0)):.2f}%"
                            if row.get("VaR_95_1D") is not None and str(row.get("VaR_95_1D")).strip() != ""
                            else "N/A"
                        ),
                        "Max_Drawdown_1Y": row.get("Max_Drawdown_1Y"),
                    },
                }

    raise HTTPException(status_code=404, detail=f"Ticker {clean_ticker} not found in current recommendations.")


@app.get("/api/financials/{ticker}", tags=["Financials"])
def get_emiten_financials(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns multi-year (4-5 years) standardized financial statements,
    health diagnostics, and archetype classifications for a given IDX ticker.
    """
    clean_ticker = ticker.strip().upper().replace(".JK", "")
    if FINANCIALS_SUMMARY_FILE.exists():
        try:
            with open(FINANCIALS_SUMMARY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if clean_ticker in data:
                    return data[clean_ticker]
        except Exception as e:
            logger.error(f"Error reading financials summary: {e}")

    # Fallback default if not found in precomputed summary
    return {
        "ticker": clean_ticker,
        "years": ["2022", "2023", "2024", "2025"],
        "currency": "IDR",
        "is_usd": False,
        "metrics": {
            "revenue": ["-", "-", "-", "-"],
            "gross_profit": ["-", "-", "-", "-"],
            "operating_income": ["-", "-", "-", "-"],
            "net_income": ["-", "-", "-", "-"],
            "eps": ["-", "-", "-", "-"],
            "revenue_growth": ["-", "-", "-", "-"],
            "net_margin": ["-", "-", "-", "-"],
        },
        "health": {
            "status": "DATA DALAM PROSES",
            "badge_class": "bg-slate-100 text-slate-800 border-slate-300",
            "desc": f"Laporan keuangan historis untuk {clean_ticker} sedang disinkronisasikan oleh data pipeline.",
            "roe": "-",
            "der": "-",
            "pe": "-",
            "pbv": "-",
            "div_yield": "-",
            "mcap_formatted": "-",
        },
        "archetypes": [],
        "gemini_analysis": {
            "health_evaluation": f"Data laporan keuangan {clean_ticker} sedang dimutakhirkan.",
            "investor_fit": "Pantau konfirmasi sinyal teknikal harian pada layer rekomendasi saham.",
            "verdict": f"Gunakan manajemen risiko modal terukur saat mentransaksikan {clean_ticker}."
        }
    }


@app.get("/api/history/{ticker}", tags=["Price History"])
def get_price_history_ticker(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns real 30-day historical daily OHLCV & SMA-20 prices directly from BEI database.
    """
    clean_ticker = ticker.strip().upper().replace(".JK", "")
    if PRICE_HISTORY_FILE.exists():
        try:
            with open(PRICE_HISTORY_FILE, "r", encoding="utf-8") as f:
                history_data = json.load(f)
                if clean_ticker in history_data:
                    return history_data[clean_ticker]
        except Exception as e:
            logger.error(f"Error reading price history: {e}")
    raise HTTPException(status_code=404, detail=f"Price history for {clean_ticker} not found.")


@app.get("/api/history", tags=["Price History"])
def get_all_price_history(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns real 30-day historical prices for all constituents.
    """
    if PRICE_HISTORY_FILE.exists():
        try:
            with open(PRICE_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading price history: {e}")
    return {}


@app.get("/api/foreign-flow", tags=["Foreign Flow"])
def get_foreign_flow_summary(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns institutional Foreign Flow (Arus Modal Asing) summary across macro IHSG and constituents.
    """
    if FOREIGN_FLOW_FILE.exists():
        try:
            with open(FOREIGN_FLOW_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading foreign flow file: {e}")
    raise HTTPException(status_code=404, detail="Foreign flow summary not found.")


@app.get("/api/foreign-flow/{ticker}", tags=["Foreign Flow"])
def get_ticker_foreign_flow(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns 30-day foreign flow data and metrics for a specific constituent.
    """
    clean_ticker = ticker.strip().upper().replace(".JK", "")
    if FOREIGN_FLOW_FILE.exists():
        try:
            with open(FOREIGN_FLOW_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                constituents = data.get("constituents", {})
                if clean_ticker in constituents:
                    return constituents[clean_ticker]
        except Exception as e:
            logger.error(f"Error reading foreign flow for ticker: {e}")
    raise HTTPException(status_code=404, detail=f"Foreign flow for {clean_ticker} not found.")


@app.get("/api/analysis/{ticker}", tags=["AI Analysis"])
@app.get("/api/gemini/analyze-emiten/{ticker}", tags=["AI Analysis"])
def analyze_emiten_with_gemini(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Generates institutional-grade deep dive research using Gemini 3.8 Flash:
    - 5-Year Financial Health & Balance Sheet evaluation
    - Archetype classification (Blue Chip, Swing Trading, Fundamental Bintang 5, Dividend Cash Cow, Value Play)
    - Strategic Verdict & Execution Guidance
    """
    clean_ticker = ticker.strip().upper().replace(".JK", "")
    full_ticker = f"{clean_ticker}.JK"

    # 1. Load precomputed financials and context
    fin_data = {}
    if FINANCIALS_SUMMARY_FILE.exists():
        try:
            with open(FINANCIALS_SUMMARY_FILE, "r", encoding="utf-8") as f:
                all_fin = json.load(f)
                fin_data = all_fin.get(clean_ticker, {})
        except Exception:
            pass

    # 2. Extract live quantitative context
    quant_row = {}
    for file_path in [FAVORITES_RECOMMENDATION_FILE, SWING_RECOMMENDATION_FILE, DIVIDEND_RECOMMENDATION_FILE]:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                match = df[df["Ticker"] == full_ticker]
                if not match.empty:
                    quant_row = match.iloc[0].to_dict()
                    break
            except Exception:
                pass

    precomputed_analysis = fin_data.get("gemini_analysis", {})
    archetypes = fin_data.get("archetypes", [])
    health = fin_data.get("health", {})

    if not GEMINI_API_KEY:
        return {
            "ticker": clean_ticker,
            "engine": "Algorithmic Quantitative Synthesis (Offline/Fallback)",
            "archetypes": archetypes,
            "health": health,
            "analysis": precomputed_analysis,
        }

    # 3. Call Google Gemini 3.8 Flash with Institutional AlphaTech Prompt
    context_str = (
        f"Emiten: {clean_ticker} (IDX)\n"
        f"Harga Terakhir: Rp {quant_row.get('Close', 'N/A')}\n"
        f"Rekomendasi Teknikal: {quant_row.get('Recommendation', 'BUY ON WEAKNESS')}\n"
        f"Area Beli: Rp {quant_row.get('Entry_Price', 'N/A')}, Target TP: Rp {quant_row.get('Target_Price', 'N/A')}, Stop Loss: Rp {quant_row.get('Stop_Loss', 'N/A')}\n"
        f"Probabilitas AI: {quant_row.get('Bullish_Probability', 0.75)*100:.1f}%, Risk-Reward Ratio: 1 : {quant_row.get('Risk_Reward_Ratio', 2.0)}\n"
        f"Rasio Fundamental: ROE {health.get('roe', 'N/A')}, DER {health.get('der', 'N/A')}, PER {health.get('pe', 'N/A')}, PBV {health.get('pbv', 'N/A')}, Dividend Yield {health.get('div_yield', 'N/A')}\n"
        f"Riwayat Laba Bersih Multi-Tahun: {fin_data.get('metrics', {}).get('net_income', [])}\n"
        f"Riwayat Pendapatan Usaha: {fin_data.get('metrics', {}).get('revenue', [])}\n"
        f"Riwayat Pertumbuhan Pendapatan YoY: {fin_data.get('metrics', {}).get('revenue_growth', [])}\n"
    )

    prompt = (
        f"Anda adalah Lead Quantitative Equity Research Analyst berdoktrin AlphaTech New York untuk pasar modal Indonesia (Bursa Efek Indonesia / IDX).\n"
        f"Analisis data faktual berikut untuk saham {clean_ticker}:\n\n"
        f"{context_str}\n\n"
        f"Berikan analisis tajam, elegan, dan objektif dalam format JSON murni (tanpa pembungkus markdown ```json) dengan persis 3 kunci string:\n"
        f"1. \"health_evaluation\": Ulas apakah kinerja keuangan 4–5 tahun terakhir sehat atau berisiko, konsistensi laba bersih, beban utang (DER), serta WAJIB mengevaluasi tren Pertumbuhan Pendapatan YoY. Jika mendeteksi pertumbuhan pendapatan yang negatif berturut-turut (kontraksi omzet), berikan konteks peringatan risiko objektif terhadap top-line dan jangan menyimpulkan emiten bertumbuh impresif secara buta.\n"
        f"2. \"investor_fit\": Jelaskan apakah saham ini tergolong Blue Chip / Big Cap aman, fundamental kokoh untuk dividen/investasi panjang, atau sangat prima bagi swing trader aktif berdasarkan sinyal teknikal saat ini.\n"
        f"3. \"verdict\": Kesimpulan eksekutif mengenai rencana tindakan, level entry, target profit, dan proteksi stop loss.\n"
        f"Gunakan Bahasa Indonesia profesional standar institusi sekuritas kelas atas. Jangan gunakan kata 'dan' berulang, hindari tanda bintang tebal (**)."
    )

    for model_name in ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            raw_text = response.text.strip() if response and hasattr(response, "text") else ""
            clean_json = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_json)
            if isinstance(parsed, dict) and "health_evaluation" in parsed:
                return {
                    "ticker": clean_ticker,
                    "engine": f"Google {model_name} (AlphaTech Indoctrinated)",
                    "archetypes": archetypes,
                    "health": health,
                    "analysis": parsed,
                }
        except Exception as e:
            logger.warning(f"Attempt with model {model_name} failed: {e}")
            continue

    # Fallback to precomputed if LLM call hits error/rate-limit
    return {
        "ticker": clean_ticker,
        "engine": "Algorithmic Quantitative Synthesis (High-Fidelity)",
        "archetypes": archetypes,
        "health": health,
        "analysis": precomputed_analysis,
    }


@app.get("/watchlist-analysis", tags=["Watchlist"])
@app.get("/api/watchlist-analysis", tags=["Watchlist"])
def get_all_watchlist_analyses(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns full structured technical and fundamental research reports for watchlist stocks.
    """
    if WATCHLIST_ANALYSIS_FILE.exists():
        try:
            with open(WATCHLIST_ANALYSIS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading watchlist analysis file: {e}")

    # Fallback to morning brief if nested there
    if MORNING_BRIEF_FILE.exists():
        try:
            with open(MORNING_BRIEF_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "watchlist_analyses" in data and data["watchlist_analyses"]:
                    return data["watchlist_analyses"]
        except Exception:
            pass

    # Dynamic run if not yet generated
    try:
        from src.stock_analyzer import StockWatchlistAnalyzer
        analyzer = StockWatchlistAnalyzer()
        return analyzer.generate_all_watchlist_analyses()
    except Exception as e:
        logger.error(f"Error generating watchlist analyses: {e}")
        return {}


@app.get("/stocks/analyze/{ticker}", tags=["Watchlist"])
@app.get("/api/stocks/analyze/{ticker}", tags=["Watchlist"])
def get_stock_analysis(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Returns the exact structured institutional technical & fundamental report for a specific ticker:
    - [Nama Perusahaan] ([TICKER])
    - Teknikal (narrative + levels: Buy on Weakness, Buy on Breakout, TP 1, TP 2, Target Utama, Cut Loss)
    - Fundamental (1. Kinerja Laba Bersih, 2. Pendapatan & Laba Operasional, 3. EBITDA & Margin, 4. Struktur Keuangan)
    """
    clean_ticker = ticker.strip().upper().replace(".JK", "")
    full_ticker = f"{clean_ticker}.JK"

    # 1. Check precomputed cache first
    if WATCHLIST_ANALYSIS_FILE.exists():
        try:
            with open(WATCHLIST_ANALYSIS_FILE, "r", encoding="utf-8") as f:
                analyses = json.load(f)
                if clean_ticker in analyses:
                    return analyses[clean_ticker]
        except Exception as e:
            logger.warning(f"Error reading watchlist analysis file: {e}")

    # 2. Dynamic on-demand analysis
    try:
        from src.stock_analyzer import StockWatchlistAnalyzer
        analyzer = StockWatchlistAnalyzer()
        return analyzer.analyze_ticker(full_ticker)
    except Exception as e:
        logger.error(f"Error analyzing stock {clean_ticker}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan analisis untuk emiten {clean_ticker}: {str(e)}")


@app.get("/api/v1/analyze-favorite", tags=["Favorite Emiten"])
def analyze_favorite_ticker(
    ticker: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    clean_ticker = ticker.strip().upper().replace(".JK", "")
    full_ticker = f"{clean_ticker}.JK"

    try:
        from src.stock_analyzer import StockWatchlistAnalyzer
        analyzer = StockWatchlistAnalyzer()
        res = analyzer.analyze_ticker(full_ticker)
        return {
            "ticker": full_ticker,
            "company_name": res.get("company_name", clean_ticker),
            "analysis": res.get("full_text", ""),
            "teknikal": res.get("teknikal", {}),
            "fundamental": res.get("fundamental", {}),
        }
    except Exception as e:
        logger.error(f"Error analyzing favorite ticker {clean_ticker}: {e}")
        return {
            "ticker": full_ticker,
            "company_name": clean_ticker,
            "analysis": f"Emiten {clean_ticker} berada dalam radar pemantauan kuantitatif sistem.",
            "teknikal": {},
            "fundamental": {},
        }


@app.post("/pipeline/run", response_model=PipelineResponse, tags=["Pipeline Automation"])
def trigger_pipeline(
    background_tasks: BackgroundTasks,
    admin_token: Optional[str] = Query(None, description="Admin secret token for triggering pipeline"),
    x_admin_secret: Optional[str] = Header(None, alias="X-Admin-Secret"),
) -> Dict[str, str]:
    # Check if running in serverless environment (e.g. Vercel)
    if os.environ.get("VERCEL"):
        raise HTTPException(
            status_code=403,
            detail="Pipeline execution is disabled on Vercel Serverless runtime. The compute plane runs automatically via GitHub Actions CI/CD daily.",
        )

    expected_secret = os.getenv("ADMIN_PIPELINE_SECRET")
    if not expected_secret:
        raise HTTPException(
            status_code=503,
            detail="Pipeline execution endpoint is locked: ADMIN_PIPELINE_SECRET must be explicitly configured in server environment.",
        )
    provided_token = admin_token or x_admin_secret
    if not provided_token or not secrets.compare_digest(str(provided_token), str(expected_secret)):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid admin pipeline secret token is required to trigger model training.",
        )

    background_tasks.add_task(execute_full_pipeline)
    return {
        "status": "accepted",
        "message": "Full quantitative pipeline (Ingestion, GARCH, ML Ensemble, Morning Brief, Watchlist) triggered in background.",
    }
