"""
Vercel Serverless Entrypoint for Stock Market Recommendation Platform.
Exposes the FastAPI application instance for Vercel Python runtime.
"""
from pathlib import Path
import sys

# Ensure root directory is on Python path for serverless imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# pyrefly: ignore [missing-import]
from api.main import app  # type: ignore # pyrefly: ignore [missing-import]

# Handler for Vercel
__all__ = ["app"]
