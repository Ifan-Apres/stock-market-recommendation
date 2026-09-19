"""
Vercel Serverless Entrypoint for AlphaTech Quantitative Platform.
Exposes the FastAPI application instance for Vercel Python runtime.
"""
import sys
from pathlib import Path

# Ensure root directory is on Python path for serverless imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import app

# Handler for Vercel
__all__ = ["app"]
