from pathlib import Path
import os
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

APP_NAME = os.getenv("APP_NAME", "JD ANALYSIS")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_VISION_MODEL = os.getenv("DEEPSEEK_VISION_MODEL", "deepseek-flash").strip()
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-flash").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
CONFIDENCE_MINIMUM = int(os.getenv("CONFIDENCE_MINIMUM", "65"))
PRICE_AGREEMENT_ABSOLUTE = float(os.getenv("PRICE_AGREEMENT_ABSOLUTE", "0.50"))
ATR_AGREEMENT_PERCENT = float(os.getenv("ATR_AGREEMENT_PERCENT", "0.20"))
# FORCE_SIGNAL=true  -> return a best-effort direction when one can be inferred; not trade-ready.
# FORCE_SIGNAL=false -> strict mode (returns WAIT when validation checks fail).
FORCE_SIGNAL = os.getenv("FORCE_SIGNAL", "true").strip().lower() in ("1", "true", "yes", "on")
UPLOAD_DIR = ROOT_DIR / "uploads"
DATA_DIR = ROOT_DIR / "data"
STATIC_DIR = ROOT_DIR / "static"
UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

if DATABASE_URL:
    # SQLAlchemy 2 + psycopg 3 URL normalization for Neon URLs.
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgres://"):]
    elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
        DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL[len("postgresql://"):]
else:
    DATABASE_URL = f"sqlite:///{(DATA_DIR / 'jd_analysis.db').as_posix()}"
