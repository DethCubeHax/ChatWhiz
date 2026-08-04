import os
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

PORTFOLIO_DATA_BASE_URL = os.getenv("PORTFOLIO_DATA_BASE_URL", "https://www.nafisui.com").rstrip("/")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

HKT = ZoneInfo("Asia/Hong_Kong")
BIRTHDATE = date(2002, 8, 17)
GRADUATION_DATE = date(2025, 6, 30)

MAX_HISTORY = 5
RATE_LIMIT = 50
RATE_WINDOW = timedelta(hours=24)
DATA_CACHE_TTL = timedelta(hours=2)

SOURCE_KEYS = ("projects", "work", "research", "resume")

SOURCE_CATALOG = {
    "projects": "Portfolio projects with descriptions, tools, screenshots, and links.",
    "work": "Professional work history with companies, roles, dates, and achievement bullets.",
    "research": "Research publications, conferences, and academic project descriptions.",
    "resume": "Full resume text covering education, experience, skills, and publications.",
}

SOURCE_STATUS_LABELS = {
    "projects": "Analysing projects...",
    "work": "Analysing work experience...",
    "research": "Analysing research...",
    "resume": "Analysing resume...",
}

WEB_SEARCH_STATUS = "Searching online..."
WEB_SEARCH_MAX_RESULTS = 5
