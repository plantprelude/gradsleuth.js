"""
Configuration settings for the biology research matching platform.
"""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
CACHE_DIR = DATA_DIR / 'cache'

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# API Configuration
class APIConfig:
    """API configuration settings."""

    # NCBI E-utilities
    NCBI_API_KEY = os.getenv('NCBI_API_KEY', '')
    NCBI_BASE_URL = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    NCBI_RATE_LIMIT = 10 if NCBI_API_KEY else 3  # requests per second

    # NIH RePORTER
    NIH_REPORTER_API_KEY = os.getenv('NIH_REPORTER_API_KEY', '')
    NIH_REPORTER_BASE_URL = 'https://api.reporter.nih.gov/v2/'
    NIH_RATE_LIMIT = int(os.getenv('NIH_RATE_LIMIT', '100'))  # requests per hour

    # Request timeouts
    REQUEST_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds


# Database Configuration
class DatabaseConfig:
    """Database configuration settings."""

    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        'postgresql://biomatch:biomatch@localhost:5432/biomatch'
    )

    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    CACHE_TTL_DAYS = int(os.getenv('CACHE_TTL_DAYS', '7'))


# Scraper Configuration
class ScraperConfig:
    """Web scraping configuration settings."""

    USER_AGENT = os.getenv(
        'SCRAPER_USER_AGENT',
        'BiologyMatchBot/1.0 (Academic Research; +https://example.com/bot)'
    )

    DOWNLOAD_DELAY = float(os.getenv('SCRAPER_DELAY', '2'))
    CONCURRENT_REQUESTS = int(os.getenv('CONCURRENT_REQUESTS', '2'))
    AUTOTHROTTLE_ENABLED = True
    RETRY_TIMES = 3
    CACHE_ENABLED = True
    CACHE_EXPIRATION_DAYS = 30

    # University-specific configurations
    UNIVERSITY_CONFIGS: Dict[str, Dict[str, Any]] = {
        "harvard": {
            "base_url": "https://www.mcb.harvard.edu/directory/",
            "selectors": {
                "faculty_list": "div.faculty-member",
                "name": "h2.faculty-name",
                "title": "span.faculty-title",
                "email": "a.email-link",
                "research": "div.research-interests",
                "lab_website": "a.lab-link",
            }
        },
        "mit": {
            "base_url": "https://biology.mit.edu/people/",
            "selectors": {
                "faculty_list": "div.person",
                "name": "h3.person-name",
                "title": "p.person-title",
                "email": "a[href^='mailto:']",
                "research": "div.person-bio",
                "lab_website": "a.website",
            }
        },
        "stanford": {
            "base_url": "https://biology.stanford.edu/people",
            "selectors": {
                "faculty_list": "div.views-row",
                "name": "h3.field-title",
                "title": "div.field-su-person-title",
                "email": "a.mailto",
                "research": "div.field-body",
                "lab_website": "a.field-link",
            }
        },
        "ucsf": {
            "base_url": "https://biochem.ucsf.edu/faculty",
            "selectors": {
                "faculty_list": "div.faculty-item",
                "name": "h3.faculty-name",
                "title": "p.faculty-title",
                "email": "a.email",
                "research": "div.faculty-research",
                "lab_website": "a.website-link",
            }
        },
        "yale": {
            "base_url": "https://medicine.yale.edu/lab/",
            "selectors": {
                "faculty_list": "div.profile",
                "name": "h2.profile-name",
                "title": "span.profile-title",
                "email": "a.profile-email",
                "research": "div.profile-interests",
                "lab_website": "a.profile-website",
            }
        },
    }


# Logging Configuration
class LoggingConfig:
    """Logging configuration settings."""

    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', str(BASE_DIR / 'logs' / 'app.log'))
    JSON_LOGGING = os.getenv('JSON_LOGGING', 'false').lower() == 'true'


# Cache Configuration
class CacheConfig:
    """Caching configuration settings."""

    CACHE_BACKEND = os.getenv('CACHE_BACKEND', 'redis')  # redis or memory
    CACHE_TTL = int(os.getenv('CACHE_TTL', str(60 * 60 * 24 * 7)))  # 7 days
    CACHE_KEY_PREFIX = 'biomatch:'


# PubMed Configuration
class PubMedConfig:
    """PubMed-specific configuration."""

    DEFAULT_MAX_RESULTS = 100
    DEFAULT_START_YEAR = 2019
    BATCH_SIZE = 200  # For batch fetching
    SEARCH_DELAY = 1.0 / APIConfig.NCBI_RATE_LIMIT  # Delay between requests


# Grant Configuration
class GrantConfig:
    """NIH grant-specific configuration."""

    DEFAULT_YEARS_LOOKBACK = 10
    BATCH_SIZE = 50
    ACTIVE_GRANT_THRESHOLD_DAYS = 30  # Consider grant active if not expired > 30 days ago


# Faculty Configuration
class FacultyConfig:
    """Faculty scraping configuration."""

    PHOTO_DOWNLOAD = True
    PHOTO_MAX_SIZE = 1024 * 1024  # 1MB
    CV_PARSING_ENABLED = False  # Requires additional dependencies
    MAX_LAB_MEMBERS = 50


# Performance Configuration
class PerformanceConfig:
    """Performance and optimization settings."""

    # Timeouts
    API_TIMEOUT = 30
    SCRAPER_TIMEOUT = 60

    # Parallelization
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    ASYNC_BATCH_SIZE = 10

    # Memory limits
    MAX_CACHE_SIZE_MB = int(os.getenv('MAX_CACHE_SIZE_MB', '512'))

    # Rate limiting
    GLOBAL_RATE_LIMIT = int(os.getenv('GLOBAL_RATE_LIMIT', '100'))  # requests per minute


# Export configuration classes
config = {
    'api': APIConfig,
    'database': DatabaseConfig,
    'scraper': ScraperConfig,
    'logging': LoggingConfig,
    'cache': CacheConfig,
    'pubmed': PubMedConfig,
    'grant': GrantConfig,
    'faculty': FacultyConfig,
    'performance': PerformanceConfig,
}


def get_config(section: str = None) -> Any:
    """
    Get configuration section or all config.

    Args:
        section: Configuration section name (optional)

    Returns:
        Configuration class or dict
    """
    if section:
        return config.get(section)
    return config


# Development/Testing flags
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
TESTING = os.getenv('TESTING', 'false').lower() == 'true'
