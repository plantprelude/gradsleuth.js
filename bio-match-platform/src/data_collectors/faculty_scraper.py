"""
Faculty profile scraper for collecting faculty information from university websites.
"""
import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path
import time

from ..config import ScraperConfig, CACHE_DIR, RAW_DATA_DIR
from ..models import FacultyInfo
from ..utils import (
    RateLimiter,
    retry_with_backoff,
    setup_logger,
    validate_email,
    validate_url,
    sanitize_text,
)

logger = setup_logger(__name__)


class FacultyProfileScraper:
    """
    Web scraper for faculty profiles from university websites.

    Supports multiple universities with configurable selectors.
    """

    def __init__(self, cache_enabled: bool = True, respect_robots: bool = True):
        """
        Initialize faculty scraper.

        Args:
            cache_enabled: Whether to use caching
            respect_robots: Whether to respect robots.txt
        """
        self.cache_enabled = cache_enabled
        self.respect_robots = respect_robots
        self.cache_dir = CACHE_DIR / 'faculty'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiter
        self.rate_limiter = RateLimiter(1 / ScraperConfig.DOWNLOAD_DELAY)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': ScraperConfig.USER_AGENT
        })

        # University configurations
        self.university_configs = ScraperConfig.UNIVERSITY_CONFIGS

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.json"

    def _get_cache(self, url: str) -> Optional[Dict]:
        """
        Get cached data if available and valid.

        Args:
            url: URL to check cache for

        Returns:
            Cached data or None
        """
        if not self.cache_enabled:
            return None

        cache_file = self._get_cache_path(url)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Check if cache is expired (30 days)
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if (datetime.now() - cached_time).days > ScraperConfig.CACHE_EXPIRATION_DAYS:
                return None

            return cached['data']
        except Exception as e:
            logger.warning(f"Cache read error for {url}: {e}")
            return None

    def _set_cache(self, url: str, data: Dict):
        """
        Cache data for a URL.

        Args:
            url: URL
            data: Data to cache
        """
        if not self.cache_enabled:
            return

        try:
            cache_file = self._get_cache_path(url)
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'url': url,
                    'data': data
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error for {url}: {e}")

    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch a web page with rate limiting and retries.

        Args:
            url: URL to fetch

        Returns:
            Page HTML or None
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        logger.debug(f"Fetching: {url}")

        try:
            response = self.session.get(
                url,
                timeout=30,
                allow_redirects=True
            )
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def scrape_university(
        self,
        university_name: str,
        department: str = "biology"
    ) -> List[FacultyInfo]:
        """
        Scrape faculty profiles from a university.

        Args:
            university_name: Name of university (must match config key)
            department: Department name

        Returns:
            List[FacultyInfo]: List of faculty profiles
        """
        if university_name.lower() not in self.university_configs:
            logger.error(f"Unknown university: {university_name}")
            return []

        config = self.university_configs[university_name.lower()]
        logger.info(f"Scraping faculty from {university_name}")

        # Fetch main directory page
        base_url = config['base_url']
        html = self._fetch_page(base_url)

        if not html:
            logger.error(f"Failed to fetch page: {base_url}")
            return []

        # Parse faculty list
        soup = BeautifulSoup(html, 'html.parser')
        selectors = config['selectors']

        faculty_list = []
        faculty_elements = soup.select(selectors.get('faculty_list', 'div.faculty'))

        logger.info(f"Found {len(faculty_elements)} faculty members")

        for i, elem in enumerate(faculty_elements):
            try:
                faculty = self._parse_faculty_element(
                    elem,
                    selectors,
                    base_url,
                    university_name,
                    department
                )
                if faculty:
                    faculty_list.append(faculty)
            except Exception as e:
                logger.warning(f"Error parsing faculty element {i}: {e}")
                continue

        logger.info(f"Successfully parsed {len(faculty_list)} faculty profiles")
        return faculty_list

    def _parse_faculty_element(
        self,
        elem: BeautifulSoup,
        selectors: Dict[str, str],
        base_url: str,
        institution: str,
        department: str
    ) -> Optional[FacultyInfo]:
        """
        Parse a single faculty element.

        Args:
            elem: BeautifulSoup element
            selectors: CSS selectors for data extraction
            base_url: Base URL for resolving relative links
            institution: Institution name
            department: Department name

        Returns:
            FacultyInfo object or None
        """
        try:
            # Extract name
            name_elem = elem.select_one(selectors.get('name', 'h3'))
            name = sanitize_text(name_elem.get_text()) if name_elem else ""

            if not name:
                return None

            # Generate faculty ID
            faculty_id = hashlib.md5(
                f"{institution}_{name}".encode()
            ).hexdigest()[:16]

            # Extract title
            title_elem = elem.select_one(selectors.get('title', 'p.title'))
            title = sanitize_text(title_elem.get_text()) if title_elem else ""

            # Extract email
            email = None
            email_elem = elem.select_one(selectors.get('email', 'a[href^="mailto:"]'))
            if email_elem:
                email_text = email_elem.get('href', '').replace('mailto:', '')
                if validate_email(email_text):
                    email = email_text

            # Extract research interests
            research_elem = elem.select_one(selectors.get('research', 'div.research'))
            research_interests = sanitize_text(
                research_elem.get_text()
            ) if research_elem else ""

            # Extract lab website
            lab_website = None
            lab_link_elem = elem.select_one(selectors.get('lab_website', 'a.website'))
            if lab_link_elem:
                lab_url = lab_link_elem.get('href', '')
                if lab_url:
                    lab_website = urljoin(base_url, lab_url)
                    if not validate_url(lab_website):
                        lab_website = None

            # Extract profile URL
            profile_url = None
            profile_link = elem.select_one('a')
            if profile_link:
                profile_href = profile_link.get('href', '')
                if profile_href:
                    profile_url = urljoin(base_url, profile_href)

            # Extract phone
            phone = None
            phone_elem = elem.select_one(selectors.get('phone', 'span.phone'))
            if phone_elem:
                phone = sanitize_text(phone_elem.get_text())

            # Extract office location
            office_location = None
            office_elem = elem.select_one(selectors.get('office', 'span.office'))
            if office_elem:
                office_location = sanitize_text(office_elem.get_text())

            # Extract photo URL
            photo_url = None
            photo_elem = elem.select_one('img')
            if photo_elem:
                photo_src = photo_elem.get('src', '')
                if photo_src:
                    photo_url = urljoin(base_url, photo_src)

            return FacultyInfo(
                faculty_id=faculty_id,
                name=name,
                title=title,
                department=department,
                institution=institution,
                email=email,
                research_interests=research_interests,
                lab_website=lab_website,
                phone=phone,
                office_location=office_location,
                photo_url=photo_url,
                profile_url=profile_url,
            )

        except Exception as e:
            logger.warning(f"Error parsing faculty element: {e}")
            return None

    def scrape_detailed_profile(self, profile_url: str) -> Dict[str, Any]:
        """
        Scrape detailed information from a faculty profile page.

        Args:
            profile_url: URL of profile page

        Returns:
            Dict with detailed profile information
        """
        # Check cache
        cached = self._get_cache(profile_url)
        if cached:
            logger.debug(f"Cache hit for profile: {profile_url}")
            return cached

        logger.debug(f"Scraping detailed profile: {profile_url}")

        html = self._fetch_page(profile_url)
        if not html:
            return {}

        soup = BeautifulSoup(html, 'html.parser')

        # Extract additional information
        details = {
            'education_history': self._extract_education(soup),
            'lab_members': self._extract_lab_members(soup),
            'recent_news': self._extract_news(soup),
            'personal_website': self._extract_personal_website(soup),
        }

        # Cache result
        self._set_cache(profile_url, details)

        return details

    def _extract_education(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract education history from profile page."""
        education = []

        # Common patterns for education sections
        education_section = soup.find(['div', 'section'], class_=lambda x: x and 'education' in x.lower())

        if education_section:
            items = education_section.find_all(['li', 'p'])
            for item in items[:5]:  # Limit to 5 entries
                text = sanitize_text(item.get_text())
                if text:
                    education.append({'degree': text})

        return education

    def _extract_lab_members(self, soup: BeautifulSoup) -> List[str]:
        """Extract lab member names from profile page."""
        members = []

        # Common patterns for lab members
        members_section = soup.find(['div', 'section'], class_=lambda x: x and 'member' in x.lower())

        if members_section:
            items = members_section.find_all(['li', 'h4', 'h5'])
            for item in items[:20]:  # Limit to 20 members
                name = sanitize_text(item.get_text())
                if name and len(name) > 3:
                    members.append(name)

        return members

    def _extract_news(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract recent news items from profile page."""
        news_items = []

        # Common patterns for news sections
        news_section = soup.find(['div', 'section'], class_=lambda x: x and 'news' in x.lower())

        if news_section:
            items = news_section.find_all(['article', 'div', 'li'])
            for item in items[:5]:  # Limit to 5 news items
                text = sanitize_text(item.get_text())
                if text:
                    news_items.append({
                        'text': text[:500],  # Limit text length
                        'date': ''
                    })

        return news_items

    def _extract_personal_website(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract personal website URL from profile page."""
        # Look for links with common patterns
        for link in soup.find_all('a'):
            text = link.get_text().lower()
            href = link.get('href', '')

            if any(keyword in text for keyword in ['personal', 'homepage', 'website']):
                if href and validate_url(href):
                    return href

        return None

    def scrape_multiple_universities(
        self,
        universities: List[str],
        department: str = "biology"
    ) -> Dict[str, List[FacultyInfo]]:
        """
        Scrape faculty from multiple universities.

        Args:
            universities: List of university names
            department: Department name

        Returns:
            Dict mapping university name to list of faculty
        """
        results = {}

        for university in universities:
            logger.info(f"Processing university: {university}")
            faculty_list = self.scrape_university(university, department)
            results[university] = faculty_list

            # Be polite - wait between universities
            time.sleep(5)

        return results

    def export_to_json(self, faculty_list: List[FacultyInfo], filepath: str):
        """
        Export faculty list to JSON file.

        Args:
            faculty_list: List of FacultyInfo objects
            filepath: Output file path
        """
        data = [faculty.to_dict() for faculty in faculty_list]

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(faculty_list)} faculty profiles to {filepath}")

    def export_to_csv(self, faculty_list: List[FacultyInfo], filepath: str):
        """
        Export faculty list to CSV file.

        Args:
            faculty_list: List of FacultyInfo objects
            filepath: Output file path
        """
        import csv

        if not faculty_list:
            return

        fieldnames = [
            'faculty_id', 'name', 'title', 'department', 'institution',
            'email', 'phone', 'lab_website', 'office_location'
        ]

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for faculty in faculty_list:
                writer.writerow({
                    'faculty_id': faculty.faculty_id,
                    'name': faculty.name,
                    'title': faculty.title,
                    'department': faculty.department,
                    'institution': faculty.institution,
                    'email': faculty.email or '',
                    'phone': faculty.phone or '',
                    'lab_website': faculty.lab_website or '',
                    'office_location': faculty.office_location or '',
                })

        logger.info(f"Exported {len(faculty_list)} faculty profiles to {filepath}")

    def validate_faculty_data(self, faculty: FacultyInfo) -> List[str]:
        """
        Validate faculty data.

        Args:
            faculty: FacultyInfo object

        Returns:
            List of validation errors
        """
        errors = []

        if not faculty.name:
            errors.append("Missing faculty name")

        if not faculty.institution:
            errors.append("Missing institution")

        if faculty.email and not validate_email(faculty.email):
            errors.append(f"Invalid email: {faculty.email}")

        if faculty.lab_website and not validate_url(faculty.lab_website):
            errors.append(f"Invalid lab website URL: {faculty.lab_website}")

        return errors
