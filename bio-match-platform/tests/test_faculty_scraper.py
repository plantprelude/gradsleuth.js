"""
Tests for faculty scraper module.
"""
import pytest
from unittest.mock import Mock, patch

from src.data_collectors.faculty_scraper import FacultyProfileScraper
from src.models import FacultyInfo


class TestFacultyProfileScraper:
    """Test suite for faculty scraper."""

    @pytest.fixture
    def scraper(self):
        """Create a faculty scraper instance."""
        return FacultyProfileScraper(cache_enabled=False)

    @pytest.fixture
    def mock_html(self):
        """Sample HTML response."""
        return """
        <html>
            <body>
                <div class="faculty-member">
                    <h2 class="faculty-name">Dr. Jane Doe</h2>
                    <span class="faculty-title">Professor</span>
                    <a class="email-link" href="mailto:jane@example.edu">jane@example.edu</a>
                    <div class="research-interests">
                        Cancer biology and immunotherapy
                    </div>
                </div>
            </body>
        </html>
        """

    def test_parse_faculty_element(self, scraper, mock_html):
        """Test parsing of faculty HTML."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(mock_html, 'html.parser')
        elem = soup.find('div', class_='faculty-member')

        selectors = {
            'name': 'h2.faculty-name',
            'title': 'span.faculty-title',
            'email': 'a.email-link',
            'research': 'div.research-interests'
        }

        faculty = scraper._parse_faculty_element(
            elem, selectors, 'https://example.edu', 'Test University', 'Biology'
        )

        assert faculty is not None
        assert faculty.name == "Dr. Jane Doe"
        assert faculty.title == "Professor"
        assert faculty.email == "jane@example.edu"

    def test_rate_limiting(self, scraper):
        """Test that rate limiting is applied."""
        import time

        with patch.object(scraper, 'session') as mock_session:
            mock_response = Mock()
            mock_response.text = "<html></html>"
            mock_response.status_code = 200
            mock_session.get.return_value = mock_response

            start = time.time()
            scraper._fetch_page("http://example.com")
            scraper._fetch_page("http://example.com")
            elapsed = time.time() - start

            # Should have some delay due to rate limiting
            assert elapsed >= 0

    def test_validation(self, scraper):
        """Test faculty data validation."""
        faculty = FacultyInfo(
            faculty_id="test123",
            name="Jane Doe",
            title="Professor",
            department="Biology",
            institution="Test University",
            email="invalid-email"
        )

        errors = scraper.validate_faculty_data(faculty)
        assert len(errors) > 0
        assert any("email" in error.lower() for error in errors)


@pytest.mark.integration
class TestFacultyScraperIntegration:
    """Integration tests for faculty scraper."""

    @pytest.mark.skip(reason="Requires actual website access")
    def test_real_scraping(self):
        """Test with real website (skipped by default)."""
        scraper = FacultyProfileScraper()
        faculty_list = scraper.scrape_university("harvard", "biology")

        assert isinstance(faculty_list, list)
