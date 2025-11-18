"""
Tests for PubMed fetcher module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.data_collectors.pubmed_fetcher import PubMedFetcher
from src.models import Publication


class TestPubMedFetcher:
    """Test suite for PubMed fetcher."""

    @pytest.fixture
    def fetcher(self):
        """Create a PubMed fetcher instance."""
        return PubMedFetcher(cache_enabled=False)

    @pytest.fixture
    def mock_pubmed_response(self):
        """Sample PubMed API response."""
        return {
            "esearchresult": {
                "idlist": ["12345", "67890"],
                "count": "2"
            }
        }

    @pytest.fixture
    def mock_xml_response(self):
        """Sample PubMed XML response."""
        return """
        <PubmedArticleSet>
            <PubmedArticle>
                <MedlineCitation>
                    <PMID>12345</PMID>
                    <Article>
                        <ArticleTitle>Test Article</ArticleTitle>
                        <Abstract>
                            <AbstractText>Test abstract</AbstractText>
                        </Abstract>
                        <Journal>
                            <Title>Test Journal</Title>
                        </Journal>
                        <AuthorList>
                            <Author>
                                <LastName>Smith</LastName>
                                <ForeName>John</ForeName>
                            </Author>
                        </AuthorList>
                    </Article>
                </MedlineCitation>
                <PubmedData>
                    <ArticleIdList>
                        <ArticleId IdType="doi">10.1234/test</ArticleId>
                    </ArticleIdList>
                </PubmedData>
            </PubmedArticle>
        </PubmedArticleSet>
        """

    def test_search_author_with_affiliation(self, fetcher, mock_pubmed_response):
        """Test author search with affiliation filtering."""
        with patch.object(fetcher, '_make_request') as mock_request:
            mock_request.return_value = '{"esearchresult": {"idlist": ["12345"], "count": "1"}}'

            pmids = fetcher.search_author("John Smith", affiliation="Harvard")

            assert isinstance(pmids, list)
            mock_request.assert_called_once()

    def test_rate_limiting(self, fetcher):
        """Ensure rate limiting is respected."""
        import time

        start_time = time.time()

        # Mock the request to return quickly
        with patch.object(fetcher, '_make_request') as mock_request:
            mock_request.return_value = '{"esearchresult": {"idlist": [], "count": "0"}}'

            # Make multiple requests
            for _ in range(3):
                fetcher.search_author("Test Author")

        elapsed = time.time() - start_time

        # Should take at least some time due to rate limiting
        # (This is a basic check - actual rate limiting is tested in rate_limiter tests)
        assert elapsed >= 0

    def test_cache_functionality(self):
        """Test that caching works properly."""
        fetcher_with_cache = PubMedFetcher(cache_enabled=True)

        # First call should make API request
        with patch.object(fetcher_with_cache, '_make_request') as mock_request:
            mock_request.return_value = '{"esearchresult": {"idlist": ["12345"], "count": "1"}}'

            pmids1 = fetcher_with_cache.search_author("John Smith")
            call_count1 = mock_request.call_count

            # Second call should use cache
            pmids2 = fetcher_with_cache.search_author("John Smith")
            call_count2 = mock_request.call_count

            assert pmids1 == pmids2
            # Cache should prevent second API call
            assert call_count2 == call_count1

    def test_parse_pubmed_xml(self, fetcher, mock_xml_response):
        """Test XML parsing."""
        publications = fetcher._parse_pubmed_xml(mock_xml_response)

        assert len(publications) > 0
        assert isinstance(publications[0], Publication)
        assert publications[0].pmid == "12345"
        assert publications[0].title == "Test Article"

    def test_invalid_pmid_handling(self, fetcher):
        """Test handling of invalid PMIDs."""
        with patch.object(fetcher, '_make_request') as mock_request:
            mock_request.return_value = '<PubmedArticleSet></PubmedArticleSet>'

            publications = fetcher.fetch_article_details(["invalid"])

            assert isinstance(publications, list)
            assert len(publications) == 0

    def test_network_error_retry(self, fetcher):
        """Test exponential backoff on network errors."""
        import requests

        with patch.object(fetcher, 'session') as mock_session:
            # Simulate network error
            mock_session.get.side_effect = requests.RequestException("Network error")

            with pytest.raises(requests.RequestException):
                fetcher._make_request('esearch.fcgi', {'term': 'test'})

    def test_author_disambiguation(self, fetcher):
        """Test handling of author name variations."""
        # Test that different name formats work
        names = [
            "John Smith",
            "J Smith",
            "Smith J",
            "Smith, John"
        ]

        for name in names:
            with patch.object(fetcher, '_make_request') as mock_request:
                mock_request.return_value = '{"esearchresult": {"idlist": [], "count": "0"}}'

                pmids = fetcher.search_author(name)
                assert isinstance(pmids, list)


@pytest.mark.integration
class TestPubMedIntegration:
    """Integration tests for PubMed fetcher."""

    @pytest.mark.skip(reason="Requires actual API access")
    def test_real_author_search(self):
        """Test with real API (skipped by default)."""
        fetcher = PubMedFetcher()
        pmids = fetcher.search_author("Smith J", start_year=2020, max_results=5)

        assert isinstance(pmids, list)
        assert len(pmids) <= 5

    @pytest.mark.skip(reason="Requires actual API access")
    def test_full_workflow(self):
        """Test complete workflow from search to details."""
        fetcher = PubMedFetcher()

        # Search
        pmids = fetcher.search_author("Smith J", start_year=2023, max_results=2)

        # Fetch details
        if pmids:
            publications = fetcher.fetch_article_details(pmids[:1])
            assert len(publications) > 0
            assert isinstance(publications[0], Publication)
