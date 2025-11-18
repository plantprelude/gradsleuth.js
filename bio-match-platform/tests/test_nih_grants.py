"""
Tests for NIH grants collector module.
"""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.data_collectors.nih_grants import NIHGrantCollector
from src.models import Grant


class TestNIHGrantCollector:
    """Test suite for NIH grant collector."""

    @pytest.fixture
    def collector(self):
        """Create a grant collector instance."""
        return NIHGrantCollector(cache_enabled=False)

    @pytest.fixture
    def mock_grant_response(self):
        """Sample NIH API response."""
        return {
            'results': [
                {
                    'project_num': '5R01CA123456-05',
                    'project_title': 'Test Grant',
                    'abstract_text': 'Test abstract',
                    'principal_investigators': [
                        {'full_name': 'John Smith'}
                    ],
                    'organization': {
                        'org_name': 'Test University'
                    },
                    'award_amount': 500000,
                    'direct_cost_amt': 400000,
                    'project_start_date': '2020-01-01',
                    'project_end_date': '2025-01-01',
                    'activity_code': 'R01'
                }
            ]
        }

    def test_search_pi_grants(self, collector, mock_grant_response):
        """Test PI grant search."""
        with patch.object(collector, '_make_request') as mock_request:
            mock_request.return_value = mock_grant_response

            grants = collector.search_pi_grants("John Smith", years=5)

            assert isinstance(grants, list)
            assert len(grants) > 0
            assert isinstance(grants[0], Grant)

    def test_parse_grant_data(self, collector, mock_grant_response):
        """Test grant data parsing."""
        grant_data = mock_grant_response['results'][0]
        grant = collector._parse_grant_data(grant_data)

        assert grant is not None
        assert grant.project_number == '5R01CA123456-05'
        assert grant.pi_name == 'John Smith'
        assert grant.total_cost == 500000

    def test_funding_gap_calculation(self, collector):
        """Test funding gap identification."""
        grants = [
            Grant(
                project_number='R01-1',
                title='Grant 1',
                abstract='',
                pi_name='Test',
                pi_institution='Test',
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 1, 1)
            ),
            Grant(
                project_number='R01-2',
                title='Grant 2',
                abstract='',
                pi_name='Test',
                pi_institution='Test',
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2027, 1, 1)
            )
        ]

        gaps = collector._calculate_funding_gaps(grants)

        assert len(gaps) > 0
        # There should be a gap between 2023 and 2024

    def test_analyze_funding_stability(self, collector):
        """Test funding stability analysis."""
        grants = [
            Grant(
                project_number='R01-1',
                title='Grant 1',
                abstract='',
                pi_name='Test',
                pi_institution='Test',
                total_cost=500000,
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 1, 1)
            )
        ]

        metrics = collector.analyze_funding_stability(grants)

        assert 'stability_score' in metrics
        assert 'avg_grant_duration' in metrics
        assert metrics['total_grants'] == 1


@pytest.mark.integration
class TestNIHGrantsIntegration:
    """Integration tests for NIH grants collector."""

    @pytest.mark.skip(reason="Requires actual API access")
    def test_real_grant_search(self):
        """Test with real API (skipped by default)."""
        collector = NIHGrantCollector()
        grants = collector.search_pi_grants("Smith J", years=5)

        assert isinstance(grants, list)
