"""
Unit tests for ResultRanker
"""

import pytest
from datetime import datetime
from src.search.result_ranker import ResultRanker
from src.search.query_processor import QueryAnalysis


class TestResultRanker:
    """Test ResultRanker functionality"""

    @pytest.fixture
    def ranker(self):
        """Create a ResultRanker instance"""
        return ResultRanker()

    @pytest.fixture
    def custom_ranker(self):
        """Create ranker with custom weights"""
        config = {
            'semantic_similarity': 0.40,
            'research_productivity': 0.30,
            'funding_status': 0.20,
            'recency': 0.10,
            'h_index': 0.0,
            'diversity_bonus': 0.0
        }
        return ResultRanker(config=config)

    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing"""
        return [
            {
                'id': 'fac1',
                'score': 0.9,
                'metadata': {
                    'name': 'Dr. Alice Smith',
                    'institution': 'MIT',
                    'department': 'Biology',
                    'h_index': 45,
                    'publication_count': 120,
                    'citation_count': 5000,
                    'active_grants': [
                        {'end_date': '2027-12-31', 'amount': 1000000}
                    ],
                    'grants': [
                        {'active': True, 'amount': 1000000, 'end_date': '2027-12-31'}
                    ],
                    'total_funding': 1500000,
                    'has_active_funding': True,
                    'recent_publications': [
                        {'publication_date': '2024-06-01', 'title': 'Recent work'}
                    ],
                    'techniques': ['CRISPR', 'RNA-seq'],
                    'research_summary': 'Gene editing in cancer cells'
                }
            },
            {
                'id': 'fac2',
                'score': 0.85,
                'metadata': {
                    'name': 'Dr. Bob Jones',
                    'institution': 'Harvard',
                    'department': 'Biology',
                    'h_index': 25,
                    'publication_count': 60,
                    'citation_count': 2000,
                    'active_grants': [],
                    'grants': [],
                    'total_funding': 500000,
                    'has_active_funding': False,
                    'recent_publications': [
                        {'publication_date': '2023-01-01', 'title': 'Older work'}
                    ],
                    'techniques': ['Western blot'],
                    'research_summary': 'Protein biochemistry'
                }
            },
            {
                'id': 'fac3',
                'score': 0.88,
                'metadata': {
                    'name': 'Dr. Carol Lee',
                    'institution': 'Stanford',
                    'department': 'Neuroscience',
                    'h_index': 35,
                    'publication_count': 90,
                    'citation_count': 3500,
                    'active_grants': [
                        {'end_date': '2026-06-30', 'amount': 800000}
                    ],
                    'grants': [
                        {'active': True, 'amount': 800000, 'end_date': '2026-06-30'}
                    ],
                    'total_funding': 1200000,
                    'has_active_funding': True,
                    'recent_publications': [
                        {'publication_date': '2024-03-15', 'title': 'Recent discovery'}
                    ],
                    'techniques': ['CRISPR', 'optogenetics'],
                    'research_summary': 'Neural circuits and behavior'
                }
            }
        ]

    @pytest.fixture
    def query_context(self):
        """Sample query context"""
        return QueryAnalysis(
            original="CRISPR gene editing",
            normalized="crispr gene editing",
            entities={'techniques': ['CRISPR']},
            intent='technique_based',
            expansions=["CRISPR gene editing", "gene editing"],
            filters={},
            boost_terms=['CRISPR']
        )

    def test_initialization(self, ranker):
        """Test ranker initializes with default config"""
        assert ranker is not None
        assert ranker.config is not None
        assert 'semantic_similarity' in ranker.config
        # Weights should sum to 1
        assert abs(sum(ranker.config.values()) - 1.0) < 0.01

    def test_custom_config(self, custom_ranker):
        """Test ranker with custom configuration"""
        assert custom_ranker.config['semantic_similarity'] == 0.40
        assert custom_ranker.config['h_index'] == 0.0

    def test_rank_results_basic(self, ranker, sample_results, query_context):
        """Test basic result ranking"""
        ranked = ranker.rank_results(sample_results, query_context)

        assert len(ranked) == len(sample_results)
        assert all('final_score' in r for r in ranked)
        assert all('component_scores' in r for r in ranked)

        # Should be sorted by final score
        scores = [r['final_score'] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_results_with_explanations(self, ranker, sample_results, query_context):
        """Test ranking includes explanations"""
        ranked = ranker.rank_results(sample_results, query_context)

        assert all('explanation' in r for r in ranked)
        assert all('rank' in r for r in ranked)
        assert ranked[0]['rank'] == 1

    def test_productivity_score_high(self, ranker):
        """Test productivity scoring for highly productive researcher"""
        metadata = {
            'h_index': 50,
            'publication_count': 150,
            'citation_count': 6000,
            'recent_publications': [{'year': 2024}] * 10
        }
        score = ranker.calculate_productivity_score(metadata)

        assert 0.7 <= score <= 1.0
        assert isinstance(score, float)

    def test_productivity_score_low(self, ranker):
        """Test productivity scoring for less productive researcher"""
        metadata = {
            'h_index': 5,
            'publication_count': 10,
            'citation_count': 50,
            'recent_publications': []
        }
        score = ranker.calculate_productivity_score(metadata)

        assert 0.0 <= score <= 0.6
        assert isinstance(score, float)

    def test_productivity_score_missing_data(self, ranker):
        """Test productivity scoring with missing data"""
        metadata = {}
        score = ranker.calculate_productivity_score(metadata)

        # Should return neutral score
        assert 0.3 <= score <= 0.7

    def test_h_index_score(self, ranker):
        """Test h-index scoring"""
        test_cases = [
            (5, 0.4, 0.6),    # Low h-index
            (15, 0.6, 0.8),   # Moderate h-index
            (35, 0.8, 0.95),  # High h-index
            (55, 0.95, 1.0),  # Very high h-index
        ]

        for h_index, min_score, max_score in test_cases:
            score = ranker.calculate_h_index_score({'h_index': h_index})
            assert min_score <= score <= max_score, f"H-index {h_index} gave score {score}"

    def test_funding_score_excellent(self, ranker):
        """Test funding scoring for well-funded researcher"""
        metadata = {
            'active_grants': [
                {'amount': 1000000, 'end_date': '2028-12-31'},
                {'amount': 500000, 'end_date': '2027-06-30'}
            ],
            'grants': [
                {'active': True, 'amount': 1000000, 'end_date': '2028-12-31'},
                {'active': True, 'amount': 500000, 'end_date': '2027-06-30'}
            ],
            'has_active_funding': True,
            'total_funding': 1500000
        }
        score = ranker.calculate_funding_score(metadata)

        assert score >= 0.7  # Should be high

    def test_funding_score_poor(self, ranker):
        """Test funding scoring for unfunded researcher"""
        metadata = {
            'active_grants': [],
            'grants': [],
            'has_active_funding': False,
            'total_funding': 0
        }
        score = ranker.calculate_funding_score(metadata)

        assert score <= 0.4  # Should be low

    def test_funding_score_missing_data(self, ranker):
        """Test funding scoring with missing data"""
        metadata = {}
        score = ranker.calculate_funding_score(metadata)

        assert 0.0 <= score <= 1.0
        assert score < 0.5  # Should penalize missing funding info

    def test_recency_score_recent(self, ranker):
        """Test recency scoring for recent publications"""
        metadata = {
            'recent_publications': [
                {'publication_date': '2024-06-01'},
                {'publication_date': '2024-01-15'}
            ]
        }
        score = ranker.calculate_recency_score(metadata)

        assert score >= 0.8  # Very recent

    def test_recency_score_old(self, ranker):
        """Test recency scoring for old publications"""
        metadata = {
            'recent_publications': [
                {'publication_date': '2015-01-01'},
                {'publication_date': '2014-06-15'}
            ]
        }
        score = ranker.calculate_recency_score(metadata)

        assert score <= 0.3  # Old publications

    def test_recency_score_with_year_only(self, ranker):
        """Test recency scoring with year-only dates"""
        metadata = {
            'recent_publications': [
                {'year': 2024},
                {'year': 2023}
            ]
        }
        score = ranker.calculate_recency_score(metadata)

        assert 0.5 <= score <= 1.0

    def test_date_to_recency_score(self, ranker):
        """Test date conversion to recency score"""
        # Current year
        current_year_score = ranker._date_to_recency_score('2024-01-01')
        assert current_year_score >= 0.9

        # 3 years ago (half-life)
        three_years_ago = ranker._date_to_recency_score('2021-01-01')
        assert 0.4 <= three_years_ago <= 0.6

        # 10 years ago
        old_score = ranker._date_to_recency_score('2014-01-01')
        assert old_score <= 0.2

    def test_diversify_results(self, ranker, sample_results):
        """Test result diversification"""
        # All from same institution initially
        for result in sample_results:
            result['metadata']['institution'] = 'MIT'

        diversified = ranker.diversify_results(sample_results, diversity_factor=0.5)

        assert len(diversified) == len(sample_results)
        # Check that it's a valid reordering
        ids_original = {r['id'] for r in sample_results}
        ids_diversified = {r['id'] for r in diversified}
        assert ids_original == ids_diversified

    def test_diversify_results_preserves_all(self, ranker, sample_results):
        """Test diversification preserves all results"""
        diversified = ranker.diversify_results(sample_results)

        assert len(diversified) == len(sample_results)

        # All original IDs should be present
        original_ids = sorted([r['id'] for r in sample_results])
        diversified_ids = sorted([r['id'] for r in diversified])
        assert original_ids == diversified_ids

    def test_apply_query_intent_boosting_technique_based(self, ranker, sample_results):
        """Test intent boosting for technique-based queries"""
        # Add final_score to results
        for result in sample_results:
            result['final_score'] = result['score']

        boosted = ranker.apply_query_intent_boosting(
            sample_results,
            intent='technique_based',
            boost_factor=1.2
        )

        # Results with many techniques should be boosted
        fac1_boosted = next(r for r in boosted if r['id'] == 'fac1')
        assert 'intent_boost' in fac1_boosted

    def test_apply_query_intent_boosting_funding_based(self, ranker, sample_results):
        """Test intent boosting for funding-based queries"""
        for result in sample_results:
            result['final_score'] = result['score']

        boosted = ranker.apply_query_intent_boosting(
            sample_results,
            intent='funding_based',
            boost_factor=1.3
        )

        # Well-funded researchers should be boosted
        fac1_boosted = next(r for r in boosted if r['id'] == 'fac1')
        assert fac1_boosted['final_score'] > sample_results[0]['score']

    def test_apply_query_intent_boosting_organism_based(self, ranker):
        """Test intent boosting for organism-based queries"""
        results = [
            {
                'id': 'fac1',
                'final_score': 0.8,
                'metadata': {'organisms': ['mouse', 'rat']}
            },
            {
                'id': 'fac2',
                'final_score': 0.85,
                'metadata': {'organisms': []}
            }
        ]

        boosted = ranker.apply_query_intent_boosting(
            results,
            intent='organism_based'
        )

        # Faculty with organisms should be boosted
        fac1 = next(r for r in boosted if r['id'] == 'fac1')
        assert 'intent_boost' in fac1

    def test_explain_ranking(self, ranker):
        """Test ranking explanation generation"""
        result = {
            'component_scores': {
                'semantic_similarity': 0.89,
                'research_productivity': 0.75,
                'funding_status': 0.85,
                'recency': 0.80,
                'h_index': 0.70
            },
            'metadata': {
                'publication_count': 45,
                'h_index': 28
            }
        }

        explanation = ranker.explain_ranking(result, rank=3)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert '#3' in explanation
        assert '0.89' in explanation  # Semantic score mentioned

    def test_explain_ranking_various_scores(self, ranker):
        """Test explanations for various score levels"""
        test_cases = [
            ({'semantic_similarity': 0.9, 'funding_status': 0.8}, "strong", "active funding"),
            ({'semantic_similarity': 0.6, 'funding_status': 0.3}, "moderate", None),
            ({'semantic_similarity': 0.85, 'research_productivity': 0.9}, "strong", "strong publication"),
        ]

        for scores, expected_term, expected_mention in test_cases:
            result = {
                'component_scores': scores,
                'metadata': {'publication_count': 50, 'h_index': 25}
            }
            explanation = ranker.explain_ranking(result, 1)

            assert expected_term.lower() in explanation.lower()
            if expected_mention:
                assert expected_mention.lower() in explanation.lower()

    def test_empty_results(self, ranker, query_context):
        """Test ranking with empty results"""
        ranked = ranker.rank_results([], query_context)
        assert ranked == []

    def test_single_result(self, ranker, sample_results, query_context):
        """Test ranking with single result"""
        ranked = ranker.rank_results([sample_results[0]], query_context)

        assert len(ranked) == 1
        assert 'final_score' in ranked[0]
        assert ranked[0]['rank'] == 1

    def test_ranking_stability(self, ranker, sample_results, query_context):
        """Test ranking is stable (same input produces same output)"""
        ranked1 = ranker.rank_results(sample_results.copy(), query_context)
        ranked2 = ranker.rank_results(sample_results.copy(), query_context)

        # Same order
        assert [r['id'] for r in ranked1] == [r['id'] for r in ranked2]

        # Same scores
        for r1, r2 in zip(ranked1, ranked2):
            assert r1['final_score'] == r2['final_score']

    def test_component_scores_normalized(self, ranker, sample_results, query_context):
        """Test all component scores are in [0, 1] range"""
        ranked = ranker.rank_results(sample_results, query_context)

        for result in ranked:
            for component, score in result['component_scores'].items():
                assert 0.0 <= score <= 1.0, f"{component} score {score} out of range"

    def test_final_scores_normalized(self, ranker, sample_results, query_context):
        """Test final scores are in [0, 1] range"""
        ranked = ranker.rank_results(sample_results, query_context)

        for result in ranked:
            assert 0.0 <= result['final_score'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
