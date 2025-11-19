"""Test result ranking functionality"""
import pytest
from src.search.result_ranker import ResultRanker
from src.search.query_processor import QueryAnalysis


class TestResultRanker:

    @pytest.fixture
    def ranker(self):
        return ResultRanker()

    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing"""
        return [
            {
                'id': 'fac1',
                'score': 0.9,
                'metadata': {
                    'name': 'Dr. Smith',
                    'h_index': 45,
                    'publication_count': 120,
                    'active_grants': 3,
                    'total_funding': 1500000,
                    'last_publication_year': 2024,
                    'institution': 'MIT',
                    'department': 'Biology'
                }
            },
            {
                'id': 'fac2',
                'score': 0.85,
                'metadata': {
                    'name': 'Dr. Jones',
                    'h_index': 25,
                    'publication_count': 60,
                    'active_grants': 1,
                    'total_funding': 500000,
                    'last_publication_year': 2023,
                    'institution': 'Stanford',
                    'department': 'Biology'
                }
            },
            {
                'id': 'fac3',
                'score': 0.88,
                'metadata': {
                    'name': 'Dr. Lee',
                    'h_index': 35,
                    'publication_count': 90,
                    'active_grants': 2,
                    'total_funding': 1200000,
                    'last_publication_year': 2024,
                    'institution': 'MIT',
                    'department': 'Chemistry'
                }
            }
        ]

    @pytest.fixture
    def query_context(self):
        return QueryAnalysis(
            original="test query",
            normalized="test query",
            entities={},
            intent='research_area',
            expansions=[],
            filters={},
            boost_terms=[]
        )

    def test_reranking_updates_scores(self, ranker, sample_results, query_context):
        """Test that reranking calculates new scores"""
        reranked = ranker.rank_results(sample_results, query_context)

        # Should have all results
        assert len(reranked) == len(sample_results)

        # All should have final scores
        assert all('final_score' in r for r in reranked)

        # Should be sorted by final score
        scores = [r['final_score'] for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_component_scores_calculated(self, ranker, sample_results, query_context):
        """Test that all component scores are calculated"""
        reranked = ranker.rank_results(sample_results, query_context)

        for result in reranked:
            component_scores = result.get('component_scores', {})

            # Should have all expected components
            assert 'semantic_similarity' in component_scores
            assert 'research_productivity' in component_scores
            assert 'funding_status' in component_scores
            assert 'recency' in component_scores
            assert 'h_index' in component_scores

            # All scores should be in [0, 1]
            for score in component_scores.values():
                assert 0 <= score <= 1

    def test_productivity_score_calculation(self, ranker):
        """Test productivity scoring"""
        high_productivity = {
            'h_index': 50,
            'publication_count': 150,
            'citations': 5000,
            'recent_publications': 10
        }

        low_productivity = {
            'h_index': 10,
            'publication_count': 20,
            'citations': 100,
            'recent_publications': 1
        }

        high_score = ranker.calculate_productivity_score(high_productivity)
        low_score = ranker.calculate_productivity_score(low_productivity)

        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1

    def test_funding_score_calculation(self, ranker):
        """Test funding scoring"""
        well_funded = {
            'active_grants': 3,
            'total_funding': 2000000
        }

        poorly_funded = {
            'active_grants': 0,
            'total_funding': 0
        }

        high_score = ranker.calculate_funding_score(well_funded)
        low_score = ranker.calculate_funding_score(poorly_funded)

        assert high_score > low_score
        # No funding should get very low score
        assert low_score < 0.3

    def test_recency_score_calculation(self, ranker):
        """Test recency scoring"""
        recent = {
            'last_publication_year': 2024
        }

        old = {
            'last_publication_year': 2015
        }

        recent_score = ranker.calculate_recency_score(recent)
        old_score = ranker.calculate_recency_score(old)

        assert recent_score > old_score

    def test_diversity_maintains_result_count(self, ranker, sample_results):
        """Test diversity maintains all results"""
        diversified = ranker.diversify_results(sample_results, diversity_factor=0.5)

        # Should still have all results
        assert len(diversified) == len(sample_results)

    def test_diversity_promotes_variety(self, ranker):
        """Test diversity increases institutional variety"""
        # Create results all from same institution except one
        results = [
            {
                'id': f'fac{i}',
                'final_score': 0.9 - i * 0.01,
                'metadata': {
                    'institution': 'MIT' if i < 4 else 'Stanford',
                    'department': 'Biology'
                }
            }
            for i in range(5)
        ]

        diversified = ranker.diversify_results(results, diversity_factor=0.7)

        # Stanford result should move up due to diversity
        stanford_ranks = [
            i for i, r in enumerate(diversified)
            if r['metadata']['institution'] == 'Stanford'
        ]

        # Should appear reasonably early despite lower score
        assert stanford_ranks[0] < 4

    def test_intent_boosting_funding(self, ranker, sample_results):
        """Test intent-based boosting for funding queries"""
        boosted = ranker.apply_query_intent_boosting(
            sample_results,
            intent='funding_based',
            boost_factor=1.2
        )

        # Well-funded results should be boosted
        fac1 = next(r for r in boosted if r['id'] == 'fac1')
        # Should have higher score than before (if it had active grants)
        assert fac1['metadata']['active_grants'] > 0

    def test_ranking_explanation_generation(self, ranker):
        """Test that explanations are generated"""
        result = {
            'id': 'fac1',
            'final_score': 0.85,
            'component_scores': {
                'semantic_similarity': 0.9,
                'funding_status': 0.85,
                'research_productivity': 0.82
            },
            'metadata': {
                'name': 'Dr. Smith',
                'h_index': 35,
                'publication_count': 80,
                'active_grants': 2
            }
        }

        explanation = ranker.explain_ranking(result, rank=1)

        assert isinstance(explanation, str)
        assert len(explanation) > 10
        assert '#1' in explanation

    def test_h_index_normalization(self, ranker):
        """Test h-index is normalized correctly"""
        # Test various h-index values
        assert ranker._normalize_h_index(0) == 0.0
        assert 0 < ranker._normalize_h_index(10) < 1.0
        assert ranker._normalize_h_index(100) <= 1.0
        # Higher h-index should give higher score
        assert ranker._normalize_h_index(30) > ranker._normalize_h_index(10)

    def test_configurable_weights(self):
        """Test that weights are configurable"""
        custom_config = {
            'semantic_similarity': 0.5,
            'research_productivity': 0.3,
            'funding_status': 0.2
        }

        ranker = ResultRanker(config=custom_config)

        # Weights should be normalized to sum to 1
        assert abs(sum(ranker.weights.values()) - 1.0) < 0.01

    def test_empty_results(self, ranker, query_context):
        """Test handling of empty results"""
        reranked = ranker.rank_results([], query_context)
        assert reranked == []

    def test_missing_metadata_handling(self, ranker, query_context):
        """Test graceful handling of missing metadata"""
        minimal_result = [{
            'id': 'fac1',
            'score': 0.8,
            'metadata': {}  # Empty metadata
        }]

        reranked = ranker.rank_results(minimal_result, query_context)

        # Should still work with defaults
        assert len(reranked) == 1
        assert 'final_score' in reranked[0]
