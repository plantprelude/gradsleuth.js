"""
Unit tests for SemanticSearchEngine
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from src.search.semantic_search import SemanticSearchEngine, SearchResults


class TestSemanticSearchEngine:
    """Test SemanticSearchEngine functionality"""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedding generator"""
        embedder = Mock()
        embedder.generate_embedding = Mock(return_value=np.random.rand(768))
        return embedder

    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store"""
        store = Mock()
        store.list_indices = Mock(return_value=['faculty_embeddings', 'publications_embeddings'])
        store.search = Mock(return_value=[
            {
                'id': 'fac1',
                'score': 0.92,
                'metadata': {
                    'name': 'Dr. Smith',
                    'institution': 'MIT',
                    'department': 'Biology',
                    'research_summary': 'CRISPR gene editing',
                    'h_index': 35,
                    'publication_count': 80,
                    'has_active_funding': True,
                    'techniques': ['CRISPR', 'RNA-seq']
                }
            },
            {
                'id': 'fac2',
                'score': 0.85,
                'metadata': {
                    'name': 'Dr. Jones',
                    'institution': 'Harvard',
                    'department': 'Biology',
                    'research_summary': 'Cancer genomics',
                    'h_index': 28,
                    'publication_count': 65,
                    'has_active_funding': True,
                    'techniques': ['RNA-seq', 'ChIP-seq']
                }
            }
        ])
        return store

    @pytest.fixture
    def mock_query_processor(self):
        """Create mock query processor"""
        from src.search.query_processor import QueryAnalysis

        processor = Mock()
        processor.process_query = Mock(return_value=QueryAnalysis(
            original="CRISPR gene editing",
            normalized="crispr gene editing",
            entities={'techniques': ['CRISPR']},
            intent='technique_based',
            expansions=["crispr gene editing", "CRISPR-Cas9", "gene editing"],
            filters={},
            boost_terms=['CRISPR']
        ))
        return processor

    @pytest.fixture
    def mock_result_ranker(self):
        """Create mock result ranker"""
        ranker = Mock()
        ranker.rank_results = Mock(side_effect=lambda results, ctx: [
            {**r, 'final_score': r['score'], 'rank': i+1, 'explanation': f"Ranked #{i+1}"}
            for i, r in enumerate(results)
        ])
        return ranker

    @pytest.fixture
    def search_engine(self, mock_embedder, mock_vector_store, mock_query_processor, mock_result_ranker):
        """Create search engine with mocked components"""
        return SemanticSearchEngine(
            embedding_generator=mock_embedder,
            vector_store=mock_vector_store,
            query_processor=mock_query_processor,
            result_ranker=mock_result_ranker
        )

    def test_initialization(self, search_engine):
        """Test search engine initializes correctly"""
        assert search_engine is not None
        assert search_engine.embedding_generator is not None
        assert search_engine.vector_store is not None
        assert search_engine.query_processor is not None
        assert search_engine.result_ranker is not None

    def test_initialization_creates_processors_if_none(self, mock_embedder, mock_vector_store):
        """Test search engine creates default processors if not provided"""
        with patch('src.search.semantic_search.QueryProcessor'):
            with patch('src.search.semantic_search.ResultRanker'):
                engine = SemanticSearchEngine(
                    embedding_generator=mock_embedder,
                    vector_store=mock_vector_store,
                    query_processor=None,
                    result_ranker=None
                )
                assert engine.query_processor is not None
                assert engine.result_ranker is not None

    def test_basic_search(self, search_engine):
        """Test basic search functionality"""
        results = search_engine.search(
            query="CRISPR gene editing",
            search_mode='faculty',
            limit=10
        )

        assert isinstance(results, SearchResults)
        assert len(results.results) > 0
        assert results.total_count > 0

    def test_search_returns_all_fields(self, search_engine):
        """Test search returns complete SearchResults"""
        results = search_engine.search(
            query="test query",
            explain=True
        )

        assert hasattr(results, 'results')
        assert hasattr(results, 'total_count')
        assert hasattr(results, 'facets')
        assert hasattr(results, 'query_interpretation')
        assert hasattr(results, 'search_metadata')

    def test_search_with_filters(self, search_engine):
        """Test search with metadata filters"""
        results = search_engine.search(
            query="cancer research",
            filters={'institution': 'MIT', 'min_h_index': 30},
            limit=10
        )

        assert results is not None
        # Vector store should be called with filters
        search_engine.vector_store.search.assert_called()

    def test_search_with_pagination(self, search_engine):
        """Test search pagination"""
        results = search_engine.search(
            query="biology",
            limit=5,
            offset=10
        )

        assert results is not None
        # Should apply offset/limit
        assert len(results.results) <= 5

    def test_search_with_explanation(self, search_engine):
        """Test search with explanations enabled"""
        results = search_engine.search(
            query="test query",
            explain=True
        )

        assert results.query_interpretation is not None
        assert results.search_metadata is not None
        assert results.facets is not None

    def test_search_without_explanation(self, search_engine):
        """Test search without explanations"""
        results = search_engine.search(
            query="test query",
            explain=False
        )

        # These should be None when explain=False
        assert results.query_interpretation is None
        assert results.search_metadata is None
        assert results.facets is None

    def test_search_calls_query_processor(self, search_engine):
        """Test search calls query processor"""
        search_engine.search(query="CRISPR")

        search_engine.query_processor.process_query.assert_called_once_with("CRISPR")

    def test_search_calls_result_ranker(self, search_engine):
        """Test search calls result ranker"""
        search_engine.search(query="test")

        search_engine.result_ranker.rank_results.assert_called()

    def test_search_generates_embeddings(self, search_engine):
        """Test search generates query embeddings"""
        search_engine.search(query="test query")

        # Should generate embedding for normalized query
        search_engine.embedding_generator.generate_embedding.assert_called()

    def test_search_merges_filters(self, search_engine):
        """Test search merges implicit and explicit filters"""
        # Mock query processor to return filters
        from src.search.query_processor import QueryAnalysis
        search_engine.query_processor.process_query.return_value = QueryAnalysis(
            original="young PI at MIT",
            normalized="principal investigator mit",
            entities={},
            intent='research_area',
            expansions=[],
            filters={'career_stage': 'assistant_professor', 'institution': 'MIT'},
            boost_terms=[]
        )

        results = search_engine.search(
            query="young PI at MIT",
            filters={'min_h_index': 20}  # Explicit filter
        )

        # Both implicit and explicit filters should be used
        assert results is not None

    def test_search_different_modes(self, search_engine):
        """Test search with different search modes"""
        modes = ['faculty', 'publications', 'grants', 'labs']

        for mode in modes:
            results = search_engine.search(
                query="test",
                search_mode=mode
            )
            assert results is not None

    def test_search_timing_metadata(self, search_engine):
        """Test search includes timing metadata"""
        results = search_engine.search(
            query="test",
            explain=True
        )

        metadata = results.search_metadata
        assert metadata is not None
        assert 'total_time_ms' in metadata
        assert 'query_processing_ms' in metadata
        assert 'embedding_generation_ms' in metadata
        assert 'vector_search_ms' in metadata
        assert 'ranking_ms' in metadata

    def test_query_interpretation(self, search_engine):
        """Test query interpretation is included"""
        results = search_engine.search(
            query="CRISPR gene editing",
            explain=True
        )

        interp = results.query_interpretation
        assert interp is not None
        assert 'original_query' in interp
        assert 'normalized_query' in interp
        assert 'detected_intent' in interp
        assert 'extracted_entities' in interp

    def test_facet_generation(self, search_engine):
        """Test facet generation"""
        results = search_engine.search(
            query="test",
            explain=True
        )

        facets = results.facets
        assert facets is not None
        assert 'institutions' in facets
        assert 'departments' in facets
        assert 'techniques' in facets

    def test_facet_counts(self, search_engine):
        """Test facets contain counts"""
        results = search_engine.search(
            query="test",
            explain=True
        )

        # Should have MIT and Harvard from mock results
        institutions = results.facets['institutions']
        assert 'MIT' in institutions or 'Harvard' in institutions

    def test_multi_query_search_union(self, search_engine):
        """Test multi-query search with union aggregation"""
        results = search_engine.multi_query_search(
            queries=["CRISPR", "gene editing"],
            aggregation='union',
            limit=10
        )

        assert isinstance(results, SearchResults)
        assert len(results.results) > 0

    def test_multi_query_search_weighted(self, search_engine):
        """Test multi-query search with weighted aggregation"""
        results = search_engine.multi_query_search(
            queries=["cancer", "genomics"],
            aggregation='weighted',
            limit=10
        )

        assert results is not None
        # Results appearing in multiple queries should be weighted higher

    def test_search_by_example(self, search_engine):
        """Test search by example profile"""
        example_profile = {
            'research_summary': 'CRISPR gene editing in cancer cells',
            'publications': [
                {'title': 'CRISPR screens', 'abstract': 'We used CRISPR...'}
            ]
        }

        results = search_engine.search_by_example(
            example_profile=example_profile,
            search_mode='faculty',
            limit=10
        )

        assert results is not None
        assert len(results.results) > 0

    def test_search_by_example_with_grants(self, search_engine):
        """Test search by example with grant information"""
        example_profile = {
            'research_summary': 'Cancer biology',
            'grants': [
                {'abstract': 'Understanding cancer mechanisms'}
            ]
        }

        results = search_engine.search_by_example(example_profile)

        assert results is not None

    def test_empty_query(self, search_engine):
        """Test handling of empty query"""
        results = search_engine.search(query="")

        assert results is not None
        # Should still return valid SearchResults

    def test_search_error_handling(self, search_engine):
        """Test search handles errors gracefully"""
        # Make vector store raise error
        search_engine.vector_store.search.side_effect = Exception("Vector store error")

        with pytest.raises(Exception):
            search_engine.search(query="test")

    def test_to_dict_conversion(self, search_engine):
        """Test SearchResults to_dict conversion"""
        results = search_engine.search(query="test", explain=True)

        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert 'results' in results_dict
        assert 'total_count' in results_dict
        assert 'facets' in results_dict
        assert 'query_interpretation' in results_dict
        assert 'search_metadata' in results_dict


class TestSearchResults:
    """Test SearchResults dataclass"""

    def test_create_search_results(self):
        """Test creating SearchResults"""
        results = SearchResults(
            results=[{'id': '1', 'score': 0.9}],
            total_count=1,
            facets={'institutions': {'MIT': 1}},
            query_interpretation={'intent': 'research_area'},
            search_metadata={'total_time_ms': 150}
        )

        assert results.results == [{'id': '1', 'score': 0.9}]
        assert results.total_count == 1
        assert results.facets is not None

    def test_search_results_to_dict(self):
        """Test SearchResults to_dict method"""
        results = SearchResults(
            results=[],
            total_count=0,
            facets=None,
            query_interpretation=None,
            search_metadata=None
        )

        results_dict = results.to_dict()

        assert isinstance(results_dict, dict)
        assert results_dict['results'] == []
        assert results_dict['total_count'] == 0


class TestSearchResultAggregation:
    """Test result aggregation methods"""

    @pytest.fixture
    def search_engine_for_aggregation(self, mock_embedder, mock_vector_store):
        """Create engine for testing aggregation"""
        with patch('src.search.semantic_search.QueryProcessor'):
            with patch('src.search.semantic_search.ResultRanker'):
                return SemanticSearchEngine(
                    embedding_generator=mock_embedder,
                    vector_store=mock_vector_store
                )

    def test_union_results(self, search_engine_for_aggregation):
        """Test union of result sets"""
        result_sets = [
            [{'id': '1', 'score': 0.9}, {'id': '2', 'score': 0.8}],
            [{'id': '2', 'score': 0.85}, {'id': '3', 'score': 0.7}]
        ]

        union = search_engine_for_aggregation._union_results(result_sets)

        # Should have 3 unique results
        assert len(union) == 3
        ids = {r['id'] for r in union}
        assert ids == {'1', '2', '3'}

    def test_intersection_results(self, search_engine_for_aggregation):
        """Test intersection of result sets"""
        result_sets = [
            [{'id': '1', 'score': 0.9}, {'id': '2', 'score': 0.8}],
            [{'id': '2', 'score': 0.85}, {'id': '3', 'score': 0.7}],
            [{'id': '2', 'score': 0.82}, {'id': '4', 'score': 0.6}]
        ]

        intersection = search_engine_for_aggregation._intersection_results(result_sets)

        # Only ID '2' appears in all sets
        assert len(intersection) == 1
        assert intersection[0]['id'] == '2'

    def test_weighted_results(self, search_engine_for_aggregation):
        """Test weighted result aggregation"""
        result_sets = [
            [{'id': '1', 'score': 0.9}, {'id': '2', 'score': 0.8}],
            [{'id': '2', 'score': 0.85}, {'id': '3', 'score': 0.7}]
        ]

        weighted = search_engine_for_aggregation._weighted_results(result_sets)

        # ID '2' should have higher score (appears twice)
        id2_result = next(r for r in weighted if r['id'] == '2')
        assert id2_result['appearances'] == 2
        assert id2_result['final_score'] > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
