"""
Integration tests for complete search pipeline

These tests verify that all components work together correctly:
- Query processing → Embedding → Search → Ranking → Results
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np


class TestSearchPipeline:
    """Test complete search flow end-to-end"""

    @pytest.fixture
    def mock_components(self):
        """Create all mocked components for pipeline testing"""
        # Mock embedding generator
        embedder = Mock()
        embedder.generate_embedding = Mock(return_value=np.random.rand(768))

        # Mock vector store with sample data
        vector_store = Mock()
        vector_store.list_indices = Mock(return_value=['faculty_embeddings'])
        vector_store.search = Mock(return_value=[
            {
                'id': 'fac001',
                'score': 0.92,
                'metadata': {
                    'name': 'Dr. Alice Smith',
                    'institution': 'MIT',
                    'department': 'Biology',
                    'research_summary': 'CRISPR gene editing in cancer cells',
                    'h_index': 42,
                    'publication_count': 95,
                    'has_active_funding': True,
                    'total_funding': 1500000,
                    'active_grants': [{'amount': 1500000, 'end_date': '2028-12-31'}],
                    'techniques': ['CRISPR', 'RNA-seq', 'flow cytometry'],
                    'organisms': ['human', 'mouse'],
                    'recent_publications': [
                        {'publication_date': '2024-03-15', 'title': 'CRISPR screens'}
                    ]
                }
            },
            {
                'id': 'fac002',
                'score': 0.88,
                'metadata': {
                    'name': 'Dr. Bob Jones',
                    'institution': 'Harvard',
                    'department': 'Genetics',
                    'research_summary': 'Cancer genomics and precision medicine',
                    'h_index': 35,
                    'publication_count': 78,
                    'has_active_funding': True,
                    'total_funding': 1200000,
                    'active_grants': [{'amount': 1200000, 'end_date': '2027-06-30'}],
                    'techniques': ['RNA-seq', 'ChIP-seq', 'single-cell'],
                    'organisms': ['human'],
                    'recent_publications': [
                        {'publication_date': '2024-01-10', 'title': 'Cancer mutations'}
                    ]
                }
            },
            {
                'id': 'fac003',
                'score': 0.85,
                'metadata': {
                    'name': 'Dr. Carol Lee',
                    'institution': 'Stanford',
                    'department': 'Bioengineering',
                    'research_summary': 'Synthetic biology and metabolic engineering',
                    'h_index': 28,
                    'publication_count': 60,
                    'has_active_funding': False,
                    'total_funding': 400000,
                    'active_grants': [],
                    'techniques': ['CRISPR', 'metabolic engineering'],
                    'organisms': ['yeast', 'bacteria'],
                    'recent_publications': [
                        {'publication_date': '2023-09-01', 'title': 'Synthetic circuits'}
                    ]
                }
            }
        ])

        return {
            'embedder': embedder,
            'vector_store': vector_store
        }

    def test_end_to_end_search_flow(self, mock_components):
        """Test complete search from query to ranked results"""
        from src.search.semantic_search import SemanticSearchEngine

        engine = SemanticSearchEngine(
            embedding_generator=mock_components['embedder'],
            vector_store=mock_components['vector_store']
        )

        # Execute search
        results = engine.search(
            query="CRISPR gene editing in cancer",
            search_mode='faculty',
            limit=10,
            explain=True
        )

        # Verify pipeline components executed
        assert mock_components['embedder'].generate_embedding.called
        assert mock_components['vector_store'].search.called

        # Verify results structure
        assert results is not None
        assert len(results.results) > 0
        assert results.total_count > 0

        # Verify query interpretation
        assert results.query_interpretation is not None
        assert 'detected_intent' in results.query_interpretation
        assert 'extracted_entities' in results.query_interpretation

        # Verify search metadata
        assert results.search_metadata is not None
        assert 'total_time_ms' in results.search_metadata

        # Verify facets
        assert results.facets is not None
        assert 'institutions' in results.facets

    def test_search_with_query_expansion(self, mock_components):
        """Test search uses query expansion"""
        from src.search.semantic_search import SemanticSearchEngine

        engine = SemanticSearchEngine(
            embedding_generator=mock_components['embedder'],
            vector_store=mock_components['vector_store']
        )

        results = engine.search(
            query="p53 mutation in cancer",
            explain=True
        )

        # Query should be expanded with synonyms
        interpretation = results.query_interpretation
        assert 'num_expansions' in interpretation
        # Should have multiple query expansions
        assert interpretation['num_expansions'] > 1

    def test_search_with_implicit_filters(self, mock_components):
        """Test search extracts and applies implicit filters"""
        from src.search.semantic_search import SemanticSearchEngine

        engine = SemanticSearchEngine(
            embedding_generator=mock_components['embedder'],
            vector_store=mock_components['vector_store']
        )

        results = engine.search(
            query="young PI at MIT studying CRISPR",
            explain=True
        )

        # Should extract implicit filters
        interpretation = results.query_interpretation
        assert 'implicit_filters' in interpretation

        filters = interpretation['implicit_filters']
        # Should detect career stage and institution
        assert 'career_stage' in filters or 'institution' in filters

    def test_search_ranking_improves_results(self, mock_components):
        """Test multi-factor ranking reorders results appropriately"""
        from src.search.semantic_search import SemanticSearchEngine

        engine = SemanticSearchEngine(
            embedding_generator=mock_components['embedder'],
            vector_store=mock_components['vector_store']
        )

        results = engine.search(
            query="well-funded cancer research",
            explain=True
        )

        # Results should be ranked by final_score, not just semantic score
        for i in range(len(results.results) - 1):
            current = results.results[i]
            next_result = results.results[i + 1]

            # Verify sorted by final score
            assert current.get('final_score', current.get('score')) >= \
                   next_result.get('final_score', next_result.get('score'))

    def test_search_intent_detection(self, mock_components):
        """Test different query intents are detected"""
        from src.search.semantic_search import SemanticSearchEngine

        engine = SemanticSearchEngine(
            embedding_generator=mock_components['embedder'],
            vector_store=mock_components['vector_store']
        )

        test_queries = [
            ("CRISPR techniques in biology", "technique_based"),
            ("well-funded cancer labs", "funding_based"),
            ("mouse models of disease", "organism_based"),
        ]

        for query, expected_intent in test_queries:
            results = engine.search(query=query, explain=True)

            detected_intent = results.query_interpretation['detected_intent']
            assert detected_intent == expected_intent, \
                f"Query '{query}' should detect '{expected_intent}', got '{detected_intent}'"

    def test_search_faceted_results(self, mock_components):
        """Test faceted search aggregations"""
        from src.search.semantic_search import SemanticSearchEngine

        engine = SemanticSearchEngine(
            embedding_generator=mock_components['embedder'],
            vector_store=mock_components['vector_store']
        )

        results = engine.search(
            query="cancer research",
            explain=True
        )

        facets = results.facets

        # Should have institution facets
        assert 'institutions' in facets
        # From mock data: MIT, Harvard, Stanford
        assert len(facets['institutions']) > 0

        # Should have technique facets
        assert 'techniques' in facets
        assert len(facets['techniques']) > 0


class TestMatchingPipeline:
    """Test complete matching flow end-to-end"""

    @pytest.fixture
    def mock_components_matching(self):
        """Create mocked components for matching pipeline"""
        # Mock embedding generator
        embedder = Mock()
        embedder.generate_embedding = Mock(return_value=np.random.rand(768))

        # Mock vector store
        vector_store = Mock()
        vector_store.list_indices = Mock(return_value=['faculty_embeddings'])
        vector_store.search = Mock(return_value=[
            {
                'id': 'fac001',
                'score': 0.90,
                'metadata': {
                    'name': 'Dr. Perfect Match',
                    'research_summary': 'CRISPR gene editing in cancer',
                    'topics': ['gene editing', 'cancer biology'],
                    'techniques': ['CRISPR', 'RNA-seq'],
                    'organisms': ['human', 'mouse'],
                    'h_index': 40,
                    'publication_count': 90,
                    'grants': [
                        {'active': True, 'amount': 1500000, 'end_date': '2028-12-31'}
                    ],
                    'active_grants': [{'amount': 1500000, 'end_date': '2028-12-31'}],
                    'total_funding': 1500000,
                    'has_active_funding': True,
                    'lab_members': ['s1', 's2', 's3'],
                    'accepting_students': True
                }
            }
        ])

        return {
            'embedder': embedder,
            'vector_store': vector_store
        }

    def test_end_to_end_matching_flow(self, mock_components_matching):
        """Test complete matching from student profile to ranked faculty"""
        from src.search.semantic_search import SemanticSearchEngine
        from src.matching.multi_factor_scorer import MultiFactorMatcher
        from src.matching.similarity_calculator import SimilarityCalculator

        # Setup
        search_engine = SemanticSearchEngine(
            embedding_generator=mock_components_matching['embedder'],
            vector_store=mock_components_matching['vector_store']
        )

        similarity_calc = SimilarityCalculator(
            embedding_generator=mock_components_matching['embedder']
        )
        matcher = MultiFactorMatcher(similarity_calculator=similarity_calc)

        # Student profile
        student = {
            'research_interests': 'CRISPR gene editing in cancer cells',
            'topics': ['gene editing', 'cancer biology'],
            'techniques': ['CRISPR', 'RNA-seq'],
            'organisms': ['human', 'mouse'],
            'career_goals': 'Academic research career'
        }

        # Step 1: Search for candidate faculty
        search_text = student['research_interests']
        search_results = search_engine.search(
            query=search_text,
            search_mode='faculty',
            limit=10
        )

        assert len(search_results.results) > 0

        # Step 2: Calculate match scores
        matches = []
        for result in search_results.results:
            faculty = result['metadata']
            faculty['id'] = result['id']

            match_score = matcher.calculate_match_score(
                student_profile=student,
                faculty_profile=faculty,
                explain=True
            )

            matches.append({
                'faculty_id': result['id'],
                'faculty_name': faculty.get('name'),
                'match_score': match_score
            })

        # Step 3: Verify matches
        assert len(matches) > 0

        first_match = matches[0]['match_score']
        assert 0.0 <= first_match.overall_score <= 1.0
        assert first_match.component_scores is not None
        assert first_match.explanation is not None
        assert len(first_match.strengths) > 0
        assert first_match.recommendation in [
            'highly_recommended', 'recommended', 'consider', 'not_recommended'
        ]

    def test_matching_considers_all_factors(self, mock_components_matching):
        """Test matching uses all six factors"""
        from src.matching.multi_factor_scorer import MultiFactorMatcher
        from src.matching.similarity_calculator import SimilarityCalculator

        similarity_calc = SimilarityCalculator(
            embedding_generator=mock_components_matching['embedder']
        )
        matcher = MultiFactorMatcher(similarity_calculator=similarity_calc)

        student = {
            'research_interests': 'cancer biology',
            'topics': ['cancer'],
            'techniques': ['CRISPR'],
            'organisms': ['mouse']
        }

        faculty = {
            'name': 'Dr. Test',
            'research_summary': 'cancer research',
            'topics': ['cancer'],
            'techniques': ['CRISPR'],
            'organisms': ['mouse'],
            'h_index': 30,
            'publication_count': 70,
            'grants': [{'active': True, 'amount': 1000000}],
            'active_grants': [{'amount': 1000000}],
            'has_active_funding': True,
            'accepting_students': True
        }

        match = matcher.calculate_match_score(student, faculty)

        # All six components should be present
        expected_components = [
            'research_alignment',
            'funding_stability',
            'productivity_match',
            'technique_match',
            'lab_environment',
            'career_development'
        ]

        for component in expected_components:
            assert component in match.component_scores, \
                f"Missing component: {component}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
