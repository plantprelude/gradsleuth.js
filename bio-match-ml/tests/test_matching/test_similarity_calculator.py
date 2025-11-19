"""Test similarity calculation functionality"""
import pytest
import numpy as np
from unittest.mock import Mock
from src.matching.similarity_calculator import SimilarityCalculator


class TestSimilarityCalculator:

    @pytest.fixture
    def calculator(self):
        # Create calculator with mock embedding generator
        mock_embedder = Mock()
        mock_embedder.generate_embedding.side_effect = lambda x: [0.1] * 768
        return SimilarityCalculator(embedding_generator=mock_embedder)

    @pytest.fixture
    def sample_student_profile(self):
        return {
            'research_interests': 'CRISPR gene editing in neurons',
            'topics': ['gene editing', 'neuroscience', 'CRISPR'],
            'techniques': ['CRISPR', 'RNA-seq', 'electrophysiology'],
            'organisms': ['mouse', 'rat']
        }

    @pytest.fixture
    def sample_faculty_profile(self):
        return {
            'research_summary': 'Developing CRISPR tools for neuroscience',
            'topics': ['gene editing', 'neuroscience', 'optogenetics'],
            'techniques': ['CRISPR', 'two-photon imaging', 'electrophysiology'],
            'organisms': ['mouse', 'zebrafish']
        }

    def test_jaccard_similarity_perfect_overlap(self, calculator):
        """Test Jaccard similarity with perfect overlap"""
        set1 = {'a', 'b', 'c'}
        set2 = {'a', 'b', 'c'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        assert similarity == 1.0

    def test_jaccard_similarity_no_overlap(self, calculator):
        """Test Jaccard similarity with no overlap"""
        set1 = {'a', 'b', 'c'}
        set2 = {'d', 'e', 'f'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        assert similarity == 0.0

    def test_jaccard_similarity_partial_overlap(self, calculator):
        """Test Jaccard similarity with partial overlap"""
        set1 = {'CRISPR', 'RNA-seq', 'PCR'}
        set2 = {'CRISPR', 'Western blot', 'PCR'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        # 2 in common (CRISPR, PCR), 4 total unique
        assert similarity == pytest.approx(0.5, abs=0.01)

    def test_jaccard_similarity_case_insensitive(self, calculator):
        """Test that Jaccard similarity is case-insensitive"""
        set1 = {'CRISPR', 'RNA-seq'}
        set2 = {'crispr', 'rna-seq'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        assert similarity == 1.0

    def test_jaccard_similarity_empty_sets(self, calculator):
        """Test Jaccard with empty sets"""
        assert calculator.calculate_jaccard_similarity(set(), {'a'}) == 0.0
        assert calculator.calculate_jaccard_similarity({'a'}, set()) == 0.0
        assert calculator.calculate_jaccard_similarity(set(), set()) == 0.0

    def test_research_similarity_all_metrics(
        self,
        calculator,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test that all similarity metrics are calculated"""
        similarities = calculator.calculate_research_similarity(
            sample_student_profile,
            sample_faculty_profile
        )

        # Should have all expected metrics
        assert 'embedding_similarity' in similarities
        assert 'topic_overlap' in similarities
        assert 'technique_overlap' in similarities
        assert 'organism_overlap' in similarities
        assert 'keyword_overlap' in similarities

        # All should be in [0, 1]
        for metric, value in similarities.items():
            assert 0 <= value <= 1, f"{metric} score out of range: {value}"

    def test_topic_overlap_calculation(
        self,
        calculator,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test topic overlap is calculated correctly"""
        similarities = calculator.calculate_research_similarity(
            sample_student_profile,
            sample_faculty_profile
        )

        topic_overlap = similarities['topic_overlap']

        # Should have some overlap (gene editing, neuroscience)
        assert topic_overlap > 0.5

    def test_technique_overlap_calculation(
        self,
        calculator,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test technique overlap is calculated correctly"""
        similarities = calculator.calculate_research_similarity(
            sample_student_profile,
            sample_faculty_profile
        )

        technique_overlap = similarities['technique_overlap']

        # Should have overlap in CRISPR and electrophysiology
        assert technique_overlap > 0.3

    def test_organism_overlap_calculation(
        self,
        calculator,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test organism overlap is calculated correctly"""
        similarities = calculator.calculate_research_similarity(
            sample_student_profile,
            sample_faculty_profile
        )

        organism_overlap = similarities['organism_overlap']

        # Should have overlap in mouse
        assert organism_overlap > 0.0

    def test_embedding_similarity_calculation(self, calculator):
        """Test embedding similarity calculation"""
        text1 = "CRISPR gene editing"
        text2 = "CRISPR genome editing"

        similarity = calculator.calculate_embedding_similarity(text1, text2)

        # Should return a valid score
        assert 0 <= similarity <= 1

    def test_embedding_similarity_caching(self, calculator):
        """Test that embeddings are cached"""
        text1 = "test text"
        text2 = "other text"

        # First call
        calculator.calculate_embedding_similarity(text1, text2)

        # Cache should have entries
        assert len(calculator._embedding_cache) > 0

        # Second call should use cache
        cache_size_before = len(calculator._embedding_cache)
        calculator.calculate_embedding_similarity(text1, text2)
        cache_size_after = len(calculator._embedding_cache)

        # Cache size shouldn't grow (reusing cached embeddings)
        assert cache_size_after == cache_size_before

    def test_weighted_overlap_calculation(self, calculator):
        """Test weighted overlap for items with weights"""
        items1 = [('topic1', 0.9), ('topic2', 0.5), ('topic3', 0.3)]
        items2 = [('topic1', 0.8), ('topic3', 0.6), ('topic4', 0.4)]

        overlap = calculator.calculate_weighted_overlap(items1, items2)

        # Should calculate weighted overlap
        assert 0 <= overlap <= 1

    def test_weighted_overlap_empty_lists(self, calculator):
        """Test weighted overlap with empty lists"""
        assert calculator.calculate_weighted_overlap([], [('a', 1.0)]) == 0.0
        assert calculator.calculate_weighted_overlap([('a', 1.0)], []) == 0.0

    def test_cosine_similarity_identical_vectors(self, calculator):
        """Test cosine similarity with identical vectors"""
        vec = np.array([1.0, 2.0, 3.0])

        similarity = calculator._cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self, calculator):
        """Test cosine similarity with orthogonal vectors"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = calculator._cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self, calculator):
        """Test cosine similarity with opposite vectors"""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])

        similarity = calculator._cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vectors(self, calculator):
        """Test handling of zero vectors"""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        similarity = calculator._cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_keyword_extraction(self, calculator):
        """Test keyword extraction from text"""
        text = "CRISPR gene editing is a powerful technique for genome modification"

        keywords = calculator._extract_keywords(text)

        # Should extract meaningful words
        assert 'crispr' in keywords
        assert 'gene' in keywords
        assert 'editing' in keywords
        # Should filter stopwords
        assert 'is' not in keywords
        assert 'a' not in keywords

    def test_batch_similarity_calculation(
        self,
        calculator,
        sample_student_profile
    ):
        """Test batch similarity calculation for multiple faculty"""
        faculty_profiles = [
            {
                'id': 'fac1',
                'research_summary': 'CRISPR in neurons',
                'topics': ['gene editing', 'neuroscience'],
                'techniques': ['CRISPR'],
                'organisms': ['mouse']
            },
            {
                'id': 'fac2',
                'research_summary': 'Cancer immunology',
                'topics': ['cancer', 'immunology'],
                'techniques': ['flow cytometry'],
                'organisms': ['human']
            }
        ]

        results = calculator.batch_similarity(sample_student_profile, faculty_profiles)

        assert len(results) == 2

        # Each result should have faculty_id and similarities
        for faculty_id, similarities in results:
            assert faculty_id in ['fac1', 'fac2']
            assert isinstance(similarities, dict)

    def test_cache_clearing(self, calculator):
        """Test cache can be cleared"""
        # Add something to cache
        calculator.calculate_embedding_similarity("test1", "test2")

        assert len(calculator._embedding_cache) > 0

        # Clear cache
        calculator.clear_cache()

        assert len(calculator._embedding_cache) == 0

    def test_missing_profile_fields(self, calculator):
        """Test handling of profiles with missing fields"""
        minimal_student = {'research_interests': 'test'}
        minimal_faculty = {'research_summary': 'test'}

        similarities = calculator.calculate_research_similarity(
            minimal_student,
            minimal_faculty
        )

        # Should still return valid similarities
        assert isinstance(similarities, dict)
        # Missing fields should get 0.0 scores
        assert similarities['topic_overlap'] == 0.0
        assert similarities['technique_overlap'] == 0.0
