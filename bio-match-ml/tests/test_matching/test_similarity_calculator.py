"""
Unit tests for SimilarityCalculator
"""

import pytest
from unittest.mock import Mock
import numpy as np
from src.matching.similarity_calculator import SimilarityCalculator


class TestSimilarityCalculator:
    """Test SimilarityCalculator functionality"""

    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedding generator"""
        embedder = Mock()
        # Return different but consistent embeddings
        def generate_embedding_side_effect(text, **kwargs):
            # Simple hash-based embedding for testing
            hash_val = hash(text) % 100
            return np.array([hash_val / 100.0] * 768)

        embedder.generate_embedding = Mock(side_effect=generate_embedding_side_effect)
        return embedder

    @pytest.fixture
    def calculator(self, mock_embedder):
        """Create calculator with mock embedder"""
        return SimilarityCalculator(embedding_generator=mock_embedder)

    @pytest.fixture
    def calculator_no_embedder(self):
        """Create calculator without embedder"""
        return SimilarityCalculator(embedding_generator=None)

    def test_initialization(self, calculator):
        """Test calculator initializes correctly"""
        assert calculator is not None
        assert calculator.embedding_generator is not None

    def test_initialization_without_embedder(self):
        """Test calculator can initialize without embedder"""
        calc = SimilarityCalculator(embedding_generator=None)
        assert calc is not None

    def test_jaccard_similarity_identical(self, calculator):
        """Test Jaccard similarity with identical sets"""
        set1 = {'CRISPR', 'RNA-seq', 'PCR'}
        set2 = {'CRISPR', 'RNA-seq', 'PCR'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        assert similarity == 1.0

    def test_jaccard_similarity_disjoint(self, calculator):
        """Test Jaccard similarity with disjoint sets"""
        set1 = {'CRISPR', 'RNA-seq'}
        set2 = {'Western blot', 'ELISA'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        assert similarity == 0.0

    def test_jaccard_similarity_partial(self, calculator):
        """Test Jaccard similarity with partial overlap"""
        set1 = {'CRISPR', 'RNA-seq', 'PCR'}
        set2 = {'CRISPR', 'Western blot', 'PCR'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        # 2 in common, 4 total unique = 0.5
        assert similarity == pytest.approx(0.5, abs=0.01)

    def test_jaccard_similarity_case_insensitive(self, calculator):
        """Test Jaccard similarity is case-insensitive"""
        set1 = {'CRISPR', 'RNA-seq'}
        set2 = {'crispr', 'rna-seq'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        assert similarity == 1.0

    def test_jaccard_similarity_empty_sets(self, calculator):
        """Test Jaccard similarity with empty sets"""
        similarity = calculator.calculate_jaccard_similarity(set(), set())
        assert similarity == 0.0

        similarity = calculator.calculate_jaccard_similarity({'a'}, set())
        assert similarity == 0.0

    def test_embedding_similarity(self, calculator):
        """Test embedding similarity calculation"""
        text1 = "CRISPR gene editing in cancer cells"
        text2 = "CRISPR gene editing in cancer cells"

        similarity = calculator.calculate_embedding_similarity(text1, text2)

        assert 0.0 <= similarity <= 1.0
        # Identical text should have high similarity
        assert similarity >= 0.9

    def test_embedding_similarity_different_texts(self, calculator):
        """Test embedding similarity with different texts"""
        text1 = "CRISPR gene editing"
        text2 = "protein crystallography"

        similarity = calculator.calculate_embedding_similarity(text1, text2)

        assert 0.0 <= similarity <= 1.0

    def test_embedding_similarity_empty_text(self, calculator):
        """Test embedding similarity with empty text"""
        similarity = calculator.calculate_embedding_similarity("", "test")
        assert similarity == 0.0

        similarity = calculator.calculate_embedding_similarity("test", "")
        assert similarity == 0.0

    def test_embedding_similarity_without_embedder(self, calculator_no_embedder):
        """Test embedding similarity without embedder returns 0"""
        similarity = calculator_no_embedder.calculate_embedding_similarity(
            "text1", "text2"
        )
        assert similarity == 0.0

    def test_research_similarity_comprehensive(self, calculator):
        """Test comprehensive research similarity calculation"""
        student = {
            'research_interests': 'CRISPR gene editing in cancer',
            'topics': ['gene editing', 'cancer biology'],
            'techniques': ['CRISPR', 'RNA-seq'],
            'organisms': ['mouse']
        }

        faculty = {
            'research_summary': 'CRISPR applications in cancer therapy',
            'topics': ['gene editing', 'cancer biology', 'immunology'],
            'techniques': ['CRISPR', 'RNA-seq', 'flow cytometry'],
            'organisms': ['mouse', 'human']
        }

        similarities = calculator.calculate_research_similarity(student, faculty)

        assert 'embedding_similarity' in similarities
        assert 'topic_overlap' in similarities
        assert 'technique_overlap' in similarities
        assert 'organism_overlap' in similarities
        assert 'keyword_overlap' in similarities

        # All should be in [0, 1]
        for score in similarities.values():
            assert 0.0 <= score <= 1.0

    def test_research_similarity_high_overlap(self, calculator):
        """Test research similarity with high overlap"""
        student = {
            'research_interests': 'neuroscience',
            'topics': ['neuroscience', 'electrophysiology'],
            'techniques': ['patch clamp', 'optogenetics'],
            'organisms': ['mouse']
        }

        faculty = {
            'research_summary': 'neuroscience research',
            'topics': ['neuroscience', 'electrophysiology'],
            'techniques': ['patch clamp', 'optogenetics'],
            'organisms': ['mouse']
        }

        similarities = calculator.calculate_research_similarity(student, faculty)

        # Should have high overlap scores
        assert similarities['topic_overlap'] == 1.0
        assert similarities['technique_overlap'] == 1.0
        assert similarities['organism_overlap'] == 1.0

    def test_research_similarity_no_overlap(self, calculator):
        """Test research similarity with no overlap"""
        student = {
            'research_interests': 'plant biology',
            'topics': ['photosynthesis'],
            'techniques': ['microscopy'],
            'organisms': ['arabidopsis']
        }

        faculty = {
            'research_summary': 'computational biology',
            'topics': ['bioinformatics'],
            'techniques': ['machine learning'],
            'organisms': ['human']
        }

        similarities = calculator.calculate_research_similarity(student, faculty)

        # Should have low overlap scores
        assert similarities['topic_overlap'] == 0.0
        assert similarities['technique_overlap'] == 0.0
        assert similarities['organism_overlap'] == 0.0

    def test_research_similarity_missing_fields(self, calculator):
        """Test research similarity with missing fields"""
        student = {'research_interests': 'biology'}
        faculty = {'research_summary': 'research'}

        similarities = calculator.calculate_research_similarity(student, faculty)

        # Should handle gracefully
        assert all(0.0 <= score <= 1.0 for score in similarities.values())

    def test_weighted_overlap(self, calculator):
        """Test weighted overlap calculation"""
        items1 = [('CRISPR', 0.9), ('RNA-seq', 0.7), ('PCR', 0.5)]
        items2 = [('CRISPR', 0.8), ('Western blot', 0.6), ('PCR', 0.4)]

        overlap = calculator.calculate_weighted_overlap(items1, items2)

        assert 0.0 <= overlap <= 1.0
        # Should be > 0 because of CRISPR and PCR overlap

    def test_weighted_overlap_no_overlap(self, calculator):
        """Test weighted overlap with no overlap"""
        items1 = [('A', 0.9), ('B', 0.8)]
        items2 = [('C', 0.7), ('D', 0.6)]

        overlap = calculator.calculate_weighted_overlap(items1, items2)

        assert overlap == 0.0

    def test_weighted_overlap_empty(self, calculator):
        """Test weighted overlap with empty lists"""
        overlap = calculator.calculate_weighted_overlap([], [('A', 1.0)])
        assert overlap == 0.0

        overlap = calculator.calculate_weighted_overlap([('A', 1.0)], [])
        assert overlap == 0.0

    def test_trajectory_alignment(self, calculator):
        """Test trajectory alignment calculation"""
        student_goals = {
            'career_goals': 'academic research career',
            'research_direction': 'cancer immunotherapy',
            'desired_skills': ['CRISPR', 'immunology', 'mouse models']
        }

        faculty_trajectory = {
            'training_focus': 'preparing students for academic careers',
            'future_directions': 'expanding into cancer immunotherapy',
            'training_opportunities': ['CRISPR', 'immunology', 'animal models']
        }

        alignment = calculator.calculate_trajectory_alignment(
            student_goals,
            faculty_trajectory
        )

        assert 0.0 <= alignment <= 1.0
        # Should have good alignment
        assert alignment > 0.5

    def test_trajectory_alignment_poor_match(self, calculator):
        """Test trajectory alignment with poor match"""
        student_goals = {
            'career_goals': 'industry biotech',
            'research_direction': 'drug discovery',
            'desired_skills': ['high-throughput screening']
        }

        faculty_trajectory = {
            'training_focus': 'basic research',
            'future_directions': 'fundamental mechanisms',
            'training_opportunities': ['theoretical modeling']
        }

        alignment = calculator.calculate_trajectory_alignment(
            student_goals,
            faculty_trajectory
        )

        assert 0.0 <= alignment <= 1.0

    def test_trajectory_alignment_missing_data(self, calculator):
        """Test trajectory alignment with missing data"""
        alignment = calculator.calculate_trajectory_alignment({}, {})

        # Should return neutral score
        assert alignment == 0.5

    def test_cosine_similarity(self, calculator):
        """Test cosine similarity calculation"""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        similarity = calculator.calculate_cosine_similarity(vec1, vec2)

        # Identical vectors
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_orthogonal(self, calculator):
        """Test cosine similarity with orthogonal vectors"""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = calculator.calculate_cosine_similarity(vec1, vec2)

        # Orthogonal vectors should have similarity ~0.5 (normalized to [0,1])
        assert 0.4 <= similarity <= 0.6

    def test_cosine_similarity_zero_vector(self, calculator):
        """Test cosine similarity with zero vector"""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        similarity = calculator.calculate_cosine_similarity(vec1, vec2)

        assert similarity == 0.0

    def test_dice_coefficient(self, calculator):
        """Test Dice coefficient calculation"""
        set1 = {'A', 'B', 'C'}
        set2 = {'B', 'C', 'D'}

        dice = calculator.calculate_dice_coefficient(set1, set2)

        # Dice = 2 * 2 / (3 + 3) = 0.667
        assert dice == pytest.approx(0.667, abs=0.01)

    def test_dice_coefficient_identical(self, calculator):
        """Test Dice coefficient with identical sets"""
        set1 = {'A', 'B'}
        dice = calculator.calculate_dice_coefficient(set1, set1)

        assert dice == 1.0

    def test_dice_coefficient_disjoint(self, calculator):
        """Test Dice coefficient with disjoint sets"""
        set1 = {'A', 'B'}
        set2 = {'C', 'D'}

        dice = calculator.calculate_dice_coefficient(set1, set2)

        assert dice == 0.0

    def test_overlap_coefficient(self, calculator):
        """Test overlap coefficient calculation"""
        set1 = {'A', 'B'}
        set2 = {'A', 'B', 'C', 'D'}

        overlap = calculator.calculate_overlap_coefficient(set1, set2)

        # All of set1 is in set2 = 2/2 = 1.0
        assert overlap == 1.0

    def test_overlap_coefficient_partial(self, calculator):
        """Test overlap coefficient with partial overlap"""
        set1 = {'A', 'B', 'C'}
        set2 = {'B', 'C', 'D', 'E'}

        overlap = calculator.calculate_overlap_coefficient(set1, set2)

        # 2 common, min size = 3, so 2/3
        assert overlap == pytest.approx(0.667, abs=0.01)

    def test_tfidf_similarity(self, calculator):
        """Test TF-IDF similarity calculation"""
        text1 = "CRISPR gene editing cancer research"
        text2 = "CRISPR gene editing biology research"

        similarity = calculator.calculate_tfidf_similarity(text1, text2)

        assert 0.0 <= similarity <= 1.0
        # Should have high similarity
        assert similarity > 0.5

    def test_tfidf_similarity_different(self, calculator):
        """Test TF-IDF similarity with very different texts"""
        text1 = "CRISPR gene editing"
        text2 = "protein crystallography methods"

        similarity = calculator.calculate_tfidf_similarity(text1, text2)

        assert 0.0 <= similarity <= 1.0

    def test_tfidf_similarity_empty(self, calculator):
        """Test TF-IDF similarity with empty text"""
        similarity = calculator.calculate_tfidf_similarity("", "test")
        assert similarity == 0.0

    def test_aggregate_similarities(self, calculator):
        """Test aggregating multiple similarities"""
        similarities = {
            'embedding_similarity': 0.8,
            'topic_overlap': 0.9,
            'technique_overlap': 0.7
        }

        aggregated = calculator.aggregate_similarities(similarities)

        # Should be average with equal weights
        assert aggregated == pytest.approx(0.8, abs=0.1)

    def test_aggregate_similarities_with_weights(self, calculator):
        """Test aggregating similarities with custom weights"""
        similarities = {
            'embedding_similarity': 0.8,
            'topic_overlap': 0.6
        }

        weights = {
            'embedding_similarity': 0.7,
            'topic_overlap': 0.3
        }

        aggregated = calculator.aggregate_similarities(similarities, weights)

        # 0.8 * 0.7 + 0.6 * 0.3 = 0.74
        assert aggregated == pytest.approx(0.74, abs=0.01)

    def test_aggregate_similarities_empty(self, calculator):
        """Test aggregating empty similarities"""
        aggregated = calculator.aggregate_similarities({})
        assert aggregated == 0.0

    def test_publication_similarity(self, calculator):
        """Test publication similarity calculation"""
        student_interests = ['CRISPR', 'gene editing', 'cancer']

        faculty_pubs = [
            {
                'title': 'CRISPR screens in cancer',
                'abstract': 'We used CRISPR to identify cancer vulnerabilities'
            },
            {
                'title': 'Gene editing applications',
                'abstract': 'Novel gene editing approaches'
            }
        ]

        similarity = calculator.calculate_publication_similarity(
            student_interests,
            faculty_pubs
        )

        assert 0.0 <= similarity <= 1.0
        # Should have good overlap
        assert similarity > 0.3

    def test_publication_similarity_no_overlap(self, calculator):
        """Test publication similarity with no overlap"""
        student_interests = ['plant biology', 'photosynthesis']

        faculty_pubs = [
            {
                'title': 'Computational neuroscience',
                'abstract': 'Neural network modeling'
            }
        ]

        similarity = calculator.calculate_publication_similarity(
            student_interests,
            faculty_pubs
        )

        assert 0.0 <= similarity <= 1.0

    def test_publication_similarity_empty(self, calculator):
        """Test publication similarity with empty inputs"""
        similarity = calculator.calculate_publication_similarity([], [])
        assert similarity == 0.0

        similarity = calculator.calculate_publication_similarity(['test'], [])
        assert similarity == 0.0

    def test_keyword_extraction(self, calculator):
        """Test keyword extraction from text"""
        text = "CRISPR gene editing is a powerful technique for studying cancer biology"

        keywords = calculator._extract_keywords(text)

        assert isinstance(keywords, set)
        assert 'crispr' in keywords
        assert 'gene' in keywords
        assert 'editing' in keywords
        assert 'powerful' in keywords
        assert 'technique' in keywords
        assert 'studying' in keywords
        assert 'cancer' in keywords
        assert 'biology' in keywords

        # Stopwords should be removed
        assert 'is' not in keywords
        assert 'a' not in keywords
        assert 'for' not in keywords

    def test_keyword_extraction_min_length(self, calculator):
        """Test keyword extraction respects min length"""
        text = "a ab abc abcd"

        keywords = calculator._extract_keywords(text, min_length=3)

        # Only 'abc' and 'abcd' should be included
        assert 'a' not in keywords
        assert 'ab' not in keywords
        assert 'abc' in keywords
        assert 'abcd' in keywords


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
