"""Test multi-factor matching functionality"""
import pytest
from unittest.mock import Mock
from src.matching.multi_factor_scorer import MultiFactorMatcher, MatchScore
from src.matching.similarity_calculator import SimilarityCalculator


class TestMultiFactorMatcher:

    @pytest.fixture
    def matcher(self):
        # Create matcher with mock similarity calculator
        mock_calculator = Mock(spec=SimilarityCalculator)
        mock_calculator.calculate_research_similarity.return_value = {
            'embedding_similarity': 0.85,
            'topic_overlap': 0.75,
            'technique_overlap': 0.80,
            'organism_overlap': 0.70,
            'keyword_overlap': 0.65
        }
        return MultiFactorMatcher(similarity_calculator=mock_calculator)

    @pytest.fixture
    def sample_student_profile(self):
        return {
            'research_interests': 'CRISPR gene editing in cancer',
            'topics': ['gene editing', 'cancer biology'],
            'techniques': ['CRISPR', 'RNA-seq'],
            'organisms': ['human', 'mouse'],
            'career_goals': 'Become independent researcher'
        }

    @pytest.fixture
    def sample_faculty_profile(self):
        return {
            'id': 'fac123',
            'name': 'Dr. Smith',
            'research_summary': 'CRISPR applications in cancer therapy',
            'topics': ['gene editing', 'cancer biology', 'immunology'],
            'techniques': ['CRISPR', 'RNA-seq', 'flow cytometry'],
            'organisms': ['human', 'mouse'],
            'publications': [{'year': 2024, 'title': 'Test'}] * 20,
            'h_index': 35,
            'publication_count': 85,
            'grants': [
                {
                    'active': True,
                    'amount': 500000,
                    'end_date': '2027-12-31',
                    'year': 2022
                }
            ],
            'active_grants': 1,
            'total_funding': 500000,
            'lab_size': 8,
            'accepting_students': True
        }

    def test_match_calculation_returns_match_score(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test basic match calculation"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile,
            explain=True
        )

        assert isinstance(match_score, MatchScore)
        assert 0 <= match_score.overall_score <= 1
        assert match_score.component_scores is not None
        assert match_score.explanation is not None

    def test_match_score_has_all_components(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test that all component scores are calculated"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile
        )

        components = match_score.component_scores

        # Should have all 6 components
        assert 'research_alignment' in components
        assert 'funding_stability' in components
        assert 'productivity_match' in components
        assert 'technique_match' in components
        assert 'lab_environment' in components
        assert 'career_development' in components

        # All should be in [0, 1]
        for component, score in components.items():
            assert 0 <= score <= 1, f"{component} out of range: {score}"

    def test_funding_stability_scoring(self, matcher):
        """Test funding stability scoring"""
        # Well-funded profile
        well_funded = {
            'grants': [
                {'active': True, 'amount': 1000000, 'end_date': '2028-12-31', 'year': 2023},
                {'active': True, 'amount': 500000, 'end_date': '2027-06-30', 'year': 2022}
            ],
            'active_grants': 2,
            'total_funding': 1500000
        }

        # Poorly funded profile
        poorly_funded = {
            'grants': [],
            'active_grants': 0,
            'total_funding': 0
        }

        good_score, _ = matcher.funding_stability_score(well_funded)
        poor_score, _ = matcher.funding_stability_score(poorly_funded)

        assert good_score > poor_score
        assert poor_score < 0.5  # Should heavily penalize no funding
        assert good_score >= 0.6  # Good funding should score well

    def test_funding_stability_details(self, matcher):
        """Test that funding details are returned"""
        profile = {
            'grants': [{'active': True, 'amount': 1000000}],
            'active_grants': 1,
            'total_funding': 1000000
        }

        score, details = matcher.funding_stability_score(profile)

        assert 'active_grants' in details
        assert 'funding_level' in details
        assert details['active_grants'] == 1

    def test_lab_environment_scoring(self, matcher):
        """Test lab environment scoring"""
        # Optimal lab
        optimal_lab = {
            'lab_size': 6,
            'accepting_students': True
        }

        # Not accepting
        not_accepting = {
            'lab_size': 8,
            'accepting_students': False
        }

        # Very large lab
        large_lab = {
            'lab_size': 20,
            'accepting_students': True
        }

        optimal_score, _ = matcher.lab_environment_score(optimal_lab)
        not_accepting_score, _ = matcher.lab_environment_score(not_accepting)
        large_score, _ = matcher.lab_environment_score(large_lab)

        assert optimal_score > large_score
        assert not_accepting_score < optimal_score
        # Not accepting should be penalized heavily
        assert not_accepting_score < 0.5

    def test_productivity_compatibility(self, matcher):
        """Test productivity compatibility scoring"""
        student = {}

        high_productivity = {
            'publication_count': 80,
            'h_index': 35,
            'recent_publications': 5
        }

        low_productivity = {
            'publication_count': 10,
            'h_index': 5,
            'recent_publications': 0
        }

        high_score, _ = matcher.productivity_compatibility(student, high_productivity)
        low_score, _ = matcher.productivity_compatibility(student, low_productivity)

        assert high_score > low_score

    def test_match_strengths_identification(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test that match strengths are identified"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile,
            explain=True
        )

        strengths = match_score.strengths

        assert isinstance(strengths, list)
        assert len(strengths) > 0
        # Each strength should be a meaningful string
        assert all(len(s) > 10 for s in strengths)

    def test_match_considerations_identification(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test that considerations are identified"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile,
            explain=True
        )

        # May or may not have considerations depending on profile
        assert isinstance(match_score.considerations, list)

    def test_recommendation_categories(self, matcher):
        """Test recommendation categorization"""
        # High score
        rec_high = matcher.get_recommendation(0.85, 0.9)
        assert rec_high in ['highly_recommended', 'recommended']

        # Medium score
        rec_medium = matcher.get_recommendation(0.65, 0.8)
        assert rec_medium in ['recommended', 'consider']

        # Low score
        rec_low = matcher.get_recommendation(0.35, 0.7)
        assert rec_low in ['consider', 'not_recommended']

    def test_explanation_generation(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test that explanations are generated"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile,
            explain=True
        )

        explanation = match_score.explanation

        assert isinstance(explanation, str)
        assert len(explanation) > 20
        # Should mention faculty name
        assert 'Smith' in explanation or 'faculty' in explanation.lower()

    def test_explanation_disabled(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test that explanation can be disabled"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile,
            explain=False
        )

        # Explanation should be empty when disabled
        assert match_score.explanation == ""
        assert len(match_score.strengths) == 0
        assert len(match_score.considerations) == 0

    def test_confidence_calculation(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test confidence calculation"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile
        )

        confidence = match_score.confidence

        assert 0 <= confidence <= 1
        # Complete profiles should have high confidence
        assert confidence > 0.5

    def test_confidence_with_incomplete_data(self, matcher):
        """Test confidence with incomplete profiles"""
        minimal_student = {'research_interests': 'test'}
        minimal_faculty = {'id': 'fac1', 'name': 'Dr. Test'}

        match_score = matcher.calculate_match_score(
            minimal_student,
            minimal_faculty
        )

        # Incomplete data should lower confidence
        assert match_score.confidence < 0.8

    def test_match_score_to_dict(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test converting match score to dictionary"""
        match_score = matcher.calculate_match_score(
            sample_student_profile,
            sample_faculty_profile
        )

        match_dict = match_score.to_dict()

        assert isinstance(match_dict, dict)
        assert 'overall_score' in match_dict
        assert 'component_scores' in match_dict
        assert 'recommendation' in match_dict

    def test_configurable_weights(self):
        """Test that matching weights are configurable"""
        custom_config = {
            'research_alignment': 0.5,
            'funding_stability': 0.3,
            'productivity_match': 0.2
        }

        matcher = MultiFactorMatcher(config=custom_config)

        # Weights should be normalized
        assert abs(sum(matcher.weights.values()) - 1.0) < 0.01

    def test_technique_compatibility_scoring(self, matcher):
        """Test technique compatibility scoring"""
        student = {
            'techniques': ['CRISPR', 'RNA-seq', 'PCR']
        }

        faculty_high_overlap = {
            'techniques': ['CRISPR', 'RNA-seq', 'Western blot']
        }

        faculty_low_overlap = {
            'techniques': ['microscopy', 'flow cytometry']
        }

        high_score, high_details = matcher.technique_compatibility(student, faculty_high_overlap)
        low_score, low_details = matcher.technique_compatibility(student, faculty_low_overlap)

        assert high_score > low_score
        assert high_details['overlap'] > 0
        assert low_details['overlap'] == 0

    def test_career_development_scoring(self, matcher):
        """Test career development potential scoring"""
        student = {}

        established_faculty = {
            'h_index': 40,
            'collaborations': 15,
            'career_stage': 'full_professor'
        }

        early_career_faculty = {
            'h_index': 10,
            'collaborations': 3,
            'career_stage': 'assistant_professor'
        }

        established_score, _ = matcher.career_development_score(student, established_faculty)
        early_score, _ = matcher.career_development_score(student, early_career_faculty)

        assert established_score > early_score

    def test_match_with_missing_similarity_calculator(self):
        """Test matching works even without similarity calculator"""
        matcher = MultiFactorMatcher(similarity_calculator=None)

        student = {'research_interests': 'test'}
        faculty = {'name': 'Dr. Test', 'grants': []}

        # Should still work with defaults
        match_score = matcher.calculate_match_score(student, faculty)

        assert isinstance(match_score, MatchScore)
        # Research alignment should default to 0.5 without calculator
        assert match_score.component_scores['research_alignment'] == 0.5

    def test_research_alignment_uses_similarity_calculator(
        self,
        matcher,
        sample_student_profile,
        sample_faculty_profile
    ):
        """Test that research alignment uses similarity calculator"""
        score, details = matcher.research_alignment_score(
            sample_student_profile,
            sample_faculty_profile
        )

        # Should have called similarity calculator
        assert matcher.similarity_calculator.calculate_research_similarity.called
        assert 'similarities' in details
