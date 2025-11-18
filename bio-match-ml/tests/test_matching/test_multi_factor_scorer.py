"""
Unit tests for MultiFactorMatcher
"""

import pytest
from unittest.mock import Mock
from src.matching.multi_factor_scorer import MultiFactorMatcher, MatchScore
from src.matching.similarity_calculator import SimilarityCalculator


class TestMultiFactorMatcher:
    """Test MultiFactorMatcher functionality"""

    @pytest.fixture
    def mock_similarity_calculator(self):
        """Create mock similarity calculator"""
        calc = Mock(spec=SimilarityCalculator)
        calc.calculate_research_similarity = Mock(return_value={
            'embedding_similarity': 0.85,
            'topic_overlap': 0.75,
            'technique_overlap': 0.80,
            'organism_overlap': 0.70,
            'keyword_overlap': 0.65
        })
        calc.calculate_trajectory_alignment = Mock(return_value=0.75)
        calc.calculate_jaccard_similarity = Mock(return_value=0.60)
        return calc

    @pytest.fixture
    def matcher(self, mock_similarity_calculator):
        """Create matcher with mock calculator"""
        return MultiFactorMatcher(similarity_calculator=mock_similarity_calculator)

    @pytest.fixture
    def custom_matcher(self, mock_similarity_calculator):
        """Create matcher with custom weights"""
        config = {
            'research_alignment': 0.50,
            'funding_stability': 0.30,
            'productivity_match': 0.10,
            'technique_match': 0.05,
            'lab_environment': 0.03,
            'career_development': 0.02
        }
        return MultiFactorMatcher(
            similarity_calculator=mock_similarity_calculator,
            config=config
        )

    @pytest.fixture
    def sample_student(self):
        """Sample student profile"""
        return {
            'research_interests': 'CRISPR gene editing in cancer cells',
            'topics': ['gene editing', 'cancer biology', 'genetics'],
            'techniques': ['CRISPR', 'RNA-seq', 'cell culture'],
            'organisms': ['human', 'mouse'],
            'career_goals': 'Academic research career focusing on cancer therapeutics'
        }

    @pytest.fixture
    def sample_faculty_excellent(self):
        """Sample faculty profile - excellent match"""
        return {
            'id': 'fac123',
            'name': 'Dr. Jane Smith',
            'research_summary': 'CRISPR applications in cancer therapy',
            'topics': ['gene editing', 'cancer biology', 'immunology'],
            'techniques': ['CRISPR', 'RNA-seq', 'flow cytometry'],
            'organisms': ['human', 'mouse'],
            'publication_count': 85,
            'h_index': 35,
            'citation_count': 4000,
            'grants': [
                {
                    'active': True,
                    'amount': 1200000,
                    'end_date': '2028-12-31',
                    'start_date': '2023-01-01'
                },
                {
                    'active': True,
                    'amount': 500000,
                    'end_date': '2027-06-30',
                    'start_date': '2022-01-01'
                }
            ],
            'active_grants': [
                {'amount': 1200000, 'end_date': '2028-12-31'},
                {'amount': 500000, 'end_date': '2027-06-30'}
            ],
            'total_funding': 1700000,
            'has_active_funding': True,
            'lab_members': ['Student1', 'Student2', 'Student3', 'Postdoc1', 'Postdoc2'],
            'lab_size': 5,
            'accepting_students': True,
            'trajectory': {
                'training_focus': 'preparing students for academic careers',
                'future_directions': 'expanding into immunotherapy'
            }
        }

    @pytest.fixture
    def sample_faculty_poor(self):
        """Sample faculty profile - poor match"""
        return {
            'id': 'fac456',
            'name': 'Dr. Bob Johnson',
            'research_summary': 'Structural biology of membrane proteins',
            'topics': ['structural biology', 'biophysics'],
            'techniques': ['X-ray crystallography', 'cryo-EM'],
            'organisms': ['bacteria'],
            'publication_count': 15,
            'h_index': 8,
            'citation_count': 200,
            'grants': [],
            'active_grants': [],
            'total_funding': 0,
            'has_active_funding': False,
            'lab_members': [],
            'accepting_students': False,
            'trajectory': {}
        }

    def test_initialization(self, matcher):
        """Test matcher initializes correctly"""
        assert matcher is not None
        assert matcher.similarity_calculator is not None
        assert matcher.config is not None

    def test_initialization_default_config(self, matcher):
        """Test matcher has correct default weights"""
        assert matcher.config['research_alignment'] == 0.35
        assert matcher.config['funding_stability'] == 0.20
        assert matcher.config['productivity_match'] == 0.15
        assert matcher.config['technique_match'] == 0.15
        assert matcher.config['lab_environment'] == 0.10
        assert matcher.config['career_development'] == 0.05

        # Weights should sum to 1
        assert abs(sum(matcher.config.values()) - 1.0) < 0.01

    def test_custom_config(self, custom_matcher):
        """Test matcher with custom configuration"""
        assert custom_matcher.config['research_alignment'] == 0.50
        assert custom_matcher.config['funding_stability'] == 0.30

    def test_calculate_match_score_excellent(self, matcher, sample_student, sample_faculty_excellent):
        """Test match calculation for excellent match"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_excellent,
            explain=True
        )

        assert isinstance(match, MatchScore)
        assert 0.0 <= match.overall_score <= 1.0
        assert match.overall_score > 0.7  # Should be high

    def test_calculate_match_score_poor(self, matcher, sample_student, sample_faculty_poor):
        """Test match calculation for poor match"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_poor,
            explain=True
        )

        assert isinstance(match, MatchScore)
        assert match.overall_score < 0.6  # Should be lower

    def test_match_score_has_all_fields(self, matcher, sample_student, sample_faculty_excellent):
        """Test MatchScore contains all required fields"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_excellent,
            explain=True
        )

        assert hasattr(match, 'overall_score')
        assert hasattr(match, 'component_scores')
        assert hasattr(match, 'confidence')
        assert hasattr(match, 'explanation')
        assert hasattr(match, 'strengths')
        assert hasattr(match, 'considerations')
        assert hasattr(match, 'recommendation')

    def test_component_scores_present(self, matcher, sample_student, sample_faculty_excellent):
        """Test all component scores are calculated"""
        match = matcher.calculate_match_score(sample_student, sample_faculty_excellent)

        assert 'research_alignment' in match.component_scores
        assert 'funding_stability' in match.component_scores
        assert 'productivity_match' in match.component_scores
        assert 'technique_match' in match.component_scores
        assert 'lab_environment' in match.component_scores
        assert 'career_development' in match.component_scores

    def test_component_scores_normalized(self, matcher, sample_student, sample_faculty_excellent):
        """Test component scores are in [0, 1] range"""
        match = matcher.calculate_match_score(sample_student, sample_faculty_excellent)

        for component, score in match.component_scores.items():
            assert 0.0 <= score <= 1.0, f"{component} score {score} out of range"

    def test_explanation_with_explain_true(self, matcher, sample_student, sample_faculty_excellent):
        """Test explanation is generated when explain=True"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_excellent,
            explain=True
        )

        assert match.explanation is not None
        assert len(match.explanation) > 0
        assert isinstance(match.explanation, str)

    def test_strengths_identified(self, matcher, sample_student, sample_faculty_excellent):
        """Test match strengths are identified"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_excellent,
            explain=True
        )

        assert isinstance(match.strengths, list)
        assert len(match.strengths) > 0
        # Each strength should be a non-empty string
        assert all(isinstance(s, str) and len(s) > 0 for s in match.strengths)

    def test_considerations_identified(self, matcher, sample_student, sample_faculty_poor):
        """Test match considerations are identified"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_poor,
            explain=True
        )

        assert isinstance(match.considerations, list)
        # Poor match should have considerations
        assert len(match.considerations) > 0

    def test_recommendation_categories(self, matcher, sample_student, sample_faculty_excellent, sample_faculty_poor):
        """Test recommendation categories are valid"""
        match_good = matcher.calculate_match_score(sample_student, sample_faculty_excellent)
        match_poor = matcher.calculate_match_score(sample_student, sample_faculty_poor)

        valid_recommendations = ['highly_recommended', 'recommended', 'consider', 'not_recommended']

        assert match_good.recommendation in valid_recommendations
        assert match_poor.recommendation in valid_recommendations

    def test_confidence_score(self, matcher, sample_student, sample_faculty_excellent):
        """Test confidence score is calculated"""
        match = matcher.calculate_match_score(sample_student, sample_faculty_excellent)

        assert 0.0 <= match.confidence <= 1.0

    def test_research_alignment_score(self, matcher, sample_student, sample_faculty_excellent):
        """Test research alignment scoring"""
        score, details = matcher.research_alignment_score(
            sample_student,
            sample_faculty_excellent
        )

        assert 0.0 <= score <= 1.0
        assert isinstance(details, dict)
        assert 'similarities' in details

    def test_funding_stability_excellent(self, matcher, sample_faculty_excellent):
        """Test funding stability for well-funded faculty"""
        score, details = matcher.funding_stability_score(sample_faculty_excellent)

        assert score >= 0.7  # Should be high
        assert details['has_active_grants'] is True
        assert 'funding_level' in details

    def test_funding_stability_poor(self, matcher, sample_faculty_poor):
        """Test funding stability for unfunded faculty"""
        score, details = matcher.funding_stability_score(sample_faculty_poor)

        assert score < 0.5  # Should be low
        assert details['has_active_grants'] is False

    def test_funding_stability_ranges(self, matcher):
        """Test funding stability scoring ranges"""
        # Excellent funding
        excellent = {
            'grants': [
                {'active': True, 'amount': 2500000, 'end_date': '2029-12-31'}
            ],
            'active_grants': [{'amount': 2500000, 'end_date': '2029-12-31'}],
            'has_active_funding': True,
            'total_funding': 2500000
        }
        score, _ = matcher.funding_stability_score(excellent)
        assert score >= 0.8, "Excellent funding should score >= 0.8"

        # Good funding
        good = {
            'grants': [
                {'active': True, 'amount': 1000000, 'end_date': '2027-12-31'}
            ],
            'active_grants': [{'amount': 1000000, 'end_date': '2027-12-31'}],
            'has_active_funding': True,
            'total_funding': 1000000
        }
        score, _ = matcher.funding_stability_score(good)
        assert 0.6 <= score < 0.8, "Good funding should score 0.6-0.8"

        # Poor funding
        poor = {
            'grants': [],
            'active_grants': [],
            'has_active_funding': False,
            'total_funding': 0
        }
        score, _ = matcher.funding_stability_score(poor)
        assert score < 0.3, "Poor funding should score < 0.3"

    def test_productivity_compatibility(self, matcher, sample_student, sample_faculty_excellent):
        """Test productivity compatibility scoring"""
        score, details = matcher.productivity_compatibility(
            sample_student,
            sample_faculty_excellent
        )

        assert 0.0 <= score <= 1.0
        assert 'faculty_productivity' in details
        assert 'compatibility' in details

    def test_productivity_levels(self, matcher, sample_student):
        """Test productivity level categorization"""
        very_high = {'publication_count': 120, 'h_index': 45}
        high = {'publication_count': 60, 'h_index': 25}
        moderate = {'publication_count': 30, 'h_index': 15}
        developing = {'publication_count': 10, 'h_index': 5}

        score_vh, details_vh = matcher.productivity_compatibility(sample_student, very_high)
        assert details_vh['faculty_productivity'] == 'very_high'

        score_h, details_h = matcher.productivity_compatibility(sample_student, high)
        assert details_h['faculty_productivity'] == 'high'

        score_m, details_m = matcher.productivity_compatibility(sample_student, moderate)
        assert details_m['faculty_productivity'] == 'moderate'

        score_d, details_d = matcher.productivity_compatibility(sample_student, developing)
        assert details_d['faculty_productivity'] == 'developing'

    def test_lab_environment_score_accepting(self, matcher, sample_faculty_excellent):
        """Test lab environment score for accepting students"""
        score, details = matcher.lab_environment_score(sample_faculty_excellent)

        assert score >= 0.5  # Should be positive
        assert details['accepting_students'] is True

    def test_lab_environment_score_not_accepting(self, matcher, sample_faculty_poor):
        """Test lab environment score for not accepting students"""
        score, details = matcher.lab_environment_score(sample_faculty_poor)

        assert score < 0.5  # Should be penalized
        assert details['accepting_students'] is False
        assert 'warning' in details

    def test_lab_size_categories(self, matcher):
        """Test lab size categorization"""
        small_lab = {'lab_members': ['1', '2', '3'], 'accepting_students': True}
        medium_lab = {'lab_members': list(range(12)), 'accepting_students': True}
        large_lab = {'lab_members': list(range(20)), 'accepting_students': True}

        score_s, details_s = matcher.lab_environment_score(small_lab)
        assert 'small' in details_s['lab_size']

        score_m, details_m = matcher.lab_environment_score(medium_lab)
        assert 'medium' in details_m['lab_size']

        score_l, details_l = matcher.lab_environment_score(large_lab)
        assert 'large' in details_l['lab_size']

    def test_generate_match_explanation(self, matcher, sample_student, sample_faculty_excellent):
        """Test match explanation generation"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_excellent,
            explain=True
        )

        explanation = match.explanation

        # Should mention faculty name
        assert 'Smith' in explanation or 'Dr.' in explanation

        # Should mention score level
        assert any(term in explanation.lower() for term in ['excellent', 'strong', 'good', 'moderate'])

        # Should be substantive
        assert len(explanation) > 50

    def test_identify_match_strengths_specific(self, matcher, sample_student, sample_faculty_excellent):
        """Test match strengths are specific"""
        match = matcher.calculate_match_score(
            sample_student,
            sample_faculty_excellent,
            explain=True
        )

        strengths = match.strengths

        # Should have multiple strengths
        assert len(strengths) >= 2

        # Should mention specific techniques or topics
        combined_text = ' '.join(strengths).lower()
        # Should mention something specific
        assert any(term in combined_text for term in ['crispr', 'funding', 'publications', 'h-index', 'active'])

    def test_identify_considerations_for_concerns(self, matcher, sample_student):
        """Test considerations identify concerns"""
        # Faculty with issues
        problematic_faculty = {
            'research_summary': 'Different field',
            'topics': ['unrelated'],
            'techniques': ['different'],
            'grants': [],
            'has_active_funding': False,
            'total_funding': 0,
            'lab_members': list(range(25)),  # Very large lab
            'accepting_students': False,
            'publication_count': 5
        }

        match = matcher.calculate_match_score(
            sample_student,
            problematic_faculty,
            explain=True
        )

        considerations = match.considerations

        # Should have multiple considerations
        assert len(considerations) >= 2

        # Should mention specific concerns
        combined_text = ' '.join(considerations).lower()
        assert any(term in combined_text for term in ['funding', 'large', 'not', 'limited'])

    def test_recommendation_highly_recommended(self, matcher, sample_student, sample_faculty_excellent):
        """Test highly_recommended recommendation"""
        match = matcher.calculate_match_score(sample_student, sample_faculty_excellent)

        # Excellent match should be highly recommended or recommended
        assert match.recommendation in ['highly_recommended', 'recommended']

    def test_recommendation_not_recommended(self, matcher, sample_student, sample_faculty_poor):
        """Test not_recommended recommendation"""
        match = matcher.calculate_match_score(sample_student, sample_faculty_poor)

        # Poor match should not be highly recommended
        assert match.recommendation in ['not_recommended', 'consider']

    def test_get_recommendation_categories(self, matcher):
        """Test get_recommendation logic"""
        # High score, high confidence
        rec = matcher.get_recommendation(overall_score=0.80, confidence=0.80)
        assert rec == 'highly_recommended'

        # Moderate score, good confidence
        rec = matcher.get_recommendation(overall_score=0.68, confidence=0.70)
        assert rec == 'recommended'

        # Moderate score, lower confidence
        rec = matcher.get_recommendation(overall_score=0.55, confidence=0.50)
        assert rec == 'consider'

        # Low score
        rec = matcher.get_recommendation(overall_score=0.40, confidence=0.70)
        assert rec == 'not_recommended'

    def test_match_score_to_dict(self, matcher, sample_student, sample_faculty_excellent):
        """Test MatchScore to_dict conversion"""
        match = matcher.calculate_match_score(sample_student, sample_faculty_excellent)

        match_dict = match.to_dict()

        assert isinstance(match_dict, dict)
        assert 'overall_score' in match_dict
        assert 'component_scores' in match_dict
        assert 'confidence' in match_dict
        assert 'explanation' in match_dict
        assert 'strengths' in match_dict
        assert 'considerations' in match_dict
        assert 'recommendation' in match_dict

    def test_missing_student_data(self, matcher, sample_faculty_excellent):
        """Test matching with incomplete student profile"""
        minimal_student = {
            'research_interests': 'biology'
        }

        match = matcher.calculate_match_score(minimal_student, sample_faculty_excellent)

        # Should still calculate a score
        assert 0.0 <= match.overall_score <= 1.0
        # Confidence should be lower
        assert match.confidence < 0.8

    def test_missing_faculty_data(self, matcher, sample_student):
        """Test matching with incomplete faculty profile"""
        minimal_faculty = {
            'name': 'Dr. Test',
            'research_summary': 'research'
        }

        match = matcher.calculate_match_score(sample_student, minimal_faculty)

        # Should still calculate a score
        assert 0.0 <= match.overall_score <= 1.0

    def test_confidence_based_on_data_completeness(self, matcher, sample_student):
        """Test confidence reflects data completeness"""
        complete_faculty = sample_student  # Use student as template
        complete_faculty.update({
            'name': 'Complete',
            'research_summary': 'detailed',
            'publications': [{'title': 'paper'}],
            'grants': [{'amount': 100000}],
            'techniques': ['CRISPR']
        })

        incomplete_faculty = {
            'name': 'Incomplete',
            'research_summary': 'brief'
        }

        match_complete = matcher.calculate_match_score(sample_student, complete_faculty)
        match_incomplete = matcher.calculate_match_score(sample_student, incomplete_faculty)

        # More complete data should have higher confidence
        assert match_complete.confidence >= match_incomplete.confidence


class TestMatchScore:
    """Test MatchScore dataclass"""

    def test_create_match_score(self):
        """Test creating MatchScore"""
        score = MatchScore(
            overall_score=0.85,
            component_scores={'research': 0.9, 'funding': 0.8},
            confidence=0.80,
            explanation="Good match",
            strengths=["High research similarity"],
            considerations=["Large lab"],
            recommendation='recommended'
        )

        assert score.overall_score == 0.85
        assert score.confidence == 0.80
        assert score.recommendation == 'recommended'

    def test_match_score_to_dict(self):
        """Test MatchScore to_dict method"""
        score = MatchScore(
            overall_score=0.75,
            component_scores={},
            confidence=0.70,
            explanation="Test",
            strengths=[],
            considerations=[],
            recommendation='consider'
        )

        score_dict = score.to_dict()

        assert isinstance(score_dict, dict)
        assert score_dict['overall_score'] == 0.75
        assert score_dict['confidence'] == 0.70
        assert score_dict['recommendation'] == 'consider'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
