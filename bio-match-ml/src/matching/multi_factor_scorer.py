"""
Multi-Factor Matcher for comprehensive faculty-student matching

Calculates match scores based on multiple factors:
- Research alignment (semantic + topic overlap)
- Funding stability
- Productivity compatibility
- Technique match
- Lab environment fit
- Career development potential
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MatchScore:
    """Data class for match results"""
    overall_score: float
    component_scores: Dict[str, float]
    confidence: float
    explanation: str
    strengths: List[str]
    considerations: List[str]
    recommendation: str

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class MultiFactorMatcher:
    """
    Comprehensive faculty-student matching system
    """

    def __init__(
        self,
        similarity_calculator: Optional[Any] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize matcher with configurable weights

        Args:
            similarity_calculator: SimilarityCalculator instance
            config: Weight configuration, defaults:
                {
                    'research_alignment': 0.35,
                    'funding_stability': 0.20,
                    'productivity_match': 0.15,
                    'technique_match': 0.15,
                    'lab_environment': 0.10,
                    'career_development': 0.05
                }
        """
        # Initialize similarity calculator
        if similarity_calculator is None:
            from .similarity_calculator import SimilarityCalculator
            self.similarity_calculator = SimilarityCalculator()
        else:
            self.similarity_calculator = similarity_calculator

        # Default weights
        default_config = {
            'research_alignment': 0.35,
            'funding_stability': 0.20,
            'productivity_match': 0.15,
            'technique_match': 0.15,
            'lab_environment': 0.10,
            'career_development': 0.05
        }

        self.config = config if config is not None else default_config

        # Normalize weights
        total_weight = sum(self.config.values())
        if total_weight > 0:
            self.config = {k: v / total_weight for k, v in self.config.items()}

        logger.info(f"MultiFactorMatcher initialized with weights: {self.config}")

    def calculate_match_score(
        self,
        student_profile: Dict,
        faculty_profile: Dict,
        explain: bool = True
    ) -> MatchScore:
        """
        Calculate comprehensive match score

        Args:
            student_profile: Student's research profile, interests, goals
            faculty_profile: Faculty's research profile, lab info, grants
            explain: Generate human-readable explanation

        Returns:
            MatchScore object with detailed scoring and recommendations
        """
        logger.debug(f"Calculating match for student and faculty {faculty_profile.get('name', 'unknown')}")

        # Calculate component scores
        component_scores = {}

        # 1. Research alignment
        research_score, research_details = self.research_alignment_score(
            student_profile,
            faculty_profile
        )
        component_scores['research_alignment'] = research_score

        # 2. Funding stability
        funding_score, funding_details = self.funding_stability_score(faculty_profile)
        component_scores['funding_stability'] = funding_score

        # 3. Productivity compatibility
        productivity_score, productivity_details = self.productivity_compatibility(
            student_profile,
            faculty_profile
        )
        component_scores['productivity_match'] = productivity_score

        # 4. Technique match
        technique_score = self._technique_match_score(student_profile, faculty_profile)
        component_scores['technique_match'] = technique_score

        # 5. Lab environment
        lab_score, lab_details = self.lab_environment_score(faculty_profile)
        component_scores['lab_environment'] = lab_score

        # 6. Career development
        career_score = self._career_development_score(student_profile, faculty_profile)
        component_scores['career_development'] = career_score

        # Calculate overall score
        overall_score = self._calculate_overall_score(component_scores)

        # Calculate confidence
        confidence = self._calculate_confidence(component_scores, student_profile, faculty_profile)

        # Generate explanation
        explanation = ""
        strengths = []
        considerations = []

        if explain:
            explanation = self.generate_match_explanation(
                overall_score,
                component_scores,
                student_profile,
                faculty_profile
            )
            strengths = self.identify_match_strengths(
                component_scores,
                student_profile,
                faculty_profile
            )
            considerations = self.identify_considerations(
                component_scores,
                faculty_profile
            )

        # Get recommendation
        recommendation = self.get_recommendation(overall_score, confidence)

        logger.debug(f"Match score calculated: {overall_score:.3f} ({recommendation})")

        return MatchScore(
            overall_score=overall_score,
            component_scores=component_scores,
            confidence=confidence,
            explanation=explanation,
            strengths=strengths,
            considerations=considerations,
            recommendation=recommendation
        )

    def _calculate_overall_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0

        for component, score in component_scores.items():
            weight = self.config.get(component, 0.0)
            total_score += weight * score

        return max(0.0, min(1.0, total_score))

    def _calculate_confidence(
        self,
        component_scores: Dict[str, float],
        student_profile: Dict,
        faculty_profile: Dict
    ) -> float:
        """
        Calculate confidence in the match prediction

        Higher confidence when:
        - More data available
        - Scores are consistent (not polarized)
        - Clear match or non-match
        """
        confidence_factors = []

        # Data completeness
        student_fields = ['research_interests', 'topics', 'techniques', 'career_goals']
        student_completeness = sum(1 for f in student_fields if student_profile.get(f)) / len(student_fields)
        confidence_factors.append(student_completeness)

        faculty_fields = ['research_summary', 'publications', 'grants', 'techniques']
        faculty_completeness = sum(1 for f in faculty_fields if faculty_profile.get(f)) / len(faculty_fields)
        confidence_factors.append(faculty_completeness)

        # Score consistency (low variance = high confidence)
        scores = list(component_scores.values())
        if scores:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            consistency = 1.0 - min(1.0, variance)
            confidence_factors.append(consistency)

        # Overall score clarity (very high or very low = confident)
        overall = self._calculate_overall_score(component_scores)
        if overall > 0.7 or overall < 0.3:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)

        return sum(confidence_factors) / len(confidence_factors)

    def research_alignment_score(
        self,
        student_interests: Dict,
        faculty_research: Dict
    ) -> Tuple[float, Dict]:
        """
        Deep research compatibility analysis

        Components:
        - Semantic similarity of research descriptions
        - Topic overlap
        - Technique compatibility
        - Organism compatibility
        - Interdisciplinary alignment

        Args:
            student_interests: Student research interests
            faculty_research: Faculty research profile

        Returns:
            Tuple of (score, details_dict)
        """
        similarities = self.similarity_calculator.calculate_research_similarity(
            student_interests,
            faculty_research
        )

        # Weight different similarity types
        weights = {
            'embedding_similarity': 0.40,  # Highest weight
            'topic_overlap': 0.25,
            'technique_overlap': 0.20,
            'organism_overlap': 0.10,
            'keyword_overlap': 0.05
        }

        # Calculate weighted score
        score = 0.0
        for sim_type, sim_score in similarities.items():
            weight = weights.get(sim_type, 0.0)
            score += weight * sim_score

        details = {
            'similarities': similarities,
            'weights': weights
        }

        return score, details

    def funding_stability_score(self, faculty_profile: Dict) -> Tuple[float, Dict]:
        """
        Evaluate funding situation

        Factors:
        - Has active grants (critical)
        - Grant runway (years of funding remaining)
        - Funding amount
        - Funding consistency (history)
        - Funding diversity (multiple sources)
        - Recent grant success rate

        Args:
            faculty_profile: Faculty profile with grant data

        Returns:
            Tuple of (score, details_dict)

        Scoring:
        - 0.0-0.3: Poor funding (red flag)
        - 0.3-0.6: Moderate funding (caution)
        - 0.6-0.8: Good funding (safe)
        - 0.8-1.0: Excellent funding (very safe)
        """
        score = 0.0
        details = {}

        active_grants = faculty_profile.get('active_grants', [])
        total_funding = faculty_profile.get('total_funding', 0)
        has_active = faculty_profile.get('has_active_funding', False) or len(active_grants) > 0

        # Critical: Has active grants (40% of score)
        if has_active:
            score += 0.4
            details['has_active_grants'] = True
        else:
            details['has_active_grants'] = False
            details['warning'] = 'No active grants'

        # Funding amount (25% of score)
        if total_funding > 2000000:  # >2M excellent
            score += 0.25
            details['funding_level'] = 'excellent'
        elif total_funding > 1000000:  # >1M good
            score += 0.20
            details['funding_level'] = 'good'
        elif total_funding > 500000:  # >500K moderate
            score += 0.15
            details['funding_level'] = 'moderate'
        else:
            score += 0.05
            details['funding_level'] = 'low'

        # Grant diversity (15% of score)
        num_grants = len(active_grants)
        if num_grants >= 3:
            score += 0.15
            details['grant_diversity'] = 'high'
        elif num_grants == 2:
            score += 0.10
            details['grant_diversity'] = 'moderate'
        elif num_grants == 1:
            score += 0.05
            details['grant_diversity'] = 'single'
        else:
            details['grant_diversity'] = 'none'

        # Grant runway (20% of score)
        max_end_year = 0
        for grant in active_grants:
            end_date = grant.get('end_date', '')
            if isinstance(end_date, str) and len(end_date) >= 4:
                try:
                    end_year = int(end_date[:4])
                    max_end_year = max(max_end_year, end_year)
                except:
                    pass

        if max_end_year > 0:
            from datetime import datetime
            current_year = datetime.now().year
            years_remaining = max_end_year - current_year

            if years_remaining >= 3:
                score += 0.20
                details['runway'] = f'{years_remaining} years'
            elif years_remaining >= 2:
                score += 0.15
                details['runway'] = f'{years_remaining} years'
            elif years_remaining >= 1:
                score += 0.10
                details['runway'] = f'{years_remaining} year(s)'
            else:
                score += 0.05
                details['runway'] = 'ending soon'
                details['warning'] = 'Grants ending soon'

        return min(1.0, score), details

    def productivity_compatibility(
        self,
        student_profile: Dict,
        faculty_metrics: Dict
    ) -> Tuple[float, Dict]:
        """
        Match productivity expectations

        Compares:
        - Publication frequency expectations
        - Journal tier preferences
        - Work pace compatibility
        - Publication authorship patterns

        Args:
            student_profile: Student goals and work style
            faculty_metrics: Faculty publication metrics

        Returns:
            Tuple of (score, details_dict)
        """
        score = 0.5  # Neutral default
        details = {}

        pub_count = faculty_metrics.get('publication_count', 0)
        h_index = faculty_metrics.get('h_index', 0)

        # Calculate faculty productivity level
        if pub_count > 100 or h_index > 30:
            faculty_productivity = 'very_high'
            score_value = 0.9
        elif pub_count > 50 or h_index > 20:
            faculty_productivity = 'high'
            score_value = 0.8
        elif pub_count > 20 or h_index > 10:
            faculty_productivity = 'moderate'
            score_value = 0.7
        else:
            faculty_productivity = 'developing'
            score_value = 0.6

        details['faculty_productivity'] = faculty_productivity

        # Match with student expectations (if provided)
        student_expectations = student_profile.get('publication_expectations', 'moderate')

        # Good match if aligned
        if student_expectations == faculty_productivity:
            score = score_value
        elif student_expectations == 'high' and faculty_productivity in ['very_high', 'high']:
            score = score_value
        elif student_expectations == 'moderate' and faculty_productivity in ['high', 'moderate']:
            score = score_value - 0.1
        else:
            score = 0.6  # Mismatch

        details['compatibility'] = 'good' if score >= 0.7 else 'moderate'

        return score, details

    def _technique_match_score(
        self,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> float:
        """Calculate technique overlap score"""
        student_techniques = set(t.lower() for t in student_profile.get('techniques', []))
        faculty_techniques = set(t.lower() for t in faculty_profile.get('techniques', []))

        if not student_techniques or not faculty_techniques:
            return 0.5  # Neutral if no data

        overlap = self.similarity_calculator.calculate_jaccard_similarity(
            student_techniques,
            faculty_techniques
        )

        # Boost score if any exact matches
        if len(student_techniques & faculty_techniques) > 0:
            overlap = min(1.0, overlap * 1.2)

        return overlap

    def lab_environment_score(self, faculty_profile: Dict) -> Tuple[float, Dict]:
        """
        Score lab environment factors

        Factors:
        - Lab size (small, medium, large)
        - Student-to-PI ratio
        - Collaboration style
        - Accepting students (boolean)
        - Mentorship indicators

        Args:
            faculty_profile: Faculty profile

        Returns:
            Tuple of (score, details_dict)
        """
        score = 0.5
        details = {}

        # Accepting students (critical)
        accepting = faculty_profile.get('accepting_students', True)  # Assume true if not specified
        if accepting:
            score += 0.3
            details['accepting_students'] = True
        else:
            score = 0.2  # Major penalty
            details['accepting_students'] = False
            details['warning'] = 'Not currently accepting students'

        # Lab size
        lab_size = len(faculty_profile.get('lab_members', []))
        if 0 < lab_size <= 8:
            score += 0.2
            details['lab_size'] = 'small (good mentorship)'
        elif 8 < lab_size <= 15:
            score += 0.15
            details['lab_size'] = 'medium'
        elif lab_size > 15:
            score += 0.1
            details['lab_size'] = 'large (less 1-on-1 time)'
        else:
            score += 0.1
            details['lab_size'] = 'unknown'

        # Collaboration indicators
        research = faculty_profile.get('research_summary', '').lower()
        if 'collaborative' in research or 'collaboration' in research:
            score += 0.1
            details['collaborative'] = True

        return min(1.0, score), details

    def _career_development_score(
        self,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> float:
        """Score career development alignment"""
        student_goals = student_profile.get('career_goals', {})
        faculty_trajectory = faculty_profile.get('trajectory', {})

        # Use trajectory alignment from similarity calculator
        try:
            alignment = self.similarity_calculator.calculate_trajectory_alignment(
                student_goals if isinstance(student_goals, dict) else {'career_goals': student_goals},
                faculty_trajectory if isinstance(faculty_trajectory, dict) else {}
            )
            return alignment
        except:
            return 0.6  # Neutral if calculation fails

    def generate_match_explanation(
        self,
        match_score: float,
        component_scores: Dict,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> str:
        """Generate human-readable match explanation"""
        faculty_name = faculty_profile.get('name', 'This faculty member')

        # Overall assessment
        if match_score >= 0.8:
            assessment = "Excellent match"
        elif match_score >= 0.7:
            assessment = "Strong match"
        elif match_score >= 0.6:
            assessment = "Good match"
        elif match_score >= 0.5:
            assessment = "Moderate match"
        else:
            assessment = "Limited match"

        # Key factors
        research_score = component_scores.get('research_alignment', 0)
        funding_score = component_scores.get('funding_stability', 0)

        explanation_parts = [f"{assessment} ({match_score:.2f}) with {faculty_name}."]

        if research_score >= 0.7:
            explanation_parts.append(f"Strong research alignment ({research_score:.2f}) in your areas of interest.")

        if funding_score >= 0.7:
            explanation_parts.append(f"Lab has strong funding stability ({funding_score:.2f}).")
        elif funding_score < 0.5:
            explanation_parts.append(f"Note: Funding situation may be limited ({funding_score:.2f}).")

        return " ".join(explanation_parts)

    def identify_match_strengths(
        self,
        component_scores: Dict,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> List[str]:
        """Identify specific strengths of this match"""
        strengths = []

        # Research alignment
        if component_scores.get('research_alignment', 0) >= 0.75:
            strengths.append(f"Very high research similarity ({component_scores['research_alignment']:.2f})")

        # Technique overlap
        if component_scores.get('technique_match', 0) >= 0.6:
            student_techs = student_profile.get('techniques', [])
            faculty_techs = faculty_profile.get('techniques', [])
            common = set(t.lower() for t in student_techs) & set(t.lower() for t in faculty_techs)
            if common:
                strengths.append(f"Strong technique overlap: {', '.join(list(common)[:3])}")

        # Funding
        if component_scores.get('funding_stability', 0) >= 0.7:
            total = faculty_profile.get('total_funding', 0)
            if total > 0:
                strengths.append(f"Active funding (${total/1000000:.1f}M)")
            else:
                strengths.append("Active funding")

        # Productivity
        pub_count = faculty_profile.get('publication_count', 0)
        h_index = faculty_profile.get('h_index', 0)
        if pub_count > 50 or h_index > 20:
            strengths.append(f"Strong publication record ({pub_count} papers, H-index: {h_index})")

        # Lab environment
        if component_scores.get('lab_environment', 0) >= 0.7:
            if faculty_profile.get('accepting_students', True):
                strengths.append("Currently accepting students")

        return strengths[:5]  # Limit to top 5

    def identify_considerations(
        self,
        component_scores: Dict,
        faculty_profile: Dict
    ) -> List[str]:
        """Identify potential concerns or considerations"""
        considerations = []

        # Funding concerns
        if component_scores.get('funding_stability', 0) < 0.5:
            considerations.append("Limited funding - verify lab stability")

        # Lab size
        lab_size = len(faculty_profile.get('lab_members', []))
        if lab_size > 15:
            considerations.append(f"Large lab ({lab_size} members) - may have less 1-on-1 time")

        # Not accepting students
        if not faculty_profile.get('accepting_students', True):
            considerations.append("May not be currently accepting students - confirm availability")

        # Low productivity match
        if component_scores.get('productivity_match', 0) < 0.5:
            considerations.append("Productivity expectations may differ")

        # Moderate research alignment
        if component_scores.get('research_alignment', 0) < 0.6:
            considerations.append("Research alignment is moderate - ensure fit")

        return considerations[:5]  # Limit to top 5

    def get_recommendation(self, overall_score: float, confidence: float) -> str:
        """Get categorical recommendation"""
        if overall_score >= 0.75 and confidence >= 0.7:
            return 'highly_recommended'
        elif overall_score >= 0.65 and confidence >= 0.6:
            return 'recommended'
        elif overall_score >= 0.5:
            return 'consider'
        else:
            return 'not_recommended'
