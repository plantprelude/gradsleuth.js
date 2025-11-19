"""
Comprehensive faculty-student matching with explainable scores
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

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
        return {
            'overall_score': self.overall_score,
            'component_scores': self.component_scores,
            'confidence': self.confidence,
            'explanation': self.explanation,
            'strengths': self.strengths,
            'considerations': self.considerations,
            'recommendation': self.recommendation
        }


class MultiFactorMatcher:
    """
    Comprehensive faculty-student matching system
    """

    def __init__(
        self,
        similarity_calculator=None,
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
        self.similarity_calculator = similarity_calculator

        default_config = {
            'research_alignment': 0.35,
            'funding_stability': 0.20,
            'productivity_match': 0.15,
            'technique_match': 0.15,
            'lab_environment': 0.10,
            'career_development': 0.05
        }

        self.config = config if config else default_config
        self.weights = self.config

        # Validate weights
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing...")
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"MultiFactorMatcher initialized with weights: {self.weights}")

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
            MatchScore object with comprehensive matching information
        """
        logger.debug(f"Calculating match for faculty {faculty_profile.get('id', 'unknown')}")

        # Calculate component scores
        component_scores = {}

        # Research alignment
        research_score, research_details = self.research_alignment_score(
            student_profile,
            faculty_profile
        )
        component_scores['research_alignment'] = research_score

        # Funding stability
        funding_score, funding_details = self.funding_stability_score(faculty_profile)
        component_scores['funding_stability'] = funding_score

        # Productivity compatibility
        productivity_score, productivity_details = self.productivity_compatibility(
            student_profile,
            faculty_profile
        )
        component_scores['productivity_match'] = productivity_score

        # Technique match
        technique_score, technique_details = self.technique_compatibility(
            student_profile,
            faculty_profile
        )
        component_scores['technique_match'] = technique_score

        # Lab environment
        lab_score, lab_details = self.lab_environment_score(faculty_profile)
        component_scores['lab_environment'] = lab_score

        # Career development potential
        career_score, career_details = self.career_development_score(
            student_profile,
            faculty_profile
        )
        component_scores['career_development'] = career_score

        # Calculate weighted overall score
        overall_score = sum(
            self.weights.get(component, 0) * score
            for component, score in component_scores.items()
        )

        # Calculate confidence based on data completeness
        confidence = self._calculate_confidence(
            student_profile,
            faculty_profile,
            component_scores
        )

        # Generate explanation if requested
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
        else:
            explanation = ""
            strengths = []
            considerations = []

        # Get recommendation category
        recommendation = self.get_recommendation(overall_score, confidence)

        match_score = MatchScore(
            overall_score=overall_score,
            component_scores=component_scores,
            confidence=confidence,
            explanation=explanation,
            strengths=strengths,
            considerations=considerations,
            recommendation=recommendation
        )

        logger.info(f"Match calculated: score={overall_score:.3f}, recommendation={recommendation}")

        return match_score

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
        if not self.similarity_calculator:
            logger.warning("No similarity calculator available")
            return 0.5, {}

        # Get all similarity metrics
        similarities = self.similarity_calculator.calculate_research_similarity(
            student_interests,
            faculty_research
        )

        # Weighted combination of similarity metrics
        weights = {
            'embedding_similarity': 0.35,
            'topic_overlap': 0.25,
            'technique_overlap': 0.20,
            'organism_overlap': 0.10,
            'keyword_overlap': 0.10
        }

        score = sum(
            weights.get(metric, 0) * value
            for metric, value in similarities.items()
        )

        # Boost for high agreement across multiple metrics
        high_scores = sum(1 for v in similarities.values() if v >= 0.7)
        if high_scores >= 3:
            score *= 1.1  # 10% bonus
            score = min(score, 1.0)

        details = {
            'similarities': similarities,
            'high_agreement_metrics': high_scores
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
        grants = faculty_profile.get('grants', [])
        active_grants = faculty_profile.get('active_grants', 0)

        details = {
            'active_grants': active_grants,
            'total_grants': len(grants),
            'funding_level': 'unknown'
        }

        # No funding - significant penalty
        if active_grants == 0 and not grants:
            details['funding_level'] = 'none'
            return 0.1, details

        scores = []

        # Active grants (most critical)
        if active_grants > 0:
            scores.append(1.0)
            details['has_active'] = True
        else:
            scores.append(0.2)
            details['has_active'] = False

        # Funding amount
        total_funding = faculty_profile.get('total_funding', 0)
        if total_funding > 0:
            # Logarithmic scaling: $500k -> 0.7, $1M -> 0.85, $2M -> 1.0
            import math
            funding_score = min(math.log(total_funding / 100000 + 1) / math.log(21), 1.0)
            scores.append(funding_score)
            details['total_funding'] = total_funding

        # Grant diversity
        if active_grants >= 3:
            scores.append(1.0)
            details['diversity'] = 'high'
        elif active_grants == 2:
            scores.append(0.8)
            details['diversity'] = 'medium'
        elif active_grants == 1:
            scores.append(0.6)
            details['diversity'] = 'low'

        # Grant runway (years remaining)
        if grants:
            current_year = datetime.now().year
            end_years = []

            for grant in grants:
                if isinstance(grant, dict) and grant.get('active'):
                    end_date = grant.get('end_date', '')
                    if end_date:
                        try:
                            # Parse year from end_date (assumes YYYY-MM-DD format)
                            end_year = int(end_date.split('-')[0])
                            end_years.append(end_year)
                        except:
                            pass

            if end_years:
                max_runway = max(end_years) - current_year
                runway_score = min(max_runway / 5.0, 1.0)  # 5+ years = 1.0
                scores.append(runway_score)
                details['max_runway_years'] = max_runway

        # Calculate final score
        if scores:
            # Heavily weight the first score (has active grants)
            final_score = (scores[0] * 2 + sum(scores[1:])) / (len(scores) + 1)
        else:
            final_score = 0.3

        # Determine funding level
        if final_score >= 0.8:
            details['funding_level'] = 'excellent'
        elif final_score >= 0.6:
            details['funding_level'] = 'good'
        elif final_score >= 0.3:
            details['funding_level'] = 'moderate'
        else:
            details['funding_level'] = 'poor'

        return final_score, details

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
        pub_count = faculty_metrics.get('publication_count', 0)
        h_index = faculty_metrics.get('h_index', 0)
        recent_pubs = faculty_metrics.get('recent_publications', 0)

        details = {
            'publication_count': pub_count,
            'h_index': h_index,
            'productivity_level': 'unknown'
        }

        # Determine faculty productivity level
        if pub_count >= 50 and h_index >= 30:
            productivity_level = 'very_high'
            score = 0.9
        elif pub_count >= 30 and h_index >= 20:
            productivity_level = 'high'
            score = 0.85
        elif pub_count >= 15 and h_index >= 10:
            productivity_level = 'moderate'
            score = 0.75
        elif pub_count >= 5:
            productivity_level = 'developing'
            score = 0.65
        else:
            productivity_level = 'low'
            score = 0.5

        details['productivity_level'] = productivity_level

        # Check recent activity
        if recent_pubs >= 3:
            score += 0.1  # Bonus for recent productivity
            details['recent_activity'] = 'active'
        else:
            details['recent_activity'] = 'moderate'

        # Student preference matching (if provided)
        student_preference = student_profile.get('productivity_preference', 'any')
        if student_preference == 'high' and productivity_level in ['high', 'very_high']:
            score += 0.05
        elif student_preference == 'balanced' and productivity_level == 'moderate':
            score += 0.05

        return min(score, 1.0), details

    def technique_compatibility(
        self,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> Tuple[float, Dict]:
        """
        Evaluate technique match and learning opportunities

        Args:
            student_profile: Student profile with techniques
            faculty_profile: Faculty profile with techniques

        Returns:
            Tuple of (score, details_dict)
        """
        student_techniques = set(student_profile.get('techniques', []))
        faculty_techniques = set(faculty_profile.get('techniques', []))

        if not student_techniques or not faculty_techniques:
            return 0.5, {'overlap': 0, 'learning_opportunities': 0}

        # Calculate overlap
        overlap = student_techniques & faculty_techniques
        student_only = student_techniques - faculty_techniques
        faculty_only = faculty_techniques - student_techniques

        overlap_ratio = len(overlap) / len(student_techniques) if student_techniques else 0

        # Score calculation
        # - Good overlap (can contribute): positive
        # - Faculty has techniques student wants to learn: positive
        # - Some overlap, some new: ideal

        if overlap_ratio >= 0.5:
            # Strong overlap
            score = 0.8 + (overlap_ratio - 0.5) * 0.4  # 0.8 to 1.0
        elif overlap_ratio >= 0.2:
            # Moderate overlap with learning opportunities
            score = 0.6 + overlap_ratio * 0.5
        else:
            # Low overlap - may not be good fit
            score = 0.3 + overlap_ratio

        # Bonus for learning opportunities
        learning_bonus = min(len(faculty_only) / 5.0, 0.2)  # Up to 0.2 bonus
        score += learning_bonus

        details = {
            'overlap': len(overlap),
            'overlap_techniques': list(overlap),
            'learning_opportunities': len(faculty_only),
            'new_techniques': list(faculty_only)[:5]  # Top 5
        }

        return min(score, 1.0), details

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
        lab_size = faculty_profile.get('lab_size', 0)
        accepting = faculty_profile.get('accepting_students', True)

        details = {
            'lab_size': lab_size,
            'accepting_students': accepting,
            'environment': 'unknown'
        }

        # Not accepting students - major issue
        if not accepting:
            details['environment'] = 'not_accepting'
            return 0.2, details

        # Lab size scoring
        if lab_size == 0:
            # No data - neutral
            score = 0.6
            details['environment'] = 'unknown'
        elif 3 <= lab_size <= 8:
            # Ideal size - good mentorship potential
            score = 0.9
            details['environment'] = 'optimal'
        elif 9 <= lab_size <= 12:
            # Medium size - good but may have less 1-on-1 time
            score = 0.75
            details['environment'] = 'medium'
        elif lab_size < 3:
            # Small lab - could be good or risky
            score = 0.7
            details['environment'] = 'small'
        else:
            # Large lab - less individual attention
            score = 0.6
            details['environment'] = 'large'

        return score, details

    def career_development_score(
        self,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> Tuple[float, Dict]:
        """
        Assess career development potential

        Args:
            student_profile: Student profile
            faculty_profile: Faculty profile

        Returns:
            Tuple of (score, details_dict)
        """
        # Factors for career development
        h_index = faculty_profile.get('h_index', 0)
        collaborations = faculty_profile.get('collaborations', 0)
        career_stage = faculty_profile.get('career_stage', 'unknown')

        details = {
            'mentorship_potential': 'unknown',
            'networking_potential': 'unknown'
        }

        scores = []

        # Faculty reputation (h-index as proxy)
        if h_index >= 30:
            scores.append(1.0)
            details['mentorship_potential'] = 'excellent'
        elif h_index >= 20:
            scores.append(0.8)
            details['mentorship_potential'] = 'strong'
        elif h_index >= 10:
            scores.append(0.6)
            details['mentorship_potential'] = 'good'
        else:
            scores.append(0.4)
            details['mentorship_potential'] = 'developing'

        # Collaboration network
        if collaborations >= 10:
            scores.append(0.9)
            details['networking_potential'] = 'excellent'
        elif collaborations >= 5:
            scores.append(0.7)
            details['networking_potential'] = 'good'
        else:
            scores.append(0.5)
            details['networking_potential'] = 'moderate'

        # Career stage considerations
        if career_stage == 'full_professor':
            scores.append(0.9)  # Established, good for training
        elif career_stage == 'associate_professor':
            scores.append(0.85)  # Good balance
        elif career_stage == 'assistant_professor':
            scores.append(0.7)  # Enthusiastic but less established
        else:
            scores.append(0.6)

        score = sum(scores) / len(scores) if scores else 0.6

        return score, details

    def generate_match_explanation(
        self,
        match_score: float,
        component_scores: Dict,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> str:
        """
        Generate human-readable match explanation

        Creates narrative explanation covering:
        - Why this is a good/poor match
        - Specific research overlaps
        - Key strengths
        - Important considerations
        - Potential collaboration topics

        Args:
            match_score: Overall match score
            component_scores: Individual component scores
            student_profile: Student profile
            faculty_profile: Faculty profile

        Returns:
            Detailed explanation string (2-4 sentences)
        """
        faculty_name = faculty_profile.get('name', 'This faculty member')

        # Determine match quality
        if match_score >= 0.8:
            quality = "Excellent"
        elif match_score >= 0.7:
            quality = "Strong"
        elif match_score >= 0.6:
            quality = "Good"
        elif match_score >= 0.5:
            quality = "Moderate"
        else:
            quality = "Limited"

        explanation_parts = [f"{quality} match ({match_score:.0%}) with {faculty_name}."]

        # Highlight top scoring components
        top_components = sorted(
            component_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]

        for component, score in top_components:
            if score >= 0.7:
                component_name = component.replace('_', ' ').title()
                explanation_parts.append(f"{component_name}: {score:.0%}.")

        # Add specific details
        research_score = component_scores.get('research_alignment', 0)
        if research_score >= 0.8:
            explanation_parts.append("Very strong research alignment.")

        funding_score = component_scores.get('funding_stability', 0)
        if funding_score >= 0.7:
            explanation_parts.append("Well-funded lab with active grants.")
        elif funding_score < 0.4:
            explanation_parts.append("Limited funding - important consideration.")

        return " ".join(explanation_parts)

    def identify_match_strengths(
        self,
        component_scores: Dict,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> List[str]:
        """
        Identify specific strengths of this match

        Returns:
            List of strength statements
        """
        strengths = []

        # Research alignment
        if component_scores.get('research_alignment', 0) >= 0.8:
            strengths.append("Excellent research alignment with shared interests")

        # Technique overlap
        if component_scores.get('technique_match', 0) >= 0.7:
            student_tech = set(student_profile.get('techniques', []))
            faculty_tech = set(faculty_profile.get('techniques', []))
            overlap = student_tech & faculty_tech
            if overlap:
                tech_str = ", ".join(list(overlap)[:3])
                strengths.append(f"Strong technique overlap: {tech_str}")

        # Funding
        if component_scores.get('funding_stability', 0) >= 0.7:
            active_grants = faculty_profile.get('active_grants', 0)
            if active_grants > 0:
                strengths.append(f"Well-funded lab with {active_grants} active grant(s)")

        # Accepting students
        if faculty_profile.get('accepting_students', False):
            strengths.append("Currently accepting graduate students")

        # Productivity
        if component_scores.get('productivity_match', 0) >= 0.8:
            pub_count = faculty_profile.get('publication_count', 0)
            h_index = faculty_profile.get('h_index', 0)
            if pub_count > 0 and h_index > 0:
                strengths.append(f"Strong publication record ({pub_count} papers, h-index: {h_index})")

        return strengths[:5]  # Top 5 strengths

    def identify_considerations(
        self,
        component_scores: Dict,
        faculty_profile: Dict
    ) -> List[str]:
        """
        Identify potential concerns or considerations

        Returns:
            List of consideration statements
        """
        considerations = []

        # Funding concerns
        if component_scores.get('funding_stability', 0) < 0.5:
            considerations.append("Limited or uncertain funding - verify grant status")

        # Lab size considerations
        lab_size = faculty_profile.get('lab_size', 0)
        if lab_size > 12:
            considerations.append("Large lab - may have less individual mentorship time")
        elif lab_size < 3 and lab_size > 0:
            considerations.append("Small lab - verify mentorship structure")

        # Not accepting students
        if not faculty_profile.get('accepting_students', True):
            considerations.append("May not be accepting students currently")

        # Low productivity
        if component_scores.get('productivity_match', 0) < 0.5:
            considerations.append("Lower publication rate - verify research activity")

        # Low research alignment
        if component_scores.get('research_alignment', 0) < 0.5:
            considerations.append("Limited research overlap - verify mutual interest")

        return considerations[:5]  # Top 5 considerations

    def get_recommendation(self, overall_score: float, confidence: float) -> str:
        """
        Get categorical recommendation

        Args:
            overall_score: Overall match score
            confidence: Confidence in prediction

        Returns:
            One of: 'highly_recommended', 'recommended', 'consider', 'not_recommended'
        """
        # Adjust score by confidence
        adjusted_score = overall_score * confidence

        if adjusted_score >= 0.75:
            return 'highly_recommended'
        elif adjusted_score >= 0.6:
            return 'recommended'
        elif adjusted_score >= 0.45:
            return 'consider'
        else:
            return 'not_recommended'

    def _calculate_confidence(
        self,
        student_profile: Dict,
        faculty_profile: Dict,
        component_scores: Dict
    ) -> float:
        """
        Calculate confidence based on data completeness

        Args:
            student_profile: Student profile
            faculty_profile: Faculty profile
            component_scores: Component scores

        Returns:
            Confidence score [0, 1]
        """
        # Check data completeness
        completeness_factors = []

        # Student data
        if student_profile.get('research_interests'):
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.5)

        if student_profile.get('techniques'):
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.7)

        # Faculty data
        required_fields = [
            'research_summary', 'publications', 'grants',
            'h_index', 'techniques'
        ]

        for field in required_fields:
            if faculty_profile.get(field):
                completeness_factors.append(1.0)
            else:
                completeness_factors.append(0.6)

        # Average completeness
        confidence = sum(completeness_factors) / len(completeness_factors)

        return min(max(confidence, 0.0), 1.0)
