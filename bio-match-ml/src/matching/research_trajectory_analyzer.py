"""
Analyze research evolution patterns and predict future directions
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryAnalysis:
    """Research trajectory analysis results"""
    trending_topics: List[str]
    declining_topics: List[str]
    stable_core: List[str]
    innovation_rate: float
    pivot_points: List['PivotPoint']
    research_phases: List['ResearchPhase']
    predicted_directions: List['PredictedDirection']


@dataclass
class ResearchPhase:
    """Single phase in research career"""
    phase_type: str
    years: Tuple[int, int]
    primary_topics: List[str]
    characteristics: str


@dataclass
class TrendData:
    """Topic trend information"""
    recent_count: int
    historical_count: int
    trend: str
    growth_rate: float
    first_appearance: int


@dataclass
class PredictedDirection:
    """Predicted future research direction"""
    topic: str
    confidence: float
    rationale: str
    timeframe: str


@dataclass
class PivotPoint:
    """Research pivot point"""
    year: int
    description: str
    old_focus: List[str]
    new_focus: List[str]


@dataclass
class AlignmentAssessment:
    """Student-faculty trajectory alignment"""
    alignment_score: float
    alignment_type: str
    opportunities: List[str]
    risks: List[str]


class ResearchTrajectoryAnalyzer:
    """
    Analyze research evolution patterns and predict future directions
    """

    def __init__(self, entity_recognizer=None, topic_classifier=None):
        """
        Initialize trajectory analyzer

        Args:
            entity_recognizer: For extracting entities from publications
            topic_classifier: For classifying research topics
        """
        self.entity_recognizer = entity_recognizer
        self.topic_classifier = topic_classifier

        logger.info("ResearchTrajectoryAnalyzer initialized")

    def analyze_trajectory(
        self,
        publication_history: List[Dict],
        time_windows: List[int] = [2, 5, 10]
    ) -> TrajectoryAnalysis:
        """
        Analyze research trajectory from publication history

        Args:
            publication_history: List of publications with:
                - title, abstract, year, keywords
            time_windows: Year windows for trend analysis [recent, medium, long]

        Returns:
            TrajectoryAnalysis object with comprehensive trajectory information
        """
        if not publication_history:
            logger.warning("Empty publication history provided")
            return self._empty_trajectory()

        logger.info(f"Analyzing trajectory for {len(publication_history)} publications")

        # Sort publications by year
        sorted_pubs = sorted(
            publication_history,
            key=lambda x: x.get('year', 0)
        )

        # Extract topics from publications over time
        topics_by_year = self._extract_topics_over_time(sorted_pubs)

        # Calculate topic trends
        trending, declining, stable = self._calculate_topic_trends(
            topics_by_year,
            time_window=time_windows[0]
        )

        # Identify research phases
        research_phases = self.identify_research_phases(sorted_pubs)

        # Calculate innovation rate
        innovation_rate = self.calculate_innovation_rate(sorted_pubs)

        # Identify pivot points
        pivot_points = self.identify_pivot_points(sorted_pubs)

        # Predict future directions
        predicted_directions = self.predict_future_research(
            TrajectoryAnalysis(
                trending_topics=trending,
                declining_topics=declining,
                stable_core=stable,
                innovation_rate=innovation_rate,
                pivot_points=pivot_points,
                research_phases=research_phases,
                predicted_directions=[]  # Will be filled
            )
        )

        trajectory = TrajectoryAnalysis(
            trending_topics=trending,
            declining_topics=declining,
            stable_core=stable,
            innovation_rate=innovation_rate,
            pivot_points=pivot_points,
            research_phases=research_phases,
            predicted_directions=predicted_directions
        )

        logger.info(f"Trajectory analysis complete: {len(trending)} trending, {len(declining)} declining")

        return trajectory

    def identify_research_phases(
        self,
        publications: List[Dict]
    ) -> List[ResearchPhase]:
        """
        Detect distinct phases in research career

        Phases:
        - Early Exploration: Trying different topics (first 3-5 years)
        - Focus Consolidation: Narrowing to core areas
        - Mature Expansion: Branching from established core
        - Late-Career Mentorship: Broader, collaborative work

        Args:
            publications: Publication history

        Returns:
            List of ResearchPhase objects
        """
        if not publications:
            return []

        phases = []

        # Get year range
        years = [p.get('year', 0) for p in publications if p.get('year')]
        if not years:
            return []

        start_year = min(years)
        end_year = max(years)
        career_length = end_year - start_year

        if career_length < 3:
            # Too short to identify phases
            phases.append(ResearchPhase(
                phase_type='Early Career',
                years=(start_year, end_year),
                primary_topics=self._extract_primary_topics(publications),
                characteristics='Establishing research program'
            ))
            return phases

        # Divide into time periods
        if career_length <= 10:
            # Early career
            phases.append(ResearchPhase(
                phase_type='Early Career',
                years=(start_year, end_year),
                primary_topics=self._extract_primary_topics(publications),
                characteristics='Building research foundation'
            ))
        else:
            # Multiple phases
            early_cutoff = start_year + 5
            mid_cutoff = start_year + (career_length * 2 // 3)

            # Early phase
            early_pubs = [p for p in publications if p.get('year', 0) < early_cutoff]
            if early_pubs:
                phases.append(ResearchPhase(
                    phase_type='Early Exploration',
                    years=(start_year, early_cutoff - 1),
                    primary_topics=self._extract_primary_topics(early_pubs),
                    characteristics='Exploring different research directions'
                ))

            # Mid phase
            mid_pubs = [
                p for p in publications
                if early_cutoff <= p.get('year', 0) < mid_cutoff
            ]
            if mid_pubs:
                phases.append(ResearchPhase(
                    phase_type='Focus Consolidation',
                    years=(early_cutoff, mid_cutoff - 1),
                    primary_topics=self._extract_primary_topics(mid_pubs),
                    characteristics='Consolidating research focus and building expertise'
                ))

            # Recent phase
            recent_pubs = [p for p in publications if p.get('year', 0) >= mid_cutoff]
            if recent_pubs:
                topics = self._extract_primary_topics(recent_pubs)
                # Determine if expanding or maintaining
                innovation = self.calculate_innovation_rate(recent_pubs, window_size=3)

                if innovation > 0.3:
                    phase_type = 'Mature Expansion'
                    characteristics = 'Expanding into new areas while maintaining core'
                else:
                    phase_type = 'Focused Expertise'
                    characteristics = 'Deepening expertise in core areas'

                phases.append(ResearchPhase(
                    phase_type=phase_type,
                    years=(mid_cutoff, end_year),
                    primary_topics=topics,
                    characteristics=characteristics
                ))

        return phases

    def calculate_topic_trends(
        self,
        publications: List[Dict],
        time_window: int = 5
    ) -> Dict[str, TrendData]:
        """
        Calculate trending vs declining topics

        Args:
            publications: Publication list
            time_window: Years to consider recent

        Returns:
            Dict mapping topic → TrendData
        """
        topics_by_year = self._extract_topics_over_time(publications)

        if not topics_by_year:
            return {}

        years = sorted(topics_by_year.keys())
        if not years:
            return {}

        current_year = max(years)
        cutoff_year = current_year - time_window

        # Count topics in recent vs historical periods
        recent_counts = Counter()
        historical_counts = Counter()
        first_appearance = {}

        for year, topics in topics_by_year.items():
            for topic in topics:
                if topic not in first_appearance:
                    first_appearance[topic] = year

                if year >= cutoff_year:
                    recent_counts[topic] += 1
                else:
                    historical_counts[topic] += 1

        # Calculate trends
        trend_data = {}

        all_topics = set(recent_counts.keys()) | set(historical_counts.keys())

        for topic in all_topics:
            recent = recent_counts.get(topic, 0)
            historical = historical_counts.get(topic, 0)

            # Determine trend
            if historical == 0 and recent > 0:
                trend = 'emerging'
                growth_rate = float('inf')
            elif recent == 0 and historical > 0:
                trend = 'declining'
                growth_rate = -1.0
            else:
                # Calculate growth rate
                if historical > 0:
                    growth_rate = (recent - historical) / historical
                else:
                    growth_rate = 0.0

                if growth_rate > 0.5:
                    trend = 'increasing'
                elif growth_rate < -0.5:
                    trend = 'declining'
                else:
                    trend = 'stable'

            trend_data[topic] = TrendData(
                recent_count=recent,
                historical_count=historical,
                trend=trend,
                growth_rate=growth_rate if growth_rate != float('inf') else 999.0,
                first_appearance=first_appearance.get(topic, 0)
            )

        return trend_data

    def predict_future_research(
        self,
        current_trajectory: TrajectoryAnalysis,
        field_trends: Optional[Dict] = None
    ) -> List[PredictedDirection]:
        """
        Predict next research directions

        Prediction based on:
        - Current trending topics
        - Citation patterns (what they're citing recently)
        - Funding agency priorities (if grant data available)
        - Collaborator influences
        - Field-wide trends

        Args:
            current_trajectory: Current trajectory analysis
            field_trends: Optional field-wide trend data

        Returns:
            List of PredictedDirection objects
        """
        predictions = []

        # Predict based on trending topics
        for topic in current_trajectory.trending_topics[:3]:
            # High confidence for topics already trending
            predictions.append(PredictedDirection(
                topic=f"Continued work in {topic}",
                confidence=0.8,
                rationale=f"Strong upward trend in {topic} over recent years",
                timeframe='1-2 years'
            ))

        # Predict combinations of stable core + emerging techniques
        if current_trajectory.stable_core and current_trajectory.trending_topics:
            core_topic = current_trajectory.stable_core[0]
            trending_topic = current_trajectory.trending_topics[0]

            if core_topic != trending_topic:
                predictions.append(PredictedDirection(
                    topic=f"{trending_topic} applied to {core_topic}",
                    confidence=0.7,
                    rationale=f"Integration of emerging {trending_topic} with established {core_topic} expertise",
                    timeframe='2-3 years'
                ))

        # Predict based on innovation rate
        if current_trajectory.innovation_rate > 0.4:
            predictions.append(PredictedDirection(
                topic="Exploration of novel research areas",
                confidence=0.6,
                rationale="High innovation rate suggests openness to new directions",
                timeframe='3-5 years'
            ))

        # Predict based on recent pivot points
        if current_trajectory.pivot_points:
            recent_pivot = current_trajectory.pivot_points[-1]
            if recent_pivot.new_focus:
                predictions.append(PredictedDirection(
                    topic=f"Expansion in {recent_pivot.new_focus[0]}",
                    confidence=0.75,
                    rationale=f"Recent pivot to {recent_pivot.new_focus[0]} in {recent_pivot.year}",
                    timeframe='1-2 years'
                ))

        # Sort by confidence
        predictions.sort(key=lambda x: x.confidence, reverse=True)

        return predictions[:5]  # Top 5 predictions

    def calculate_innovation_rate(
        self,
        publications: List[Dict],
        window_size: int = 3
    ) -> float:
        """
        Calculate rate of new topic adoption

        Higher rate = more exploratory, lower = more focused

        Args:
            publications: Publication history
            window_size: Years per window

        Returns:
            Innovation rate [0, 1]
            0 = all work in same topics
            1 = every paper in completely new topic
        """
        if not publications or len(publications) < 2:
            return 0.5

        topics_by_year = self._extract_topics_over_time(publications)

        if not topics_by_year:
            return 0.5

        years = sorted(topics_by_year.keys())

        # Calculate rolling window innovation
        innovation_scores = []

        for i in range(len(years) - window_size):
            window_years = years[i:i + window_size]
            next_year = years[i + window_size] if i + window_size < len(years) else None

            if not next_year:
                continue

            # Topics in window
            window_topics = set()
            for year in window_years:
                window_topics.update(topics_by_year.get(year, []))

            # Topics in next period
            next_topics = set(topics_by_year.get(next_year, []))

            # Calculate proportion of new topics
            if next_topics:
                new_topics = next_topics - window_topics
                innovation = len(new_topics) / len(next_topics)
                innovation_scores.append(innovation)

        if innovation_scores:
            return sum(innovation_scores) / len(innovation_scores)
        else:
            return 0.5

    def identify_pivot_points(
        self,
        publications: List[Dict],
        threshold: float = 0.3
    ) -> List[PivotPoint]:
        """
        Identify years with significant research shifts

        Detects:
        - New technique adoption
        - Organism changes
        - Topic shifts
        - Methodology changes

        Args:
            publications: Publication list
            threshold: Similarity threshold for detecting pivots

        Returns:
            List of PivotPoint objects
        """
        if len(publications) < 2:
            return []

        topics_by_year = self._extract_topics_over_time(publications)
        years = sorted(topics_by_year.keys())

        pivot_points = []

        for i in range(1, len(years)):
            prev_year = years[i - 1]
            curr_year = years[i]

            prev_topics = set(topics_by_year.get(prev_year, []))
            curr_topics = set(topics_by_year.get(curr_year, []))

            if not prev_topics or not curr_topics:
                continue

            # Calculate Jaccard similarity
            overlap = len(prev_topics & curr_topics)
            union = len(prev_topics | curr_topics)
            similarity = overlap / union if union > 0 else 0

            # Significant shift if similarity is low
            if similarity < threshold:
                # Identify what changed
                new_topics = list(curr_topics - prev_topics)
                old_topics = list(prev_topics - curr_topics)

                if new_topics or old_topics:
                    description = self._describe_pivot(old_topics, new_topics, curr_year)

                    pivot_points.append(PivotPoint(
                        year=curr_year,
                        description=description,
                        old_focus=old_topics[:3],
                        new_focus=new_topics[:3]
                    ))

        return pivot_points

    def assess_student_alignment(
        self,
        student_interests: Dict,
        faculty_trajectory: TrajectoryAnalysis
    ) -> AlignmentAssessment:
        """
        Assess how student interests align with faculty trajectory

        Considers:
        - Alignment with current trending topics (good)
        - Alignment with declining topics (warning)
        - Alignment with predicted future (excellent)
        - Alignment with stable core (safe)

        Args:
            student_interests: Student research interests
            faculty_trajectory: Faculty trajectory analysis

        Returns:
            AlignmentAssessment with detailed analysis
        """
        # Extract student topics
        student_topics = set()

        if isinstance(student_interests, dict):
            student_topics.update(student_interests.get('topics', []))
            student_topics.update(student_interests.get('techniques', []))

            # Extract from research interests text
            if student_interests.get('research_interests'):
                text_topics = self._extract_topics_from_text(
                    student_interests['research_interests']
                )
                student_topics.update(text_topics)
        elif isinstance(student_interests, str):
            text_topics = self._extract_topics_from_text(student_interests)
            student_topics.update(text_topics)

        if not student_topics:
            return AlignmentAssessment(
                alignment_score=0.5,
                alignment_type='unknown',
                opportunities=[],
                risks=[]
            )

        # Calculate alignments with different trajectory components
        trending_alignment = self._calculate_overlap(
            student_topics,
            set(faculty_trajectory.trending_topics)
        )

        declining_alignment = self._calculate_overlap(
            student_topics,
            set(faculty_trajectory.declining_topics)
        )

        stable_alignment = self._calculate_overlap(
            student_topics,
            set(faculty_trajectory.stable_core)
        )

        # Future alignment
        predicted_topics = {
            p.topic for p in faculty_trajectory.predicted_directions
        }
        future_alignment = self._calculate_overlap(student_topics, predicted_topics)

        # Determine primary alignment type and score
        alignments = {
            'trending': trending_alignment * 1.3,  # Bonus
            'stable': stable_alignment * 0.9,
            'future': future_alignment * 1.5,  # Highest bonus
            'declining': declining_alignment * 0.5  # Penalty
        }

        best_type = max(alignments.items(), key=lambda x: x[1])
        alignment_score = min(best_type[1], 1.0)

        # Generate opportunities and risks
        opportunities = []
        risks = []

        if trending_alignment > 0.3:
            opportunities.append("Join faculty's expansion into trending research areas")
            opportunities.append("Contribute to growing research direction")

        if stable_alignment > 0.4:
            opportunities.append("Build on faculty's established expertise")
            opportunities.append("Access to proven methodologies and infrastructure")

        if future_alignment > 0.2:
            opportunities.append("Align with predicted future research directions")
            opportunities.append("Be at forefront of faculty's research evolution")

        if declining_alignment > 0.5:
            risks.append("Interest aligns with declining research areas")
            risks.append("Faculty may be moving away from your core interests")

        if faculty_trajectory.innovation_rate > 0.6 and stable_alignment > declining_alignment:
            risks.append("Faculty frequently pivots - your interests may not remain focal")

        if len(faculty_trajectory.pivot_points) > 2 and max(alignments.values()) < 0.4:
            risks.append("Multiple pivot points suggest evolving research focus")

        return AlignmentAssessment(
            alignment_score=alignment_score,
            alignment_type=best_type[0],
            opportunities=opportunities[:4],
            risks=risks[:3]
        )

    # Helper methods

    def _extract_topics_over_time(self, publications: List[Dict]) -> Dict[int, List[str]]:
        """Extract topics from publications grouped by year"""
        topics_by_year = defaultdict(list)

        for pub in publications:
            year = pub.get('year')
            if not year:
                continue

            # Extract topics from various sources
            topics = []

            # Keywords
            if pub.get('keywords'):
                keywords = pub['keywords']
                if isinstance(keywords, list):
                    topics.extend(keywords)
                elif isinstance(keywords, str):
                    topics.extend([k.strip() for k in keywords.split(',')])

            # Extract from title and abstract
            text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
            if text.strip() and self.entity_recognizer:
                try:
                    entities = self.entity_recognizer.extract_all_entities(text)
                    # Add techniques
                    if entities.get('techniques'):
                        topics.extend([t['name'] for t in entities['techniques']])
                except:
                    pass

            # Use topic classifier
            if self.topic_classifier and text.strip():
                try:
                    classification = self.topic_classifier.classify_research_area(text)
                    if classification.get('primary_area'):
                        topics.append(classification['primary_area'])
                    topics.extend(classification.get('specific_topics', []))
                except:
                    pass

            # Deduplicate and add to year
            topics_by_year[year].extend(list(set(topics)))

        return dict(topics_by_year)

    def _calculate_topic_trends(
        self,
        topics_by_year: Dict[int, List[str]],
        time_window: int
    ) -> Tuple[List[str], List[str], List[str]]:
        """Calculate trending, declining, and stable topics"""
        if not topics_by_year:
            return [], [], []

        years = sorted(topics_by_year.keys())
        if not years:
            return [], [], []

        current_year = max(years)
        cutoff_year = current_year - time_window

        recent_topics = Counter()
        historical_topics = Counter()

        for year, topics in topics_by_year.items():
            for topic in topics:
                if year >= cutoff_year:
                    recent_topics[topic] += 1
                else:
                    historical_topics[topic] += 1

        trending = []
        declining = []
        stable = []

        all_topics = set(recent_topics.keys()) | set(historical_topics.keys())

        for topic in all_topics:
            recent = recent_topics.get(topic, 0)
            historical = historical_topics.get(topic, 0)

            if recent > historical * 1.5:
                trending.append(topic)
            elif recent < historical * 0.5:
                declining.append(topic)
            else:
                if recent > 0 and historical > 0:
                    stable.append(topic)

        # Sort by frequency
        trending.sort(key=lambda x: recent_topics[x], reverse=True)
        declining.sort(key=lambda x: historical_topics[x], reverse=True)
        stable.sort(key=lambda x: recent_topics[x] + historical_topics[x], reverse=True)

        return trending[:10], declining[:10], stable[:10]

    def _extract_primary_topics(self, publications: List[Dict], limit: int = 5) -> List[str]:
        """Extract most common topics from publications"""
        topics = []

        for pub in publications:
            if pub.get('keywords'):
                keywords = pub['keywords']
                if isinstance(keywords, list):
                    topics.extend(keywords)
                elif isinstance(keywords, str):
                    topics.extend([k.strip() for k in keywords.split(',')])

        # Count and return most common
        topic_counts = Counter(topics)
        return [topic for topic, count in topic_counts.most_common(limit)]

    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from text using simple keyword matching"""
        # Simple keyword extraction
        keywords = []

        # Biology domain keywords
        domain_terms = [
            'crispr', 'gene editing', 'rna-seq', 'cancer', 'neuroscience',
            'immunology', 'genomics', 'proteomics', 'cell biology', 'development',
            'metabolism', 'signaling', 'stem cell', 'organoid', 'mouse', 'human'
        ]

        text_lower = text.lower()

        for term in domain_terms:
            if term in text_lower:
                keywords.append(term)

        return keywords

    def _calculate_overlap(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between sets"""
        if not set1 or not set2:
            return 0.0

        # Normalize to lowercase for comparison
        set1_lower = {str(s).lower() for s in set1}
        set2_lower = {str(s).lower() for s in set2}

        intersection = len(set1_lower & set2_lower)
        union = len(set1_lower | set2_lower)

        return intersection / union if union > 0 else 0.0

    def _describe_pivot(self, old_topics: List[str], new_topics: List[str], year: int) -> str:
        """Generate description of research pivot"""
        if new_topics and old_topics:
            return f"Shift from {old_topics[0]} to {new_topics[0]}"
        elif new_topics:
            return f"New focus on {new_topics[0]}"
        elif old_topics:
            return f"Moving away from {old_topics[0]}"
        else:
            return f"Research direction change in {year}"

    def _empty_trajectory(self) -> TrajectoryAnalysis:
        """Return empty trajectory analysis"""
        return TrajectoryAnalysis(
            trending_topics=[],
            declining_topics=[],
            stable_core=[],
            innovation_rate=0.5,
            pivot_points=[],
            research_phases=[],
            predicted_directions=[]
        )
