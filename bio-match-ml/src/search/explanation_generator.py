"""
Generate natural language explanations for search results and matches
"""
import logging
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generate human-readable explanations for search results
    """

    def __init__(self, entity_recognizer=None):
        """
        Initialize explanation generator

        Args:
            entity_recognizer: BioEntityRecognizer for entity extraction
        """
        self.entity_recognizer = entity_recognizer

        logger.info("ExplanationGenerator initialized")

    def explain_search_result(
        self,
        query: str,
        result: Dict,
        rank: int,
        query_entities: Optional[Dict] = None
    ) -> str:
        """
        Generate explanation for why this result matched the query

        Args:
            query: Original search query
            result: Search result with metadata and scores
            rank: Result position (1-indexed)
            query_entities: Entities extracted from query

        Returns:
            Natural language explanation

        Example output:
            "Ranked #2 with 89% match. Dr. Smith's research on CRISPR-mediated
             gene editing in neuronal cells directly aligns with your query.
             Strong expertise in CRISPR (15 papers) and neuroscience (h-index: 32).
             Currently accepting PhD students with active NIH funding through 2027."
        """
        metadata = result.get('metadata', {})
        score = result.get('final_score', result.get('score', 0))

        parts = []

        # Opening with rank and score
        parts.append(f"Ranked #{rank} with {score:.0%} match.")

        # Faculty name and research alignment
        name = metadata.get('name', 'This researcher')
        research = metadata.get('research_summary', '')

        if research:
            # Extract key overlap
            common_ground = self._extract_key_overlap(query, research, query_entities, metadata)

            if common_ground:
                parts.append(f"{name}'s research on {common_ground} aligns with your query.")
            else:
                parts.append(f"{name}'s work is relevant to your search.")

        # Expertise indicators
        expertise_parts = []

        # Publication metrics
        pub_count = metadata.get('publication_count', 0)
        h_index = metadata.get('h_index', 0)

        if pub_count > 30 and h_index > 20:
            expertise_parts.append(f"{pub_count} papers, h-index: {h_index}")

        # Funding
        active_grants = metadata.get('active_grants', 0)
        if active_grants > 0:
            funding_info = self._describe_funding(metadata)
            if funding_info:
                expertise_parts.append(funding_info)

        # Accepting students
        if metadata.get('accepting_students'):
            expertise_parts.append("currently accepting students")

        if expertise_parts:
            parts.append(" ".join(expertise_parts).capitalize() + ".")

        return " ".join(parts)

    def explain_match_score(
        self,
        student_profile: Dict,
        faculty_profile: Dict,
        match_score: Any,  # MatchScore object
        detail_level: str = 'standard'
    ) -> str:
        """
        Generate detailed match explanation

        Args:
            student_profile: Student research profile
            faculty_profile: Faculty profile
            match_score: MatchScore object with component scores
            detail_level: 'brief', 'standard', or 'detailed'

        Returns:
            Formatted explanation

        Example output (standard):
            "Excellent match (87%) with Dr. Smith's lab based on:

             Research Alignment (92%): Very strong overlap in CRISPR gene editing
             and cancer biology. Your interest in therapeutic applications aligns
             perfectly with Dr. Smith's recent focus on CRISPR screens for cancer
             vulnerabilities.

             Funding Stability (85%): Well-funded lab with $1.5M NIH R01 through
             2028, ensuring secure research support.

             Lab Environment (75%): Medium-sized lab (10 members) with good
             mentorship opportunities. Currently accepting 1-2 PhD students.

             Recommendations:
             • Reach out about ongoing CRISPR-Cas9 projects
             • Prepare strong application - competitive lab
             • Consider taking lab's graduate seminar next semester"
        """
        if detail_level == 'brief':
            return self._brief_explanation(student_profile, faculty_profile, match_score)
        elif detail_level == 'detailed':
            return self._detailed_explanation(student_profile, faculty_profile, match_score)
        else:
            return self._standard_explanation(student_profile, faculty_profile, match_score)

    def _brief_explanation(
        self,
        student_profile: Dict,
        faculty_profile: Dict,
        match_score: Any
    ) -> str:
        """Generate brief explanation (1-2 sentences)"""
        overall = match_score.overall_score
        faculty_name = faculty_profile.get('name', 'This faculty member')

        quality = self._score_to_quality(overall)

        # Get top strength
        strengths = match_score.strengths
        top_strength = strengths[0] if strengths else "good research fit"

        return f"{quality} match ({overall:.0%}) with {faculty_name}. {top_strength}."

    def _standard_explanation(
        self,
        student_profile: Dict,
        faculty_profile: Dict,
        match_score: Any
    ) -> str:
        """Generate standard explanation (2-3 paragraphs)"""
        lines = []

        overall = match_score.overall_score
        faculty_name = faculty_profile.get('name', 'This faculty member')
        quality = self._score_to_quality(overall)

        # Header
        lines.append(f"{quality} match ({overall:.0%}) with {faculty_name}'s lab.\n")

        # Component breakdowns (top 3)
        component_scores = match_score.component_scores
        top_components = sorted(
            component_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        for component, score in top_components:
            component_name = component.replace('_', ' ').title()
            explanation = self._explain_component(
                component,
                score,
                student_profile,
                faculty_profile
            )
            lines.append(f"\n{component_name} ({score:.0%}): {explanation}")

        # Recommendations
        if match_score.strengths:
            lines.append("\n\nKey Strengths:")
            for strength in match_score.strengths[:3]:
                lines.append(f"• {strength}")

        if match_score.considerations:
            lines.append("\n\nConsiderations:")
            for consideration in match_score.considerations[:2]:
                lines.append(f"• {consideration}")

        return "".join(lines)

    def _detailed_explanation(
        self,
        student_profile: Dict,
        faculty_profile: Dict,
        match_score: Any
    ) -> str:
        """Generate detailed explanation (full breakdown)"""
        lines = []

        overall = match_score.overall_score
        faculty_name = faculty_profile.get('name', 'This faculty member')
        quality = self._score_to_quality(overall)

        # Header
        lines.append(f"## Match Analysis: {quality} Match ({overall:.0%})\n\n")
        lines.append(f"Faculty: {faculty_name}\n")
        lines.append(f"Overall Score: {overall:.1%}\n")
        lines.append(f"Confidence: {match_score.confidence:.1%}\n")
        lines.append(f"Recommendation: {match_score.recommendation.replace('_', ' ').title()}\n\n")

        # All component scores
        lines.append("### Component Scores\n\n")
        component_scores = match_score.component_scores

        for component, score in sorted(component_scores.items(), key=lambda x: x[1], reverse=True):
            component_name = component.replace('_', ' ').title()
            bar = self._score_bar(score)
            explanation = self._explain_component(
                component,
                score,
                student_profile,
                faculty_profile
            )
            lines.append(f"**{component_name}**: {score:.1%} {bar}\n")
            lines.append(f"  {explanation}\n\n")

        # Strengths
        if match_score.strengths:
            lines.append("### Strengths\n\n")
            for strength in match_score.strengths:
                lines.append(f"✓ {strength}\n")
            lines.append("\n")

        # Considerations
        if match_score.considerations:
            lines.append("### Considerations\n\n")
            for consideration in match_score.considerations:
                lines.append(f"⚠ {consideration}\n")
            lines.append("\n")

        # Action items
        actions = self.generate_recommendation_action_items(overall, faculty_profile)
        if actions:
            lines.append("### Recommended Actions\n\n")
            for action in actions:
                lines.append(f"• {action}\n")

        return "".join(lines)

    def _explain_component(
        self,
        component: str,
        score: float,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> str:
        """Explain a specific component score"""
        if component == 'research_alignment':
            return self._explain_research_alignment(score, student_profile, faculty_profile)
        elif component == 'funding_stability':
            return self._explain_funding(score, faculty_profile)
        elif component == 'technique_match':
            return self._explain_techniques(score, student_profile, faculty_profile)
        elif component == 'lab_environment':
            return self._explain_lab_environment(score, faculty_profile)
        elif component == 'productivity_match':
            return self._explain_productivity(score, faculty_profile)
        elif component == 'career_development':
            return self._explain_career_development(score, faculty_profile)
        else:
            return self._generic_score_explanation(score)

    def _explain_research_alignment(
        self,
        score: float,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> str:
        """Explain research alignment score"""
        if score >= 0.8:
            quality = "Excellent overlap"
        elif score >= 0.6:
            quality = "Strong overlap"
        elif score >= 0.4:
            quality = "Moderate overlap"
        else:
            quality = "Limited overlap"

        # Find common ground
        common_ground = self.identify_common_ground(
            student_profile.get('topics', []),
            faculty_profile.get('topics', [])
        )

        if common_ground.get('topics'):
            topics_str = ", ".join(common_ground['topics'][:3])
            return f"{quality} in {topics_str}."
        else:
            return f"{quality} in research interests."

    def _explain_funding(self, score: float, faculty_profile: Dict) -> str:
        """Explain funding score"""
        active_grants = faculty_profile.get('active_grants', 0)
        total_funding = faculty_profile.get('total_funding', 0)

        if score >= 0.8:
            if active_grants > 0:
                return f"Excellent funding with {active_grants} active grant(s) totaling ${total_funding:,.0f}."
            else:
                return "Excellent funding status."
        elif score >= 0.6:
            return "Good funding with active grants."
        elif score >= 0.4:
            return "Moderate funding - verify current status."
        else:
            return "Limited or uncertain funding - important to verify."

    def _explain_techniques(
        self,
        score: float,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> str:
        """Explain technique compatibility"""
        student_tech = set(student_profile.get('techniques', []))
        faculty_tech = set(faculty_profile.get('techniques', []))
        overlap = student_tech & faculty_tech

        if score >= 0.7 and overlap:
            overlap_str = ", ".join(list(overlap)[:3])
            return f"Strong match in {overlap_str}. Good opportunities to contribute and learn."
        elif score >= 0.5:
            return "Moderate technique overlap with learning opportunities."
        else:
            return "Limited technique overlap - different methodological focus."

    def _explain_lab_environment(self, score: float, faculty_profile: Dict) -> str:
        """Explain lab environment score"""
        lab_size = faculty_profile.get('lab_size', 0)
        accepting = faculty_profile.get('accepting_students', True)

        if not accepting:
            return "Not currently accepting students."

        if score >= 0.8:
            return f"Optimal lab environment (size: {lab_size}) with good mentorship potential."
        elif score >= 0.6:
            return f"Good lab environment (size: {lab_size})."
        else:
            return f"Lab size ({lab_size}) may impact individual attention."

    def _explain_productivity(self, score: float, faculty_profile: Dict) -> str:
        """Explain productivity compatibility"""
        pub_count = faculty_profile.get('publication_count', 0)
        h_index = faculty_profile.get('h_index', 0)

        if score >= 0.8:
            return f"Highly productive lab ({pub_count} papers, h-index: {h_index})."
        elif score >= 0.6:
            return f"Good publication record ({pub_count} papers)."
        else:
            return "Developing publication record."

    def _explain_career_development(self, score: float, faculty_profile: Dict) -> str:
        """Explain career development potential"""
        h_index = faculty_profile.get('h_index', 0)

        if score >= 0.8:
            return f"Excellent mentorship potential (h-index: {h_index}). Strong network for career development."
        elif score >= 0.6:
            return "Good career development opportunities."
        else:
            return "Developing network and reputation."

    def _generic_score_explanation(self, score: float) -> str:
        """Generic explanation for unknown components"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Strong"
        elif score >= 0.4:
            return "Moderate"
        else:
            return "Limited"

    def identify_common_ground(
        self,
        student_entities: Any,
        faculty_entities: Any
    ) -> Dict[str, List[str]]:
        """
        Identify specific overlaps between student and faculty

        Args:
            student_entities: Entities from student profile (can be list or dict)
            faculty_entities: Entities from faculty profile (can be list or dict)

        Returns:
            Dict of common entities by type:
            {
                'techniques': ['CRISPR', 'RNA-seq'],
                'organisms': ['mouse'],
                'topics': ['gene editing', 'cancer']
            }
        """
        common = {
            'techniques': [],
            'organisms': [],
            'topics': []
        }

        # Handle both list and dict formats
        if isinstance(student_entities, dict) and isinstance(faculty_entities, dict):
            for category in ['techniques', 'organisms', 'topics']:
                student_set = set(student_entities.get(category, []))
                faculty_set = set(faculty_entities.get(category, []))
                overlap = student_set & faculty_set
                common[category] = list(overlap)
        elif isinstance(student_entities, list) and isinstance(faculty_entities, list):
            # Simple list overlap
            student_set = set(student_entities)
            faculty_set = set(faculty_entities)
            overlap = student_set & faculty_set
            common['topics'] = list(overlap)

        return common

    def generate_recommendation_action_items(
        self,
        match_score: float,
        faculty_profile: Dict
    ) -> List[str]:
        """
        Generate actionable recommendations for student

        Args:
            match_score: Overall match score
            faculty_profile: Faculty information

        Returns:
            List of action items

        Example:
            [
                "Email Dr. Smith expressing interest in CRISPR projects",
                "Read recent papers: 'CRISPR screens...' (2024)",
                "Prepare research statement highlighting RNA-seq experience",
                "Attend lab's weekly seminar (Thursdays 4pm)"
            ]
        """
        actions = []
        name = faculty_profile.get('name', 'the faculty member')

        # Contact recommendation based on match quality
        if match_score >= 0.7:
            actions.append(f"Reach out to {name} expressing interest in their research")
            actions.append("Prepare a strong research statement highlighting relevant experience")
        elif match_score >= 0.5:
            actions.append(f"Consider reaching out to {name} to learn more")
        else:
            actions.append(f"Research {name}'s work further to assess fit")

        # Read recent papers
        publications = faculty_profile.get('publications', [])
        if publications:
            recent_pubs = [p for p in publications if isinstance(p, dict) and p.get('year', 0) >= 2023]
            if recent_pubs:
                pub_title = recent_pubs[0].get('title', '')[:50]
                actions.append(f"Read recent papers, including '{pub_title}...'")

        # Lab-specific actions
        if faculty_profile.get('lab_website'):
            actions.append("Visit lab website to learn about ongoing projects")

        if faculty_profile.get('accepting_students'):
            actions.append("Lab is accepting students - timely to apply")

        # Funding-related
        active_grants = faculty_profile.get('active_grants', 0)
        if active_grants > 0:
            actions.append("Verify current grant status and project timelines")

        return actions[:5]  # Return top 5

    def explain_ranking_factors(
        self,
        result: Dict,
        component_scores: Dict[str, float]
    ) -> str:
        """
        Break down ranking score components

        Args:
            result: Search result
            component_scores: Individual score components

        Returns:
            Formatted explanation of scoring

        Example:
            "Ranking Breakdown:
             • Semantic Similarity: 0.89 (Very High)
             • Research Productivity: 0.82 (High - 45 papers, h-index: 28)
             • Funding Status: 0.95 (Excellent - 3 active grants)
             • Recency: 0.88 (Active - published 4 papers in 2024)"
        """
        lines = ["Ranking Breakdown:"]

        for component, score in sorted(component_scores.items(), key=lambda x: x[1], reverse=True):
            component_name = component.replace('_', ' ').title()
            quality = self._score_to_quality(score)

            # Add context based on metadata
            context = ""
            metadata = result.get('metadata', {})

            if component == 'research_productivity' and metadata.get('publication_count'):
                pub_count = metadata['publication_count']
                h_index = metadata.get('h_index', 0)
                context = f" - {pub_count} papers, h-index: {h_index}"
            elif component == 'funding_status' and metadata.get('active_grants'):
                grants = metadata['active_grants']
                context = f" - {grants} active grant{'s' if grants > 1 else ''}"
            elif component == 'recency' and metadata.get('last_publication_year'):
                year = metadata['last_publication_year']
                context = f" - last published {year}"

            lines.append(f"  • {component_name}: {score:.2f} ({quality}{context})")

        return "\n".join(lines)

    def format_explanation(
        self,
        sections: List[Tuple[str, str]],
        format_type: str = 'markdown'
    ) -> str:
        """
        Format explanation sections

        Args:
            sections: List of (heading, content) tuples
            format_type: 'markdown', 'html', or 'plain'

        Returns:
            Formatted explanation
        """
        if format_type == 'markdown':
            lines = []
            for heading, content in sections:
                lines.append(f"### {heading}\n")
                lines.append(f"{content}\n\n")
            return "".join(lines)

        elif format_type == 'html':
            lines = []
            for heading, content in sections:
                lines.append(f"<h3>{heading}</h3>")
                lines.append(f"<p>{content}</p>")
            return "".join(lines)

        else:  # plain
            lines = []
            for heading, content in sections:
                lines.append(f"{heading}:")
                lines.append(f"{content}\n")
            return "\n".join(lines)

    def _score_to_quality(self, score: float) -> str:
        """Convert numeric score to quality descriptor"""
        if score >= 0.9:
            return "Outstanding"
        elif score >= 0.8:
            return "Excellent"
        elif score >= 0.7:
            return "Strong"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.5:
            return "Moderate"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Limited"

    def _score_bar(self, score: float, width: int = 20) -> str:
        """Generate a visual score bar"""
        filled = int(score * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"

    def _extract_key_overlap(
        self,
        query: str,
        text: str,
        query_entities: Optional[Dict],
        metadata: Dict
    ) -> str:
        """Extract key overlapping concepts for explanation"""
        # Extract techniques mentioned
        techniques = metadata.get('techniques', [])
        if techniques:
            technique_names = [
                t if isinstance(t, str) else t.get('name', '')
                for t in techniques[:2]
            ]
            if technique_names:
                return ", ".join(technique_names)

        # Extract organisms
        organisms = metadata.get('organisms', [])
        if organisms:
            org_names = [
                o if isinstance(o, str) else o.get('common_name', '')
                for o in organisms[:1]
            ]
            if org_names:
                return f"{org_names[0]} research"

        # Fallback to primary research area
        primary_area = metadata.get('primary_research_area', '')
        if primary_area:
            return primary_area.lower()

        return "related research"

    def _describe_funding(self, metadata: Dict) -> str:
        """Generate funding description"""
        active_grants = metadata.get('active_grants', 0)

        if active_grants >= 2:
            return f"well-funded ({active_grants} active grants)"
        elif active_grants == 1:
            # Check amount if available
            total_funding = metadata.get('total_funding', 0)
            if total_funding > 1000000:
                return "well-funded (active R01)"
            else:
                return "active funding"
        else:
            return None
