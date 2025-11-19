"""
Multi-factor result ranking for search results
"""
import logging
import math
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class ResultRanker:
    """
    Multi-factor result ranking for search results
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ranker with weights

        Args:
            config: Configuration with ranking weights, defaults:
                {
                    'semantic_similarity': 0.30,
                    'research_productivity': 0.20,
                    'funding_status': 0.15,
                    'recency': 0.15,
                    'h_index': 0.10,
                    'diversity_bonus': 0.10
                }
        """
        default_config = {
            'semantic_similarity': 0.30,
            'research_productivity': 0.20,
            'funding_status': 0.15,
            'recency': 0.15,
            'h_index': 0.10,
            'diversity_bonus': 0.10
        }

        self.config = config if config else default_config
        self.weights = self.config

        # Validate weights sum to approximately 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Ranking weights sum to {total_weight}, normalizing...")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        logger.info(f"ResultRanker initialized with weights: {self.weights}")

    def rank_results(
        self,
        results: List[Dict],
        query_context: Any,  # QueryAnalysis object
        user_preferences: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Rerank results using multi-factor scoring

        Args:
            results: Initial results from vector search, each with:
                - id: faculty/publication ID
                - score: vector similarity score
                - metadata: all available metadata
            query_context: Processed query information
            user_preferences: Optional user-specific preferences

        Returns:
            Reranked results with updated scores and ranking explanations

        Process:
            1. Calculate component scores for each result
            2. Combine scores using weighted sum
            3. Apply diversity penalties if needed
            4. Sort by final score
            5. Add ranking explanation to each result
        """
        if not results:
            return []

        logger.info(f"Ranking {len(results)} results")

        # Calculate component scores for each result
        ranked_results = []
        for result in results:
            component_scores = self._calculate_component_scores(result, query_context)

            # Calculate final weighted score
            final_score = sum(
                self.weights.get(component, 0) * score
                for component, score in component_scores.items()
            )

            # Create enhanced result
            enhanced_result = result.copy()
            enhanced_result['final_score'] = final_score
            enhanced_result['component_scores'] = component_scores
            enhanced_result['original_score'] = result.get('score', 0)

            ranked_results.append(enhanced_result)

        # Apply intent-based boosting
        if hasattr(query_context, 'intent'):
            ranked_results = self.apply_query_intent_boosting(
                ranked_results,
                query_context.intent
            )

        # Sort by final score
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Add rank position and explanations
        for rank, result in enumerate(ranked_results, 1):
            result['rank'] = rank
            result['ranking_explanation'] = self.explain_ranking(result, rank)

        logger.info(f"Ranking complete. Top score: {ranked_results[0]['final_score']:.3f}")

        return ranked_results

    def _calculate_component_scores(
        self,
        result: Dict,
        query_context: Any
    ) -> Dict[str, float]:
        """Calculate all component scores for a result"""
        metadata = result.get('metadata', {})

        component_scores = {}

        # Semantic similarity (from vector search)
        component_scores['semantic_similarity'] = result.get('score', 0.5)

        # Research productivity
        component_scores['research_productivity'] = self.calculate_productivity_score(metadata)

        # Funding status
        component_scores['funding_status'] = self.calculate_funding_score(metadata)

        # Recency
        component_scores['recency'] = self.calculate_recency_score(metadata)

        # H-index
        h_index = metadata.get('h_index', 0)
        component_scores['h_index'] = self._normalize_h_index(h_index)

        # Diversity bonus (placeholder, actual calculation in diversify_results)
        component_scores['diversity_bonus'] = 0.5

        return component_scores

    def calculate_productivity_score(self, metadata: Dict) -> float:
        """
        Score based on research productivity

        Factors:
        - Publication count (normalized)
        - Citation count (normalized)
        - H-index (normalized)
        - Publication frequency (recent activity)

        Args:
            metadata: Faculty/publication metadata

        Returns:
            Productivity score [0.0, 1.0]
        """
        scores = []

        # Publication count
        pub_count = metadata.get('publication_count', 0)
        pub_score = min(pub_count / 100.0, 1.0)  # Cap at 100 publications
        scores.append(pub_score)

        # Citation count
        citations = metadata.get('citations', metadata.get('total_citations', 0))
        citation_score = min(citations / 10000.0, 1.0)  # Cap at 10k citations
        scores.append(citation_score)

        # H-index
        h_index = metadata.get('h_index', 0)
        h_score = self._normalize_h_index(h_index)
        scores.append(h_score)

        # Publication frequency (publications in last 3 years)
        recent_pubs = metadata.get('recent_publications', 0)
        freq_score = min(recent_pubs / 10.0, 1.0)  # Cap at 10 recent pubs
        scores.append(freq_score)

        # Average of all scores
        if scores:
            return sum(scores) / len(scores)
        return 0.0

    def calculate_funding_score(self, metadata: Dict) -> float:
        """
        Score based on funding status

        Factors:
        - Has active grants (binary)
        - Total funding amount (normalized)
        - Grant diversity (multiple sources bonus)
        - Grant recency

        Args:
            metadata: Faculty metadata with grant info

        Returns:
            Funding score [0.0, 1.0]
        """
        # Check for active grants
        active_grants = metadata.get('active_grants', 0)
        grants = metadata.get('grants', [])

        if active_grants == 0 and not grants:
            return 0.1  # Low score for no funding

        scores = []

        # Active grants binary
        has_active = 1.0 if active_grants > 0 else 0.0
        scores.append(has_active)

        # Total funding amount
        total_funding = metadata.get('total_funding', 0)
        funding_score = min(total_funding / 2000000.0, 1.0)  # Cap at $2M
        scores.append(funding_score)

        # Grant diversity (multiple sources)
        grant_count = max(active_grants, len(grants))
        diversity_score = min(grant_count / 3.0, 1.0)  # Cap at 3 grants
        scores.append(diversity_score)

        # Grant recency (if grant data available)
        if grants:
            current_year = datetime.now().year
            recent_grants = [
                g for g in grants
                if isinstance(g, dict) and g.get('year', 0) >= current_year - 3
            ]
            recency_score = min(len(recent_grants) / 2.0, 1.0)
            scores.append(recency_score)

        # Weighted average (higher weight on having active grants)
        if scores:
            # First score (has_active) gets 2x weight
            weighted_sum = has_active * 2 + sum(scores[1:])
            return weighted_sum / (len(scores) + 1)

        return 0.5

    def calculate_recency_score(self, metadata: Dict) -> float:
        """
        Score based on recency of work

        Uses exponential decay: more recent = higher score

        Args:
            metadata: Metadata with publication dates

        Returns:
            Recency score [0.0, 1.0]
        """
        # Get most recent publication year
        last_pub_year = metadata.get('last_publication_year')

        if not last_pub_year:
            # Try to extract from publications list
            publications = metadata.get('publications', [])
            if publications:
                years = [
                    p.get('year', 0) for p in publications
                    if isinstance(p, dict) and p.get('year')
                ]
                last_pub_year = max(years) if years else None

        if not last_pub_year:
            return 0.5  # Neutral score if no data

        # Calculate years since last publication
        current_year = datetime.now().year
        years_ago = current_year - last_pub_year

        # Exponential decay: score = e^(-λ * years_ago)
        # λ = 0.2 gives reasonable decay (0.67 after 2 years, 0.45 after 4 years)
        lambda_decay = 0.2
        score = math.exp(-lambda_decay * years_ago)

        return min(max(score, 0.0), 1.0)

    def diversify_results(
        self,
        results: List[Dict],
        diversity_factor: float = 0.3
    ) -> List[Dict]:
        """
        Ensure result diversity using Maximal Marginal Relevance (MMR)

        Promotes diversity in:
        - Institutions
        - Research sub-areas
        - Career stages
        - Geographic locations

        Args:
            results: Ranked results
            diversity_factor: Weight of diversity vs relevance (0=pure relevance, 1=pure diversity)

        Returns:
            Diversified results
        """
        if len(results) <= 1 or diversity_factor == 0:
            return results

        # Track what we've selected
        selected: List[Dict] = []
        remaining = results.copy()

        # Diversity features to track
        selected_institutions: Set[str] = set()
        selected_departments: Set[str] = set()
        selected_areas: Set[str] = set()

        while remaining:
            best_score = -1
            best_idx = 0

            for idx, result in enumerate(remaining):
                metadata = result.get('metadata', {})

                # Relevance score
                relevance = result.get('final_score', result.get('score', 0))

                # Diversity score
                diversity = 0.0

                institution = metadata.get('institution', '').lower()
                if institution and institution not in selected_institutions:
                    diversity += 0.4

                department = metadata.get('department', '').lower()
                if department and department not in selected_departments:
                    diversity += 0.3

                # Research area diversity
                primary_area = metadata.get('primary_research_area', '').lower()
                if primary_area and primary_area not in selected_areas:
                    diversity += 0.3

                # Combined MMR score
                mmr_score = (1 - diversity_factor) * relevance + diversity_factor * diversity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Add best to selected
            best_result = remaining.pop(best_idx)
            selected.append(best_result)

            # Update diversity tracking
            best_metadata = best_result.get('metadata', {})
            if best_metadata.get('institution'):
                selected_institutions.add(best_metadata['institution'].lower())
            if best_metadata.get('department'):
                selected_departments.add(best_metadata['department'].lower())
            if best_metadata.get('primary_research_area'):
                selected_areas.add(best_metadata['primary_research_area'].lower())

        return selected

    def apply_query_intent_boosting(
        self,
        results: List[Dict],
        intent: str,
        boost_factor: float = 1.2
    ) -> List[Dict]:
        """
        Boost results matching query intent

        Examples:
        - 'technique_based' intent → boost faculty with matching techniques
        - 'funding_based' intent → boost well-funded faculty
        - 'collaborative' intent → boost interdisciplinary researchers

        Args:
            results: Current results
            intent: Query intent from QueryProcessor
            boost_factor: Multiplier for matching results

        Returns:
            Results with intent-based boosting applied
        """
        for result in results:
            metadata = result.get('metadata', {})
            boost = 1.0

            if intent == 'funding_based':
                # Boost well-funded researchers
                if metadata.get('active_grants', 0) > 0:
                    boost = boost_factor

            elif intent == 'technique_based':
                # Boost high-technique diversity
                techniques = metadata.get('techniques', [])
                if len(techniques) >= 3:
                    boost = boost_factor

            elif intent == 'collaborative':
                # Boost interdisciplinary researchers
                if metadata.get('is_interdisciplinary', False):
                    boost = boost_factor

            elif intent == 'organism_based':
                # Boost organism specialists (already handled by semantic similarity)
                pass

            elif intent == 'specific_person':
                # For person searches, boost exact name matches
                # This would need name matching logic
                pass

            # Apply boost to final score
            if 'final_score' in result:
                result['final_score'] *= boost

        return results

    def explain_ranking(self, result: Dict, rank: int) -> str:
        """
        Generate human-readable ranking explanation

        Args:
            result: Result with component scores
            rank: Position in results

        Returns:
            Explanation string

        Example:
            "Ranked #3 due to strong semantic match (0.89) and active funding.
             Strong publication record with 45 papers and H-index of 28."
        """
        component_scores = result.get('component_scores', {})
        metadata = result.get('metadata', {})

        # Build explanation parts
        parts = [f"Ranked #{rank}"]

        # Semantic similarity
        sem_score = component_scores.get('semantic_similarity', 0)
        if sem_score >= 0.8:
            parts.append(f"strong semantic match ({sem_score:.2f})")
        elif sem_score >= 0.6:
            parts.append(f"good semantic match ({sem_score:.2f})")

        # Funding
        funding_score = component_scores.get('funding_status', 0)
        if funding_score >= 0.7:
            active_grants = metadata.get('active_grants', 0)
            if active_grants > 0:
                parts.append(f"active funding ({active_grants} grant{'s' if active_grants > 1 else ''})")

        # Productivity
        prod_score = component_scores.get('research_productivity', 0)
        if prod_score >= 0.7:
            pub_count = metadata.get('publication_count', 0)
            h_index = metadata.get('h_index', 0)
            if pub_count > 0 and h_index > 0:
                parts.append(f"strong publication record ({pub_count} papers, h-index: {h_index})")

        # Recency
        recency_score = component_scores.get('recency', 0)
        if recency_score >= 0.8:
            parts.append("recent activity")

        # Combine parts
        if len(parts) == 1:
            return f"{parts[0]} with overall score {result.get('final_score', 0):.3f}."

        explanation = f"{parts[0]} due to " + ", ".join(parts[1:]) + "."

        return explanation

    def _normalize_h_index(self, h_index: int) -> float:
        """
        Normalize h-index to [0, 1] scale

        Uses logarithmic scaling since h-index growth is non-linear

        Args:
            h_index: H-index value

        Returns:
            Normalized score [0, 1]
        """
        if h_index <= 0:
            return 0.0

        # Logarithmic scaling: log(h+1) / log(101)
        # This maps h=0 to 0.0, h=10 to ~0.52, h=30 to ~0.75, h=100 to ~1.0
        normalized = math.log(h_index + 1) / math.log(101)

        return min(max(normalized, 0.0), 1.0)
