"""
Result Ranker for multi-factor search result ranking

Ranks search results using multiple factors beyond vector similarity:
- Semantic similarity (from vector search)
- Research productivity (publications, citations, h-index)
- Funding status (active grants, funding amount)
- Recency (recent publications and activity)
- Diversity (ensure diverse results)
- Query intent boosting (boost results matching user intent)
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math

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
        # Default weights
        default_config = {
            'semantic_similarity': 0.30,
            'research_productivity': 0.20,
            'funding_status': 0.15,
            'recency': 0.15,
            'h_index': 0.10,
            'diversity_bonus': 0.10
        }

        self.config = config if config is not None else default_config

        # Normalize weights
        total_weight = sum(self.config.values())
        if total_weight > 0:
            self.config = {k: v / total_weight for k, v in self.config.items()}

        logger.info(f"ResultRanker initialized with weights: {self.config}")

    def rank_results(
        self,
        results: List[Dict],
        query_context: Any = None,  # QueryAnalysis object
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

        logger.debug(f"Ranking {len(results)} results")

        # Step 1: Calculate component scores
        ranked_results = []
        for result in results:
            component_scores = self._calculate_component_scores(result, query_context)

            # Calculate weighted final score
            final_score = self._calculate_final_score(component_scores)

            # Add to result
            result_copy = result.copy()
            result_copy['component_scores'] = component_scores
            result_copy['final_score'] = final_score
            result_copy['original_score'] = result.get('score', 0.0)

            ranked_results.append(result_copy)

        # Step 2: Apply query intent boosting
        if query_context and hasattr(query_context, 'intent'):
            ranked_results = self.apply_query_intent_boosting(
                ranked_results,
                query_context.intent
            )

        # Step 3: Sort by final score
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)

        # Step 4: Apply diversity if needed
        if self.config.get('diversity_bonus', 0) > 0:
            ranked_results = self.diversify_results(ranked_results)

        # Step 5: Add ranking explanations
        for rank, result in enumerate(ranked_results, 1):
            result['rank'] = rank
            result['explanation'] = self.explain_ranking(result, rank)

        logger.debug(f"Ranking complete. Top score: {ranked_results[0]['final_score']:.3f}")
        return ranked_results

    def _calculate_component_scores(
        self,
        result: Dict,
        query_context: Any
    ) -> Dict[str, float]:
        """
        Calculate all component scores for a result

        Args:
            result: Search result with metadata
            query_context: Query analysis

        Returns:
            Dictionary of component scores
        """
        metadata = result.get('metadata', {})
        scores = {}

        # 1. Semantic similarity (already calculated)
        scores['semantic_similarity'] = result.get('score', 0.0)

        # 2. Research productivity
        scores['research_productivity'] = self.calculate_productivity_score(metadata)

        # 3. Funding status
        scores['funding_status'] = self.calculate_funding_score(metadata)

        # 4. Recency
        scores['recency'] = self.calculate_recency_score(metadata)

        # 5. H-index score
        scores['h_index'] = self.calculate_h_index_score(metadata)

        return scores

    def _calculate_final_score(self, component_scores: Dict[str, float]) -> float:
        """
        Calculate weighted final score

        Args:
            component_scores: Dictionary of component scores

        Returns:
            Final weighted score
        """
        final_score = 0.0

        for component, score in component_scores.items():
            weight = self.config.get(component, 0.0)
            final_score += weight * score

        return final_score

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
        score = 0.0
        components = 0

        # Publication count (log scale, capped at 200)
        pub_count = metadata.get('publication_count', 0)
        if pub_count > 0:
            pub_score = min(1.0, math.log(pub_count + 1) / math.log(200))
            score += pub_score
            components += 1

        # Citation count (log scale, capped at 10000)
        citation_count = metadata.get('citation_count', 0)
        if citation_count > 0:
            citation_score = min(1.0, math.log(citation_count + 1) / math.log(10000))
            score += citation_score
            components += 1

        # H-index (linear scale, capped at 50)
        h_index = metadata.get('h_index', 0)
        if h_index > 0:
            h_score = min(1.0, h_index / 50.0)
            score += h_score
            components += 1

        # Publication frequency (publications per year)
        recent_pubs = metadata.get('recent_publications', [])
        if recent_pubs and len(recent_pubs) > 0:
            # Calculate publication rate over last 3 years
            pubs_per_year = len(recent_pubs) / 3.0
            freq_score = min(1.0, pubs_per_year / 10.0)  # Cap at 10 per year
            score += freq_score
            components += 1

        # Average component scores
        if components > 0:
            return score / components
        else:
            return 0.5  # Neutral score if no data

    def calculate_h_index_score(self, metadata: Dict) -> float:
        """
        Score based on h-index

        Args:
            metadata: Faculty metadata

        Returns:
            H-index score [0.0, 1.0]
        """
        h_index = metadata.get('h_index', 0)

        if h_index == 0:
            return 0.3  # Low but not zero for new faculty

        # Normalize h-index (typical ranges: 10-20 good, 30+ excellent, 50+ outstanding)
        if h_index < 10:
            return 0.4 + (h_index / 10.0) * 0.2  # 0.4-0.6
        elif h_index < 30:
            return 0.6 + ((h_index - 10) / 20.0) * 0.2  # 0.6-0.8
        elif h_index < 50:
            return 0.8 + ((h_index - 30) / 20.0) * 0.15  # 0.8-0.95
        else:
            return min(1.0, 0.95 + (h_index - 50) / 100.0)  # 0.95-1.0

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
        score = 0.0

        # Check for active grants
        active_grants = metadata.get('active_grants', [])
        grants = metadata.get('grants', [])

        if not active_grants and not grants:
            return 0.3  # Low score if no funding info

        # Has active grants (binary, high weight)
        has_active = len(active_grants) > 0 if active_grants else metadata.get('has_active_funding', False)
        if has_active:
            score += 0.4
        else:
            score += 0.1

        # Total funding amount (normalized)
        total_funding = metadata.get('total_funding', 0)
        if total_funding > 0:
            # Log scale, typical R01 is ~1.5M for 5 years
            funding_score = min(0.3, math.log(total_funding + 1) / math.log(5000000) * 0.3)
            score += funding_score

        # Grant diversity (multiple funding sources)
        num_grants = len(active_grants) if active_grants else len(grants)
        if num_grants > 1:
            diversity_bonus = min(0.2, num_grants * 0.05)
            score += diversity_bonus
        elif num_grants == 1:
            score += 0.05

        # Grant recency
        if active_grants and len(active_grants) > 0:
            # Check how recent the grants are
            recent_grant = False
            for grant in active_grants:
                start_date = grant.get('start_date')
                if isinstance(start_date, str):
                    try:
                        start_year = int(start_date[:4])
                        if datetime.now().year - start_year <= 2:
                            recent_grant = True
                            break
                    except:
                        pass

            if recent_grant:
                score += 0.1

        return min(1.0, score)

    def calculate_recency_score(self, metadata: Dict) -> float:
        """
        Score based on recency of work

        Uses exponential decay: more recent = higher score

        Args:
            metadata: Metadata with publication dates

        Returns:
            Recency score [0.0, 1.0]
        """
        recent_pubs = metadata.get('recent_publications', [])

        if not recent_pubs or len(recent_pubs) == 0:
            # Check for last_publication_date
            last_pub_date = metadata.get('last_publication_date')
            if last_pub_date:
                return self._date_to_recency_score(last_pub_date)
            else:
                return 0.5  # Neutral if no data

        # Calculate average recency from recent publications
        recency_scores = []

        for pub in recent_pubs[:5]:  # Check top 5 recent
            pub_date = pub.get('publication_date') or pub.get('date') or pub.get('year')

            if pub_date:
                recency = self._date_to_recency_score(pub_date)
                recency_scores.append(recency)

        if recency_scores:
            # Weight more recent publications higher
            recency_scores.sort(reverse=True)
            weighted_avg = sum(score * (1.0 / (i + 1)) for i, score in enumerate(recency_scores))
            weight_sum = sum(1.0 / (i + 1) for i in range(len(recency_scores)))
            return weighted_avg / weight_sum
        else:
            return 0.5

    def _date_to_recency_score(self, date_input: Any) -> float:
        """
        Convert date to recency score using exponential decay

        Args:
            date_input: Date (string, datetime, or year)

        Returns:
            Recency score [0.0, 1.0]
        """
        try:
            # Extract year
            if isinstance(date_input, str):
                year = int(date_input[:4])
            elif isinstance(date_input, datetime):
                year = date_input.year
            elif isinstance(date_input, (int, float)):
                year = int(date_input)
            else:
                return 0.5

            current_year = datetime.now().year
            years_ago = current_year - year

            # Exponential decay: half-life of 3 years
            decay_rate = math.log(2) / 3.0
            recency_score = math.exp(-decay_rate * years_ago)

            return max(0.0, min(1.0, recency_score))

        except Exception as e:
            logger.warning(f"Error calculating recency: {e}")
            return 0.5

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
        if len(results) <= 1:
            return results

        # Track diversity dimensions
        seen_institutions = Counter()
        seen_departments = Counter()
        seen_keywords = Counter()

        diversified = []

        # Keep track of remaining results
        remaining = results.copy()

        while remaining:
            best_idx = 0
            best_score = -1

            for idx, result in enumerate(remaining):
                metadata = result.get('metadata', {})

                # Base relevance score
                relevance = result.get('final_score', 0.0)

                # Calculate diversity penalty
                institution = metadata.get('institution', 'unknown')
                department = metadata.get('department', 'unknown')

                # Extract keywords from research summary
                research = metadata.get('research_summary', metadata.get('research_interests', ''))
                keywords = set(research.lower().split()[:10])  # First 10 words as proxy

                # Diversity penalties (higher count = higher penalty)
                inst_penalty = seen_institutions[institution] / (len(diversified) + 1)
                dept_penalty = seen_departments[department] / (len(diversified) + 1)

                # Keyword diversity
                keyword_overlap = len(keywords & set(seen_keywords.elements())) / (len(keywords) + 1)

                # Total diversity penalty
                diversity_penalty = (inst_penalty + dept_penalty + keyword_overlap) / 3.0

                # MMR score
                mmr_score = (1 - diversity_factor) * relevance - diversity_factor * diversity_penalty

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Add best result to diversified list
            selected = remaining.pop(best_idx)
            diversified.append(selected)

            # Update counters
            metadata = selected.get('metadata', {})
            seen_institutions[metadata.get('institution', 'unknown')] += 1
            seen_departments[metadata.get('department', 'unknown')] += 1

            research = metadata.get('research_summary', metadata.get('research_interests', ''))
            for word in research.lower().split()[:10]:
                seen_keywords[word] += 1

        return diversified

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
            boost_applied = False

            if intent == 'technique_based':
                # Boost if faculty has many techniques listed
                techniques = metadata.get('techniques', [])
                if len(techniques) >= 3:
                    result['final_score'] *= boost_factor
                    boost_applied = True

            elif intent == 'funding_based':
                # Boost well-funded faculty
                has_funding = metadata.get('has_active_funding', False)
                total_funding = metadata.get('total_funding', 0)
                if has_funding or total_funding > 1000000:
                    result['final_score'] *= boost_factor
                    boost_applied = True

            elif intent == 'organism_based':
                # Boost if faculty works with specific organisms
                organisms = metadata.get('organisms', [])
                if len(organisms) >= 1:
                    result['final_score'] *= boost_factor
                    boost_applied = True

            elif intent == 'collaborative':
                # Boost interdisciplinary researchers
                research = metadata.get('research_summary', '').lower()
                collab_keywords = ['interdisciplinary', 'collaborative', 'cross-disciplinary',
                                   'multidisciplinary', 'translational']
                if any(kw in research for kw in collab_keywords):
                    result['final_score'] *= boost_factor
                    boost_applied = True

            elif intent == 'specific_person':
                # No boosting for specific person searches
                pass

            if boost_applied:
                result['intent_boost'] = True

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

        # Start with rank
        explanation_parts = [f"Ranked #{rank}"]

        # Semantic match
        sem_score = component_scores.get('semantic_similarity', 0)
        if sem_score >= 0.8:
            explanation_parts.append(f"strong semantic match ({sem_score:.2f})")
        elif sem_score >= 0.6:
            explanation_parts.append(f"good semantic match ({sem_score:.2f})")
        else:
            explanation_parts.append(f"moderate match ({sem_score:.2f})")

        # Funding
        funding_score = component_scores.get('funding_status', 0)
        if funding_score >= 0.7:
            explanation_parts.append("active funding")

        # Productivity
        prod_score = component_scores.get('research_productivity', 0)
        pub_count = metadata.get('publication_count', 0)
        h_index = metadata.get('h_index', 0)

        if prod_score >= 0.7:
            prod_str = f"strong publication record"
            if pub_count > 0:
                prod_str += f" with {pub_count} papers"
            if h_index > 0:
                prod_str += f" and H-index of {h_index}"
            explanation_parts.append(prod_str)

        # Recency
        recency_score = component_scores.get('recency', 0)
        if recency_score >= 0.8:
            explanation_parts.append("recent publications")

        # Combine parts
        if len(explanation_parts) > 1:
            explanation = explanation_parts[0] + " due to " + ", ".join(explanation_parts[1:]) + "."
        else:
            explanation = explanation_parts[0] + "."

        return explanation
