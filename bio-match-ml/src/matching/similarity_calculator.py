"""
Calculate various similarity metrics for faculty-student matching
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """
    Calculate various similarity metrics for matching
    """

    def __init__(self, embedding_generator=None):
        """
        Initialize calculator

        Args:
            embedding_generator: EmbeddingGenerator for on-demand embeddings
        """
        self.embedding_generator = embedding_generator
        self._embedding_cache = {}

        logger.info("SimilarityCalculator initialized")

    def calculate_research_similarity(
        self,
        student_profile: Dict,
        faculty_profile: Dict
    ) -> Dict[str, float]:
        """
        Calculate research alignment scores

        Returns multiple similarity metrics:
        - embedding_similarity: Cosine similarity of research embeddings
        - topic_overlap: Jaccard similarity of research topics
        - technique_overlap: Overlap in experimental techniques
        - organism_overlap: Overlap in model organisms
        - keyword_overlap: Overlap in key research terms

        Args:
            student_profile: Student research profile with:
                - research_interests: str
                - topics: List[str]
                - techniques: List[str]
                - organisms: List[str]
            faculty_profile: Faculty profile with same structure

        Returns:
            Dict of similarity scores, all in [0, 1]
        """
        similarities = {}

        # Embedding similarity
        student_text = student_profile.get('research_interests', '')
        faculty_text = faculty_profile.get('research_summary', '')

        if student_text and faculty_text:
            embedding_sim = self.calculate_embedding_similarity(student_text, faculty_text)
            similarities['embedding_similarity'] = embedding_sim
        else:
            similarities['embedding_similarity'] = 0.0

        # Topic overlap
        student_topics = set(student_profile.get('topics', []))
        faculty_topics = set(faculty_profile.get('topics', []))

        if student_topics and faculty_topics:
            topic_overlap = self.calculate_jaccard_similarity(student_topics, faculty_topics)
            similarities['topic_overlap'] = topic_overlap
        else:
            similarities['topic_overlap'] = 0.0

        # Technique overlap
        student_techniques = set(student_profile.get('techniques', []))
        faculty_techniques = set(faculty_profile.get('techniques', []))

        if student_techniques and faculty_techniques:
            technique_overlap = self.calculate_jaccard_similarity(
                student_techniques,
                faculty_techniques
            )
            similarities['technique_overlap'] = technique_overlap
        else:
            similarities['technique_overlap'] = 0.0

        # Organism overlap
        student_organisms = set(student_profile.get('organisms', []))
        faculty_organisms = set(faculty_profile.get('organisms', []))

        if student_organisms and faculty_organisms:
            organism_overlap = self.calculate_jaccard_similarity(
                student_organisms,
                faculty_organisms
            )
            similarities['organism_overlap'] = organism_overlap
        else:
            similarities['organism_overlap'] = 0.0

        # Keyword overlap (extract from text)
        student_keywords = self._extract_keywords(student_text)
        faculty_keywords = self._extract_keywords(faculty_text)

        if student_keywords and faculty_keywords:
            keyword_overlap = self.calculate_jaccard_similarity(
                student_keywords,
                faculty_keywords
            )
            similarities['keyword_overlap'] = keyword_overlap
        else:
            similarities['keyword_overlap'] = 0.0

        logger.debug(f"Research similarity calculated: {similarities}")

        return similarities

    def calculate_embedding_similarity(
        self,
        text1: str,
        text2: str,
        model: str = 'pubmedbert'
    ) -> float:
        """
        Calculate cosine similarity between text embeddings

        Args:
            text1: First text
            text2: Second text
            model: Embedding model to use

        Returns:
            Cosine similarity score [0, 1]
        """
        if not text1 or not text2:
            return 0.0

        if not self.embedding_generator:
            logger.warning("No embedding generator available, returning default similarity")
            return 0.5

        try:
            # Check cache
            cache_key1 = (text1, model)
            cache_key2 = (text2, model)

            if cache_key1 in self._embedding_cache:
                emb1 = self._embedding_cache[cache_key1]
            else:
                emb1 = self.embedding_generator.generate_embedding(text1, model=model)
                self._embedding_cache[cache_key1] = emb1

            if cache_key2 in self._embedding_cache:
                emb2 = self._embedding_cache[cache_key2]
            else:
                emb2 = self.embedding_generator.generate_embedding(text2, model=model)
                self._embedding_cache[cache_key2] = emb2

            # Convert to numpy arrays
            emb1 = np.array(emb1)
            emb2 = np.array(emb2)

            # Calculate cosine similarity
            similarity = self._cosine_similarity(emb1, emb2)

            # Normalize to [0, 1] range (cosine similarity is [-1, 1])
            normalized = (similarity + 1) / 2

            return float(normalized)

        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            return 0.5

    def calculate_jaccard_similarity(
        self,
        set1: Set[str],
        set2: Set[str]
    ) -> float:
        """
        Calculate Jaccard similarity between two sets

        Jaccard = |intersection| / |union|

        Args:
            set1: First set
            set2: Second set

        Returns:
            Jaccard similarity [0, 1]
        """
        if not set1 or not set2:
            return 0.0

        # Normalize to lowercase for comparison
        set1_lower = {s.lower() for s in set1}
        set2_lower = {s.lower() for s in set2}

        intersection = len(set1_lower & set2_lower)
        union = len(set1_lower | set2_lower)

        if union == 0:
            return 0.0

        return intersection / union

    def calculate_weighted_overlap(
        self,
        items1: List[Tuple[str, float]],
        items2: List[Tuple[str, float]]
    ) -> float:
        """
        Calculate weighted overlap for weighted items

        Useful for topics with confidence scores or importance weights

        Args:
            items1: List of (item, weight) tuples
            items2: List of (item, weight) tuples

        Returns:
            Weighted overlap score [0, 1]
        """
        if not items1 or not items2:
            return 0.0

        # Create weight dictionaries (normalized to lowercase)
        weights1 = {item.lower(): weight for item, weight in items1}
        weights2 = {item.lower(): weight for item, weight in items2}

        # Find common items
        common_items = set(weights1.keys()) & set(weights2.keys())

        if not common_items:
            return 0.0

        # Calculate weighted overlap
        overlap_score = sum(
            min(weights1[item], weights2[item])
            for item in common_items
        )

        # Normalize by maximum possible overlap
        max_possible = sum(max(weights1.get(item, 0), weights2.get(item, 0))
                          for item in set(weights1.keys()) | set(weights2.keys()))

        if max_possible == 0:
            return 0.0

        return overlap_score / max_possible

    def calculate_trajectory_alignment(
        self,
        student_goals: Dict,
        faculty_trajectory: Dict
    ) -> float:
        """
        Score alignment between student career goals and faculty research trajectory

        Considers:
        - Student's desired research direction
        - Faculty's current vs future research
        - Training opportunities in the lab

        Args:
            student_goals: Student career goals and interests
            faculty_trajectory: Faculty research evolution data

        Returns:
            Trajectory alignment score [0, 1]
        """
        if not student_goals or not faculty_trajectory:
            return 0.5

        scores = []

        # Align student interests with faculty's trending topics
        student_interests = set(student_goals.get('research_interests', []))
        if isinstance(student_interests, str):
            student_interests = {student_interests}

        faculty_trending = set(faculty_trajectory.get('trending_topics', []))

        if student_interests and faculty_trending:
            trending_alignment = self.calculate_jaccard_similarity(
                student_interests,
                faculty_trending
            )
            scores.append(trending_alignment * 1.2)  # Bonus for trending alignment

        # Align with stable core (good for foundational training)
        faculty_core = set(faculty_trajectory.get('stable_core', []))

        if student_interests and faculty_core:
            core_alignment = self.calculate_jaccard_similarity(
                student_interests,
                faculty_core
            )
            scores.append(core_alignment * 0.8)  # Lower weight for core

        # Check if student interests align with predicted future directions
        predicted_directions = faculty_trajectory.get('predicted_directions', [])

        if isinstance(predicted_directions, list):
            predicted_topics = {
                d.get('topic') if isinstance(d, dict) else d
                for d in predicted_directions
            }

            if student_interests and predicted_topics:
                future_alignment = self.calculate_jaccard_similarity(
                    student_interests,
                    predicted_topics
                )
                scores.append(future_alignment * 1.3)  # Highest bonus for future alignment

        # Average the scores
        if scores:
            return min(sum(scores) / len(scores), 1.0)

        return 0.5

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity
        """
        # Handle edge cases
        if vec1.size == 0 or vec2.size == 0:
            return 0.0

        # Ensure same dimensionality
        if vec1.shape != vec2.shape:
            logger.warning(f"Vector shape mismatch: {vec1.shape} vs {vec2.shape}")
            return 0.0

        # Calculate norms
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        return float(similarity)

    def _extract_keywords(self, text: str, min_length: int = 3) -> Set[str]:
        """
        Extract keywords from text (simple approach)

        Args:
            text: Input text
            min_length: Minimum word length

        Returns:
            Set of keywords
        """
        if not text:
            return set()

        # Simple keyword extraction: lowercase words, filter stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        # Extract words
        words = text.lower().split()

        # Filter and clean
        keywords = {
            word.strip('.,!?;:()[]{}')
            for word in words
            if len(word) >= min_length and word.lower() not in stopwords
        }

        return keywords

    def batch_similarity(
        self,
        student_profile: Dict,
        faculty_profiles: List[Dict]
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Calculate similarities for multiple faculty members efficiently

        Args:
            student_profile: Student profile
            faculty_profiles: List of faculty profiles

        Returns:
            List of (faculty_id, similarities_dict) tuples
        """
        results = []

        for faculty_profile in faculty_profiles:
            faculty_id = faculty_profile.get('id', 'unknown')

            similarities = self.calculate_research_similarity(
                student_profile,
                faculty_profile
            )

            results.append((faculty_id, similarities))

        return results

    def clear_cache(self):
        """Clear embedding cache to free memory"""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")
