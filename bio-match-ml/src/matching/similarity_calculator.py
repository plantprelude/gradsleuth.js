"""
Similarity Calculator for faculty-student matching

Calculates various similarity metrics:
- Embedding similarity (semantic)
- Jaccard similarity (set overlap)
- Weighted overlap (with confidence scores)
- Topic overlap
- Technique overlap
- Organism overlap
- Keyword overlap
- Trajectory alignment
"""

import logging
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
from collections import Counter

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

        # Load embedding generator if not provided
        if self.embedding_generator is None:
            try:
                from ..embeddings.embedding_generator import EmbeddingGenerator
                self.embedding_generator = EmbeddingGenerator()
                logger.info("Loaded EmbeddingGenerator")
            except Exception as e:
                logger.warning(f"Could not load EmbeddingGenerator: {e}")

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

        # 1. Embedding similarity (semantic alignment)
        if 'research_interests' in student_profile and 'research_interests' in faculty_profile:
            student_text = student_profile['research_interests']
            faculty_text = faculty_profile.get('research_summary', faculty_profile.get('research_interests', ''))

            if student_text and faculty_text:
                similarities['embedding_similarity'] = self.calculate_embedding_similarity(
                    student_text,
                    faculty_text
                )
            else:
                similarities['embedding_similarity'] = 0.0
        else:
            similarities['embedding_similarity'] = 0.0

        # 2. Topic overlap (Jaccard)
        student_topics = set(t.lower() for t in student_profile.get('topics', []))
        faculty_topics = set(t.lower() for t in faculty_profile.get('topics', faculty_profile.get('research_topics', [])))
        similarities['topic_overlap'] = self.calculate_jaccard_similarity(student_topics, faculty_topics)

        # 3. Technique overlap
        student_techniques = set(t.lower() for t in student_profile.get('techniques', []))
        faculty_techniques = set(t.lower() for t in faculty_profile.get('techniques', []))
        similarities['technique_overlap'] = self.calculate_jaccard_similarity(
            student_techniques,
            faculty_techniques
        )

        # 4. Organism overlap
        student_organisms = set(o.lower() for o in student_profile.get('organisms', []))
        faculty_organisms = set(o.lower() for o in faculty_profile.get('organisms', []))
        similarities['organism_overlap'] = self.calculate_jaccard_similarity(
            student_organisms,
            faculty_organisms
        )

        # 5. Keyword overlap (from research text)
        similarities['keyword_overlap'] = self._calculate_keyword_overlap(
            student_profile.get('research_interests', ''),
            faculty_profile.get('research_summary', faculty_profile.get('research_interests', ''))
        )

        logger.debug(f"Research similarities: {similarities}")
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

        if self.embedding_generator is None:
            logger.warning("No embedding generator available")
            return 0.0

        try:
            # Generate embeddings
            emb1 = self.embedding_generator.generate_embedding(text1, model_name=model)
            emb2 = self.embedding_generator.generate_embedding(text2, model_name=model)

            # Calculate cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(emb1, emb2))

            # Ensure in [0, 1] range
            similarity = max(0.0, min(1.0, similarity))

            return similarity

        except Exception as e:
            logger.error(f"Embedding similarity calculation failed: {e}")
            return 0.0

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

        # Convert to lowercase for case-insensitive comparison
        set1_lower = set(s.lower() if isinstance(s, str) else s for s in set1)
        set2_lower = set(s.lower() if isinstance(s, str) else s for s in set2)

        intersection = set1_lower & set2_lower
        union = set1_lower | set2_lower

        if len(union) == 0:
            return 0.0

        similarity = len(intersection) / len(union)
        return float(similarity)

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

        # Create dictionaries for easier lookup
        dict1 = {item.lower(): weight for item, weight in items1}
        dict2 = {item.lower(): weight for item, weight in items2}

        # Calculate weighted overlap
        overlap_score = 0.0
        total_weight = 0.0

        all_items = set(dict1.keys()) | set(dict2.keys())

        for item in all_items:
            weight1 = dict1.get(item, 0.0)
            weight2 = dict2.get(item, 0.0)

            # Use minimum weight for overlap
            overlap_score += min(weight1, weight2)
            # Use maximum weight for normalization
            total_weight += max(weight1, weight2)

        if total_weight == 0:
            return 0.0

        return overlap_score / total_weight

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
        alignment_score = 0.0
        components = 0

        # 1. Career goals alignment
        student_career_goal = student_goals.get('career_goals', '').lower()
        faculty_training = faculty_trajectory.get('training_focus', '').lower()

        if student_career_goal and faculty_training:
            # Check for keywords
            academic_keywords = ['academic', 'professor', 'research', 'postdoc', 'phd']
            industry_keywords = ['industry', 'biotech', 'pharma', 'commercial', 'startup']
            clinical_keywords = ['clinical', 'medical', 'physician', 'md', 'translational']

            student_type = self._classify_career_goal(student_career_goal, [
                academic_keywords, industry_keywords, clinical_keywords
            ])
            faculty_type = self._classify_career_goal(faculty_training, [
                academic_keywords, industry_keywords, clinical_keywords
            ])

            if student_type == faculty_type and student_type is not None:
                alignment_score += 1.0
            elif student_type is not None:
                alignment_score += 0.5  # Some alignment

            components += 1

        # 2. Research direction alignment
        student_future = student_goals.get('research_direction', '').lower()
        faculty_future = faculty_trajectory.get('future_directions', '').lower()

        if student_future and faculty_future:
            # Calculate keyword overlap
            overlap = self._calculate_keyword_overlap(student_future, faculty_future)
            alignment_score += overlap
            components += 1

        # 3. Skill development alignment
        student_skills = set(s.lower() for s in student_goals.get('desired_skills', []))
        faculty_skills = set(s.lower() for s in faculty_trajectory.get('training_opportunities', []))

        if student_skills and faculty_skills:
            skill_overlap = self.calculate_jaccard_similarity(student_skills, faculty_skills)
            alignment_score += skill_overlap
            components += 1

        # Normalize by number of components
        if components > 0:
            return alignment_score / components
        else:
            return 0.5  # Neutral score if no data

    def _calculate_keyword_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate keyword overlap between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Keyword overlap score [0, 1]
        """
        if not text1 or not text2:
            return 0.0

        # Extract keywords (simple approach: meaningful words)
        keywords1 = self._extract_keywords(text1)
        keywords2 = self._extract_keywords(text2)

        return self.calculate_jaccard_similarity(keywords1, keywords2)

    def _extract_keywords(self, text: str, min_length: int = 3) -> Set[str]:
        """
        Extract keywords from text

        Args:
            text: Input text
            min_length: Minimum keyword length

        Returns:
            Set of keywords
        """
        import re

        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-z]+\b', text.lower())

        # Filter by length and remove common stopwords
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
                     'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'can',
                     'this', 'that', 'these', 'those', 'it', 'its'}

        keywords = {w for w in words if len(w) >= min_length and w not in stopwords}

        return keywords

    def _classify_career_goal(
        self,
        text: str,
        keyword_groups: List[List[str]]
    ) -> Optional[int]:
        """
        Classify career goal based on keyword groups

        Args:
            text: Text to classify
            keyword_groups: List of keyword lists for each category

        Returns:
            Index of matching category or None
        """
        for i, keywords in enumerate(keyword_groups):
            if any(kw in text for kw in keywords):
                return i
        return None

    def calculate_cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two vectors

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity [-1, 1], normalized to [0, 1]
        """
        if vec1 is None or vec2 is None:
            return 0.0

        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)

        # Normalize to [0, 1]
        similarity = (similarity + 1) / 2

        return float(similarity)

    def calculate_dice_coefficient(
        self,
        set1: Set[str],
        set2: Set[str]
    ) -> float:
        """
        Calculate Dice coefficient (similar to Jaccard but different weighting)

        Dice = 2 * |intersection| / (|set1| + |set2|)

        Args:
            set1: First set
            set2: Second set

        Returns:
            Dice coefficient [0, 1]
        """
        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2

        if len(set1) + len(set2) == 0:
            return 0.0

        dice = 2 * len(intersection) / (len(set1) + len(set2))
        return float(dice)

    def calculate_overlap_coefficient(
        self,
        set1: Set[str],
        set2: Set[str]
    ) -> float:
        """
        Calculate overlap coefficient (Szymkiewicz–Simpson coefficient)

        Overlap = |intersection| / min(|set1|, |set2|)

        Useful when sets are very different in size

        Args:
            set1: First set
            set2: Second set

        Returns:
            Overlap coefficient [0, 1]
        """
        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        min_size = min(len(set1), len(set2))

        if min_size == 0:
            return 0.0

        overlap = len(intersection) / min_size
        return float(overlap)

    def calculate_tfidf_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate TF-IDF based similarity

        Args:
            text1: First text
            text2: Second text

        Returns:
            TF-IDF cosine similarity [0, 1]
        """
        if not text1 or not text2:
            return 0.0

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(stop_words='english', min_df=1)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])

            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            return float(max(0.0, similarity))

        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {e}")
            # Fallback to keyword overlap
            return self._calculate_keyword_overlap(text1, text2)

    def aggregate_similarities(
        self,
        similarities: Dict[str, float],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Aggregate multiple similarity scores into single score

        Args:
            similarities: Dictionary of similarity scores
            weights: Optional weights for each similarity type

        Returns:
            Weighted average similarity [0, 1]
        """
        if not similarities:
            return 0.0

        if weights is None:
            # Default equal weights
            weights = {key: 1.0 for key in similarities.keys()}

        # Calculate weighted average
        total_weighted = 0.0
        total_weight = 0.0

        for key, score in similarities.items():
            weight = weights.get(key, 1.0)
            total_weighted += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted / total_weight

    def calculate_publication_similarity(
        self,
        student_interests: List[str],
        faculty_publications: List[Dict]
    ) -> float:
        """
        Calculate similarity between student interests and faculty publications

        Args:
            student_interests: List of student research interests/keywords
            faculty_publications: List of publication dicts with 'title' and 'abstract'

        Returns:
            Publication similarity score [0, 1]
        """
        if not student_interests or not faculty_publications:
            return 0.0

        # Combine student interests into single text
        student_text = ' '.join(student_interests).lower()

        # Extract keywords from student interests
        student_keywords = self._extract_keywords(student_text)

        # Calculate overlap with publications
        pub_scores = []

        for pub in faculty_publications[:10]:  # Limit to recent 10 publications
            pub_text = f"{pub.get('title', '')} {pub.get('abstract', '')}".lower()
            pub_keywords = self._extract_keywords(pub_text)

            overlap = self.calculate_jaccard_similarity(student_keywords, pub_keywords)
            pub_scores.append(overlap)

        if not pub_scores:
            return 0.0

        # Return average of top 3 publications
        pub_scores.sort(reverse=True)
        top_scores = pub_scores[:3]

        return float(np.mean(top_scores))
