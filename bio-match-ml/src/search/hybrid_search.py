"""
Hybrid search combining semantic (vector) and lexical (keyword) search
"""
import logging
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Hybrid search combining semantic (vector) and lexical (keyword) search
    """

    def __init__(
        self,
        vector_store,
        embedding_generator,
        text_index: Optional[Any] = None
    ):
        """
        Initialize hybrid search engine

        Args:
            vector_store: Vector store for semantic search
            embedding_generator: For generating query embeddings
            text_index: Optional text search index (BM25, TF-IDF)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.text_index = text_index

        # If no text index provided, we'll use simple in-memory TF-IDF
        self.use_simple_search = text_index is None

        logger.info(f"HybridSearchEngine initialized (text_index={'simple' if self.use_simple_search else 'external'})")

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        index_name: str,
        k: int = 20,
        alpha: float = 0.5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining vector and keyword results

        Args:
            query: Text query
            query_embedding: Query embedding vector
            index_name: Index to search
            k: Number of results
            alpha: Weight between vector (1.0) and keyword (0.0)
                   alpha=0.5 means equal weighting
            filters: Metadata filters

        Returns:
            Combined and reranked results

        Algorithm:
            1. Perform vector search → get top k*2 results
            2. Perform keyword search → get top k*2 results
            3. Combine using Reciprocal Rank Fusion (RRF)
            4. Apply alpha weighting
            5. Return top k

        Example:
            >>> engine.hybrid_search(
            ...     query="CRISPR in neurons",
            ...     query_embedding=embedding,
            ...     index_name="faculty",
            ...     k=10,
            ...     alpha=0.6  # Prefer semantic over keyword
            ... )
        """
        logger.debug(f"Hybrid search: query='{query}', k={k}, alpha={alpha}")

        # Fetch more results for better fusion
        fetch_k = k * 2

        # 1. Vector search
        try:
            vector_results = self.vector_store.search(
                query_embedding=query_embedding,
                index_name=index_name,
                k=fetch_k,
                filters=filters
            )
            logger.debug(f"Vector search returned {len(vector_results)} results")
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            vector_results = []

        # 2. Keyword search
        try:
            keyword_results = self._keyword_search(
                query=query,
                index_name=index_name,
                k=fetch_k,
                filters=filters
            )
            logger.debug(f"Keyword search returned {len(keyword_results)} results")
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            keyword_results = []

        # 3. Combine using alpha weighting and RRF
        if alpha == 1.0:
            # Pure vector search
            combined = vector_results
        elif alpha == 0.0:
            # Pure keyword search
            combined = keyword_results
        else:
            # Hybrid combination
            combined = self._combine_with_alpha(
                vector_results,
                keyword_results,
                alpha
            )

        # Sort by final score
        combined.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

        # Return top k
        return combined[:k]

    def _keyword_search(
        self,
        query: str,
        index_name: str,
        k: int,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform keyword/BM25 search

        If text_index is available, use it.
        Otherwise, fall back to simple TF-IDF search on metadata.

        Args:
            query: Text query
            index_name: Index name
            k: Number of results
            filters: Metadata filters

        Returns:
            Keyword search results with BM25 scores
        """
        if self.text_index is not None:
            # Use external text index (e.g., Elasticsearch)
            return self._external_text_search(query, index_name, k, filters)
        else:
            # Use simple in-memory TF-IDF
            return self._simple_tfidf_search(query, index_name, k, filters)

    def _simple_tfidf_search(
        self,
        query: str,
        index_name: str,
        k: int,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Simple TF-IDF based keyword search

        This is a fallback when no text index is available.
        Searches metadata text fields.

        Args:
            query: Search query
            index_name: Index name
            k: Number of results
            filters: Filters

        Returns:
            Keyword search results
        """
        # Get all documents from vector store (or a sample)
        # This is simplified - in production, you'd have a proper index
        try:
            # Get more documents for filtering
            all_docs = self.vector_store.search(
                query_embedding=np.zeros(768),  # Dummy embedding
                index_name=index_name,
                k=1000,  # Fetch many for keyword matching
                filters=filters
            )
        except:
            logger.warning("Could not fetch documents for keyword search")
            return []

        # Extract query terms
        query_terms = self._tokenize(query.lower())
        query_term_counts = Counter(query_terms)

        # Score each document
        scored_docs = []
        for doc in all_docs:
            metadata = doc.get('metadata', {})

            # Combine searchable text fields
            text_fields = []
            if metadata.get('name'):
                text_fields.append(metadata['name'])
            if metadata.get('research_summary'):
                text_fields.append(metadata['research_summary'])
            if metadata.get('title'):
                text_fields.append(metadata['title'])

            doc_text = ' '.join(text_fields).lower()
            doc_terms = self._tokenize(doc_text)

            # Calculate TF-IDF-like score
            score = self._calculate_bm25_score(query_term_counts, doc_terms)

            if score > 0:
                doc_copy = doc.copy()
                doc_copy['keyword_score'] = score
                scored_docs.append(doc_copy)

        # Sort by score
        scored_docs.sort(key=lambda x: x['keyword_score'], reverse=True)

        return scored_docs[:k]

    def _calculate_bm25_score(
        self,
        query_term_counts: Counter,
        doc_terms: List[str],
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """
        Simplified BM25 scoring

        Args:
            query_term_counts: Counter of query terms
            doc_terms: List of document terms
            k1: BM25 parameter
            b: BM25 parameter

        Returns:
            BM25 score
        """
        doc_term_counts = Counter(doc_terms)
        doc_length = len(doc_terms)
        avg_doc_length = 100  # Assumed average

        score = 0.0
        for term, query_count in query_term_counts.items():
            if term in doc_term_counts:
                tf = doc_term_counts[term]

                # Simplified BM25 formula (without IDF component)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))

                score += (numerator / denominator) * query_count

        return score

    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        # Remove stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'
        }

        return [t for t in tokens if t and t not in stopwords]

    def _external_text_search(
        self,
        query: str,
        index_name: str,
        k: int,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Use external text search index (placeholder)

        Args:
            query: Search query
            index_name: Index name
            k: Number of results
            filters: Filters

        Returns:
            Search results
        """
        # This would integrate with Elasticsearch, Whoosh, etc.
        # For now, return empty
        logger.warning("External text index not implemented")
        return []

    def reciprocal_rank_fusion(
        self,
        result_sets: List[List[Dict]],
        k: int = 60
    ) -> List[Dict]:
        """
        Combine multiple ranked lists using RRF

        RRF score = sum(1 / (k + rank_i)) for each list

        Args:
            result_sets: List of ranked result lists
            k: RRF parameter (typically 60)

        Returns:
            Combined ranked results
        """
        rrf_scores = defaultdict(float)
        result_map = {}

        for result_set in result_sets:
            for rank, result in enumerate(result_set, 1):
                result_id = result.get('id', str(hash(str(result))))

                # Accumulate RRF score
                rrf_scores[result_id] += 1.0 / (k + rank)

                # Store result
                if result_id not in result_map:
                    result_map[result_id] = result

        # Create merged results
        merged = []
        for result_id, score in rrf_scores.items():
            result = result_map[result_id].copy()
            result['rrf_score'] = score
            merged.append(result)

        # Sort by RRF score
        merged.sort(key=lambda x: x['rrf_score'], reverse=True)

        return merged

    def _combine_with_alpha(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        alpha: float
    ) -> List[Dict]:
        """
        Combine results with alpha weighting

        Final score = alpha * vector_score + (1-alpha) * keyword_score

        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            alpha: Weight parameter

        Returns:
            Combined results
        """
        # Normalize scores to [0, 1] range
        vector_results = self._normalize_scores(vector_results, 'score')
        keyword_results = self._normalize_scores(keyword_results, 'keyword_score')

        # Create score maps
        vector_scores = {
            r.get('id', str(hash(str(r)))): r.get('score', 0)
            for r in vector_results
        }

        keyword_scores = {
            r.get('id', str(hash(str(r)))): r.get('keyword_score', 0)
            for r in keyword_results
        }

        # Combine all unique IDs
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())

        # Create result map
        result_map = {}
        for r in vector_results + keyword_results:
            result_id = r.get('id', str(hash(str(r))))
            if result_id not in result_map:
                result_map[result_id] = r

        # Calculate hybrid scores
        combined = []
        for result_id in all_ids:
            result = result_map.get(result_id)
            if not result:
                continue

            vector_score = vector_scores.get(result_id, 0)
            keyword_score = keyword_scores.get(result_id, 0)

            # Weighted combination
            hybrid_score = alpha * vector_score + (1 - alpha) * keyword_score

            result = result.copy()
            result['hybrid_score'] = hybrid_score
            result['vector_score'] = vector_score
            result['keyword_score'] = keyword_score
            combined.append(result)

        return combined

    def _normalize_scores(
        self,
        results: List[Dict],
        score_key: str
    ) -> List[Dict]:
        """
        Normalize scores to [0, 1] range

        Args:
            results: Results to normalize
            score_key: Key containing score

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        scores = [r.get(score_key, 0) for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores the same
            for r in results:
                r[score_key] = 1.0
            return results

        # Min-max normalization
        for r in results:
            original_score = r.get(score_key, 0)
            normalized = (original_score - min_score) / (max_score - min_score)
            r[score_key] = normalized

        return results

    def optimize_alpha(
        self,
        validation_queries: List[Tuple[str, List[str]]],
        alpha_range: List[float] = [0.3, 0.5, 0.7, 0.9]
    ) -> float:
        """
        Find optimal alpha value using validation data

        Args:
            validation_queries: List of (query, relevant_doc_ids)
            alpha_range: Alpha values to test

        Returns:
            Best alpha value based on NDCG
        """
        logger.info(f"Optimizing alpha over {len(alpha_range)} values")

        best_alpha = 0.5
        best_score = 0.0

        for alpha in alpha_range:
            total_ndcg = 0.0

            for query, relevant_ids in validation_queries:
                # Generate embedding
                try:
                    embedding = self.embedding_generator.generate_embedding(query)

                    # Perform hybrid search
                    results = self.hybrid_search(
                        query=query,
                        query_embedding=embedding,
                        index_name='faculty_embeddings',
                        k=20,
                        alpha=alpha
                    )

                    # Calculate NDCG
                    ndcg = self._calculate_ndcg(results, relevant_ids)
                    total_ndcg += ndcg

                except Exception as e:
                    logger.error(f"Error in alpha optimization: {e}")

            avg_ndcg = total_ndcg / len(validation_queries) if validation_queries else 0

            logger.info(f"Alpha={alpha:.2f}, NDCG={avg_ndcg:.4f}")

            if avg_ndcg > best_score:
                best_score = avg_ndcg
                best_alpha = alpha

        logger.info(f"Optimal alpha: {best_alpha:.2f} (NDCG={best_score:.4f})")

        return best_alpha

    def _calculate_ndcg(
        self,
        results: List[Dict],
        relevant_ids: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain

        Args:
            results: Search results
            relevant_ids: List of relevant document IDs
            k: Top k results to consider

        Returns:
            NDCG@k score
        """
        # Get result IDs
        result_ids = [r.get('id') for r in results[:k]]

        # Calculate DCG
        dcg = 0.0
        for i, result_id in enumerate(result_ids, 1):
            if result_id in relevant_ids:
                # Relevance = 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 1)

        # Calculate ideal DCG (all relevant docs at top)
        num_relevant = min(len(relevant_ids), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))

        # NDCG
        if idcg == 0:
            return 0.0

        return dcg / idcg
