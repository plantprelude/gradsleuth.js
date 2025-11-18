"""
Semantic Search Engine - Main orchestrator for intelligent search

Coordinates:
- Query processing (understanding, expansion)
- Embedding generation
- Vector search
- Multi-factor ranking
- Result diversification
- Faceted search
"""

import logging
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class SearchResults:
    """Data class for search results"""
    results: List[Dict]
    total_count: int
    facets: Optional[Dict] = None
    query_interpretation: Optional[Dict] = None
    search_metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            'results': self.results,
            'total_count': self.total_count,
            'facets': self.facets,
            'query_interpretation': self.query_interpretation,
            'search_metadata': self.search_metadata
        }


class SemanticSearchEngine:
    """
    Main search orchestrator for biology research matching platform
    """

    def __init__(
        self,
        embedding_generator,
        vector_store,
        query_processor: Optional[Any] = None,
        result_ranker: Optional[Any] = None
    ):
        """
        Initialize search engine with required components

        Args:
            embedding_generator: EmbeddingGenerator instance
            vector_store: Vector store instance (FAISS, etc.)
            query_processor: QueryProcessor instance (created if None)
            result_ranker: ResultRanker instance (created if None)
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store

        # Initialize query processor if not provided
        if query_processor is None:
            from .query_processor import QueryProcessor
            self.query_processor = QueryProcessor(use_ner=True)
        else:
            self.query_processor = query_processor

        # Initialize result ranker if not provided
        if result_ranker is None:
            from .result_ranker import ResultRanker
            self.result_ranker = ResultRanker()
        else:
            self.result_ranker = result_ranker

        logger.info("SemanticSearchEngine initialized")

    def search(
        self,
        query: str,
        search_mode: str = 'faculty',
        filters: Optional[Dict] = None,
        limit: int = 20,
        offset: int = 0,
        explain: bool = False,
        diversity_factor: float = 0.3
    ) -> SearchResults:
        """
        Complete search pipeline

        Pipeline:
        1. Process query (understand, expand, extract entities)
        2. Generate embeddings for query and expansions
        3. Search vector store with filters
        4. Merge results from multiple query variants
        5. Rerank using multi-factor scoring
        6. Apply diversity if requested
        7. Generate explanations if requested
        8. Return formatted results

        Args:
            query: User's search query
            search_mode: 'faculty', 'publications', 'grants', 'labs'
            filters: Additional metadata filters
            limit: Number of results to return
            offset: Offset for pagination
            explain: Include ranking explanations
            diversity_factor: Diversity weight (0=pure relevance, 1=max diversity)

        Returns:
            SearchResults object with:
            - results: List of ranked results
            - total_count: Total matching documents
            - facets: Aggregated facet data
            - query_interpretation: How query was understood
            - search_metadata: Timing, models used, etc.

        Example:
            >>> engine.search(
            ...     "CRISPR gene editing in stem cells",
            ...     search_mode='faculty',
            ...     limit=10,
            ...     explain=True
            ... )
        """
        start_time = time.time()

        logger.info(f"Search request: query='{query}', mode={search_mode}, limit={limit}")

        # Step 1: Process query
        query_start = time.time()
        query_analysis = self.query_processor.process_query(query)
        query_time = time.time() - query_start

        logger.debug(f"Query processed in {query_time:.3f}s: intent={query_analysis.intent}")

        # Step 2: Merge filters from query analysis and explicit filters
        combined_filters = self._merge_filters(query_analysis.filters, filters)

        # Step 3: Generate embeddings for query variants
        embedding_start = time.time()
        query_embeddings = self._generate_query_embeddings(query_analysis)
        embedding_time = time.time() - embedding_start

        logger.debug(f"Generated {len(query_embeddings)} query embeddings in {embedding_time:.3f}s")

        # Step 4: Search vector store with multiple query variants
        search_start = time.time()
        index_name = f"{search_mode}_embeddings"
        raw_results = self._multi_query_search(
            query_embeddings,
            index_name,
            combined_filters,
            limit=limit * 2  # Get more for reranking
        )
        search_time = time.time() - search_start

        logger.debug(f"Vector search returned {len(raw_results)} results in {search_time:.3f}s")

        # Step 5: Rerank using multi-factor scoring
        rank_start = time.time()
        ranked_results = self.result_ranker.rank_results(
            raw_results,
            query_context=query_analysis
        )
        rank_time = time.time() - rank_start

        logger.debug(f"Results reranked in {rank_time:.3f}s")

        # Step 6: Apply pagination
        total_count = len(ranked_results)
        paginated_results = ranked_results[offset:offset + limit]

        # Step 7: Generate facets
        facets = self._generate_facets(ranked_results) if explain else None

        # Step 8: Build query interpretation
        query_interpretation = {
            'original_query': query,
            'normalized_query': query_analysis.normalized,
            'detected_intent': query_analysis.intent,
            'extracted_entities': query_analysis.entities,
            'implicit_filters': query_analysis.filters,
            'num_expansions': len(query_analysis.expansions)
        } if explain else None

        # Step 9: Build search metadata
        total_time = time.time() - start_time
        search_metadata = {
            'total_time_ms': round(total_time * 1000, 2),
            'query_processing_ms': round(query_time * 1000, 2),
            'embedding_generation_ms': round(embedding_time * 1000, 2),
            'vector_search_ms': round(search_time * 1000, 2),
            'ranking_ms': round(rank_time * 1000, 2),
            'results_returned': len(paginated_results),
            'total_matches': total_count
        } if explain else None

        logger.info(f"Search completed in {total_time:.3f}s, returning {len(paginated_results)} results")

        return SearchResults(
            results=paginated_results,
            total_count=total_count,
            facets=facets,
            query_interpretation=query_interpretation,
            search_metadata=search_metadata
        )

    def _generate_query_embeddings(self, query_analysis: Any) -> List[Dict]:
        """
        Generate embeddings for query and its expansions

        Args:
            query_analysis: QueryAnalysis object

        Returns:
            List of dicts with 'text' and 'embedding' keys
        """
        query_embeddings = []

        # Generate embedding for normalized query
        try:
            main_embedding = self.embedding_generator.generate_embedding(
                query_analysis.normalized
            )
            query_embeddings.append({
                'text': query_analysis.normalized,
                'embedding': main_embedding,
                'weight': 1.0
            })
        except Exception as e:
            logger.error(f"Failed to generate main query embedding: {e}")
            # Use original query as fallback
            try:
                fallback_embedding = self.embedding_generator.generate_embedding(
                    query_analysis.original
                )
                query_embeddings.append({
                    'text': query_analysis.original,
                    'embedding': fallback_embedding,
                    'weight': 1.0
                })
            except Exception as e2:
                logger.error(f"Failed to generate fallback embedding: {e2}")
                raise

        # Generate embeddings for top expansions (limit to 3)
        for i, expansion in enumerate(query_analysis.expansions[:3]):
            if expansion.lower() != query_analysis.normalized.lower():
                try:
                    exp_embedding = self.embedding_generator.generate_embedding(expansion)
                    query_embeddings.append({
                        'text': expansion,
                        'embedding': exp_embedding,
                        'weight': 0.7 / (i + 1)  # Decreasing weight
                    })
                except Exception as e:
                    logger.warning(f"Failed to generate expansion embedding: {e}")

        return query_embeddings

    def _multi_query_search(
        self,
        query_embeddings: List[Dict],
        index_name: str,
        filters: Optional[Dict],
        limit: int
    ) -> List[Dict]:
        """
        Search with multiple query embeddings and merge results

        Args:
            query_embeddings: List of query embeddings with weights
            index_name: Name of vector index
            filters: Metadata filters
            limit: Number of results

        Returns:
            Merged and deduplicated results
        """
        all_results = []

        for query_emb in query_embeddings:
            try:
                # Search with this embedding
                results = self.vector_store.search(
                    query_emb['embedding'],
                    index_name=index_name,
                    k=limit,
                    filters=filters
                )

                # Weight the scores
                weight = query_emb.get('weight', 1.0)
                for result in results:
                    result['score'] *= weight
                    result['query_variant'] = query_emb['text']

                all_results.extend(results)

            except Exception as e:
                logger.error(f"Search failed for query variant: {e}")

        # Merge and deduplicate using Reciprocal Rank Fusion
        merged_results = self._merge_search_results_rrf(all_results)

        return merged_results[:limit]

    def _merge_search_results_rrf(
        self,
        results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Merge results using Reciprocal Rank Fusion (RRF)

        RRF score = sum(1 / (k + rank)) for each appearance

        Args:
            results: List of results from different queries
            k: RRF constant (typically 60)

        Returns:
            Merged and sorted results
        """
        # Group results by ID
        result_groups = defaultdict(list)

        for result in results:
            result_id = result.get('id')
            result_groups[result_id].append(result)

        # Calculate RRF scores
        merged = []

        for result_id, group in result_groups.items():
            # Sort group by score
            group.sort(key=lambda x: x.get('score', 0), reverse=True)

            # Calculate RRF score
            rrf_score = 0.0
            for rank, result in enumerate(group, 1):
                rrf_score += 1.0 / (k + rank)

            # Take the first result as representative
            merged_result = group[0].copy()
            merged_result['score'] = rrf_score
            merged_result['appearances'] = len(group)

            merged.append(merged_result)

        # Sort by RRF score
        merged.sort(key=lambda x: x['score'], reverse=True)

        return merged

    def _merge_filters(
        self,
        implicit_filters: Dict,
        explicit_filters: Optional[Dict]
    ) -> Dict:
        """
        Merge implicit (from query) and explicit filters

        Args:
            implicit_filters: Filters extracted from query
            explicit_filters: Filters provided by user

        Returns:
            Merged filters
        """
        combined = implicit_filters.copy()

        if explicit_filters:
            # Explicit filters override implicit ones
            combined.update(explicit_filters)

        return combined

    def _generate_facets(self, results: List[Dict]) -> Dict[str, Dict]:
        """
        Generate faceted search data

        Facets:
        - institutions: Count by institution
        - departments: Count by department
        - techniques: Count by technique
        - organisms: Count by organism
        - funding_status: Active vs inactive grants
        - career_stage: Distribution

        Args:
            results: Search results

        Returns:
            Facet data with counts
        """
        facets = {
            'institutions': Counter(),
            'departments': Counter(),
            'techniques': Counter(),
            'organisms': Counter(),
            'career_stages': Counter(),
            'funding_status': Counter()
        }

        for result in results:
            metadata = result.get('metadata', {})

            # Institution
            inst = metadata.get('institution')
            if inst:
                facets['institutions'][inst] += 1

            # Department
            dept = metadata.get('department')
            if dept:
                facets['departments'][dept] += 1

            # Techniques
            techniques = metadata.get('techniques', [])
            for tech in techniques:
                facets['techniques'][tech] += 1

            # Organisms
            organisms = metadata.get('organisms', [])
            for org in organisms:
                facets['organisms'][org] += 1

            # Career stage (from title)
            title = metadata.get('title', '')
            if 'assistant' in title.lower():
                facets['career_stages']['Assistant Professor'] += 1
            elif 'associate' in title.lower():
                facets['career_stages']['Associate Professor'] += 1
            elif 'professor' in title.lower() and 'assistant' not in title.lower():
                facets['career_stages']['Full Professor'] += 1

            # Funding status
            has_funding = metadata.get('has_active_funding', False)
            facets['funding_status']['Active Funding' if has_funding else 'No Active Funding'] += 1

        # Convert Counters to dicts with top N
        facet_limit = 10
        return {
            facet_name: dict(counter.most_common(facet_limit))
            for facet_name, counter in facets.items()
        }

    def multi_query_search(
        self,
        queries: List[str],
        aggregation: str = 'union',
        **kwargs
    ) -> SearchResults:
        """
        Search with multiple query variants

        Args:
            queries: List of query strings
            aggregation: How to combine results:
                - 'union': All results from all queries
                - 'intersection': Only results appearing in all queries
                - 'weighted': Weighted by how many queries matched
            **kwargs: Additional search parameters

        Returns:
            Aggregated SearchResults
        """
        all_results = []

        for query in queries:
            try:
                results = self.search(query, **kwargs)
                all_results.append(results.results)
            except Exception as e:
                logger.error(f"Search failed for query '{query}': {e}")

        # Aggregate based on strategy
        if aggregation == 'union':
            merged = self._union_results(all_results)
        elif aggregation == 'intersection':
            merged = self._intersection_results(all_results)
        elif aggregation == 'weighted':
            merged = self._weighted_results(all_results)
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")

        return SearchResults(
            results=merged,
            total_count=len(merged),
            facets=self._generate_facets(merged),
            query_interpretation={'multi_query': queries, 'aggregation': aggregation},
            search_metadata={'num_queries': len(queries)}
        )

    def _union_results(self, result_sets: List[List[Dict]]) -> List[Dict]:
        """Union of all result sets"""
        seen_ids = set()
        union = []

        for result_set in result_sets:
            for result in result_set:
                result_id = result.get('id')
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    union.append(result)

        return union

    def _intersection_results(self, result_sets: List[List[Dict]]) -> List[Dict]:
        """Intersection of result sets"""
        if not result_sets:
            return []

        # Get IDs from first set
        id_sets = [set(r.get('id') for r in rs) for rs in result_sets]
        common_ids = set.intersection(*id_sets)

        # Return results from first set that are in common
        return [r for r in result_sets[0] if r.get('id') in common_ids]

    def _weighted_results(self, result_sets: List[List[Dict]]) -> List[Dict]:
        """Weighted results by number of appearances"""
        result_map = {}

        for result_set in result_sets:
            for result in result_set:
                result_id = result.get('id')
                if result_id in result_map:
                    # Increase weight
                    result_map[result_id]['final_score'] += result.get('final_score', result.get('score', 0))
                    result_map[result_id]['appearances'] += 1
                else:
                    result_copy = result.copy()
                    result_copy['appearances'] = 1
                    result_map[result_id] = result_copy

        # Sort by weighted score
        weighted = list(result_map.values())
        weighted.sort(key=lambda x: x.get('final_score', 0), reverse=True)

        return weighted

    def search_by_example(
        self,
        example_profile: Dict,
        search_mode: str = 'faculty',
        **kwargs
    ) -> SearchResults:
        """
        Find similar profiles to provided example

        Args:
            example_profile: Example faculty/publication profile with:
                - research_summary: text
                - publications: list of titles/abstracts
                - grants: list of grant abstracts
            search_mode: Type of search
            **kwargs: Additional parameters

        Returns:
            SearchResults with similar profiles
        """
        # Combine profile text
        profile_texts = []

        if 'research_summary' in example_profile:
            profile_texts.append(example_profile['research_summary'])

        if 'publications' in example_profile:
            for pub in example_profile['publications'][:5]:
                if isinstance(pub, dict):
                    profile_texts.append(pub.get('title', '') + ' ' + pub.get('abstract', ''))
                else:
                    profile_texts.append(str(pub))

        combined_text = ' '.join(profile_texts)

        # Use combined text as query
        return self.search(
            query=combined_text,
            search_mode=search_mode,
            **kwargs
        )
