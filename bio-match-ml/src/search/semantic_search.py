"""
Main search orchestrator for biology research matching platform
"""
import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter

from .query_processor import QueryProcessor, QueryAnalysis
from .result_ranker import ResultRanker

logger = logging.getLogger(__name__)


@dataclass
class SearchResults:
    """Data class for search results"""
    results: List[Dict]
    total_count: int
    facets: Optional[Dict] = None
    query_interpretation: Optional[Dict] = None
    search_metadata: Optional[Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        return {
            'results': self.results,
            'total_count': self.total_count,
            'facets': self.facets or {},
            'query_interpretation': self.query_interpretation or {},
            'search_metadata': self.search_metadata or {}
        }


class SemanticSearchEngine:
    """
    Main search orchestrator for biology research matching platform
    """

    def __init__(
        self,
        embedding_generator,
        vector_store,
        query_processor: Optional[QueryProcessor] = None,
        result_ranker: Optional[ResultRanker] = None
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

        # Initialize or use provided components
        self.query_processor = query_processor if query_processor else QueryProcessor()
        self.result_ranker = result_ranker if result_ranker else ResultRanker()

        # Query embedding cache for performance
        self._query_cache = {}
        self._cache_max_size = 1000

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
            SearchResults object with comprehensive search information
        """
        start_time = time.time()

        logger.info(f"Search started: query='{query}', mode={search_mode}, limit={limit}")

        # Step 1: Process query
        query_context = self.query_processor.process_query(query)

        # Merge implicit filters with explicit filters
        combined_filters = {**(filters or {}), **query_context.filters}

        # Step 2: Generate embeddings for query variants
        query_embeddings = self._generate_query_embeddings(query_context)

        # Step 3: Search vector store with multiple query variants
        index_name = self._get_index_name(search_mode)

        # Fetch more results than limit for better ranking
        fetch_limit = limit * 3

        all_results = []
        for query_text, embedding in query_embeddings:
            try:
                results = self.vector_store.search(
                    query_embedding=embedding,
                    index_name=index_name,
                    k=fetch_limit,
                    filters=combined_filters
                )
                all_results.append(results)
            except Exception as e:
                logger.error(f"Vector search error for '{query_text}': {e}")

        # Step 4: Merge results from multiple searches
        if all_results:
            merged_results = self._merge_search_results(all_results, aggregation='weighted')
        else:
            logger.warning("No search results obtained")
            merged_results = []

        # Step 5: Rerank using multi-factor scoring
        if merged_results:
            ranked_results = self.result_ranker.rank_results(
                results=merged_results,
                query_context=query_context
            )
        else:
            ranked_results = []

        # Step 6: Apply diversity if requested
        if diversity_factor > 0 and ranked_results:
            ranked_results = self.result_ranker.diversify_results(
                ranked_results,
                diversity_factor=diversity_factor
            )

        # Step 7: Paginate results
        total_count = len(ranked_results)
        paginated_results = ranked_results[offset:offset + limit]

        # Step 8: Generate facets
        facets = self._generate_facets(ranked_results)

        # Prepare query interpretation
        query_interpretation = {
            'original_query': query_context.original,
            'normalized_query': query_context.normalized,
            'intent': query_context.intent,
            'entities': query_context.entities,
            'applied_filters': combined_filters,
            'expansions_used': len(query_context.expansions)
        }

        # Search metadata
        search_time = time.time() - start_time
        search_metadata = {
            'search_time_ms': round(search_time * 1000, 2),
            'results_before_ranking': len(merged_results),
            'results_after_ranking': total_count,
            'query_variants_used': len(query_embeddings),
            'model_used': 'pubmedbert'  # Default model
        }

        logger.info(f"Search completed: {total_count} results in {search_time:.3f}s")

        return SearchResults(
            results=paginated_results,
            total_count=total_count,
            facets=facets,
            query_interpretation=query_interpretation,
            search_metadata=search_metadata
        )

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
        logger.info(f"Multi-query search with {len(queries)} queries, aggregation={aggregation}")

        all_search_results = []
        for query in queries:
            results = self.search(query, **kwargs)
            all_search_results.append(results.results)

        # Merge based on aggregation strategy
        if aggregation == 'union':
            # Simple union with deduplication
            merged = self._deduplicate_results(
                [r for result_list in all_search_results for r in result_list]
            )
        elif aggregation == 'intersection':
            # Only results in all queries
            merged = self._intersect_results(all_search_results)
        elif aggregation == 'weighted':
            # Weight by number of queries matched
            merged = self._weighted_merge(all_search_results)
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation}")

        # Sort by score
        merged.sort(key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)

        return SearchResults(
            results=merged[:kwargs.get('limit', 20)],
            total_count=len(merged),
            query_interpretation={'queries': queries, 'aggregation': aggregation}
        )

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
        logger.info(f"Search by example: mode={search_mode}")

        # Construct query from example profile
        query_parts = []

        if example_profile.get('research_summary'):
            query_parts.append(example_profile['research_summary'])

        if example_profile.get('publications'):
            # Use recent publication titles/abstracts
            for pub in example_profile['publications'][:5]:
                if isinstance(pub, dict):
                    title = pub.get('title', '')
                    abstract = pub.get('abstract', '')
                    query_parts.append(f"{title} {abstract}")

        if example_profile.get('grants'):
            # Use grant abstracts
            for grant in example_profile['grants'][:3]:
                if isinstance(grant, dict):
                    abstract = grant.get('abstract', '')
                    query_parts.append(abstract)

        # Combine and truncate
        combined_query = " ".join(query_parts)[:2000]  # Limit length

        # Perform search
        return self.search(combined_query, search_mode=search_mode, **kwargs)

    def _generate_query_embeddings(
        self,
        query_context: QueryAnalysis
    ) -> List[Tuple[str, List[float]]]:
        """
        Generate embeddings for query and its expansions

        Args:
            query_context: Processed query information

        Returns:
            List of (query_text, embedding) tuples
        """
        embeddings = []

        # Always include original query
        queries_to_embed = [query_context.normalized]

        # Add expansions (up to 3)
        if query_context.expansions:
            queries_to_embed.extend(query_context.expansions[:3])

        for query_text in queries_to_embed:
            # Check cache
            cache_key = query_text.lower()
            if cache_key in self._query_cache:
                embedding = self._query_cache[cache_key]
            else:
                try:
                    embedding = self.embedding_generator.generate_embedding(query_text)

                    # Cache the embedding
                    if len(self._query_cache) >= self._cache_max_size:
                        # Simple cache eviction: remove oldest
                        self._query_cache.pop(next(iter(self._query_cache)))

                    self._query_cache[cache_key] = embedding

                except Exception as e:
                    logger.error(f"Error generating embedding for '{query_text}': {e}")
                    continue

            embeddings.append((query_text, embedding))

        return embeddings

    def _merge_search_results(
        self,
        result_sets: List[List[Dict]],
        aggregation: str = 'union'
    ) -> List[Dict]:
        """
        Merge results from multiple searches

        Uses Reciprocal Rank Fusion (RRF) for weighted aggregation

        Args:
            result_sets: List of result lists from different queries
            aggregation: Aggregation strategy

        Returns:
            Merged and deduplicated results
        """
        if not result_sets:
            return []

        if len(result_sets) == 1:
            return result_sets[0]

        if aggregation == 'weighted':
            return self._reciprocal_rank_fusion(result_sets)
        else:
            # Simple union with deduplication
            all_results = [r for result_set in result_sets for r in result_set]
            return self._deduplicate_results(all_results)

    def _reciprocal_rank_fusion(
        self,
        result_sets: List[List[Dict]],
        k: int = 60
    ) -> List[Dict]:
        """
        Combine ranked lists using Reciprocal Rank Fusion

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

                # Add RRF score
                rrf_scores[result_id] += 1.0 / (k + rank)

                # Store result (use first occurrence)
                if result_id not in result_map:
                    result_map[result_id] = result

        # Create merged results with RRF scores
        merged = []
        for result_id, rrf_score in rrf_scores.items():
            result = result_map[result_id].copy()
            result['rrf_score'] = rrf_score
            result['score'] = rrf_score  # Use RRF as main score
            merged.append(result)

        # Sort by RRF score
        merged.sort(key=lambda x: x['rrf_score'], reverse=True)

        return merged

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Deduplicate results by ID"""
        seen = set()
        deduped = []

        for result in results:
            result_id = result.get('id')
            if result_id and result_id not in seen:
                seen.add(result_id)
                deduped.append(result)
            elif not result_id:
                # No ID, keep it
                deduped.append(result)

        return deduped

    def _intersect_results(self, result_sets: List[List[Dict]]) -> List[Dict]:
        """Get intersection of result sets"""
        if not result_sets:
            return []

        # Get IDs from each set
        id_sets = [
            {r.get('id') for r in result_set if r.get('id')}
            for result_set in result_sets
        ]

        # Find intersection
        common_ids = set.intersection(*id_sets) if id_sets else set()

        # Get results for common IDs
        id_to_result = {}
        for result_set in result_sets:
            for result in result_set:
                result_id = result.get('id')
                if result_id in common_ids and result_id not in id_to_result:
                    id_to_result[result_id] = result

        return list(id_to_result.values())

    def _weighted_merge(self, result_sets: List[List[Dict]]) -> List[Dict]:
        """Weight results by how many queries they appeared in"""
        result_counts = Counter()
        result_map = {}

        for result_set in result_sets:
            for result in result_set:
                result_id = result.get('id', str(hash(str(result))))
                result_counts[result_id] += 1

                if result_id not in result_map:
                    result_map[result_id] = result

        # Create weighted results
        weighted = []
        for result_id, count in result_counts.items():
            result = result_map[result_id].copy()
            # Boost score by number of queries matched
            original_score = result.get('score', 0.5)
            result['score'] = original_score * (1 + 0.2 * (count - 1))
            result['query_match_count'] = count
            weighted.append(result)

        return weighted

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
            'funding_status': Counter(),
            'career_stage': Counter(),
            'research_areas': Counter()
        }

        for result in results:
            metadata = result.get('metadata', {})

            # Institution
            if metadata.get('institution'):
                facets['institutions'][metadata['institution']] += 1

            # Department
            if metadata.get('department'):
                facets['departments'][metadata['department']] += 1

            # Techniques
            techniques = metadata.get('techniques', [])
            for tech in techniques:
                tech_name = tech if isinstance(tech, str) else tech.get('name', '')
                if tech_name:
                    facets['techniques'][tech_name] += 1

            # Organisms
            organisms = metadata.get('organisms', [])
            for org in organisms:
                org_name = org if isinstance(org, str) else org.get('common_name', '')
                if org_name:
                    facets['organisms'][org_name] += 1

            # Funding status
            has_funding = metadata.get('active_grants', 0) > 0
            facets['funding_status']['funded' if has_funding else 'unfunded'] += 1

            # Career stage
            if metadata.get('career_stage'):
                facets['career_stage'][metadata['career_stage']] += 1

            # Research areas
            if metadata.get('primary_research_area'):
                facets['research_areas'][metadata['primary_research_area']] += 1

        # Convert to regular dicts and get top items
        facet_output = {}
        for facet_name, counter in facets.items():
            facet_output[facet_name] = dict(counter.most_common(10))

        return facet_output

    def _get_index_name(self, search_mode: str) -> str:
        """
        Get vector store index name for search mode

        Args:
            search_mode: Search mode

        Returns:
            Index name
        """
        index_mapping = {
            'faculty': 'faculty_embeddings',
            'publications': 'publication_embeddings',
            'grants': 'grant_embeddings',
            'labs': 'lab_embeddings'
        }

        return index_mapping.get(search_mode, 'faculty_embeddings')

    def clear_cache(self):
        """Clear query embedding cache"""
        self._query_cache.clear()
        logger.info("Query cache cleared")
