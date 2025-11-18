# Claude Code Prompt: Core Search Engine & Matching System - Batch 1

## Context

You are working on the `bio-match-ml` component of a biology research matching platform. The foundational infrastructure exists (embeddings, vector stores, NER, API), but the **intelligent search engine and matching system are missing**.

Currently, the API endpoints directly call vector stores, resulting in:
- No query understanding or expansion
- No multi-factor ranking
- Hardcoded match scores
- Generic explanations

Your task is to build the **core search and matching intelligence** that transforms basic vector similarity into a production-ready semantic search and faculty-student matching system.

## Existing Infrastructure (Already Built)

```
bio-match-ml/
├── src/
│   ├── embeddings/
│   │   ├── embedding_generator.py     ✅ Complete - generates embeddings
│   │   ├── model_manager.py           ✅ Complete - manages models
│   │   ├── cache_manager.py           ✅ Complete - caches embeddings
│   │   └── fine_tuner.py              ✅ Complete - fine-tunes models
│   ├── vector_stores/
│   │   ├── base_store.py              ✅ Complete - abstract interface
│   │   └── faiss_store.py             ✅ Complete - FAISS implementation
│   ├── ner/
│   │   ├── bio_entity_recognizer.py   ✅ Complete - extracts entities
│   │   └── research_topic_classifier.py ✅ Complete - classifies topics
│   └── api/
│       └── main.py                     ✅ Complete - API endpoints (needs integration)
```

## Your Deliverables - Batch 1

Build the following 5 files in the `bio-match-ml/src/` directory:

### 1. `search/semantic_search.py` - Main Search Orchestrator
The central search engine that coordinates all search components.

### 2. `search/query_processor.py` - Query Intelligence
Query understanding, expansion, entity extraction, and intent detection.

### 3. `search/result_ranker.py` - Multi-Factor Ranking
Sophisticated ranking beyond simple vector similarity.

### 4. `matching/similarity_calculator.py` - Similarity Metrics
Calculate various similarity metrics for faculty-student matching.

### 5. `matching/multi_factor_scorer.py` - Comprehensive Matching
Multi-dimensional faculty-student matching with explainable scores.

---

## Detailed Specifications

## File 1: `src/search/query_processor.py`

**Purpose**: Transform raw user queries into optimized search parameters.

**Key Classes & Methods**:

```python
class QueryProcessor:
    """
    Intelligent query understanding and expansion for biology domain
    """

    def __init__(self, entity_recognizer=None, topic_classifier=None):
        """
        Initialize with existing NER and classification models

        Args:
            entity_recognizer: BioEntityRecognizer instance
            topic_classifier: ResearchTopicClassifier instance
        """
        pass

    def process_query(self, raw_query: str) -> QueryAnalysis:
        """
        Complete query analysis pipeline

        Args:
            raw_query: User's search query

        Returns:
            QueryAnalysis object containing:
            - original: raw query
            - normalized: cleaned query
            - entities: extracted biological entities
            - intent: detected search intent
            - expansions: query variations
            - filters: implicit filters from query
            - boost_terms: important terms for boosting

        Example:
            Input: "young PI studying CRISPR in neurons at Harvard"
            Output: QueryAnalysis(
                original="young PI studying CRISPR in neurons at Harvard",
                normalized="principal investigator crispr neurons harvard",
                entities={'techniques': ['CRISPR'], 'cell_types': ['neurons']},
                intent='technique_based',
                expansions=["CRISPR cas9", "gene editing neurons", ...],
                filters={'institution': 'Harvard', 'career_stage': 'assistant_professor'},
                boost_terms=['CRISPR', 'neurons']
            )
        """
        pass

    def expand_query_scientifically(self, query: str, entities: Dict) -> List[str]:
        """
        Generate scientific query expansions using domain knowledge

        Expansions include:
        - Synonyms from biology ontologies
        - Related techniques
        - Organism variants (e.g., "mouse" → "Mus musculus")
        - Protein/gene aliases (e.g., "p53" → "TP53", "tumor protein 53")

        Args:
            query: Normalized query
            entities: Extracted entities from NER

        Returns:
            List of expanded query strings

        Example:
            Input: "p53 cancer"
            Output: ["p53 cancer", "TP53 tumor", "tumor protein 53 neoplasm", ...]
        """
        pass

    def detect_query_intent(self, query: str, entities: Dict) -> str:
        """
        Classify the user's search intent

        Intent types:
        - 'specific_person': Looking for a known researcher (has name)
        - 'research_area': Exploring a research field (broad terms)
        - 'technique_based': Finding technique experts (methods mentioned)
        - 'collaborative': Seeking collaboration partners (interdisciplinary)
        - 'funding_based': Grant-related search (funding keywords)
        - 'organism_based': Focused on model organism

        Args:
            query: User query
            entities: Extracted entities

        Returns:
            Intent classification string
        """
        pass

    def extract_implicit_filters(self, query: str) -> Dict[str, Any]:
        """
        Parse filters from natural language

        Recognize patterns like:
        - "young PI" → career_stage: assistant_professor
        - "well-funded" → has_active_funding: true
        - "at MIT" → institution: MIT
        - "prolific researcher" → min_publications: high_threshold

        Args:
            query: User query

        Returns:
            Dictionary of extracted filters
        """
        pass

    def normalize_biology_terms(self, text: str) -> str:
        """
        Normalize biological terminology

        - Standardize organism names
        - Expand common abbreviations
        - Handle case sensitivity appropriately

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        pass

class QueryAnalysis:
    """Data class for query analysis results"""

    def __init__(
        self,
        original: str,
        normalized: str,
        entities: Dict[str, List],
        intent: str,
        expansions: List[str],
        filters: Dict[str, Any],
        boost_terms: List[str]
    ):
        self.original = original
        self.normalized = normalized
        self.entities = entities
        self.intent = intent
        self.expansions = expansions
        self.filters = filters
        self.boost_terms = boost_terms
```

**Implementation Requirements**:
- Use the existing `BioEntityRecognizer` from `src/ner/bio_entity_recognizer.py`
- Use the existing `ResearchTopicClassifier` from `src/ner/research_topic_classifier.py`
- Handle edge cases: empty queries, very short queries, non-English characters
- Include comprehensive docstrings
- Add logging for debugging

**Biology-Specific Knowledge**:
Include hardcoded dictionaries for common mappings:
```python
CAREER_STAGE_KEYWORDS = {
    'young': 'assistant_professor',
    'new': 'assistant_professor',
    'junior': 'assistant_professor',
    'established': 'associate_professor',
    'senior': 'full_professor',
    'emeritus': 'emeritus'
}

COMMON_GENE_ALIASES = {
    'p53': ['TP53', 'tumor protein 53'],
    'egfr': ['epidermal growth factor receptor'],
    # Add more as needed
}

TECHNIQUE_SYNONYMS = {
    'crispr': ['CRISPR-Cas9', 'gene editing', 'genome editing'],
    'rna-seq': ['RNA sequencing', 'transcriptomics'],
    'pcr': ['polymerase chain reaction'],
    # Add more
}
```

---

## File 2: `src/search/result_ranker.py`

**Purpose**: Rank search results using multiple factors beyond vector similarity.

**Key Classes & Methods**:

```python
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
        pass

    def rank_results(
        self,
        results: List[Dict],
        query_context: QueryAnalysis,
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
        pass

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
        pass

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
        pass

    def calculate_recency_score(self, metadata: Dict) -> float:
        """
        Score based on recency of work

        Uses exponential decay: more recent = higher score

        Args:
            metadata: Metadata with publication dates

        Returns:
            Recency score [0.0, 1.0]
        """
        pass

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
        pass

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
        pass

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
        pass
```

**Implementation Requirements**:
- Normalize all scores to [0, 1] range
- Handle missing metadata gracefully (default to neutral scores)
- Make ranking weights configurable
- Add detailed logging for ranking decisions
- Include methods for A/B testing different weight configurations

---

## File 3: `src/search/semantic_search.py`

**Purpose**: Main search orchestrator that ties everything together.

**Key Classes & Methods**:

```python
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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

class SearchResults:
    """Data class for search results"""

    def __init__(
        self,
        results: List[Dict],
        total_count: int,
        facets: Optional[Dict] = None,
        query_interpretation: Optional[Dict] = None,
        search_metadata: Optional[Dict] = None
    ):
        self.results = results
        self.total_count = total_count
        self.facets = facets
        self.query_interpretation = query_interpretation
        self.search_metadata = search_metadata

    def to_dict(self) -> Dict:
        """Convert to dictionary for API responses"""
        pass
```

**Implementation Requirements**:
- Comprehensive error handling
- Performance logging (query time, vector search time, ranking time)
- Cache query embeddings for repeated searches
- Support pagination efficiently
- Handle empty results gracefully

---

## File 4: `src/matching/similarity_calculator.py`

**Purpose**: Calculate various similarity metrics for faculty-student matching.

**Key Classes & Methods**:

```python
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
        pass

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
        pass

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
            Cosine similarity score
        """
        pass

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
        pass

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
            Weighted overlap score
        """
        pass

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
        pass
```

**Implementation Requirements**:
- Efficient computation (batch operations where possible)
- Handle missing data gracefully
- Normalize all scores to [0, 1]
- Add caching for expensive similarity calculations

---

## File 5: `src/matching/multi_factor_scorer.py`

**Purpose**: Comprehensive faculty-student matching with explainable scores.

**Key Classes & Methods**:

```python
class MultiFactorMatcher:
    """
    Comprehensive faculty-student matching system
    """

    def __init__(
        self,
        similarity_calculator: Optional[SimilarityCalculator] = None,
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
        pass

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
            MatchScore object with:
            - overall_score: Weighted combination [0, 1]
            - component_scores: Dict of individual factor scores
            - confidence: Confidence in the match prediction
            - explanation: Human-readable explanation
            - strengths: List of match strengths
            - considerations: List of potential concerns
            - recommendation: 'highly_recommended', 'recommended', 'consider', 'not_recommended'

        Example:
            MatchScore(
                overall_score=0.87,
                component_scores={
                    'research_alignment': 0.92,
                    'funding_stability': 0.85,
                    'productivity_match': 0.88,
                    'technique_match': 0.90,
                    'lab_environment': 0.75,
                    'career_development': 0.82
                },
                confidence=0.85,
                explanation="Excellent research fit with 92% alignment...",
                strengths=[
                    "Very high research similarity (0.92)",
                    "Strong technique overlap: CRISPR, RNA-seq",
                    "Active funding through 2027",
                    "Faculty accepting students"
                ],
                considerations=[
                    "Lab is large (12 members), may have less 1-on-1 time",
                    "Very competitive - strong application required"
                ],
                recommendation='highly_recommended'
            )
        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass

    def get_recommendation(self, overall_score: float, confidence: float) -> str:
        """
        Get categorical recommendation

        Args:
            overall_score: Overall match score
            confidence: Confidence in prediction

        Returns:
            One of: 'highly_recommended', 'recommended', 'consider', 'not_recommended'
        """
        pass

class MatchScore:
    """Data class for match results"""

    def __init__(
        self,
        overall_score: float,
        component_scores: Dict[str, float],
        confidence: float,
        explanation: str,
        strengths: List[str],
        considerations: List[str],
        recommendation: str
    ):
        self.overall_score = overall_score
        self.component_scores = component_scores
        self.confidence = confidence
        self.explanation = explanation
        self.strengths = strengths
        self.considerations = considerations
        self.recommendation = recommendation

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        pass
```

**Implementation Requirements**:
- Make all weights configurable
- Handle missing data gracefully (use reasonable defaults)
- Add confidence estimation based on data completeness
- Include detailed logging for debugging
- Make explanations specific and actionable

**Scoring Philosophy**:
- Be conservative: missing data should not inflate scores
- Funding is critical: heavily penalize lack of funding
- Research fit is most important: highest weight
- Balance quantitative metrics with qualitative factors

---

## Integration with Existing API

After building these files, update `src/api/main.py` to use the new components:

```python
# Replace lines 154-216 in main.py
@app.post("/api/v1/search/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    try:
        # Use new search engine instead of direct vector store
        search_engine = get_search_engine()

        results = search_engine.search(
            query=request.query,
            search_mode=request.mode,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            explain=request.explain
        )

        # Convert SearchResults to API response format
        return results.to_dict()

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Replace lines 242-340 in main.py
@app.post("/api/v1/match/calculate", response_model=MatchResponse)
async def calculate_match(request: MatchRequest):
    try:
        # Use new multi-factor matcher
        matcher = get_matcher()

        # Get faculty profiles (from vector store or database)
        faculty_profiles = _get_faculty_profiles(request.faculty_ids)

        # Calculate matches
        matches = []
        for faculty_profile in faculty_profiles[:request.top_k]:
            match_score = matcher.calculate_match_score(
                student_profile=request.student_profile,
                faculty_profile=faculty_profile,
                explain=(len(matches) < request.explain_top_n)
            )

            matches.append(
                MatchResult(
                    faculty_id=faculty_profile['id'],
                    faculty_name=faculty_profile['name'],
                    overall_score=match_score.overall_score,
                    component_scores=match_score.component_scores,
                    explanation=match_score.explanation,
                    strengths=match_score.strengths,
                    considerations=match_score.considerations
                )
            )

        # Sort by score
        matches.sort(key=lambda x: x.overall_score, reverse=True)

        return MatchResponse(
            matches=matches,
            student_profile_summary={...}
        )

    except Exception as e:
        logger.error(f"Match error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Testing Strategy

### Unit Tests

Create `bio-match-ml/tests/test_search/` with:

#### 1. `test_query_processor.py`
```python
"""Test query processing functionality"""
import pytest
from src.search.query_processor import QueryProcessor, QueryAnalysis

class TestQueryProcessor:

    @pytest.fixture
    def processor(self):
        return QueryProcessor()

    def test_basic_query_processing(self, processor):
        """Test basic query is processed correctly"""
        result = processor.process_query("CRISPR in neurons")

        assert isinstance(result, QueryAnalysis)
        assert result.original == "CRISPR in neurons"
        assert len(result.entities) > 0
        assert result.intent is not None

    def test_entity_extraction(self, processor):
        """Test biological entities are extracted"""
        result = processor.process_query("p53 mutation in breast cancer")

        # Should recognize p53 as gene and breast cancer as disease
        assert 'genes' in result.entities or 'proteins' in result.entities
        assert 'diseases' in result.entities

    def test_implicit_filter_extraction(self, processor):
        """Test implicit filters are extracted"""
        result = processor.process_query("young PI at Harvard studying CRISPR")

        assert 'institution' in result.filters
        assert result.filters['institution'] == 'Harvard'
        assert 'career_stage' in result.filters

    def test_scientific_expansion(self, processor):
        """Test query expansion with scientific synonyms"""
        expansions = processor.expand_query_scientifically(
            "p53 cancer",
            {'genes': ['p53'], 'diseases': ['cancer']}
        )

        assert any('TP53' in exp for exp in expansions)
        assert any('tumor' in exp.lower() for exp in expansions)

    def test_intent_detection(self, processor):
        """Test various query intents are detected correctly"""
        cases = [
            ("Jane Smith biology", "specific_person"),
            ("CRISPR techniques", "technique_based"),
            ("neuroscience collaboration", "collaborative"),
            ("well-funded cancer research", "funding_based"),
        ]

        for query, expected_intent in cases:
            result = processor.process_query(query)
            assert result.intent == expected_intent

    def test_empty_query(self, processor):
        """Test handling of empty or invalid queries"""
        result = processor.process_query("")
        assert result is not None

        result = processor.process_query("   ")
        assert result is not None
```

#### 2. `test_result_ranker.py`
```python
"""Test result ranking functionality"""
import pytest
from src.search.result_ranker import ResultRanker

class TestResultRanker:

    @pytest.fixture
    def ranker(self):
        return ResultRanker()

    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing"""
        return [
            {
                'id': 'fac1',
                'score': 0.9,
                'metadata': {
                    'name': 'Dr. Smith',
                    'h_index': 45,
                    'publication_count': 120,
                    'active_grants': 3,
                    'total_funding': 1500000,
                    'last_publication_year': 2024
                }
            },
            {
                'id': 'fac2',
                'score': 0.85,
                'metadata': {
                    'name': 'Dr. Jones',
                    'h_index': 25,
                    'publication_count': 60,
                    'active_grants': 1,
                    'total_funding': 500000,
                    'last_publication_year': 2023
                }
            },
            {
                'id': 'fac3',
                'score': 0.88,
                'metadata': {
                    'name': 'Dr. Lee',
                    'h_index': 35,
                    'publication_count': 90,
                    'active_grants': 2,
                    'total_funding': 1200000,
                    'last_publication_year': 2024
                }
            }
        ]

    def test_reranking_improves_results(self, ranker, sample_results):
        """Test that reranking considers multiple factors"""
        from src.search.query_processor import QueryAnalysis

        query_context = QueryAnalysis(
            original="test",
            normalized="test",
            entities={},
            intent='research_area',
            expansions=[],
            filters={},
            boost_terms=[]
        )

        reranked = ranker.rank_results(sample_results, query_context)

        # Should have all results
        assert len(reranked) == len(sample_results)

        # Scores should be updated
        assert all('final_score' in r for r in reranked)

        # Should be sorted by final score
        scores = [r['final_score'] for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_productivity_score_calculation(self, ranker):
        """Test productivity scoring"""
        high_productivity = {
            'h_index': 50,
            'publication_count': 150,
            'citations': 5000
        }

        low_productivity = {
            'h_index': 10,
            'publication_count': 20,
            'citations': 100
        }

        high_score = ranker.calculate_productivity_score(high_productivity)
        low_score = ranker.calculate_productivity_score(low_productivity)

        assert high_score > low_score
        assert 0 <= high_score <= 1
        assert 0 <= low_score <= 1

    def test_funding_score_calculation(self, ranker):
        """Test funding scoring"""
        well_funded = {
            'active_grants': 3,
            'total_funding': 2000000
        }

        poorly_funded = {
            'active_grants': 0,
            'total_funding': 0
        }

        high_score = ranker.calculate_funding_score(well_funded)
        low_score = ranker.calculate_funding_score(poorly_funded)

        assert high_score > low_score

    def test_diversity_application(self, ranker, sample_results):
        """Test diversity increases variety"""
        # Add institution metadata
        for i, result in enumerate(sample_results):
            result['metadata']['institution'] = f'Institution{i % 2}'

        diversified = ranker.diversify_results(sample_results, diversity_factor=0.5)

        # Should still have all results
        assert len(diversified) == len(sample_results)

        # Top results should be from different institutions
        institutions = [r['metadata']['institution'] for r in diversified[:2]]
        assert len(set(institutions)) == 2  # Different institutions
```

#### 3. `test_semantic_search.py`
```python
"""Test semantic search engine"""
import pytest
from src.search.semantic_search import SemanticSearchEngine

class TestSemanticSearchEngine:

    @pytest.fixture
    def search_engine(self):
        # Use mock components for testing
        from unittest.mock import Mock

        mock_embedder = Mock()
        mock_embedder.generate_embedding.return_value = [0.1] * 768

        mock_vector_store = Mock()
        mock_vector_store.search.return_value = []
        mock_vector_store.list_indices.return_value = ['faculty_embeddings']

        return SemanticSearchEngine(
            embedding_generator=mock_embedder,
            vector_store=mock_vector_store
        )

    def test_basic_search(self, search_engine):
        """Test basic search functionality"""
        results = search_engine.search(
            query="CRISPR gene editing",
            search_mode='faculty',
            limit=10
        )

        assert results is not None
        assert hasattr(results, 'results')
        assert hasattr(results, 'total_count')

    def test_search_with_filters(self, search_engine):
        """Test search with metadata filters"""
        results = search_engine.search(
            query="cancer research",
            filters={'institution': 'MIT', 'min_h_index': 30},
            limit=10
        )

        assert results is not None

    def test_search_with_explanation(self, search_engine):
        """Test search with explanations enabled"""
        results = search_engine.search(
            query="test query",
            explain=True
        )

        assert results.query_interpretation is not None
```

#### 4. `test_matching.py`
```python
"""Test matching functionality"""
import pytest
from src.matching.multi_factor_scorer import MultiFactorMatcher, MatchScore
from src.matching.similarity_calculator import SimilarityCalculator

class TestSimilarityCalculator:

    @pytest.fixture
    def calculator(self):
        from unittest.mock import Mock
        mock_embedder = Mock()
        mock_embedder.generate_embedding.side_effect = lambda x: [0.1] * 768
        return SimilarityCalculator(embedding_generator=mock_embedder)

    def test_jaccard_similarity(self, calculator):
        """Test Jaccard similarity calculation"""
        set1 = {'CRISPR', 'RNA-seq', 'PCR'}
        set2 = {'CRISPR', 'Western blot', 'PCR'}

        similarity = calculator.calculate_jaccard_similarity(set1, set2)

        # 2 in common, 4 total unique
        assert similarity == pytest.approx(0.5, abs=0.01)

    def test_research_similarity(self, calculator):
        """Test research similarity calculation"""
        student = {
            'research_interests': 'CRISPR gene editing in neurons',
            'topics': ['gene editing', 'neuroscience'],
            'techniques': ['CRISPR', 'electrophysiology'],
            'organisms': ['mouse']
        }

        faculty = {
            'research_summary': 'Developing CRISPR tools for neuroscience',
            'topics': ['gene editing', 'neuroscience', 'optogenetics'],
            'techniques': ['CRISPR', 'two-photon imaging'],
            'organisms': ['mouse', 'zebrafish']
        }

        similarities = calculator.calculate_research_similarity(student, faculty)

        assert 'topic_overlap' in similarities
        assert 'technique_overlap' in similarities
        assert 'organism_overlap' in similarities
        assert all(0 <= v <= 1 for v in similarities.values())

class TestMultiFactorMatcher:

    @pytest.fixture
    def matcher(self):
        return MultiFactorMatcher()

    @pytest.fixture
    def sample_profiles(self):
        """Sample student and faculty profiles"""
        student = {
            'research_interests': 'CRISPR gene editing in cancer',
            'topics': ['gene editing', 'cancer biology'],
            'techniques': ['CRISPR', 'RNA-seq'],
            'career_goals': 'Become independent researcher'
        }

        faculty = {
            'id': 'fac123',
            'name': 'Dr. Smith',
            'research_summary': 'CRISPR applications in cancer therapy',
            'topics': ['gene editing', 'cancer biology', 'immunology'],
            'techniques': ['CRISPR', 'RNA-seq', 'flow cytometry'],
            'publications': [{'year': 2024}] * 20,
            'h_index': 35,
            'grants': [
                {
                    'active': True,
                    'amount': 500000,
                    'end_date': '2027-12-31'
                }
            ],
            'lab_size': 8,
            'accepting_students': True
        }

        return student, faculty

    def test_match_calculation(self, matcher, sample_profiles):
        """Test basic match calculation"""
        student, faculty = sample_profiles

        match_score = matcher.calculate_match_score(student, faculty, explain=True)

        assert isinstance(match_score, MatchScore)
        assert 0 <= match_score.overall_score <= 1
        assert match_score.component_scores is not None
        assert match_score.explanation is not None
        assert len(match_score.strengths) > 0
        assert match_score.recommendation in [
            'highly_recommended', 'recommended', 'consider', 'not_recommended'
        ]

    def test_funding_scoring(self, matcher):
        """Test funding stability scoring"""
        well_funded = {
            'grants': [
                {'active': True, 'amount': 1000000, 'end_date': '2028-12-31'},
                {'active': True, 'amount': 500000, 'end_date': '2027-06-30'}
            ]
        }

        poorly_funded = {
            'grants': []
        }

        good_score, _ = matcher.funding_stability_score(well_funded)
        poor_score, _ = matcher.funding_stability_score(poorly_funded)

        assert good_score > poor_score
        assert poor_score < 0.5  # Should heavily penalize no funding

    def test_match_explanation_quality(self, matcher, sample_profiles):
        """Test that explanations are informative"""
        student, faculty = sample_profiles

        match_score = matcher.calculate_match_score(student, faculty, explain=True)

        explanation = match_score.explanation

        # Should mention specific research areas
        assert any(term in explanation.lower() for term in ['crispr', 'gene', 'cancer'])

        # Should be substantive
        assert len(explanation) > 50

        # Strengths should be specific
        assert all(len(s) > 10 for s in match_score.strengths)
```

### Integration Tests

Create `bio-match-ml/tests/test_integration/test_search_pipeline.py`:

```python
"""Integration tests for complete search pipeline"""
import pytest

class TestSearchPipeline:
    """Test the complete search flow"""

    def test_end_to_end_search(self):
        """Test complete search from query to results"""
        # This would test:
        # 1. Query processing
        # 2. Embedding generation
        # 3. Vector search
        # 4. Result ranking
        # 5. Facet generation
        # 6. Response formatting
        pass

    def test_end_to_end_matching(self):
        """Test complete matching from student profile to ranked faculty"""
        # This would test:
        # 1. Profile embedding
        # 2. Similarity calculation
        # 3. Multi-factor scoring
        # 4. Ranking
        # 5. Explanation generation
        pass
```

### Manual Testing

Create test scripts in `bio-match-ml/scripts/`:

#### `test_search_manual.py`
```python
"""Manual test script for search functionality"""
from src.search.semantic_search import SemanticSearchEngine
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.model_manager import ModelManager
from src.vector_stores.faiss_store import FaissStore

def main():
    print("Initializing search engine...")

    model_manager = ModelManager()
    embedder = EmbeddingGenerator(model_manager=model_manager)
    vector_store = FaissStore()

    engine = SemanticSearchEngine(
        embedding_generator=embedder,
        vector_store=vector_store
    )

    # Test queries
    test_queries = [
        "CRISPR gene editing in neurons",
        "young PI studying cancer immunology",
        "well-funded structural biology",
        "p53 mutation in breast cancer",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)

        results = engine.search(query, explain=True, limit=5)

        print(f"Found {results.total_count} results")
        print(f"\nQuery Interpretation:")
        print(f"  Intent: {results.query_interpretation.get('intent')}")
        print(f"  Entities: {results.query_interpretation.get('entities')}")

        print(f"\nTop 5 Results:")
        for i, result in enumerate(results.results[:5], 1):
            print(f"\n{i}. {result.get('name')} - Score: {result.get('score'):.3f}")
            if 'explanation' in result:
                print(f"   {result['explanation']}")

if __name__ == '__main__':
    main()
```

#### `test_matching_manual.py`
```python
"""Manual test script for matching functionality"""
from src.matching.multi_factor_scorer import MultiFactorMatcher

def main():
    print("Testing faculty-student matching...\n")

    matcher = MultiFactorMatcher()

    # Sample student profile
    student = {
        'research_interests': 'I am interested in using CRISPR to study gene regulation in cancer cells',
        'topics': ['gene editing', 'cancer biology', 'gene regulation'],
        'techniques': ['CRISPR', 'RNA-seq', 'ChIP-seq'],
        'organisms': ['human', 'mouse'],
        'career_goals': 'Academic research career, focus on cancer therapeutics'
    }

    # Sample faculty profiles
    faculty_list = [
        {
            'id': 'fac1',
            'name': 'Dr. Jane Smith',
            'research_summary': 'CRISPR screens for cancer vulnerabilities',
            'topics': ['gene editing', 'cancer', 'functional genomics'],
            'techniques': ['CRISPR', 'RNA-seq', 'proteomics'],
            'h_index': 42,
            'grants': [{'active': True, 'amount': 1500000, 'end_date': '2028-01-01'}],
            'lab_size': 10,
            'accepting_students': True
        },
        {
            'id': 'fac2',
            'name': 'Dr. John Doe',
            'research_summary': 'Structural biology of membrane proteins',
            'topics': ['structural biology', 'biophysics'],
            'techniques': ['X-ray crystallography', 'cryo-EM'],
            'h_index': 35,
            'grants': [],
            'lab_size': 5,
            'accepting_students': True
        }
    ]

    print("Student Profile:")
    print(f"  Interests: {student['research_interests'][:100]}...")
    print(f"  Techniques: {', '.join(student['techniques'])}")
    print(f"  Organisms: {', '.join(student['organisms'])}\n")

    print("="*80)

    for faculty in faculty_list:
        print(f"\nFaculty: {faculty['name']}")
        print(f"Research: {faculty['research_summary']}")
        print("-"*80)

        match = matcher.calculate_match_score(student, faculty, explain=True)

        print(f"Overall Score: {match.overall_score:.3f} - {match.recommendation.upper()}")
        print(f"Confidence: {match.confidence:.3f}\n")

        print("Component Scores:")
        for component, score in match.component_scores.items():
            print(f"  {component}: {score:.3f}")

        print(f"\nExplanation:\n  {match.explanation}")

        print(f"\nStrengths:")
        for strength in match.strengths:
            print(f"  + {strength}")

        print(f"\nConsiderations:")
        for consideration in match.considerations:
            print(f"  - {consideration}")

        print("="*80)

if __name__ == '__main__':
    main()
```

---

## Quality Requirements

### Code Quality
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Comprehensive docstrings following Google style
- **Error handling**: Graceful handling of edge cases and missing data
- **Logging**: Detailed logging for debugging and monitoring
- **Configuration**: Make key parameters configurable via config dicts

### Performance
- **Query processing**: < 50ms
- **Search (with ranking)**: < 200ms for 1000 documents
- **Matching calculation**: < 100ms per match

### Data Handling
- **Missing data**: Never crash on missing fields, use sensible defaults
- **Normalization**: All scores normalized to [0, 1]
- **Validation**: Validate input data shapes and types

---

## Success Criteria

After implementation, you should be able to:

1. ✅ Process complex biology queries with entity extraction and expansion
2. ✅ Search with intelligent ranking beyond simple vector similarity
3. ✅ Generate explainable faculty-student match scores
4. ✅ Handle real-world messy data (missing fields, inconsistent formatting)
5. ✅ Run the API and get intelligent results (not hardcoded)
6. ✅ Pass all unit tests
7. ✅ Run manual test scripts successfully

---

## Notes

- **Start with query_processor.py** - it's foundational for everything else
- **Use existing components** - BioEntityRecognizer, ResearchTopicClassifier, EmbeddingGenerator
- **Test incrementally** - test each file as you build it
- **Focus on biology domain** - use biology-specific logic and knowledge
- **Make it explainable** - users need to understand why they got specific results

---

## Questions to Clarify

Before starting, consider:
1. Should query expansion be aggressive or conservative?
2. What should be the default ranking weights?
3. How important is funding vs research fit in matching?
4. Should we penalize very large labs in matching?

Start with reasonable defaults, but make everything configurable for tuning.

---

Build production-quality code with comprehensive error handling, logging, and documentation. This is the intelligence layer that transforms basic vector search into a sophisticated biology research matching platform.
