# Claude Code Prompt: Search Enhancement & Advanced Matching - Batch 2

## Prerequisites

**Required**: Batch 1 must be completed first. You should have:
- ✅ `src/search/semantic_search.py`
- ✅ `src/search/query_processor.py`
- ✅ `src/search/result_ranker.py`
- ✅ `src/matching/similarity_calculator.py`
- ✅ `src/matching/multi_factor_scorer.py`

## Context

Batch 1 provided the core search and matching functionality. Now you'll add:
1. **Hybrid search** combining vector and keyword search
2. **Explanation generation** for search results
3. **Research trajectory analysis** for advanced matching

These components enhance the system with production features and deeper matching intelligence.

---

## Your Deliverables - Batch 2

Build the following 3 files:

### 1. `src/search/hybrid_search.py` - Vector + Keyword Search
Combine semantic search with traditional keyword search for better precision.

### 2. `src/search/explanation_generator.py` - Result Explanations
Generate human-readable explanations for why results matched.

### 3. `src/matching/research_trajectory_analyzer.py` - Research Evolution
Analyze how faculty research evolves over time and predict future directions.

---

## File 1: `src/search/hybrid_search.py`

**Purpose**: Combine vector similarity search with keyword/BM25 search for optimal results.

**Key Classes & Methods**:

```python
class HybridSearchEngine:
    """
    Hybrid search combining semantic (vector) and lexical (keyword) search
    """

    def __init__(
        self,
        vector_store,
        embedding_generator,
        text_index: Optional[Any] = None  # Could be Elasticsearch, Whoosh, etc.
    ):
        """
        Initialize hybrid search engine

        Args:
            vector_store: Vector store for semantic search
            embedding_generator: For generating query embeddings
            text_index: Optional text search index (BM25, TF-IDF)
        """
        pass

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
        pass

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
        pass

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
        pass

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
        pass

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
        pass
```

**Implementation Requirements**:
- If no text index is available, implement simple in-memory TF-IDF
- Use Reciprocal Rank Fusion for combining results (well-studied algorithm)
- Make alpha configurable per query based on query type
- Add detailed logging of score contributions

---

## File 2: `src/search/explanation_generator.py`

**Purpose**: Generate natural language explanations for search results and matches.

**Key Classes & Methods**:

```python
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
        pass

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
        pass

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
        pass

    def identify_common_ground(
        self,
        student_entities: Dict,
        faculty_entities: Dict
    ) -> Dict[str, List[str]]:
        """
        Identify specific overlaps between student and faculty

        Args:
            student_entities: Entities from student profile
            faculty_entities: Entities from faculty profile

        Returns:
            Dict of common entities by type:
            {
                'techniques': ['CRISPR', 'RNA-seq'],
                'organisms': ['mouse'],
                'topics': ['gene editing', 'cancer']
            }
        """
        pass

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
        pass

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
        pass

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
        pass
```

**Implementation Requirements**:
- Extract specific overlapping entities (techniques, organisms, topics)
- Use templates but make them feel natural (vary sentence structure)
- Include actionable information (contact info, lab meetings, papers to read)
- Adapt tone based on match quality (enthusiastic for high matches, cautious for low)
- Make explanations scannable with clear sections

**Explanation Quality Guidelines**:
- Be specific: Mention actual techniques, not just "similar research"
- Be honest: Don't oversell poor matches
- Be actionable: Include next steps
- Be concise: 2-4 sentences for brief, 1-2 paragraphs for detailed

---

## File 3: `src/matching/research_trajectory_analyzer.py`

**Purpose**: Analyze research evolution over time and predict future directions.

**Key Classes & Methods**:

```python
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
        pass

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
            TrajectoryAnalysis object with:
            - trending_topics: Topics increasing in frequency
            - declining_topics: Topics decreasing
            - stable_core: Consistent research areas
            - innovation_rate: How frequently new topics appear
            - pivot_points: Years with significant research shifts
            - research_phases: Identified career phases
            - predicted_directions: Likely future research areas

        Example:
            trajectory = analyzer.analyze_trajectory(pubs)
            print(trajectory.trending_topics)
            # ['single-cell RNA-seq', 'spatial transcriptomics']
            print(trajectory.predicted_directions)
            # ['Likely to expand into multi-omics integration']
        """
        pass

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
            List of ResearchPhase objects with:
            - phase_type: str
            - years: tuple (start, end)
            - primary_topics: List[str]
            - characteristics: str

        Uses clustering on publication embeddings over time
        """
        pass

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
            Dict mapping topic → TrendData:
            {
                'CRISPR': TrendData(
                    recent_count=15,
                    historical_count=5,
                    trend='increasing',
                    growth_rate=2.0,
                    first_appearance=2018
                )
            }
        """
        pass

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
            List of PredictedDirection objects:
            [
                PredictedDirection(
                    topic='spatial multi-omics',
                    confidence=0.75,
                    rationale='Recent citations in spatial transcriptomics
                              + established omics expertise',
                    timeframe='1-2 years'
                )
            ]
        """
        pass

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
        pass

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
            List of PivotPoint objects with year and description
        """
        pass

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
            AlignmentAssessment with:
            - alignment_score: float
            - alignment_type: 'core', 'trending', 'future', 'declining'
            - opportunities: List[str]
            - risks: List[str]

        Example:
            assessment.alignment_type = 'trending'
            assessment.opportunities = [
                "Join faculty's expansion into spatial transcriptomics",
                "Contribute fresh perspective on single-cell methods"
            ]
            assessment.risks = [
                "Field is rapidly evolving - need to stay current"
            ]
        """
        pass

# Data classes
class TrajectoryAnalysis:
    """Research trajectory analysis results"""
    def __init__(self, ...):
        self.trending_topics: List[str]
        self.declining_topics: List[str]
        self.stable_core: List[str]
        self.innovation_rate: float
        self.pivot_points: List[PivotPoint]
        self.research_phases: List[ResearchPhase]
        self.predicted_directions: List[PredictedDirection]

class ResearchPhase:
    """Single phase in research career"""
    def __init__(self, phase_type, years, primary_topics, characteristics):
        self.phase_type = phase_type
        self.years = years
        self.primary_topics = primary_topics
        self.characteristics = characteristics

class TrendData:
    """Topic trend information"""
    def __init__(self, recent_count, historical_count, trend, growth_rate, first_appearance):
        ...

class PredictedDirection:
    """Predicted future research direction"""
    def __init__(self, topic, confidence, rationale, timeframe):
        ...

class PivotPoint:
    """Research pivot point"""
    def __init__(self, year, description, old_focus, new_focus):
        ...

class AlignmentAssessment:
    """Student-faculty trajectory alignment"""
    def __init__(self, alignment_score, alignment_type, opportunities, risks):
        ...
```

**Implementation Requirements**:
- Use temporal clustering to identify phases
- Calculate topic frequency in sliding time windows
- Extract entities from publication titles/abstracts for topic tracking
- Use simple trend detection (linear regression on topic counts)
- Make predictions conservative (high confidence threshold)

**Algorithm for Trend Detection**:
```python
def detect_trend(topic_counts_by_year):
    recent_avg = mean(topic_counts[-5:])  # Last 5 years
    historical_avg = mean(topic_counts[:-5])  # Earlier

    if recent_avg > historical_avg * 1.5:
        return 'increasing'
    elif recent_avg < historical_avg * 0.5:
        return 'declining'
    else:
        return 'stable'
```

---

## Integration with Batch 1 Components

### Update `MultiFactorMatcher` to use trajectory analysis:

```python
# In src/matching/multi_factor_scorer.py

def calculate_match_score(self, student_profile, faculty_profile, explain=True):
    # ... existing code ...

    # Add trajectory analysis if publication history available
    if 'publications' in faculty_profile:
        trajectory_analyzer = ResearchTrajectoryAnalyzer()
        trajectory = trajectory_analyzer.analyze_trajectory(
            faculty_profile['publications']
        )

        # Assess alignment with trajectory
        alignment = trajectory_analyzer.assess_student_alignment(
            student_profile,
            trajectory
        )

        # Add to component scores
        component_scores['trajectory_alignment'] = alignment.alignment_score

        # Add trajectory insights to explanation
        if explain:
            explanation_gen = ExplanationGenerator()
            trajectory_explanation = explanation_gen.explain_trajectory_alignment(
                alignment
            )
            # Append to main explanation
```

### Update `SemanticSearchEngine` to use hybrid search:

```python
# In src/search/semantic_search.py

def search(self, query, ...):
    # ... existing code ...

    # Use hybrid search if available
    if hasattr(self, 'hybrid_engine'):
        results = self.hybrid_engine.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            index_name=index_name,
            k=limit * 2,  # Get more for ranking
            alpha=0.6  # Prefer semantic
        )
    else:
        # Fall back to vector-only search
        results = self.vector_store.search(...)
```

---

## Testing Strategy - Batch 2

### Unit Tests

```python
# tests/test_search/test_hybrid_search.py
class TestHybridSearch:
    def test_rrf_combination(self):
        """Test Reciprocal Rank Fusion"""
        ...

    def test_alpha_weighting(self):
        """Test alpha parameter effects"""
        ...

# tests/test_search/test_explanation_generator.py
class TestExplanationGenerator:
    def test_explanation_quality(self):
        """Test explanations are informative"""
        ...

    def test_common_ground_identification(self):
        """Test overlap detection"""
        ...

# tests/test_matching/test_trajectory_analyzer.py
class TestTrajectoryAnalyzer:
    def test_trend_detection(self):
        """Test trending vs declining topic detection"""
        ...

    def test_phase_identification(self):
        """Test career phase detection"""
        ...

    def test_pivot_detection(self):
        """Test pivot point identification"""
        ...
```

### Manual Testing

```python
# scripts/test_trajectory.py
"""Test trajectory analysis on sample publication history"""

sample_pubs = [
    {'year': 2015, 'title': 'PCR methods...', 'keywords': ['PCR', 'genetics']},
    {'year': 2016, 'title': 'More PCR...', 'keywords': ['PCR', 'genomics']},
    {'year': 2018, 'title': 'CRISPR screening...', 'keywords': ['CRISPR', 'screening']},
    {'year': 2019, 'title': 'CRISPR in cancer...', 'keywords': ['CRISPR', 'cancer']},
    {'year': 2020, 'title': 'Single-cell CRISPR...', 'keywords': ['CRISPR', 'single-cell']},
    {'year': 2024, 'title': 'Spatial omics...', 'keywords': ['spatial', 'omics', 'CRISPR']},
]

analyzer = ResearchTrajectoryAnalyzer()
trajectory = analyzer.analyze_trajectory(sample_pubs)

print("Trending Topics:", trajectory.trending_topics)
# Expected: ['CRISPR', 'single-cell', 'spatial omics']

print("Declining Topics:", trajectory.declining_topics)
# Expected: ['PCR']

print("Pivot Points:")
for pivot in trajectory.pivot_points:
    print(f"  {pivot.year}: {pivot.description}")
# Expected: 2018 - Shift from PCR to CRISPR
```

---

## Success Criteria - Batch 2

After completion:

1. ✅ Hybrid search improves results over vector-only search
2. ✅ Explanations are specific and mention actual techniques/topics
3. ✅ Trajectory analysis correctly identifies trends from publication history
4. ✅ Research phases are logically segmented
5. ✅ Future predictions are reasonable given current trends
6. ✅ All unit tests pass
7. ✅ Manual test scripts produce sensible output

---

## Estimated Effort

- **Hybrid Search**: 4-6 hours
- **Explanation Generator**: 4-6 hours
- **Trajectory Analyzer**: 6-8 hours
- **Integration & Testing**: 3-4 hours

**Total**: 17-24 hours (2-3 work days)

---

Start with hybrid_search.py (smallest scope), then explanation_generator.py, then trajectory_analyzer.py (most complex).
