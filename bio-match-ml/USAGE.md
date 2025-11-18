# Usage Guide for Bio-Match-ML Search and Matching Systems

This guide explains how to use the bio-match-ml search engine and matching systems.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Semantic Search](#semantic-search)
- [Faculty-Student Matching](#faculty-student-matching)
- [Query Processing](#query-processing)
- [Result Ranking](#result-ranking)
- [API Integration](#api-integration)
- [Examples](#examples)

## Overview

The bio-match-ml system provides:

1. **Semantic Search Engine** - Intelligent search for faculty, publications, and grants
2. **Multi-Factor Matching** - Comprehensive faculty-student matching
3. **Query Intelligence** - Natural language understanding with domain knowledge
4. **Explainable Results** - Detailed explanations for rankings and matches

## Quick Start

### Installation

```bash
cd bio-match-ml
pip install -r requirements.txt
```

### Basic Search

```python
from src.search.semantic_search import SemanticSearchEngine
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectorstore.pinecone_client import PineconeVectorStore

# Initialize components
embedder = EmbeddingGenerator()
vector_store = PineconeVectorStore()
search_engine = SemanticSearchEngine(embedder, vector_store)

# Search for faculty
results = search_engine.search(
    query="CRISPR gene editing in cancer",
    search_mode='faculty',
    limit=10
)

# Display results
for result in results.results:
    faculty = result['metadata']
    print(f"{faculty['name']} - {faculty['institution']}")
    print(f"Score: {result['final_score']:.3f}")
    print(f"Research: {faculty['research_summary'][:100]}...")
    print()
```

### Basic Matching

```python
from src.matching.multi_factor_scorer import MultiFactorMatcher
from src.matching.similarity_calculator import SimilarityCalculator

# Initialize components
similarity_calc = SimilarityCalculator()
matcher = MultiFactorMatcher(similarity_calc)

# Define profiles
student = {
    'research_interests': 'CRISPR gene editing in cancer cells',
    'topics': ['gene editing', 'cancer biology'],
    'techniques': ['CRISPR-Cas9', 'cell culture'],
    'organisms': ['human', 'mouse'],
    'career_goals': 'Academic research career'
}

faculty = {
    'name': 'Dr. Alice Chen',
    'research_summary': 'CRISPR applications in cancer therapy',
    'topics': ['gene editing', 'cancer biology'],
    'techniques': ['CRISPR-Cas9', 'RNA-seq'],
    'organisms': ['human', 'mouse'],
    'h_index': 45,
    'publication_count': 120,
    'has_active_funding': True,
    'total_funding': 2500000,
    'accepting_students': True
}

# Calculate match score
match = matcher.calculate_match_score(student, faculty, explain=True)

print(f"Match Score: {match.overall_score:.3f}")
print(f"Recommendation: {match.recommendation}")
print(f"\nExplanation: {match.explanation}")
print(f"\nStrengths:")
for strength in match.strengths:
    print(f"  • {strength}")
```

## Semantic Search

### Search Modes

The search engine supports three modes:

```python
# Search for faculty members
results = search_engine.search(
    query="cancer genomics",
    search_mode='faculty',
    limit=10
)

# Search for publications
results = search_engine.search(
    query="CRISPR therapeutics",
    search_mode='publications',
    limit=20
)

# Search for grants
results = search_engine.search(
    query="NIH cancer research",
    search_mode='grants',
    limit=15
)
```

### Advanced Search Options

```python
results = search_engine.search(
    query="young PI studying neural circuits",
    search_mode='faculty',
    limit=10,
    explain=True,  # Include ranking explanations
    filters={
        'institution': 'MIT',
        'has_active_funding': True,
        'accepting_students': True
    },
    boost_recent=True,  # Boost recent publications
    diversity_weight=0.3  # Apply result diversification
)
```

### Multi-Query Search

Search with multiple queries simultaneously:

```python
results = search_engine.multi_query_search(
    queries=[
        "CRISPR gene editing",
        "cancer therapeutics",
        "precision medicine"
    ],
    aggregation_mode='weighted',  # 'union', 'intersection', or 'weighted'
    query_weights=[0.5, 0.3, 0.2],
    limit=10
)
```

### Search by Example

Find similar profiles:

```python
# Find faculty similar to a specific faculty member
results = search_engine.search_by_example(
    example_id='fac12345',
    search_mode='faculty',
    limit=10
)
```

### Understanding Search Results

```python
results = search_engine.search(query="...", explain=True)

# Total results found
print(f"Found {results.total_count} results")

# Query interpretation
interpretation = results.query_interpretation
print(f"Intent: {interpretation['detected_intent']}")
print(f"Entities: {interpretation['extracted_entities']}")
print(f"Expansions: {interpretation['num_expansions']}")

# Individual results
for result in results.results:
    print(f"Name: {result['metadata']['name']}")
    print(f"Score: {result['final_score']:.3f}")
    print(f"Explanation: {result.get('ranking_explanation', 'N/A')}")

# Facets (aggregations)
facets = results.facets
print(f"\nTop Institutions:")
for facet in facets['institutions'][:5]:
    print(f"  {facet['value']}: {facet['count']}")

# Performance metrics
print(f"\nSearch Time: {results.search_metadata['total_time_ms']:.1f}ms")
```

## Faculty-Student Matching

### Complete Matching Pipeline

```python
from src.search.semantic_search import SemanticSearchEngine
from src.matching.multi_factor_scorer import MultiFactorMatcher

# Step 1: Search for candidate faculty
search_results = search_engine.search(
    query=student['research_interests'],
    search_mode='faculty',
    limit=20
)

# Step 2: Calculate match scores
matches = []
for result in search_results.results:
    faculty = result['metadata']
    match_score = matcher.calculate_match_score(
        student_profile=student,
        faculty_profile=faculty,
        explain=True
    )
    matches.append({
        'faculty': faculty,
        'match_score': match_score
    })

# Step 3: Sort by match score
matches.sort(key=lambda x: x['match_score'].overall_score, reverse=True)

# Display top matches
for i, match in enumerate(matches[:5], 1):
    faculty = match['faculty']
    score = match['match_score']

    print(f"\n{i}. {faculty['name']} - {faculty['institution']}")
    print(f"   Overall Score: {score.overall_score:.3f}")
    print(f"   Recommendation: {score.recommendation}")
    print(f"   Confidence: {score.confidence:.2f}")
    print(f"\n   Component Scores:")
    for component, value in score.component_scores.items():
        print(f"     • {component}: {value:.3f}")
```

### Match Score Components

The matching system evaluates six factors:

1. **Research Alignment (35%)** - Topic, technique, organism overlap
2. **Funding Stability (20%)** - Active grants, funding amount, runway
3. **Productivity Match (15%)** - Publication count, h-index, citations
4. **Technique Match (15%)** - Experimental technique overlap
5. **Lab Environment (10%)** - Lab size, culture, accepting students
6. **Career Development (5%)** - Training focus, trajectory alignment

```python
match = matcher.calculate_match_score(student, faculty)

# Access individual components
research_score = match.component_scores['research_alignment']
funding_score = match.component_scores['funding_stability']

# Overall score is weighted average
overall = match.overall_score  # 0.0 to 1.0
```

### Match Recommendations

The system provides four recommendation levels:

```python
if match.recommendation == 'highly_recommended':
    # Score ≥ 0.8, high confidence
    print("🌟 Highly Recommended")
elif match.recommendation == 'recommended':
    # Score ≥ 0.6, good confidence
    print("✅ Recommended")
elif match.recommendation == 'consider':
    # Score ≥ 0.4
    print("🤔 Consider")
else:  # 'not_recommended'
    # Score < 0.4
    print("❌ Not Recommended")
```

### Match Explanations

```python
match = matcher.calculate_match_score(student, faculty, explain=True)

# Human-readable explanation
print(match.explanation)
# "Excellent match with strong research alignment in CRISPR and cancer biology..."

# Specific strengths
for strength in match.strengths:
    print(f"✅ {strength}")
# "Very strong alignment in CRISPR gene editing techniques"
# "Excellent funding stability with multiple active grants"

# Considerations
for consideration in match.considerations:
    print(f"⚠️ {consideration}")
# "Large lab size may mean less individual attention"

# Confidence score
print(f"Confidence: {match.confidence:.2f}")
```

## Query Processing

### Query Intelligence

The query processor automatically:
- Extracts entities (genes, techniques, organisms, topics)
- Detects query intent
- Expands queries with biological synonyms
- Identifies implicit filters

```python
from src.search.query_processor import QueryProcessor

processor = QueryProcessor()
analysis = processor.process_query("young PI at MIT studying p53 in mice")

print(f"Original: {analysis.original_query}")
print(f"Normalized: {analysis.normalized_query}")
print(f"Intent: {analysis.detected_intent}")
print(f"Entities: {analysis.extracted_entities}")
print(f"Filters: {analysis.implicit_filters}")
print(f"Expansions: {len(analysis.query_expansions)}")
```

### Intent Detection

Six query intent types:

```python
# technique_based: "CRISPR applications in biology"
# organism_based: "mouse models of disease"
# funding_based: "well-funded cancer labs"
# disease_based: "Alzheimer's research"
# career_stage_based: "early career investigators"
# general: fallback for other queries
```

### Query Expansion

Automatically expands queries with synonyms:

```python
# Input: "p53 mutation in cancer"
# Expansions:
# - "p53 mutation in cancer"
# - "TP53 mutation in cancer" (gene alias)
# - "tumor protein 53 mutation in cancer"

# Input: "CRISPR gene editing"
# Expansions:
# - "CRISPR gene editing"
# - "Cas9 genome editing" (technique synonym)
# - "gene editing with CRISPR-Cas9"
```

## Result Ranking

### Multi-Factor Ranking

Results are ranked using multiple factors beyond semantic similarity:

```python
from src.search.result_ranker import ResultRanker

ranker = ResultRanker(
    weights={
        'semantic_score': 0.30,      # Vector similarity
        'productivity_score': 0.20,  # Publications, citations
        'funding_score': 0.15,       # Active grants
        'recency_score': 0.15,       # Recent publications
        'h_index_score': 0.10,       # H-index
        'diversity_bonus': 0.10      # Result diversity
    }
)

# Rank results
ranked_results = ranker.rank_results(
    results=raw_results,
    query_analysis=query_analysis
)
```

### Ranking Explanations

```python
# Get ranking explanation for a result
explanation = ranker.explain_ranking(result, query_analysis)
print(explanation)
# "Highly relevant (0.92 similarity) with strong productivity (h-index: 45)
#  and excellent recent activity (3 publications in 2024)."
```

### Result Diversification

Apply Maximal Marginal Relevance (MMR) for diverse results:

```python
diverse_results = ranker.diversify_results(
    results=ranked_results,
    diversity_weight=0.3  # 0.0 = no diversity, 1.0 = max diversity
)
```

## API Integration

### FastAPI Endpoints

The system integrates with FastAPI:

```python
# Start the API server
uvicorn src.api.main:app --reload
```

### Search Endpoint

```bash
# POST /api/v1/search/semantic
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CRISPR gene editing in cancer",
    "search_mode": "faculty",
    "limit": 10,
    "explain": true
  }'
```

### Matching Endpoint

```bash
# POST /api/v1/match/calculate
curl -X POST "http://localhost:8000/api/v1/match/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "student_profile": {
      "research_interests": "CRISPR gene editing",
      "topics": ["gene editing", "cancer"],
      "techniques": ["CRISPR-Cas9"],
      "organisms": ["human", "mouse"]
    },
    "faculty_profiles": [
      {
        "name": "Dr. Alice Chen",
        "research_summary": "CRISPR applications",
        ...
      }
    ],
    "explain": true
  }'
```

## Examples

### Example 1: Finding Well-Funded Cancer Labs

```python
results = search_engine.search(
    query="well-funded cancer research",
    search_mode='faculty',
    filters={'has_active_funding': True},
    limit=10,
    explain=True
)

for result in results.results:
    faculty = result['metadata']
    funding = faculty.get('total_funding', 0)
    print(f"{faculty['name']}: ${funding:,}")
```

### Example 2: Matching Student with Career Goals

```python
student = {
    'research_interests': 'computational genomics and cancer',
    'topics': ['genomics', 'cancer', 'bioinformatics'],
    'techniques': ['RNA-seq', 'machine learning'],
    'organisms': ['human'],
    'career_goals': 'Industry career in computational biology',
    'desired_skills': ['bioinformatics', 'data science']
}

# Search for faculty
search_results = search_engine.search(
    query=student['research_interests'],
    search_mode='faculty',
    limit=20
)

# Find matches
for result in search_results.results:
    faculty = result['metadata']
    match = matcher.calculate_match_score(student, faculty, explain=True)

    # Filter by career development alignment
    career_score = match.component_scores.get('career_development', 0)

    if career_score > 0.7:
        print(f"{faculty['name']}: Career Development Score = {career_score:.3f}")
        print(f"  Training Focus: {faculty.get('training_focus', 'N/A')}")
```

### Example 3: Technique-Based Search

```python
# Search for labs using specific techniques
results = search_engine.search(
    query="optogenetics and two-photon imaging",
    search_mode='faculty',
    limit=10
)

# Verify technique overlap
for result in results.results:
    faculty = result['metadata']
    techniques = faculty.get('techniques', [])

    print(f"{faculty['name']}")
    print(f"  Techniques: {', '.join(techniques)}")
```

### Example 4: Multi-Query Search for Interdisciplinary Research

```python
# Search across multiple related topics
results = search_engine.multi_query_search(
    queries=[
        "CRISPR gene editing",
        "cancer immunotherapy",
        "precision medicine"
    ],
    aggregation_mode='weighted',
    query_weights=[0.4, 0.4, 0.2],
    limit=15
)

print(f"Found {results.total_count} interdisciplinary researchers")
```

## Configuration

### Custom Ranking Weights

```python
# Emphasize recent publications
ranker = ResultRanker(weights={
    'semantic_score': 0.25,
    'recency_score': 0.30,  # Increased
    'productivity_score': 0.20,
    'funding_score': 0.15,
    'h_index_score': 0.10
})
```

### Custom Matching Weights

```python
# Emphasize funding for industry-focused students
matcher = MultiFactorMatcher(
    similarity_calculator=similarity_calc,
    weights={
        'research_alignment': 0.30,
        'funding_stability': 0.30,  # Increased
        'productivity_match': 0.15,
        'technique_match': 0.10,
        'lab_environment': 0.10,
        'career_development': 0.05
    }
)
```

## Performance

Expected performance metrics:
- Query processing: < 50ms
- Semantic search (10 results): < 200ms
- Match score calculation: < 100ms
- End-to-end search + match: < 500ms

## Troubleshooting

See [TESTING.md](TESTING.md) for detailed troubleshooting guide.

## Additional Resources

- [TESTING.md](TESTING.md) - Complete testing guide
- [API Documentation](src/api/README.md) - API reference
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Technical details
