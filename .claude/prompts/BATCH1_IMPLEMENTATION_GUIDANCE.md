# Batch 1 Implementation Guidance

## Answers to Your Questions

### 1. Scope & Priority ✅

**YES - Implement all 5 files in this specific order:**

```
Day 1-2: query_processor.py (foundational)
         ↓
Day 2-3: similarity_calculator.py (standalone, can parallel with query_processor)
         ↓
Day 4-5: result_ranker.py (depends on query_processor)
         ↓
Day 6-7: semantic_search.py (orchestrates search components)
         ↓
Day 8-10: multi_factor_scorer.py (depends on similarity_calculator)
```

**Your dependency analysis is correct.** This order allows incremental testing.

**Can parallelize**:
- `similarity_calculator.py` and `query_processor.py` (no dependencies)
- If multiple developers: one person on search (files 1,3,4), another on matching (files 2,5)

---

### 2. Testing ✅

**YES - Create comprehensive tests:**

#### Test Structure to Create:
```
bio-match-ml/
├── tests/
│   ├── __init__.py
│   ├── test_search/
│   │   ├── __init__.py
│   │   ├── test_query_processor.py      ← Create this
│   │   ├── test_result_ranker.py        ← Create this
│   │   └── test_semantic_search.py      ← Create this
│   ├── test_matching/
│   │   ├── __init__.py
│   │   ├── test_similarity_calculator.py ← Create this
│   │   └── test_multi_factor_scorer.py   ← Create this
│   └── test_integration/
│       ├── __init__.py
│       └── test_search_pipeline.py      ← Create this
└── scripts/
    ├── test_search_manual.py            ← Create this
    └── test_matching_manual.py          ← Create this
```

**For each module**:
- ✅ Unit tests (as specified in the prompt)
- ✅ Manual test script (for debugging)
- ✅ Integration test (for the full pipeline)

**Testing priority**:
1. **Unit tests** - Write alongside each module
2. **Manual scripts** - Create after module works
3. **Integration tests** - Create at the end

**Use pytest** (already in requirements.txt)

---

### 3. Data Formats ✅

**Use the existing data models from bio-match-platform:**

#### Faculty Profile Format:
```python
# From bio-match-platform/src/models/faculty.py
{
    'faculty_id': str,          # Unique ID
    'name': str,                # "Dr. Jane Smith"
    'title': str,               # "Assistant Professor"
    'department': str,          # "Biology"
    'institution': str,         # "MIT"
    'email': Optional[str],
    'research_interests': str,  # Long text description
    'lab_website': Optional[str],
    'lab_members': List[str],   # Lab size
    'profile_url': str,

    # You'll need to add these (not in base model):
    'publications': List[Dict],  # Publication objects
    'grants': List[Dict],        # Grant objects
    'h_index': int,             # Add this
    'total_funding': float,      # Sum from grants
    'active_grants': int,        # Count active grants
    'accepting_students': bool   # Add this
}
```

#### Publication Format:
```python
# From bio-match-platform/src/models/publication.py
{
    'pmid': str,                # PubMed ID
    'title': str,
    'abstract': str,
    'authors': List[Dict],      # [{name, affiliation, is_first, is_last}]
    'journal': str,
    'publication_date': datetime,
    'keywords': List[str],
    'mesh_terms': List[str],    # MeSH headings
    'citation_count': int
}
```

#### Grant Format:
```python
# From bio-match-platform/src/models/grant.py
{
    'project_number': str,      # "R01-..."
    'title': str,
    'abstract': str,
    'pi_name': str,
    'pi_institution': str,
    'total_cost': float,
    'start_date': datetime,
    'end_date': datetime,
    'activity_code': str,       # "R01", "R21", etc.
    'keywords': List[str]
}
```

#### Student Profile Format (you'll define this):
```python
{
    'student_id': str,
    'name': str,
    'research_interests': str,  # Long text description
    'topics': List[str],        # ["gene editing", "neuroscience"]
    'techniques': List[str],    # ["CRISPR", "RNA-seq"]
    'organisms': List[str],     # ["mouse", "human"]
    'career_goals': str,        # "Academic research career"
    'preferred_institution': Optional[str],
    'min_funding': Optional[float]
}
```

**Action**: Create a `src/models/` directory in bio-match-ml and define:
- `search_models.py` - QueryAnalysis, SearchResults
- `matching_models.py` - MatchScore, AlignmentAssessment

**Or** just use dictionaries for now (simpler, faster). Add proper models later.

---

### 4. Biology Knowledge Bases ✅

**Start with MINIMAL starter sets, make them expandable:**

#### Approach:
```python
# In src/search/biology_knowledge.py (create this helper file)

"""Biology domain knowledge for query processing"""

# MINIMAL starter set - can be expanded
CAREER_STAGE_KEYWORDS = {
    'young': 'assistant_professor',
    'new': 'assistant_professor',
    'junior': 'assistant_professor',
    'early career': 'assistant_professor',
    'established': 'associate_professor',
    'mid career': 'associate_professor',
    'senior': 'full_professor',
    'emeritus': 'emeritus'
}

# Top 20 most common gene aliases
COMMON_GENE_ALIASES = {
    'p53': ['TP53', 'tumor protein 53', 'tumor protein p53'],
    'egfr': ['epidermal growth factor receptor', 'ERBB1'],
    'brca1': ['breast cancer 1', 'BRCA1'],
    'myc': ['c-myc', 'MYC'],
    'ras': ['KRAS', 'NRAS', 'HRAS'],
    'akt': ['protein kinase B', 'PKB'],
    'vegf': ['vascular endothelial growth factor'],
    'il6': ['interleukin 6', 'interleukin-6'],
    'tnf': ['tumor necrosis factor', 'TNF-alpha'],
    'nf-kb': ['nuclear factor kappa B', 'NFKB'],
    # Add more as needed
}

# Top 30 technique synonyms
TECHNIQUE_SYNONYMS = {
    'crispr': ['CRISPR-Cas9', 'gene editing', 'genome editing', 'cas9'],
    'rna-seq': ['RNA sequencing', 'transcriptomics', 'RNA-sequencing'],
    'pcr': ['polymerase chain reaction', 'RT-PCR', 'qPCR'],
    'western blot': ['western blotting', 'immunoblot', 'protein blot'],
    'flow cytometry': ['FACS', 'fluorescence activated cell sorting'],
    'elisa': ['enzyme-linked immunosorbent assay'],
    'chip-seq': ['chromatin immunoprecipitation sequencing', 'ChIP sequencing'],
    'microscopy': ['imaging', 'confocal microscopy', 'fluorescence microscopy'],
    'mass spec': ['mass spectrometry', 'LC-MS', 'proteomics'],
    'cloning': ['molecular cloning', 'gene cloning'],
    # Add more as needed
}

# Common organism mappings
ORGANISM_MAPPINGS = {
    'mouse': ['Mus musculus', 'mice'],
    'rat': ['Rattus norvegicus', 'rats'],
    'human': ['Homo sapiens', 'humans'],
    'fly': ['Drosophila melanogaster', 'fruit fly', 'drosophila'],
    'worm': ['C. elegans', 'Caenorhabditis elegans'],
    'zebrafish': ['Danio rerio', 'zebra fish'],
    'yeast': ['Saccharomyces cerevisiae', 'S. cerevisiae'],
    'e. coli': ['Escherichia coli', 'E. coli'],
    # Add more as needed
}

# Disease synonyms
DISEASE_SYNONYMS = {
    'cancer': ['tumor', 'neoplasm', 'malignancy', 'carcinoma'],
    'alzheimers': ["alzheimer's disease", 'AD', 'dementia'],
    'parkinsons': ["parkinson's disease", 'PD'],
    'diabetes': ['diabetes mellitus', 'type 2 diabetes', 'T2D'],
    'covid': ['COVID-19', 'coronavirus', 'SARS-CoV-2'],
    # Add more as needed
}

def expand_biology_term(term: str) -> List[str]:
    """
    Expand a biology term to include synonyms

    Args:
        term: Input term (gene, technique, organism, disease)

    Returns:
        List of synonyms/expansions including original term
    """
    term_lower = term.lower()
    expansions = [term]  # Always include original

    # Check all dictionaries
    for knowledge_dict in [COMMON_GENE_ALIASES, TECHNIQUE_SYNONYMS,
                          ORGANISM_MAPPINGS, DISEASE_SYNONYMS]:
        if term_lower in knowledge_dict:
            expansions.extend(knowledge_dict[term_lower])

    return list(set(expansions))  # Remove duplicates
```

**Why minimal**:
- ✅ Faster to implement
- ✅ Easier to test
- ✅ Can expand incrementally
- ✅ Covers 80% of common cases

**Expansion strategy** (for later):
- Add more as you test with real queries
- Eventually load from external files (JSON/YAML)
- Could integrate with actual ontologies (MeSH, GO) in Batch 3

---

### 5. External Dependencies ✅

**You can add packages - they're already in requirements.txt!**

#### Already Available (use freely):
```python
# NLP & Text Processing
import spacy                    # Already installed
import nltk                     # Already installed
from sklearn.feature_extraction.text import TfidfVectorizer  # Already installed

# Biology-specific
import scispacy                 # Already installed
from bioservices import BioMart  # For ontology lookups
import pronto                   # For OBO ontologies

# Vector operations
import numpy as np
from scipy.spatial.distance import cosine

# Data processing
import pandas as pd

# Testing
import pytest
```

#### You DON'T need to add:
- ✅ All biology/ML packages already there
- ✅ Testing framework (pytest) already there
- ✅ NLP libraries already there

#### If you find you need something new:
1. Check if it's already in requirements.txt first
2. If not, add it (but mention it in your implementation notes)
3. Prefer packages already in requirements over new ones

---

### 6. Integration with API ✅

**YES - Please update src/api/main.py as final step:**

#### After implementing all 5 files, update main.py:

**Replace these sections**:

1. **Lines 90-110** - Add search engine initialization:
```python
def get_search_engine():
    """Get or initialize search engine"""
    global _search_engine
    if _search_engine is None:
        from ..search.semantic_search import SemanticSearchEngine
        from ..search.query_processor import QueryProcessor
        from ..search.result_ranker import ResultRanker

        embedder = get_embedding_generator()
        vector_store = get_vector_store()

        query_processor = QueryProcessor()
        result_ranker = ResultRanker()

        _search_engine = SemanticSearchEngine(
            embedding_generator=embedder,
            vector_store=vector_store,
            query_processor=query_processor,
            result_ranker=result_ranker
        )
        logger.info("Initialized SemanticSearchEngine")
    return _search_engine

def get_matcher():
    """Get or initialize matcher"""
    global _matcher
    if _matcher is None:
        from ..matching.multi_factor_scorer import MultiFactorMatcher
        from ..matching.similarity_calculator import SimilarityCalculator

        embedder = get_embedding_generator()
        similarity_calc = SimilarityCalculator(embedding_generator=embedder)
        _matcher = MultiFactorMatcher(similarity_calculator=similarity_calc)
        logger.info("Initialized MultiFactorMatcher")
    return _matcher
```

2. **Lines 140-216** - Replace semantic_search endpoint:
```python
@app.post("/api/v1/search/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Semantic search using intelligent query processing and ranking"""
    try:
        search_engine = get_search_engine()

        # Use new intelligent search
        results = search_engine.search(
            query=request.query,
            search_mode=request.mode,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            explain=request.explain
        )

        # Convert SearchResults to API response
        return SearchResponse(
            results=[
                FacultyResult(
                    id=r['id'],
                    name=r['metadata']['name'],
                    institution=r['metadata']['institution'],
                    department=r['metadata']['department'],
                    research_summary=r['metadata']['research_summary'],
                    match_score=r['final_score'],
                    explanation=r.get('explanation')
                )
                for r in results.results
            ],
            total=results.total_count,
            facets=results.facets,
            query_interpretation=results.query_interpretation
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

3. **Lines 242-340** - Replace calculate_match endpoint:
```python
@app.post("/api/v1/match/calculate", response_model=MatchResponse)
async def calculate_match(request: MatchRequest):
    """Calculate faculty-student matches using multi-factor scoring"""
    try:
        matcher = get_matcher()

        # Get faculty profiles (from vector store or database)
        # For now, search similar faculty using embeddings
        search_engine = get_search_engine()

        # Convert student profile to search query
        student_text = request.student_profile.get('research_interests', '')

        # Get candidate faculty
        candidate_results = search_engine.search(
            query=student_text,
            search_mode='faculty',
            filters=None,
            limit=request.top_k * 2,  # Get more candidates
            explain=False
        )

        # Calculate match scores
        matches = []
        for result in candidate_results.results[:request.top_k]:
            faculty_profile = result['metadata']

            match_score = matcher.calculate_match_score(
                student_profile=request.student_profile,
                faculty_profile=faculty_profile,
                explain=(len(matches) < request.explain_top_n)
            )

            matches.append(
                MatchResult(
                    faculty_id=result['id'],
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
            student_profile_summary={
                'research_areas': request.student_profile.get('topics', []),
                'techniques': request.student_profile.get('techniques', []),
                'career_goals': request.student_profile.get('career_goals', '')
            }
        )
    except Exception as e:
        logger.error(f"Match error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

**Do this as your LAST step** after all 5 files are working and tested.

---

## Recommended Workflow

### Week 1: Foundation
**Day 1-2**:
- ✅ Create `src/search/biology_knowledge.py` with minimal dictionaries
- ✅ Implement `query_processor.py`
- ✅ Write unit tests for query_processor
- ✅ Test with manual script

**Day 3**:
- ✅ Implement `similarity_calculator.py` (standalone)
- ✅ Write unit tests
- ✅ Test with manual script

### Week 2: Search
**Day 4-5**:
- ✅ Implement `result_ranker.py`
- ✅ Write unit tests
- ✅ Test integration with query_processor

**Day 6-7**:
- ✅ Implement `semantic_search.py` (orchestrator)
- ✅ Write unit tests
- ✅ Test full search pipeline
- ✅ Create manual test script

### Week 3: Matching
**Day 8-10**:
- ✅ Implement `multi_factor_scorer.py`
- ✅ Write unit tests
- ✅ Test full matching pipeline
- ✅ Create manual test script

### Week 4: Integration
**Day 11-12**:
- ✅ Integration tests (full search + match pipeline)
- ✅ Update API endpoints in main.py
- ✅ End-to-end API testing
- ✅ Bug fixes and refinement

**Day 13-14**:
- ✅ Documentation
- ✅ Performance testing
- ✅ Code review and cleanup

---

## Mock Data for Testing

Since you don't have real faculty data yet, create mock data:

```python
# tests/fixtures/mock_data.py

MOCK_FACULTY_PROFILES = [
    {
        'faculty_id': 'fac001',
        'name': 'Dr. Jane Smith',
        'title': 'Assistant Professor',
        'department': 'Biology',
        'institution': 'MIT',
        'research_interests': 'My lab uses CRISPR-Cas9 gene editing to study cancer vulnerabilities in human cell lines and mouse models.',
        'publications': [
            {
                'pmid': '12345',
                'title': 'CRISPR screens identify cancer dependencies',
                'abstract': '...',
                'year': 2024,
                'keywords': ['CRISPR', 'cancer', 'screening']
            }
        ],
        'grants': [
            {
                'project_number': 'R01-CA12345',
                'active': True,
                'total_cost': 1500000,
                'end_date': '2028-01-01'
            }
        ],
        'h_index': 28,
        'total_funding': 1500000,
        'active_grants': 1,
        'accepting_students': True
    },
    # Add 10-20 more varied profiles
]

MOCK_STUDENT_PROFILE = {
    'student_id': 'stu001',
    'name': 'Alex Chen',
    'research_interests': 'I want to use gene editing to develop cancer therapies',
    'topics': ['gene editing', 'cancer biology'],
    'techniques': ['CRISPR', 'RNA-seq'],
    'organisms': ['human', 'mouse'],
    'career_goals': 'Academic research career in cancer biology'
}
```

Use these in your tests!

---

## Final Checklist

Before considering Batch 1 complete:

- [ ] All 5 files implemented and working
- [ ] Unit tests for each file (>70% coverage)
- [ ] Manual test scripts working
- [ ] Integration test passing
- [ ] API endpoints updated and tested
- [ ] Mock data created for testing
- [ ] Biology knowledge dictionaries created
- [ ] Documentation strings complete
- [ ] Code follows existing style (use black, flake8)
- [ ] No hardcoded values (use config)
- [ ] Error handling for edge cases
- [ ] Logging added for debugging

---

## Questions?

**Q: Should I handle edge cases thoroughly?**
A: YES - especially missing data. Faculty profiles might be incomplete.

**Q: How much logging?**
A: Liberal logging at INFO level for pipeline steps, DEBUG for details.

**Q: Code style?**
A: Follow existing code style. Use type hints. Run `black` formatter.

**Q: Performance?**
A: Don't optimize prematurely, but avoid obvious inefficiencies (e.g., O(n²) loops).

**Q: What if I need clarification on the biology logic?**
A: Use reasonable defaults. Document assumptions. Can refine later.

---

## Summary

✅ **Implement all 5 files** in the specified order
✅ **Create comprehensive tests** (unit + integration + manual)
✅ **Use existing data models** from bio-match-platform
✅ **Start with minimal biology dictionaries** (expandable)
✅ **Use existing dependencies** (all in requirements.txt)
✅ **Update API as final step** after all modules work

**Your instincts are correct** - explore the codebase, understand the data models, then implement systematically with testing at each step.

**Time estimate**: 2-4 weeks for thorough implementation with tests.

Good luck! 🚀
