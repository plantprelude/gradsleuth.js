# Testing Guide for Bio-Match-ML

This guide explains how to test the bio-match-ml search and matching systems.

## Table of Contents

- [Overview](#overview)
- [Running Tests](#running-tests)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Manual Testing](#manual-testing)
- [Test Coverage](#test-coverage)
- [Mock Data](#mock-data)

## Overview

The bio-match-ml testing suite includes:
- **Unit tests** for individual components (5 test suites, 200+ tests)
- **Integration tests** for end-to-end pipelines (2 test suites)
- **Manual test scripts** for interactive testing
- **Mock data fixtures** for consistent test data

## Running Tests

### All Tests

Run all tests with pytest:

```bash
cd bio-match-ml
pytest tests/
```

### With Coverage

Run tests with coverage report:

```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term
```

View the HTML coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Specific Test Suites

Run specific test modules:

```bash
# Search tests
pytest tests/test_search/

# Matching tests
pytest tests/test_matching/

# Integration tests
pytest tests/test_integration/
```

### Specific Test Files

```bash
# Query processor tests
pytest tests/test_search/test_query_processor.py

# Semantic search tests
pytest tests/test_search/test_semantic_search.py

# Result ranker tests
pytest tests/test_search/test_result_ranker.py

# Similarity calculator tests
pytest tests/test_matching/test_similarity_calculator.py

# Multi-factor scorer tests
pytest tests/test_matching/test_multi_factor_scorer.py

# Integration pipeline tests
pytest tests/test_integration/test_search_pipeline.py
```

### Verbose Output

Run tests with detailed output:

```bash
pytest tests/ -v
```

Show print statements:

```bash
pytest tests/ -s
```

## Unit Tests

### Search Component Tests

#### 1. Query Processor Tests (`test_query_processor.py`)

Tests intelligent query understanding and expansion:

```bash
pytest tests/test_search/test_query_processor.py -v
```

Key test categories:
- Entity extraction (techniques, topics, genes, organisms)
- Intent detection (technique_based, organism_based, funding_based, etc.)
- Query expansion with biological synonyms
- Implicit filter extraction
- Normalization and validation

Example tests:
```python
# Test entity extraction
def test_extract_entities_with_techniques()
def test_extract_entities_with_genes()
def test_extract_entities_with_organisms()

# Test intent detection
def test_detect_intent_technique_based()
def test_detect_intent_funding_based()

# Test query expansion
def test_expand_query_with_gene_synonyms()
def test_expand_query_with_technique_synonyms()
```

#### 2. Result Ranker Tests (`test_result_ranker.py`)

Tests multi-factor ranking beyond semantic similarity:

```bash
pytest tests/test_search/test_result_ranker.py -v
```

Key test categories:
- Productivity scoring (publications, citations, h-index)
- Funding scoring (active grants, amount, diversity)
- Recency scoring (publication dates)
- H-index normalization
- Result diversification (MMR algorithm)
- Ranking explanations

#### 3. Semantic Search Tests (`test_semantic_search.py`)

Tests complete search orchestration:

```bash
pytest tests/test_search/test_semantic_search.py -v
```

Key test categories:
- End-to-end search pipeline
- Query interpretation
- Facet generation (institutions, departments, techniques)
- Multi-query search (union, intersection, weighted)
- Search-by-example
- Result aggregation (RRF)

### Matching Component Tests

#### 4. Similarity Calculator Tests (`test_similarity_calculator.py`)

Tests various similarity metrics:

```bash
pytest tests/test_matching/test_similarity_calculator.py -v
```

Key test categories:
- Embedding similarity (cosine similarity)
- Jaccard similarity (set overlap)
- Weighted overlap
- Research similarity (multi-component)
- Trajectory alignment
- Dice coefficient
- TF-IDF similarity

#### 5. Multi-Factor Scorer Tests (`test_multi_factor_scorer.py`)

Tests comprehensive faculty-student matching:

```bash
pytest tests/test_matching/test_multi_factor_scorer.py -v
```

Key test categories:
- Overall match score calculation
- All 6 component scores:
  - Research alignment (35%)
  - Funding stability (20%)
  - Productivity compatibility (15%)
  - Technique match (15%)
  - Lab environment (10%)
  - Career development (5%)
- Match explanations
- Strengths and considerations
- Recommendations (highly_recommended, recommended, consider, not_recommended)

## Integration Tests

### Search Pipeline Tests (`test_search_pipeline.py`)

Tests complete search flow end-to-end:

```bash
pytest tests/test_integration/test_search_pipeline.py -v
```

Key test categories:
- Complete search flow (query → embedding → search → rank → results)
- Query expansion integration
- Implicit filter extraction and application
- Multi-factor ranking improvement
- Intent detection
- Faceted search results

### Matching Pipeline Tests

Tests complete matching flow end-to-end:

```bash
pytest tests/test_integration/test_search_pipeline.py::TestMatchingPipeline -v
```

Key test categories:
- Complete matching flow (profile → search → match → rank)
- All six matching factors integration
- Match explanation generation

## Manual Testing

### Interactive Search Testing

Run the interactive search test console:

```bash
python scripts/test_search_manual.py
```

Or with predefined queries:

```bash
python scripts/test_search_manual.py --mode predefined
```

Or both:

```bash
python scripts/test_search_manual.py --mode both
```

**Interactive Commands:**
- Type a search query to search
- `modes` - See available search modes
- `examples` - See example queries
- `quit` - Exit

**Example Queries:**
- `CRISPR gene editing in cancer`
- `well-funded immunology labs at MIT`
- `young PI studying neural circuits with optogenetics`
- `synthetic biology and metabolic engineering`
- `mouse models of disease`

### Interactive Matching Testing

Run the interactive matching test console:

```bash
python scripts/test_matching_manual.py
```

Or with predefined scenarios:

```bash
python scripts/test_matching_manual.py --mode predefined
```

**Interactive Commands:**
- `1-3` - Select a student profile
- `list` - List all profiles
- `match` - Run matching for selected student
- `all` - Run matching for all students
- `quit` - Exit

## Test Coverage

Expected coverage targets:

| Component | Target | Status |
|-----------|--------|--------|
| Query Processor | > 90% | ✅ |
| Result Ranker | > 85% | ✅ |
| Semantic Search | > 85% | ✅ |
| Similarity Calculator | > 90% | ✅ |
| Multi-Factor Scorer | > 90% | ✅ |

Generate coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Mock Data

### Using Mock Data Fixtures

Import pre-built mock data:

```python
from tests.fixtures.mock_data import (
    get_mock_faculty_profiles,
    get_mock_student_profiles,
    get_mock_embeddings,
    get_mock_search_results,
    get_high_match_pair,
    get_medium_match_pair
)

# Get sample faculty profiles
faculty = get_mock_faculty_profiles()

# Get sample student profiles
students = get_mock_student_profiles()

# Get specific match pairs
student, faculty = get_high_match_pair()  # Should match highly
```

### Available Mock Data Functions

- `get_mock_faculty_profiles()` - 5 comprehensive faculty profiles
- `get_mock_student_profiles()` - 4 comprehensive student profiles
- `get_mock_embeddings(dimension, count)` - Generate normalized embeddings
- `get_mock_search_results()` - Search results with scores
- `get_mock_query_analysis()` - Query analysis result
- `get_mock_match_score()` - Match score result
- `get_mock_publications()` - Publication data
- `get_mock_grants()` - Grant data
- `get_high_match_pair()` - Student + faculty that should match highly
- `get_medium_match_pair()` - Student + faculty with moderate match
- `get_low_match_pair()` - Student + faculty with poor match
- `get_computational_match_pair()` - Computational research pair

## Writing New Tests

### Test Structure

Follow this structure for new tests:

```python
import pytest
from unittest.mock import Mock, patch
from tests.fixtures.mock_data import get_mock_faculty_profiles

class TestMyComponent:
    """Test suite for MyComponent"""

    @pytest.fixture
    def mock_dependency(self):
        """Create mock dependency"""
        mock = Mock()
        # Configure mock
        return mock

    def test_basic_functionality(self, mock_dependency):
        """Test basic functionality"""
        # Arrange
        component = MyComponent(mock_dependency)

        # Act
        result = component.do_something()

        # Assert
        assert result is not None
        assert mock_dependency.called
```

### Testing Best Practices

1. **Use fixtures** for reusable test data
2. **Mock external dependencies** (embeddings, databases, APIs)
3. **Test edge cases** (empty inputs, missing data, errors)
4. **Use descriptive test names** (`test_calculate_score_with_missing_data`)
5. **Include docstrings** explaining what each test verifies
6. **Test both happy path and error cases**
7. **Verify both behavior and values**

### Example Test

```python
def test_calculate_match_score_with_complete_profiles(self):
    """Test match score calculation with complete student and faculty profiles"""
    # Arrange
    student, faculty = get_high_match_pair()
    matcher = MultiFactorMatcher()

    # Act
    match_score = matcher.calculate_match_score(student, faculty)

    # Assert
    assert 0.0 <= match_score.overall_score <= 1.0
    assert len(match_score.component_scores) == 6
    assert match_score.confidence > 0.7
    assert len(match_score.strengths) > 0
    assert match_score.recommendation in [
        'highly_recommended', 'recommended', 'consider', 'not_recommended'
    ]
```

## Continuous Integration

Tests are run automatically on:
- Pull requests to main branch
- Commits to main branch

CI configuration: See `.github/workflows/test.yml`

## Troubleshooting

### Common Issues

**Issue: Import errors**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:** Run from bio-match-ml directory:
```bash
cd bio-match-ml
pytest tests/
```

**Issue: Mock embeddings not working**

**Solution:** Ensure you're using mock components:
```python
from unittest.mock import Mock
embedder = Mock()
embedder.generate_embedding = Mock(return_value=np.random.rand(768))
```

**Issue: Slow tests**

**Solution:** Run specific test files or use markers:
```bash
pytest tests/ -m "not slow"
```

## Performance Testing

Performance benchmarks are included in some tests:

```bash
# Run with timing
pytest tests/ --durations=10
```

Expected performance:
- Query processing: < 50ms
- Semantic search: < 200ms (with mocked embeddings)
- Match score calculation: < 100ms

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py docs](https://coverage.readthedocs.io/)

## Getting Help

For questions or issues:
1. Check this documentation
2. Review existing tests for examples
3. Check test output for error messages
4. Run tests in verbose mode: `pytest -v -s`
