"""
Manual test script for search functionality

This script demonstrates how to use the semantic search engine with real queries.
Run from bio-match-ml directory: python scripts/test_search_manual.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from search.semantic_search import SemanticSearchEngine
from search.query_processor import QueryProcessor
from search.result_ranker import ResultRanker
from unittest.mock import Mock
import numpy as np


def create_mock_components():
    """Create mock components for testing without full infrastructure"""
    # Mock embedding generator
    mock_embedder = Mock()
    mock_embedder.generate_embedding = lambda x: np.random.rand(768).tolist()

    # Mock vector store with sample data
    mock_vector_store = Mock()

    def mock_search(query_embedding, index_name, k, filters=None):
        # Return sample faculty results
        return [
            {
                'id': f'fac{i}',
                'score': 0.9 - i * 0.05,
                'metadata': {
                    'name': f'Dr. Sample{i}',
                    'research_summary': f'Research in CRISPR and gene editing',
                    'h_index': 35 - i * 5,
                    'publication_count': 80 - i * 10,
                    'active_grants': 2 if i < 3 else 1,
                    'total_funding': 1000000 - i * 100000,
                    'last_publication_year': 2024,
                    'institution': 'MIT' if i % 2 == 0 else 'Stanford',
                    'department': 'Biology',
                    'techniques': ['CRISPR', 'RNA-seq', 'PCR'],
                    'organisms': ['mouse', 'human'],
                    'accepting_students': True,
                    'lab_size': 8 + i,
                    'primary_research_area': 'Molecular Biology'
                }
            }
            for i in range(k)
        ]

    mock_vector_store.search = mock_search

    return mock_embedder, mock_vector_store


def main():
    print("="*80)
    print("Semantic Search Engine - Manual Test")
    print("="*80)
    print()

    # Create mock components
    print("Initializing search engine...")
    embedder, vector_store = create_mock_components()

    # Create query processor and result ranker
    query_processor = QueryProcessor()
    result_ranker = ResultRanker()

    # Create search engine
    engine = SemanticSearchEngine(
        embedding_generator=embedder,
        vector_store=vector_store,
        query_processor=query_processor,
        result_ranker=result_ranker
    )

    print("Search engine initialized!\n")

    # Test queries
    test_queries = [
        "CRISPR gene editing in neurons",
        "young PI studying cancer immunology",
        "well-funded structural biology lab",
        "p53 mutation in breast cancer",
        "machine learning in genomics"
    ]

    for query in test_queries:
        print('='*80)
        print(f"Query: {query}")
        print('='*80)

        try:
            results = engine.search(
                query=query,
                search_mode='faculty',
                limit=5,
                explain=True,
                diversity_factor=0.3
            )

            print(f"\nFound {results.total_count} results")

            # Query interpretation
            print(f"\nQuery Interpretation:")
            print(f"  Intent: {results.query_interpretation.get('intent')}")
            print(f"  Entities: {results.query_interpretation.get('entities')}")
            print(f"  Filters: {results.query_interpretation.get('applied_filters')}")

            # Top results
            print(f"\nTop 5 Results:")
            for i, result in enumerate(results.results[:5], 1):
                metadata = result.get('metadata', {})
                score = result.get('final_score', result.get('score', 0))

                print(f"\n{i}. {metadata.get('name', 'Unknown')} - Score: {score:.3f}")
                print(f"   Institution: {metadata.get('institution', 'N/A')}")
                print(f"   H-index: {metadata.get('h_index', 0)}, "
                      f"Publications: {metadata.get('publication_count', 0)}")
                print(f"   Active Grants: {metadata.get('active_grants', 0)}")

                if result.get('ranking_explanation'):
                    print(f"   Explanation: {result['ranking_explanation']}")

            # Facets
            if results.facets:
                print(f"\nFacet Aggregations:")
                for facet_name, facet_values in results.facets.items():
                    if facet_values:
                        print(f"  {facet_name.title()}: {dict(list(facet_values.items())[:3])}")

            # Search metadata
            print(f"\nSearch Metadata:")
            print(f"  Search time: {results.search_metadata.get('search_time_ms', 0):.2f}ms")
            print(f"  Results before ranking: {results.search_metadata.get('results_before_ranking', 0)}")
            print(f"  Query variants used: {results.search_metadata.get('query_variants_used', 0)}")

        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Test multi-query search
    print('='*80)
    print("Testing Multi-Query Search")
    print('='*80)

    try:
        multi_results = engine.multi_query_search(
            queries=["CRISPR gene editing", "genome editing techniques"],
            aggregation='weighted',
            limit=5
        )

        print(f"\nMulti-query search found {multi_results.total_count} results")
        print("\nTop 3 Combined Results:")
        for i, result in enumerate(multi_results.results[:3], 1):
            metadata = result.get('metadata', {})
            score = result.get('final_score', result.get('score', 0))
            print(f"{i}. {metadata.get('name')} - Score: {score:.3f}")

    except Exception as e:
        print(f"Error during multi-query search: {e}")

    print("\n" + "="*80)
    print("Manual test complete!")
    print("="*80)


if __name__ == '__main__':
    main()
