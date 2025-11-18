#!/usr/bin/env python3
"""
Manual test script for search functionality

This script allows interactive testing of the semantic search engine.
Run: python scripts/test_search_manual.py

Features:
- Interactive query testing
- Multiple search modes (faculty, publications, grants)
- Result explanation and facet display
- Performance timing
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from typing import Dict, List, Any
from unittest.mock import Mock
import numpy as np


def create_mock_components():
    """Create mock components for testing"""
    # Mock embedding generator
    embedder = Mock()
    embedder.generate_embedding = Mock(return_value=np.random.rand(768))

    # Mock vector store with diverse sample data
    vector_store = Mock()
    vector_store.list_indices = Mock(return_value=['faculty_embeddings', 'publication_embeddings'])

    # Sample faculty data
    sample_results = [
        {
            'id': 'fac001',
            'score': 0.92,
            'metadata': {
                'name': 'Dr. Alice Chen',
                'institution': 'MIT',
                'department': 'Biology',
                'research_summary': 'CRISPR gene editing in cancer cells, developing new therapeutic approaches',
                'h_index': 45,
                'publication_count': 120,
                'citation_count': 5800,
                'has_active_funding': True,
                'total_funding': 2500000,
                'active_grants': [
                    {'amount': 1500000, 'end_date': '2028-12-31', 'title': 'R01: CRISPR therapy'},
                    {'amount': 1000000, 'end_date': '2027-06-30', 'title': 'NSF: Gene regulation'}
                ],
                'techniques': ['CRISPR-Cas9', 'RNA-seq', 'flow cytometry', 'single-cell sequencing'],
                'organisms': ['human', 'mouse'],
                'career_stage': 'mid_career',
                'accepting_students': True,
                'lab_size': 12,
                'recent_publications': [
                    {'publication_date': '2024-11-01', 'title': 'CRISPR screens in cancer', 'citations': 15},
                    {'publication_date': '2024-09-15', 'title': 'Therapeutic gene editing', 'citations': 8}
                ]
            }
        },
        {
            'id': 'fac002',
            'score': 0.88,
            'metadata': {
                'name': 'Dr. Robert Martinez',
                'institution': 'Harvard',
                'department': 'Genetics',
                'research_summary': 'Cancer genomics and precision medicine using next-generation sequencing',
                'h_index': 38,
                'publication_count': 95,
                'citation_count': 4200,
                'has_active_funding': True,
                'total_funding': 1800000,
                'active_grants': [
                    {'amount': 1800000, 'end_date': '2027-12-31', 'title': 'R01: Cancer mutations'}
                ],
                'techniques': ['RNA-seq', 'ChIP-seq', 'whole-genome sequencing', 'bioinformatics'],
                'organisms': ['human'],
                'career_stage': 'mid_career',
                'accepting_students': True,
                'lab_size': 8,
                'recent_publications': [
                    {'publication_date': '2024-10-20', 'title': 'Genomic drivers of cancer', 'citations': 22},
                    {'publication_date': '2024-08-05', 'title': 'Precision medicine approaches', 'citations': 12}
                ]
            }
        },
        {
            'id': 'fac003',
            'score': 0.85,
            'metadata': {
                'name': 'Dr. Sarah Lee',
                'institution': 'Stanford',
                'department': 'Bioengineering',
                'research_summary': 'Synthetic biology and metabolic engineering for biofuel production',
                'h_index': 32,
                'publication_count': 72,
                'citation_count': 3100,
                'has_active_funding': True,
                'total_funding': 1200000,
                'active_grants': [
                    {'amount': 1200000, 'end_date': '2026-12-31', 'title': 'DOE: Biofuel engineering'}
                ],
                'techniques': ['CRISPR-Cas9', 'metabolic engineering', 'synthetic circuits', 'fermentation'],
                'organisms': ['yeast', 'bacteria', 'E. coli'],
                'career_stage': 'early_career',
                'accepting_students': True,
                'lab_size': 6,
                'recent_publications': [
                    {'publication_date': '2024-09-30', 'title': 'Engineered yeast for biofuel', 'citations': 18},
                    {'publication_date': '2024-07-15', 'title': 'Synthetic metabolic pathways', 'citations': 10}
                ]
            }
        },
        {
            'id': 'fac004',
            'score': 0.82,
            'metadata': {
                'name': 'Dr. James Wilson',
                'institution': 'UCSF',
                'department': 'Immunology',
                'research_summary': 'T cell biology and immunotherapy for cancer treatment',
                'h_index': 28,
                'publication_count': 65,
                'citation_count': 2400,
                'has_active_funding': True,
                'total_funding': 900000,
                'active_grants': [
                    {'amount': 900000, 'end_date': '2026-06-30', 'title': 'R21: CAR-T therapy'}
                ],
                'techniques': ['flow cytometry', 'cell culture', 'immunofluorescence', 'ELISA'],
                'organisms': ['human', 'mouse'],
                'career_stage': 'early_career',
                'accepting_students': True,
                'lab_size': 5,
                'recent_publications': [
                    {'publication_date': '2024-08-20', 'title': 'CAR-T cell engineering', 'citations': 25},
                    {'publication_date': '2024-06-10', 'title': 'T cell exhaustion in cancer', 'citations': 14}
                ]
            }
        },
        {
            'id': 'fac005',
            'score': 0.78,
            'metadata': {
                'name': 'Dr. Maria Garcia',
                'institution': 'Johns Hopkins',
                'department': 'Neuroscience',
                'research_summary': 'Neural circuits and behavior using optogenetics and imaging',
                'h_index': 24,
                'publication_count': 48,
                'citation_count': 1800,
                'has_active_funding': False,
                'total_funding': 400000,
                'active_grants': [],
                'techniques': ['optogenetics', 'two-photon imaging', 'electrophysiology', 'behavioral assays'],
                'organisms': ['mouse', 'zebrafish'],
                'career_stage': 'early_career',
                'accepting_students': False,
                'lab_size': 4,
                'recent_publications': [
                    {'publication_date': '2023-12-15', 'title': 'Neural circuits of learning', 'citations': 20},
                    {'publication_date': '2023-10-05', 'title': 'Optogenetic control', 'citations': 16}
                ]
            }
        }
    ]

    vector_store.search = Mock(return_value=sample_results)

    return {
        'embedder': embedder,
        'vector_store': vector_store
    }


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_result(result: Dict, rank: int):
    """Print formatted search result"""
    metadata = result.get('metadata', {})

    print(f"\n{rank}. {metadata.get('name', 'Unknown')} - {metadata.get('institution', 'N/A')}")
    print(f"   Score: {result.get('final_score', result.get('score', 0.0)):.3f}")
    print(f"   Department: {metadata.get('department', 'N/A')}")
    print(f"   Research: {metadata.get('research_summary', 'N/A')[:100]}...")
    print(f"   H-Index: {metadata.get('h_index', 'N/A')} | Publications: {metadata.get('publication_count', 'N/A')}")
    print(f"   Funding: ${metadata.get('total_funding', 0):,} | Active Grants: {len(metadata.get('active_grants', []))}")
    print(f"   Techniques: {', '.join(metadata.get('techniques', [])[:3])}")

    # Show ranking explanation if available
    if 'ranking_explanation' in result:
        print(f"   💡 {result['ranking_explanation']}")


def print_facets(facets: Dict):
    """Print facet aggregations"""
    print("\n📊 Facets:")

    for facet_name, facet_values in facets.items():
        if facet_values:
            print(f"\n  {facet_name.replace('_', ' ').title()}:")
            for item in facet_values[:5]:  # Top 5
                print(f"    • {item['value']}: {item['count']}")


def print_query_interpretation(interpretation: Dict):
    """Print query interpretation"""
    print("\n🔍 Query Interpretation:")
    print(f"  Original: {interpretation.get('original_query', 'N/A')}")
    print(f"  Normalized: {interpretation.get('normalized_query', 'N/A')}")
    print(f"  Detected Intent: {interpretation.get('detected_intent', 'general')}")

    if interpretation.get('extracted_entities'):
        print(f"  Extracted Entities:")
        for entity_type, entities in interpretation['extracted_entities'].items():
            if entities:
                print(f"    • {entity_type}: {', '.join(entities[:3])}")

    if interpretation.get('implicit_filters'):
        print(f"  Implicit Filters: {interpretation['implicit_filters']}")

    if interpretation.get('num_expansions', 0) > 1:
        print(f"  Query Expansions: {interpretation['num_expansions']}")


def test_search_interactive():
    """Interactive search testing"""
    from src.search.semantic_search import SemanticSearchEngine

    print_header("Semantic Search - Interactive Testing")

    # Create mock components
    components = create_mock_components()

    # Initialize search engine
    engine = SemanticSearchEngine(
        embedding_generator=components['embedder'],
        vector_store=components['vector_store']
    )

    print("Welcome to the Semantic Search Test Console!")
    print("\nCommands:")
    print("  - Type a search query to search")
    print("  - 'modes' to see available search modes")
    print("  - 'examples' to see example queries")
    print("  - 'quit' to exit")

    while True:
        print("\n" + "-" * 80)
        query = input("\n🔍 Enter search query (or command): ").strip()

        if not query:
            continue

        if query.lower() == 'quit':
            print("\nGoodbye!")
            break

        if query.lower() == 'modes':
            print("\nAvailable Search Modes:")
            print("  • faculty - Search for faculty members (default)")
            print("  • publications - Search for publications")
            print("  • grants - Search for grants")
            continue

        if query.lower() == 'examples':
            print("\nExample Queries:")
            print("  • CRISPR gene editing in cancer")
            print("  • well-funded immunology labs at MIT")
            print("  • young PI studying neural circuits with optogenetics")
            print("  • synthetic biology and metabolic engineering")
            print("  • mouse models of disease")
            continue

        # Execute search
        print(f"\nSearching for: '{query}'...")
        start_time = time.time()

        try:
            results = engine.search(
                query=query,
                search_mode='faculty',
                limit=5,
                explain=True
            )

            elapsed = (time.time() - start_time) * 1000

            # Print results
            print_header(f"Search Results ({results.total_count} found in {elapsed:.1f}ms)")

            # Print query interpretation
            if results.query_interpretation:
                print_query_interpretation(results.query_interpretation)

            # Print results
            print(f"\n📄 Top {len(results.results)} Results:")
            for i, result in enumerate(results.results, 1):
                print_result(result, i)

            # Print facets
            if results.facets:
                print_facets(results.facets)

            # Print metadata
            if results.search_metadata:
                meta = results.search_metadata
                print(f"\n⏱️  Performance:")
                print(f"  Total Time: {meta.get('total_time_ms', elapsed):.1f}ms")
                print(f"  Query Processing: {meta.get('query_processing_ms', 0):.1f}ms")
                print(f"  Vector Search: {meta.get('vector_search_ms', 0):.1f}ms")
                print(f"  Ranking: {meta.get('ranking_ms', 0):.1f}ms")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def test_predefined_queries():
    """Test with predefined queries"""
    from src.search.semantic_search import SemanticSearchEngine

    print_header("Semantic Search - Predefined Query Tests")

    # Create mock components
    components = create_mock_components()

    # Initialize search engine
    engine = SemanticSearchEngine(
        embedding_generator=components['embedder'],
        vector_store=components['vector_store']
    )

    # Test queries
    test_queries = [
        ("CRISPR gene editing in cancer", "Research topic query"),
        ("well-funded cancer labs at Harvard", "Funding + institution filter"),
        ("young PI studying mouse models", "Career stage + organism filter"),
        ("RNA-seq and bioinformatics", "Technique-based query"),
        ("p53 mutations in cancer", "Gene-specific query"),
    ]

    for query, description in test_queries:
        print(f"\n{'─' * 80}")
        print(f"Query: {query}")
        print(f"Type: {description}")
        print(f"{'─' * 80}")

        start_time = time.time()
        results = engine.search(query=query, limit=3, explain=True)
        elapsed = (time.time() - start_time) * 1000

        print(f"\n✅ Found {results.total_count} results in {elapsed:.1f}ms")

        if results.query_interpretation:
            print(f"   Intent: {results.query_interpretation.get('detected_intent', 'N/A')}")
            entities = results.query_interpretation.get('extracted_entities', {})
            if entities:
                print(f"   Entities: {sum(len(v) for v in entities.values())} extracted")

        print(f"\n   Top 3 Results:")
        for i, result in enumerate(results.results[:3], 1):
            metadata = result.get('metadata', {})
            print(f"   {i}. {metadata.get('name', 'Unknown')} ({result.get('final_score', result.get('score', 0)):.3f})")

    print(f"\n\n{'=' * 80}")
    print("✅ All predefined query tests completed!")
    print(f"{'=' * 80}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Manual test script for semantic search')
    parser.add_argument('--mode', choices=['interactive', 'predefined', 'both'],
                       default='interactive',
                       help='Test mode')

    args = parser.parse_args()

    try:
        if args.mode in ['predefined', 'both']:
            test_predefined_queries()

        if args.mode in ['interactive', 'both']:
            test_search_interactive()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
