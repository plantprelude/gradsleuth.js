"""
Manual test script for matching functionality

This script demonstrates faculty-student matching with detailed explanations.
Run from bio-match-ml directory: python scripts/test_matching_manual.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from matching.multi_factor_scorer import MultiFactorMatcher
from matching.similarity_calculator import SimilarityCalculator
from search.explanation_generator import ExplanationGenerator
from unittest.mock import Mock
import numpy as np


def create_mock_similarity_calculator():
    """Create mock similarity calculator"""
    mock_calc = Mock(spec=SimilarityCalculator)

    def mock_similarity(student, faculty):
        # Generate realistic similarity scores
        return {
            'embedding_similarity': 0.82,
            'topic_overlap': 0.75,
            'technique_overlap': 0.85,
            'organism_overlap': 0.70,
            'keyword_overlap': 0.68
        }

    mock_calc.calculate_research_similarity = mock_similarity
    return mock_calc


def main():
    print("="*80)
    print("Faculty-Student Matching System - Manual Test")
    print("="*80)
    print()

    # Initialize components
    print("Initializing matcher...")
    similarity_calc = create_mock_similarity_calculator()
    matcher = MultiFactorMatcher(similarity_calculator=similarity_calc)
    explainer = ExplanationGenerator()

    print("Matcher initialized!\n")

    # Sample student profile
    student = {
        'research_interests': 'I am interested in using CRISPR to study gene regulation in cancer cells and developing new therapeutic approaches.',
        'topics': ['gene editing', 'cancer biology', 'gene regulation', 'therapeutics'],
        'techniques': ['CRISPR', 'RNA-seq', 'ChIP-seq', 'cell culture'],
        'organisms': ['human', 'mouse'],
        'career_goals': 'Academic research career focused on cancer therapeutics'
    }

    # Sample faculty profiles
    faculty_list = [
        {
            'id': 'fac1',
            'name': 'Dr. Jane Smith',
            'research_summary': 'CRISPR screens for cancer vulnerabilities and therapeutic target discovery',
            'topics': ['gene editing', 'cancer', 'functional genomics', 'drug discovery'],
            'techniques': ['CRISPR', 'RNA-seq', 'proteomics', 'high-throughput screening'],
            'organisms': ['human', 'mouse'],
            'h_index': 42,
            'publication_count': 95,
            'active_grants': 2,
            'total_funding': 1500000,
            'grants': [
                {
                    'active': True,
                    'amount': 1000000,
                    'end_date': '2028-01-01',
                    'year': 2023
                },
                {
                    'active': True,
                    'amount': 500000,
                    'end_date': '2026-12-31',
                    'year': 2021
                }
            ],
            'lab_size': 10,
            'accepting_students': True,
            'career_stage': 'associate_professor',
            'collaborations': 12,
            'last_publication_year': 2024,
            'recent_publications': 6,
            'publications': [
                {'year': 2024, 'title': 'CRISPR screens reveal cancer vulnerabilities'},
                {'year': 2023, 'title': 'Therapeutic targets in cancer'},
                {'year': 2023, 'title': 'Gene regulation in oncogenesis'}
            ]
        },
        {
            'id': 'fac2',
            'name': 'Dr. John Doe',
            'research_summary': 'Structural biology of membrane proteins and drug design',
            'topics': ['structural biology', 'biophysics', 'drug design'],
            'techniques': ['X-ray crystallography', 'cryo-EM', 'molecular dynamics'],
            'organisms': ['bacteria', 'yeast'],
            'h_index': 35,
            'publication_count': 70,
            'active_grants': 1,
            'total_funding': 800000,
            'grants': [
                {
                    'active': True,
                    'amount': 800000,
                    'end_date': '2027-06-30',
                    'year': 2022
                }
            ],
            'lab_size': 5,
            'accepting_students': True,
            'career_stage': 'assistant_professor',
            'collaborations': 8,
            'last_publication_year': 2024,
            'recent_publications': 4,
            'publications': [
                {'year': 2024, 'title': 'Membrane protein structures'},
                {'year': 2023, 'title': 'Cryo-EM methodology'}
            ]
        },
        {
            'id': 'fac3',
            'name': 'Dr. Maria Garcia',
            'research_summary': 'Cancer immunology and immunotherapy development',
            'topics': ['cancer', 'immunology', 'immunotherapy', 'T cells'],
            'techniques': ['flow cytometry', 'immunoassays', 'in vivo models'],
            'organisms': ['human', 'mouse'],
            'h_index': 38,
            'publication_count': 82,
            'active_grants': 3,
            'total_funding': 2000000,
            'grants': [
                {
                    'active': True,
                    'amount': 1200000,
                    'end_date': '2029-01-01',
                    'year': 2024
                },
                {
                    'active': True,
                    'amount': 800000,
                    'end_date': '2027-12-31',
                    'year': 2022
                }
            ],
            'lab_size': 12,
            'accepting_students': True,
            'career_stage': 'full_professor',
            'collaborations': 20,
            'last_publication_year': 2024,
            'recent_publications': 8,
            'publications': [
                {'year': 2024, 'title': 'CAR-T cell therapy for solid tumors'},
                {'year': 2024, 'title': 'Tumor microenvironment and immunity'},
                {'year': 2023, 'title': 'Novel immunotherapy approaches'}
            ]
        }
    ]

    print("Student Profile:")
    print(f"  Interests: {student['research_interests'][:100]}...")
    print(f"  Techniques: {', '.join(student['techniques'])}")
    print(f"  Organisms: {', '.join(student['organisms'])}")
    print(f"  Career Goals: {student['career_goals']}")
    print()

    print("="*80)
    print("Testing Faculty Matches")
    print("="*80)

    for faculty in faculty_list:
        print()
        print("="*80)
        print(f"Faculty: {faculty['name']}")
        print("="*80)
        print(f"Research: {faculty['research_summary']}")
        print(f"H-index: {faculty['h_index']}, Publications: {faculty['publication_count']}")
        print(f"Active Grants: {faculty['active_grants']}, Total Funding: ${faculty['total_funding']:,}")
        print(f"Lab Size: {faculty['lab_size']}, Career Stage: {faculty['career_stage']}")
        print()

        try:
            # Calculate match
            match = matcher.calculate_match_score(student, faculty, explain=True)

            print(f"MATCH RESULTS:")
            print(f"Overall Score: {match.overall_score:.1%} - {match.recommendation.upper().replace('_', ' ')}")
            print(f"Confidence: {match.confidence:.1%}")
            print()

            print("Component Scores:")
            for component, score in sorted(match.component_scores.items(), key=lambda x: x[1], reverse=True):
                component_name = component.replace('_', ' ').title()
                bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
                print(f"  {component_name:.<30} {score:.1%} [{bar}]")

            if match.explanation:
                print(f"\nExplanation:")
                print(f"  {match.explanation}")

            if match.strengths:
                print(f"\nKey Strengths:")
                for strength in match.strengths:
                    print(f"  ✓ {strength}")

            if match.considerations:
                print(f"\nConsiderations:")
                for consideration in match.considerations:
                    print(f"  ⚠ {consideration}")

            # Generate detailed explanation
            print(f"\nDetailed Match Analysis:")
            print("-" * 80)
            detailed = explainer.explain_match_score(
                student,
                faculty,
                match,
                detail_level='standard'
            )
            print(detailed)

            # Action items
            actions = explainer.generate_recommendation_action_items(
                match.overall_score,
                faculty
            )
            if actions:
                print(f"\nRecommended Actions:")
                for action in actions:
                    print(f"  • {action}")

        except Exception as e:
            print(f"Error calculating match: {e}")
            import traceback
            traceback.print_exc()

        print()

    # Summary comparison
    print("="*80)
    print("Match Summary - Ranked by Overall Score")
    print("="*80)

    try:
        matches = []
        for faculty in faculty_list:
            match = matcher.calculate_match_score(student, faculty, explain=False)
            matches.append({
                'faculty': faculty,
                'match': match
            })

        # Sort by score
        matches.sort(key=lambda x: x['match'].overall_score, reverse=True)

        print()
        for i, item in enumerate(matches, 1):
            faculty = item['faculty']
            match = item['match']
            print(f"{i}. {faculty['name']}")
            print(f"   Score: {match.overall_score:.1%} | Recommendation: {match.recommendation.replace('_', ' ').title()}")
            print(f"   Top Components: Research={match.component_scores['research_alignment']:.1%}, "
                  f"Funding={match.component_scores['funding_stability']:.1%}, "
                  f"Techniques={match.component_scores['technique_match']:.1%}")
            print()

    except Exception as e:
        print(f"Error generating summary: {e}")

    print("="*80)
    print("Manual test complete!")
    print("="*80)


if __name__ == '__main__':
    main()
