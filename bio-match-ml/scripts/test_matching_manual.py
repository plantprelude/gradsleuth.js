#!/usr/bin/env python3
"""
Manual test script for matching functionality

This script allows interactive testing of the faculty-student matching system.
Run: python scripts/test_matching_manual.py

Features:
- Interactive matching with custom profiles
- Predefined test scenarios
- Detailed match explanations
- Component score breakdowns
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

    return {'embedder': embedder}


def get_sample_student_profiles() -> List[Dict]:
    """Get sample student profiles for testing"""
    return [
        {
            'id': 'student001',
            'name': 'Alex Johnson',
            'research_interests': 'CRISPR gene editing in cancer cells for therapeutic applications',
            'topics': ['gene editing', 'cancer biology', 'therapeutics'],
            'techniques': ['CRISPR-Cas9', 'cell culture', 'flow cytometry'],
            'organisms': ['human', 'mouse'],
            'career_goals': 'Academic research career, tenure-track position',
            'desired_skills': ['advanced gene editing', 'in vivo models', 'grant writing'],
            'preferred_lab_size': 'medium',
            'productivity_preference': 'high'
        },
        {
            'id': 'student002',
            'name': 'Maria Chen',
            'research_interests': 'Neural circuits and behavior using optogenetics',
            'topics': ['neuroscience', 'behavior', 'neural circuits'],
            'techniques': ['optogenetics', 'two-photon imaging', 'electrophysiology'],
            'organisms': ['mouse', 'zebrafish'],
            'career_goals': 'Industry career in neurotechnology',
            'desired_skills': ['imaging', 'data analysis', 'optogenetics'],
            'preferred_lab_size': 'small',
            'productivity_preference': 'moderate'
        },
        {
            'id': 'student003',
            'name': 'David Kim',
            'research_interests': 'Cancer genomics and precision medicine approaches',
            'topics': ['cancer genomics', 'precision medicine', 'bioinformatics'],
            'techniques': ['RNA-seq', 'whole-genome sequencing', 'bioinformatics'],
            'organisms': ['human'],
            'career_goals': 'Translational research bridging academia and industry',
            'desired_skills': ['computational biology', 'clinical data analysis', 'machine learning'],
            'preferred_lab_size': 'medium',
            'productivity_preference': 'high'
        }
    ]


def get_sample_faculty_profiles() -> List[Dict]:
    """Get sample faculty profiles for testing"""
    return [
        {
            'id': 'fac001',
            'name': 'Dr. Alice Chen',
            'institution': 'MIT',
            'department': 'Biology',
            'research_summary': 'CRISPR gene editing in cancer cells, developing new therapeutic approaches',
            'topics': ['gene editing', 'cancer biology', 'therapeutics'],
            'techniques': ['CRISPR-Cas9', 'RNA-seq', 'flow cytometry', 'single-cell sequencing'],
            'organisms': ['human', 'mouse'],
            'h_index': 45,
            'publication_count': 120,
            'citation_count': 5800,
            'grants': [
                {'active': True, 'amount': 1500000, 'end_date': '2028-12-31', 'title': 'R01: CRISPR therapy'},
                {'active': True, 'amount': 1000000, 'end_date': '2027-06-30', 'title': 'NSF: Gene regulation'}
            ],
            'active_grants': [
                {'amount': 1500000, 'end_date': '2028-12-31'},
                {'amount': 1000000, 'end_date': '2027-06-30'}
            ],
            'total_funding': 2500000,
            'has_active_funding': True,
            'lab_size': 12,
            'accepting_students': True,
            'lab_members': ['postdoc1', 'postdoc2', 'phd1', 'phd2', 'phd3', 'phd4'],
            'career_stage': 'mid_career',
            'collaboration_count': 45,
            'training_focus': 'academic research careers',
            'recent_publications': [
                {'publication_date': '2024-11-01', 'title': 'CRISPR screens in cancer', 'citations': 15},
                {'publication_date': '2024-09-15', 'title': 'Therapeutic gene editing', 'citations': 8}
            ]
        },
        {
            'id': 'fac002',
            'name': 'Dr. Maria Garcia',
            'institution': 'Johns Hopkins',
            'department': 'Neuroscience',
            'research_summary': 'Neural circuits and behavior using optogenetics and imaging',
            'topics': ['neuroscience', 'behavior', 'neural circuits', 'optogenetics'],
            'techniques': ['optogenetics', 'two-photon imaging', 'electrophysiology', 'behavioral assays'],
            'organisms': ['mouse', 'zebrafish'],
            'h_index': 24,
            'publication_count': 48,
            'citation_count': 1800,
            'grants': [],
            'active_grants': [],
            'total_funding': 400000,
            'has_active_funding': False,
            'lab_size': 4,
            'accepting_students': False,
            'lab_members': ['postdoc1', 'phd1', 'phd2'],
            'career_stage': 'early_career',
            'collaboration_count': 15,
            'training_focus': 'diverse career paths',
            'recent_publications': [
                {'publication_date': '2023-12-15', 'title': 'Neural circuits of learning', 'citations': 20}
            ]
        },
        {
            'id': 'fac003',
            'name': 'Dr. Robert Martinez',
            'institution': 'Harvard',
            'department': 'Genetics',
            'research_summary': 'Cancer genomics and precision medicine using next-generation sequencing',
            'topics': ['cancer genomics', 'precision medicine', 'computational biology'],
            'techniques': ['RNA-seq', 'ChIP-seq', 'whole-genome sequencing', 'bioinformatics'],
            'organisms': ['human'],
            'h_index': 38,
            'publication_count': 95,
            'citation_count': 4200,
            'grants': [
                {'active': True, 'amount': 1800000, 'end_date': '2027-12-31', 'title': 'R01: Cancer mutations'}
            ],
            'active_grants': [
                {'amount': 1800000, 'end_date': '2027-12-31'}
            ],
            'total_funding': 1800000,
            'has_active_funding': True,
            'lab_size': 8,
            'accepting_students': True,
            'lab_members': ['postdoc1', 'postdoc2', 'phd1', 'phd2', 'phd3'],
            'career_stage': 'mid_career',
            'collaboration_count': 35,
            'training_focus': 'translational research careers',
            'recent_publications': [
                {'publication_date': '2024-10-20', 'title': 'Genomic drivers of cancer', 'citations': 22},
                {'publication_date': '2024-08-05', 'title': 'Precision medicine approaches', 'citations': 12}
            ]
        }
    ]


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'=' * 80}")
    print(f"  {text}")
    print(f"{'=' * 80}\n")


def print_match_result(student: Dict, faculty: Dict, match_score, rank: int):
    """Print formatted match result"""
    print(f"\n{rank}. Match with {faculty['name']} at {faculty['institution']}")
    print(f"   {'─' * 76}")
    print(f"   Overall Score: {match_score.overall_score:.3f} ({match_score.recommendation.upper()})")
    print(f"   Confidence: {match_score.confidence:.2f}")

    print(f"\n   📊 Component Scores:")
    for component, score in match_score.component_scores.items():
        bar_length = int(score * 20)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        print(f"      {component:.<30} {bar} {score:.3f}")

    print(f"\n   ✅ Strengths:")
    for strength in match_score.strengths:
        print(f"      • {strength}")

    if match_score.considerations:
        print(f"\n   ⚠️  Considerations:")
        for consideration in match_score.considerations:
            print(f"      • {consideration}")

    print(f"\n   💡 Explanation:")
    print(f"      {match_score.explanation}")


def print_student_profile(student: Dict):
    """Print student profile"""
    print(f"\n👤 Student Profile: {student.get('name', 'N/A')}")
    print(f"   {'─' * 76}")
    print(f"   Research Interests: {student['research_interests']}")
    print(f"   Topics: {', '.join(student.get('topics', []))}")
    print(f"   Techniques: {', '.join(student.get('techniques', []))}")
    print(f"   Organisms: {', '.join(student.get('organisms', []))}")
    print(f"   Career Goals: {student.get('career_goals', 'N/A')}")


def print_faculty_profile(faculty: Dict):
    """Print faculty profile"""
    print(f"\n🎓 Faculty Profile: {faculty['name']} - {faculty['institution']}")
    print(f"   {'─' * 76}")
    print(f"   Department: {faculty['department']}")
    print(f"   Research: {faculty['research_summary']}")
    print(f"   Topics: {', '.join(faculty.get('topics', []))}")
    print(f"   Techniques: {', '.join(faculty.get('techniques', []))}")
    print(f"   H-Index: {faculty.get('h_index')} | Publications: {faculty.get('publication_count')}")
    print(f"   Funding: ${faculty.get('total_funding', 0):,} | Active: {faculty.get('has_active_funding')}")
    print(f"   Lab Size: {faculty.get('lab_size')} | Accepting: {faculty.get('accepting_students')}")


def test_matching_interactive():
    """Interactive matching testing"""
    from src.matching.multi_factor_scorer import MultiFactorMatcher
    from src.matching.similarity_calculator import SimilarityCalculator

    print_header("Faculty-Student Matching - Interactive Testing")

    # Create components
    components = create_mock_components()
    similarity_calc = SimilarityCalculator(embedding_generator=components['embedder'])
    matcher = MultiFactorMatcher(similarity_calculator=similarity_calc)

    # Load sample profiles
    students = get_sample_student_profiles()
    faculty = get_sample_faculty_profiles()

    print("Welcome to the Matching Test Console!")
    print("\nAvailable Commands:")
    print("  1-3   - Select a student profile")
    print("  list  - List all profiles")
    print("  match - Run matching for selected student")
    print("  all   - Run matching for all students")
    print("  quit  - Exit")

    selected_student = None

    while True:
        print("\n" + "-" * 80)

        if selected_student:
            print(f"Selected: {selected_student['name']}")

        command = input("\n🎯 Enter command: ").strip().lower()

        if not command:
            continue

        if command == 'quit':
            print("\nGoodbye!")
            break

        if command == 'list':
            print("\n📋 Available Student Profiles:")
            for i, student in enumerate(students, 1):
                print(f"\n{i}. {student['name']}")
                print(f"   Interests: {student['research_interests'][:60]}...")
                print(f"   Career: {student['career_goals']}")

            print("\n📋 Available Faculty Profiles:")
            for i, fac in enumerate(faculty, 1):
                print(f"\n{i}. {fac['name']} - {fac['institution']}")
                print(f"   Research: {fac['research_summary'][:60]}...")
                print(f"   Accepting: {fac['accepting_students']}")
            continue

        if command in ['1', '2', '3']:
            idx = int(command) - 1
            if 0 <= idx < len(students):
                selected_student = students[idx]
                print_student_profile(selected_student)
            else:
                print("Invalid student number")
            continue

        if command == 'match':
            if not selected_student:
                print("❌ Please select a student first (use commands 1-3)")
                continue

            print_header(f"Matching {selected_student['name']} with Faculty")

            matches = []
            for fac in faculty:
                start_time = time.time()
                match_score = matcher.calculate_match_score(
                    student_profile=selected_student,
                    faculty_profile=fac,
                    explain=True
                )
                elapsed = (time.time() - start_time) * 1000

                matches.append({
                    'faculty': fac,
                    'match_score': match_score,
                    'time_ms': elapsed
                })

            # Sort by match score
            matches.sort(key=lambda x: x['match_score'].overall_score, reverse=True)

            # Display matches
            for i, match in enumerate(matches, 1):
                print_match_result(
                    selected_student,
                    match['faculty'],
                    match['match_score'],
                    i
                )
                print(f"\n   ⏱️  Calculation Time: {match['time_ms']:.1f}ms")

            # Summary statistics
            scores = [m['match_score'].overall_score for m in matches]
            print(f"\n📈 Summary Statistics:")
            print(f"   Highest Score: {max(scores):.3f}")
            print(f"   Average Score: {sum(scores)/len(scores):.3f}")
            print(f"   Lowest Score: {min(scores):.3f}")

            continue

        if command == 'all':
            print_header("Matching All Students with All Faculty")

            for student in students:
                print(f"\n{'═' * 80}")
                print_student_profile(student)
                print(f"{'═' * 80}")

                matches = []
                for fac in faculty:
                    match_score = matcher.calculate_match_score(
                        student_profile=student,
                        faculty_profile=fac,
                        explain=True
                    )
                    matches.append({'faculty': fac, 'match_score': match_score})

                # Sort and show top 2
                matches.sort(key=lambda x: x['match_score'].overall_score, reverse=True)

                print(f"\n   Top 2 Matches:")
                for i, match in enumerate(matches[:2], 1):
                    fac = match['faculty']
                    score = match['match_score']
                    print(f"\n   {i}. {fac['name']} ({score.overall_score:.3f} - {score.recommendation})")
                    print(f"      {score.explanation[:100]}...")

            continue

        print(f"Unknown command: {command}")


def test_predefined_scenarios():
    """Test predefined matching scenarios"""
    from src.matching.multi_factor_scorer import MultiFactorMatcher
    from src.matching.similarity_calculator import SimilarityCalculator

    print_header("Faculty-Student Matching - Predefined Scenarios")

    # Create components
    components = create_mock_components()
    similarity_calc = SimilarityCalculator(embedding_generator=components['embedder'])
    matcher = MultiFactorMatcher(similarity_calculator=similarity_calc)

    # Load profiles
    students = get_sample_student_profiles()
    faculty = get_sample_faculty_profiles()

    # Test scenarios
    scenarios = [
        (students[0], faculty[0], "Perfect alignment: CRISPR cancer researcher"),
        (students[1], faculty[1], "Good match but funding concern"),
        (students[2], faculty[2], "Excellent for computational genomics"),
    ]

    for student, fac, description in scenarios:
        print(f"\n{'─' * 80}")
        print(f"Scenario: {description}")
        print(f"Student: {student['name']}")
        print(f"Faculty: {fac['name']}")
        print(f"{'─' * 80}")

        start_time = time.time()
        match_score = matcher.calculate_match_score(
            student_profile=student,
            faculty_profile=fac,
            explain=True
        )
        elapsed = (time.time() - start_time) * 1000

        print(f"\n✅ Match Score: {match_score.overall_score:.3f} ({match_score.recommendation})")
        print(f"   Confidence: {match_score.confidence:.2f}")
        print(f"   Time: {elapsed:.1f}ms")

        print(f"\n   Top 3 Components:")
        sorted_components = sorted(
            match_score.component_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for component, score in sorted_components[:3]:
            print(f"      • {component}: {score:.3f}")

        print(f"\n   Key Strength: {match_score.strengths[0] if match_score.strengths else 'N/A'}")
        if match_score.considerations:
            print(f"   Main Consideration: {match_score.considerations[0]}")

    print(f"\n\n{'=' * 80}")
    print("✅ All predefined scenarios completed!")
    print(f"{'=' * 80}\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Manual test script for faculty-student matching')
    parser.add_argument('--mode', choices=['interactive', 'predefined', 'both'],
                       default='interactive',
                       help='Test mode')

    args = parser.parse_args()

    try:
        if args.mode in ['predefined', 'both']:
            test_predefined_scenarios()

        if args.mode in ['interactive', 'both']:
            test_matching_interactive()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
