"""
Manual test script for research trajectory analysis

This script demonstrates trajectory analysis from publication history.
Run from bio-match-ml directory: python scripts/test_trajectory_manual.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from matching.research_trajectory_analyzer import ResearchTrajectoryAnalyzer


def main():
    print("="*80)
    print("Research Trajectory Analyzer - Manual Test")
    print("="*80)
    print()

    # Create analyzer
    print("Initializing trajectory analyzer...")
    analyzer = ResearchTrajectoryAnalyzer()
    print("Analyzer initialized!\n")

    # Sample publication history showing research evolution
    sample_pubs = [
        # Early career - PCR methods
        {
            'year': 2010,
            'title': 'Novel PCR techniques for amplification',
            'abstract': 'We developed new PCR methods for genomic DNA amplification',
            'keywords': ['PCR', 'genetics', 'methodology']
        },
        {
            'year': 2011,
            'title': 'PCR optimization for difficult templates',
            'abstract': 'Optimizing polymerase chain reaction conditions',
            'keywords': ['PCR', 'genomics', 'optimization']
        },
        {
            'year': 2012,
            'title': 'High-throughput PCR screening',
            'abstract': 'PCR screening methods for large scale studies',
            'keywords': ['PCR', 'high-throughput', 'screening']
        },

        # Transition to CRISPR
        {
            'year': 2014,
            'title': 'CRISPR-Cas9 system for genome editing',
            'abstract': 'Applying CRISPR-Cas9 for targeted genome editing in mammalian cells',
            'keywords': ['CRISPR', 'genome editing', 'gene editing']
        },
        {
            'year': 2015,
            'title': 'CRISPR screening for gene function',
            'abstract': 'Using CRISPR screens to identify gene function',
            'keywords': ['CRISPR', 'screening', 'functional genomics']
        },

        # Focus on CRISPR in cancer
        {
            'year': 2017,
            'title': 'CRISPR screens identify cancer vulnerabilities',
            'abstract': 'CRISPR screening reveals therapeutic targets in cancer',
            'keywords': ['CRISPR', 'cancer', 'screening', 'therapeutics']
        },
        {
            'year': 2018,
            'title': 'CRISPR-mediated cancer therapy development',
            'abstract': 'Developing CRISPR-based therapeutic approaches for cancer',
            'keywords': ['CRISPR', 'cancer', 'therapy', 'gene editing']
        },
        {
            'year': 2019,
            'title': 'Synthetic lethality screens in cancer',
            'abstract': 'Identifying synthetic lethal interactions using CRISPR',
            'keywords': ['CRISPR', 'cancer', 'synthetic lethality']
        },

        # Integration with single-cell methods
        {
            'year': 2020,
            'title': 'Single-cell CRISPR screens',
            'abstract': 'Combining single-cell RNA-seq with CRISPR screening',
            'keywords': ['CRISPR', 'single-cell', 'RNA-seq', 'cancer']
        },
        {
            'year': 2021,
            'title': 'Single-cell analysis of CRISPR perturbations',
            'abstract': 'Single-cell methods reveal heterogeneous responses to CRISPR',
            'keywords': ['single-cell', 'CRISPR', 'RNA-seq']
        },

        # Recent: spatial transcriptomics
        {
            'year': 2023,
            'title': 'Spatial CRISPR screens in tumor microenvironment',
            'abstract': 'Spatially resolved CRISPR screens reveal microenvironmental effects',
            'keywords': ['spatial', 'CRISPR', 'cancer', 'microenvironment']
        },
        {
            'year': 2024,
            'title': 'Multi-omics integration in CRISPR screens',
            'abstract': 'Integrating proteomics and transcriptomics with CRISPR',
            'keywords': ['CRISPR', 'multi-omics', 'proteomics', 'transcriptomics']
        },
        {
            'year': 2024,
            'title': 'Spatial transcriptomics of cancer evolution',
            'abstract': 'Tracking cancer evolution using spatial transcriptomics',
            'keywords': ['spatial transcriptomics', 'cancer', 'evolution']
        },
    ]

    print(f"Analyzing trajectory from {len(sample_pubs)} publications")
    print(f"Years: {min(p['year'] for p in sample_pubs)} - {max(p['year'] for p in sample_pubs)}")
    print()

    print("="*80)
    print("Publication Timeline")
    print("="*80)
    for pub in sample_pubs[:3]:
        print(f"{pub['year']}: {pub['title']}")
    print("...")
    for pub in sample_pubs[-3:]:
        print(f"{pub['year']}: {pub['title']}")
    print()

    try:
        # Analyze trajectory
        print("="*80)
        print("Analyzing Research Trajectory...")
        print("="*80)
        print()

        trajectory = analyzer.analyze_trajectory(sample_pubs)

        # Trending topics
        print("🔥 Trending Topics (Increasing in Recent Years):")
        if trajectory.trending_topics:
            for topic in trajectory.trending_topics[:5]:
                print(f"  • {topic}")
        else:
            print("  None identified")
        print()

        # Declining topics
        print("📉 Declining Topics (Decreasing in Recent Years):")
        if trajectory.declining_topics:
            for topic in trajectory.declining_topics[:5]:
                print(f"  • {topic}")
        else:
            print("  None identified")
        print()

        # Stable core
        print("🎯 Stable Core Topics (Consistent Throughout):")
        if trajectory.stable_core:
            for topic in trajectory.stable_core[:5]:
                print(f"  • {topic}")
        else:
            print("  None identified")
        print()

        # Innovation rate
        print(f"💡 Innovation Rate: {trajectory.innovation_rate:.2f}")
        if trajectory.innovation_rate > 0.5:
            print("  → High innovation: Frequently explores new topics")
        elif trajectory.innovation_rate > 0.3:
            print("  → Moderate innovation: Balances core focus with exploration")
        else:
            print("  → Low innovation: Focused on established topics")
        print()

        # Research phases
        print("📊 Research Career Phases:")
        if trajectory.research_phases:
            for phase in trajectory.research_phases:
                print(f"\n  Phase: {phase.phase_type}")
                print(f"  Years: {phase.years[0]} - {phase.years[1]}")
                print(f"  Primary Topics: {', '.join(phase.primary_topics[:3])}")
                print(f"  Characteristics: {phase.characteristics}")
        else:
            print("  None identified")
        print()

        # Pivot points
        print("🔄 Major Pivot Points (Significant Research Shifts):")
        if trajectory.pivot_points:
            for pivot in trajectory.pivot_points:
                print(f"\n  Year {pivot.year}: {pivot.description}")
                if pivot.old_focus:
                    print(f"    From: {', '.join(pivot.old_focus[:2])}")
                if pivot.new_focus:
                    print(f"    To: {', '.join(pivot.new_focus[:2])}")
        else:
            print("  None identified")
        print()

        # Predicted directions
        print("🔮 Predicted Future Research Directions:")
        if trajectory.predicted_directions:
            for i, prediction in enumerate(trajectory.predicted_directions, 1):
                print(f"\n  {i}. {prediction.topic}")
                print(f"     Confidence: {prediction.confidence:.0%}")
                print(f"     Rationale: {prediction.rationale}")
                print(f"     Timeframe: {prediction.timeframe}")
        else:
            print("  None predicted")
        print()

        # Test student alignment
        print("="*80)
        print("Student-Faculty Trajectory Alignment")
        print("="*80)
        print()

        # Sample student interests
        student_interests = {
            'research_interests': 'Interested in using CRISPR and spatial transcriptomics to study cancer',
            'topics': ['CRISPR', 'spatial transcriptomics', 'cancer'],
            'techniques': ['CRISPR', 'spatial transcriptomics', 'RNA-seq']
        }

        print("Student Interests:")
        print(f"  {student_interests['research_interests']}")
        print(f"  Topics: {', '.join(student_interests['topics'])}")
        print()

        alignment = analyzer.assess_student_alignment(student_interests, trajectory)

        print(f"Alignment Score: {alignment.alignment_score:.1%}")
        print(f"Alignment Type: {alignment.alignment_type.upper()}")
        print()

        if alignment.opportunities:
            print("✓ Opportunities:")
            for opp in alignment.opportunities:
                print(f"  • {opp}")
            print()

        if alignment.risks:
            print("⚠ Considerations:")
            for risk in alignment.risks:
                print(f"  • {risk}")
            print()

        # Interpretation
        print("Interpretation:")
        if alignment.alignment_type == 'trending':
            print("  Student interests align with faculty's current trending research.")
            print("  This is a strong match - joining faculty's growth areas.")
        elif alignment.alignment_type == 'future':
            print("  Student interests align with predicted future directions.")
            print("  Excellent match - be at the forefront of research evolution.")
        elif alignment.alignment_type == 'stable':
            print("  Student interests align with faculty's core expertise.")
            print("  Safe match - established methods and infrastructure.")
        elif alignment.alignment_type == 'declining':
            print("  Warning: Student interests align with declining research areas.")
            print("  Faculty may be moving away from these topics.")
        print()

    except Exception as e:
        print(f"Error during trajectory analysis: {e}")
        import traceback
        traceback.print_exc()

    print("="*80)
    print("Manual test complete!")
    print("="*80)


if __name__ == '__main__':
    main()
