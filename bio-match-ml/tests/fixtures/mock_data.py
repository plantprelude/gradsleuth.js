"""
Mock data fixtures for testing

This module provides consistent mock data for all tests:
- Sample faculty profiles
- Sample student profiles
- Sample publications
- Sample grants
- Mock embeddings and search results
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta


def get_mock_faculty_profiles() -> List[Dict[str, Any]]:
    """
    Get comprehensive mock faculty profiles for testing

    Returns:
        List of faculty profile dictionaries with complete metadata
    """
    return [
        {
            'id': 'fac001',
            'name': 'Dr. Alice Chen',
            'institution': 'MIT',
            'department': 'Biology',
            'email': 'alicechen@mit.edu',
            'research_summary': 'CRISPR gene editing in cancer cells, developing new therapeutic approaches for precision oncology',
            'research_interests': 'CRISPR gene editing, cancer therapeutics, precision medicine',
            'topics': ['gene editing', 'cancer biology', 'therapeutics', 'precision medicine'],
            'research_topics': ['gene editing', 'cancer biology', 'therapeutics'],
            'techniques': ['CRISPR-Cas9', 'RNA-seq', 'flow cytometry', 'single-cell sequencing', 'organoid culture'],
            'organisms': ['human', 'mouse'],
            'h_index': 45,
            'publication_count': 120,
            'citation_count': 5800,
            'total_funding': 2500000,
            'has_active_funding': True,
            'grants': [
                {
                    'id': 'grant001',
                    'active': True,
                    'amount': 1500000,
                    'start_date': '2023-01-01',
                    'end_date': '2028-12-31',
                    'title': 'R01: CRISPR-based cancer therapies',
                    'agency': 'NIH'
                },
                {
                    'id': 'grant002',
                    'active': True,
                    'amount': 1000000,
                    'start_date': '2022-07-01',
                    'end_date': '2027-06-30',
                    'title': 'NSF: Gene regulation mechanisms',
                    'agency': 'NSF'
                }
            ],
            'active_grants': [
                {'amount': 1500000, 'end_date': '2028-12-31'},
                {'amount': 1000000, 'end_date': '2027-06-30'}
            ],
            'lab_size': 12,
            'lab_members': ['postdoc1', 'postdoc2', 'postdoc3', 'phd1', 'phd2', 'phd3', 'phd4', 'tech1', 'tech2'],
            'accepting_students': True,
            'career_stage': 'mid_career',
            'years_since_phd': 15,
            'collaboration_count': 45,
            'training_focus': 'academic research careers',
            'recent_publications': [
                {
                    'id': 'pub001',
                    'title': 'CRISPR screens identify novel cancer vulnerabilities',
                    'publication_date': '2024-11-01',
                    'citations': 15,
                    'journal': 'Nature',
                    'impact_factor': 49.962
                },
                {
                    'id': 'pub002',
                    'title': 'Therapeutic gene editing for precision oncology',
                    'publication_date': '2024-09-15',
                    'citations': 8,
                    'journal': 'Cell',
                    'impact_factor': 38.637
                },
                {
                    'id': 'pub003',
                    'title': 'CRISPR-Cas9 delivery methods for cancer therapy',
                    'publication_date': '2024-06-20',
                    'citations': 12,
                    'journal': 'Science',
                    'impact_factor': 47.728
                }
            ],
            'website': 'https://chenlab.mit.edu',
            'lab_culture': 'collaborative, mentorship-focused',
            'diversity_statement': True
        },
        {
            'id': 'fac002',
            'name': 'Dr. Robert Martinez',
            'institution': 'Harvard',
            'department': 'Genetics',
            'email': 'rmartinez@harvard.edu',
            'research_summary': 'Cancer genomics and precision medicine using next-generation sequencing and computational approaches',
            'research_interests': 'cancer genomics, precision medicine, computational biology',
            'topics': ['cancer genomics', 'precision medicine', 'computational biology', 'bioinformatics'],
            'research_topics': ['cancer genomics', 'precision medicine', 'computational biology'],
            'techniques': ['RNA-seq', 'ChIP-seq', 'whole-genome sequencing', 'bioinformatics', 'machine learning'],
            'organisms': ['human'],
            'h_index': 38,
            'publication_count': 95,
            'citation_count': 4200,
            'total_funding': 1800000,
            'has_active_funding': True,
            'grants': [
                {
                    'id': 'grant003',
                    'active': True,
                    'amount': 1800000,
                    'start_date': '2022-01-01',
                    'end_date': '2027-12-31',
                    'title': 'R01: Genomic drivers of cancer progression',
                    'agency': 'NIH'
                }
            ],
            'active_grants': [
                {'amount': 1800000, 'end_date': '2027-12-31'}
            ],
            'lab_size': 8,
            'lab_members': ['postdoc1', 'postdoc2', 'phd1', 'phd2', 'phd3', 'analyst1'],
            'accepting_students': True,
            'career_stage': 'mid_career',
            'years_since_phd': 12,
            'collaboration_count': 35,
            'training_focus': 'translational research and industry careers',
            'recent_publications': [
                {
                    'id': 'pub004',
                    'title': 'Genomic drivers of metastatic cancer',
                    'publication_date': '2024-10-20',
                    'citations': 22,
                    'journal': 'Nature Genetics',
                    'impact_factor': 31.616
                },
                {
                    'id': 'pub005',
                    'title': 'Precision medicine approaches in oncology',
                    'publication_date': '2024-08-05',
                    'citations': 12,
                    'journal': 'Nature Medicine',
                    'impact_factor': 87.241
                }
            ],
            'website': 'https://martinezlab.harvard.edu',
            'lab_culture': 'interdisciplinary, computational focus',
            'diversity_statement': True
        },
        {
            'id': 'fac003',
            'name': 'Dr. Sarah Lee',
            'institution': 'Stanford',
            'department': 'Bioengineering',
            'email': 'sarahlee@stanford.edu',
            'research_summary': 'Synthetic biology and metabolic engineering for sustainable biofuel production',
            'research_interests': 'synthetic biology, metabolic engineering, biofuels',
            'topics': ['synthetic biology', 'metabolic engineering', 'biofuels', 'sustainability'],
            'research_topics': ['synthetic biology', 'metabolic engineering'],
            'techniques': ['CRISPR-Cas9', 'metabolic engineering', 'synthetic circuits', 'fermentation', 'proteomics'],
            'organisms': ['yeast', 'bacteria', 'E. coli'],
            'h_index': 32,
            'publication_count': 72,
            'citation_count': 3100,
            'total_funding': 1200000,
            'has_active_funding': True,
            'grants': [
                {
                    'id': 'grant004',
                    'active': True,
                    'amount': 1200000,
                    'start_date': '2021-01-01',
                    'end_date': '2026-12-31',
                    'title': 'DOE: Engineered microbes for biofuel production',
                    'agency': 'DOE'
                }
            ],
            'active_grants': [
                {'amount': 1200000, 'end_date': '2026-12-31'}
            ],
            'lab_size': 6,
            'lab_members': ['postdoc1', 'phd1', 'phd2', 'phd3', 'tech1'],
            'accepting_students': True,
            'career_stage': 'early_career',
            'years_since_phd': 7,
            'collaboration_count': 20,
            'training_focus': 'academic and industry careers',
            'recent_publications': [
                {
                    'id': 'pub006',
                    'title': 'Engineered yeast strains for enhanced biofuel production',
                    'publication_date': '2024-09-30',
                    'citations': 18,
                    'journal': 'Nature Biotechnology',
                    'impact_factor': 68.164
                },
                {
                    'id': 'pub007',
                    'title': 'Synthetic metabolic pathways for sustainable chemistry',
                    'publication_date': '2024-07-15',
                    'citations': 10,
                    'journal': 'PNAS',
                    'impact_factor': 11.205
                }
            ],
            'website': 'https://leelab.stanford.edu',
            'lab_culture': 'innovative, entrepreneurial',
            'diversity_statement': True
        },
        {
            'id': 'fac004',
            'name': 'Dr. James Wilson',
            'institution': 'UCSF',
            'department': 'Immunology',
            'email': 'jwilson@ucsf.edu',
            'research_summary': 'T cell biology and immunotherapy for cancer treatment',
            'research_interests': 'T cell biology, immunotherapy, CAR-T cells',
            'topics': ['immunology', 'cancer immunotherapy', 'T cells', 'CAR-T'],
            'research_topics': ['immunology', 'cancer immunotherapy'],
            'techniques': ['flow cytometry', 'cell culture', 'immunofluorescence', 'ELISA', 'T cell assays'],
            'organisms': ['human', 'mouse'],
            'h_index': 28,
            'publication_count': 65,
            'citation_count': 2400,
            'total_funding': 900000,
            'has_active_funding': True,
            'grants': [
                {
                    'id': 'grant005',
                    'active': True,
                    'amount': 900000,
                    'start_date': '2021-07-01',
                    'end_date': '2026-06-30',
                    'title': 'R21: CAR-T cell therapy optimization',
                    'agency': 'NIH'
                }
            ],
            'active_grants': [
                {'amount': 900000, 'end_date': '2026-06-30'}
            ],
            'lab_size': 5,
            'lab_members': ['postdoc1', 'phd1', 'phd2', 'tech1'],
            'accepting_students': True,
            'career_stage': 'early_career',
            'years_since_phd': 6,
            'collaboration_count': 18,
            'training_focus': 'translational and clinical research',
            'recent_publications': [
                {
                    'id': 'pub008',
                    'title': 'CAR-T cell engineering for solid tumors',
                    'publication_date': '2024-08-20',
                    'citations': 25,
                    'journal': 'Cancer Cell',
                    'impact_factor': 50.3
                },
                {
                    'id': 'pub009',
                    'title': 'T cell exhaustion mechanisms in cancer',
                    'publication_date': '2024-06-10',
                    'citations': 14,
                    'journal': 'Immunity',
                    'impact_factor': 43.474
                }
            ],
            'website': 'https://wilsonlab.ucsf.edu',
            'lab_culture': 'clinical focus, translational',
            'diversity_statement': True
        },
        {
            'id': 'fac005',
            'name': 'Dr. Maria Garcia',
            'institution': 'Johns Hopkins',
            'department': 'Neuroscience',
            'email': 'mgarcia@jhu.edu',
            'research_summary': 'Neural circuits and behavior using optogenetics and advanced imaging',
            'research_interests': 'neural circuits, behavior, optogenetics, imaging',
            'topics': ['neuroscience', 'behavior', 'neural circuits', 'optogenetics'],
            'research_topics': ['neuroscience', 'behavior', 'neural circuits'],
            'techniques': ['optogenetics', 'two-photon imaging', 'electrophysiology', 'behavioral assays', 'calcium imaging'],
            'organisms': ['mouse', 'zebrafish'],
            'h_index': 24,
            'publication_count': 48,
            'citation_count': 1800,
            'total_funding': 400000,
            'has_active_funding': False,
            'grants': [],
            'active_grants': [],
            'lab_size': 4,
            'lab_members': ['postdoc1', 'phd1', 'phd2'],
            'accepting_students': False,
            'career_stage': 'early_career',
            'years_since_phd': 5,
            'collaboration_count': 15,
            'training_focus': 'diverse career paths including industry',
            'recent_publications': [
                {
                    'id': 'pub010',
                    'title': 'Neural circuits underlying learning and memory',
                    'publication_date': '2023-12-15',
                    'citations': 20,
                    'journal': 'Neuron',
                    'impact_factor': 16.2
                },
                {
                    'id': 'pub011',
                    'title': 'Optogenetic control of behavior',
                    'publication_date': '2023-10-05',
                    'citations': 16,
                    'journal': 'Nature Neuroscience',
                    'impact_factor': 28.771
                }
            ],
            'website': 'https://garcialab.jhu.edu',
            'lab_culture': 'tight-knit, supportive',
            'diversity_statement': True
        }
    ]


def get_mock_student_profiles() -> List[Dict[str, Any]]:
    """
    Get comprehensive mock student profiles for testing

    Returns:
        List of student profile dictionaries
    """
    return [
        {
            'id': 'student001',
            'name': 'Alex Johnson',
            'email': 'alex.johnson@university.edu',
            'research_interests': 'CRISPR gene editing in cancer cells for therapeutic applications',
            'topics': ['gene editing', 'cancer biology', 'therapeutics', 'CRISPR'],
            'techniques': ['CRISPR-Cas9', 'cell culture', 'flow cytometry', 'molecular cloning'],
            'organisms': ['human', 'mouse'],
            'career_goals': 'Academic research career, tenure-track position at R1 university',
            'desired_skills': ['advanced gene editing', 'in vivo models', 'grant writing', 'mentorship'],
            'research_direction': 'Developing CRISPR-based cancer therapies',
            'preferred_lab_size': 'medium',
            'productivity_preference': 'high',
            'gpa': 3.85,
            'gre_scores': {'verbal': 165, 'quantitative': 168, 'analytical': 5.0},
            'research_experience_years': 2,
            'publications': 1,
            'awards': ['Dean\'s List', 'Summer Research Fellowship'],
            'programming_skills': ['Python', 'R'],
            'preferred_institutions': ['MIT', 'Harvard', 'Stanford'],
            'preferred_locations': ['Boston', 'Bay Area']
        },
        {
            'id': 'student002',
            'name': 'Maria Chen',
            'email': 'maria.chen@university.edu',
            'research_interests': 'Neural circuits and behavior using optogenetics and imaging',
            'topics': ['neuroscience', 'behavior', 'neural circuits', 'imaging'],
            'techniques': ['optogenetics', 'two-photon imaging', 'electrophysiology', 'behavioral assays'],
            'organisms': ['mouse', 'zebrafish'],
            'career_goals': 'Industry career in neurotechnology and brain-computer interfaces',
            'desired_skills': ['advanced imaging', 'data analysis', 'optogenetics', 'signal processing'],
            'research_direction': 'Understanding neural basis of behavior',
            'preferred_lab_size': 'small',
            'productivity_preference': 'moderate',
            'gpa': 3.92,
            'gre_scores': {'verbal': 162, 'quantitative': 170, 'analytical': 5.5},
            'research_experience_years': 3,
            'publications': 2,
            'awards': ['NSF Graduate Fellowship', 'Best Poster Award'],
            'programming_skills': ['Python', 'MATLAB', 'Julia'],
            'preferred_institutions': ['Johns Hopkins', 'UCSF', 'Caltech'],
            'preferred_locations': ['Baltimore', 'San Francisco', 'Pasadena']
        },
        {
            'id': 'student003',
            'name': 'David Kim',
            'email': 'david.kim@university.edu',
            'research_interests': 'Cancer genomics and precision medicine using computational approaches',
            'topics': ['cancer genomics', 'precision medicine', 'bioinformatics', 'computational biology'],
            'techniques': ['RNA-seq', 'whole-genome sequencing', 'bioinformatics', 'machine learning', 'data analysis'],
            'organisms': ['human'],
            'career_goals': 'Translational research bridging academia and industry, computational biology',
            'desired_skills': ['advanced bioinformatics', 'machine learning', 'clinical data analysis', 'pipeline development'],
            'research_direction': 'Identifying genomic biomarkers for cancer treatment',
            'preferred_lab_size': 'medium',
            'productivity_preference': 'high',
            'gpa': 3.88,
            'gre_scores': {'verbal': 158, 'quantitative': 170, 'analytical': 4.5},
            'research_experience_years': 2.5,
            'publications': 3,
            'awards': ['Goldwater Scholar', 'Bioinformatics Hackathon Winner'],
            'programming_skills': ['Python', 'R', 'Bash', 'SQL'],
            'preferred_institutions': ['Harvard', 'Stanford', 'MIT'],
            'preferred_locations': ['Boston', 'Bay Area', 'flexible']
        },
        {
            'id': 'student004',
            'name': 'Emily Rodriguez',
            'email': 'emily.rodriguez@university.edu',
            'research_interests': 'Immunotherapy and T cell engineering for cancer treatment',
            'topics': ['immunology', 'cancer immunotherapy', 'T cells', 'cell engineering'],
            'techniques': ['flow cytometry', 'cell culture', 'CRISPR-Cas9', 'immunofluorescence'],
            'organisms': ['human', 'mouse'],
            'career_goals': 'Translational research and clinical applications',
            'desired_skills': ['CAR-T engineering', 'immunology techniques', 'translational research'],
            'research_direction': 'Improving CAR-T cell therapies for solid tumors',
            'preferred_lab_size': 'medium',
            'productivity_preference': 'moderate',
            'gpa': 3.80,
            'gre_scores': {'verbal': 160, 'quantitative': 165, 'analytical': 5.0},
            'research_experience_years': 1.5,
            'publications': 0,
            'awards': ['Undergraduate Research Award'],
            'programming_skills': ['Python', 'basic R'],
            'preferred_institutions': ['UCSF', 'Stanford', 'Johns Hopkins'],
            'preferred_locations': ['flexible']
        }
    ]


def get_mock_embeddings(dimension: int = 768, count: int = 1) -> np.ndarray:
    """
    Generate mock embeddings for testing

    Args:
        dimension: Embedding dimension (default 768 for PubMedBERT)
        count: Number of embeddings to generate

    Returns:
        Array of normalized random embeddings
    """
    np.random.seed(42)  # For reproducibility
    embeddings = np.random.randn(count, dimension)

    # Normalize to unit length (common for embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    if count == 1:
        return embeddings[0]
    return embeddings


def get_mock_search_results() -> List[Dict[str, Any]]:
    """
    Get mock search results with faculty profiles

    Returns:
        List of search result dictionaries with scores
    """
    faculty = get_mock_faculty_profiles()

    # Add semantic similarity scores
    results = []
    scores = [0.92, 0.88, 0.85, 0.82, 0.78]

    for fac, score in zip(faculty, scores):
        results.append({
            'id': fac['id'],
            'score': score,
            'metadata': fac
        })

    return results


def get_mock_query_analysis() -> Dict[str, Any]:
    """
    Get mock query analysis result

    Returns:
        QueryAnalysis-like dictionary
    """
    return {
        'original_query': 'CRISPR gene editing in cancer',
        'normalized_query': 'crispr gene editing cancer',
        'extracted_entities': {
            'techniques': ['CRISPR'],
            'topics': ['gene editing', 'cancer'],
            'genes': [],
            'organisms': []
        },
        'detected_intent': 'technique_based',
        'query_expansions': [
            'CRISPR gene editing in cancer',
            'Cas9 genome editing in cancer',
            'gene editing therapeutic applications'
        ],
        'num_expansions': 3,
        'implicit_filters': {},
        'boost_terms': ['CRISPR', 'gene editing']
    }


def get_mock_match_score() -> Dict[str, Any]:
    """
    Get mock match score result

    Returns:
        MatchScore-like dictionary
    """
    return {
        'overall_score': 0.85,
        'component_scores': {
            'research_alignment': 0.92,
            'funding_stability': 0.88,
            'productivity_match': 0.85,
            'technique_match': 0.90,
            'lab_environment': 0.82,
            'career_development': 0.75
        },
        'confidence': 0.88,
        'explanation': 'Excellent match with strong research alignment in CRISPR and cancer biology. Well-funded lab with robust research program.',
        'strengths': [
            'Very strong alignment in CRISPR gene editing techniques',
            'Excellent funding stability with multiple active grants',
            'High productivity lab with recent high-impact publications'
        ],
        'considerations': [
            'Large lab size may mean less individual attention',
            'Highly competitive environment'
        ],
        'recommendation': 'highly_recommended'
    }


def get_mock_publications() -> List[Dict[str, Any]]:
    """
    Get mock publications for testing

    Returns:
        List of publication dictionaries
    """
    return [
        {
            'id': 'pub001',
            'title': 'CRISPR screens identify novel cancer vulnerabilities',
            'abstract': 'We performed genome-wide CRISPR screens to identify genetic dependencies in cancer cells...',
            'authors': ['Chen, A.', 'Smith, J.', 'Johnson, K.'],
            'publication_date': '2024-11-01',
            'journal': 'Nature',
            'volume': 625,
            'pages': '123-130',
            'doi': '10.1038/nature12345',
            'pmid': '39012345',
            'citations': 15,
            'impact_factor': 49.962,
            'keywords': ['CRISPR', 'cancer', 'genetic screens', 'therapeutics']
        },
        {
            'id': 'pub002',
            'title': 'Therapeutic gene editing for precision oncology',
            'abstract': 'Gene editing approaches offer promising therapeutic strategies for cancer treatment...',
            'authors': ['Chen, A.', 'Williams, R.', 'Brown, T.'],
            'publication_date': '2024-09-15',
            'journal': 'Cell',
            'volume': 187,
            'pages': '456-468',
            'doi': '10.1016/j.cell.2024.09.012',
            'pmid': '39012346',
            'citations': 8,
            'impact_factor': 38.637,
            'keywords': ['gene editing', 'precision medicine', 'cancer therapy', 'CRISPR']
        }
    ]


def get_mock_grants() -> List[Dict[str, Any]]:
    """
    Get mock grants for testing

    Returns:
        List of grant dictionaries
    """
    return [
        {
            'id': 'grant001',
            'title': 'R01: CRISPR-based cancer therapies',
            'agency': 'NIH',
            'amount': 1500000,
            'start_date': '2023-01-01',
            'end_date': '2028-12-31',
            'active': True,
            'pi': 'Dr. Alice Chen',
            'institution': 'MIT',
            'abstract': 'This project aims to develop novel CRISPR-based therapeutic strategies for cancer treatment...'
        },
        {
            'id': 'grant002',
            'title': 'NSF: Gene regulation mechanisms',
            'agency': 'NSF',
            'amount': 1000000,
            'start_date': '2022-07-01',
            'end_date': '2027-06-30',
            'active': True,
            'pi': 'Dr. Alice Chen',
            'institution': 'MIT',
            'abstract': 'Understanding fundamental mechanisms of gene regulation in mammalian cells...'
        }
    ]


# Convenience functions for common test scenarios

def get_high_match_pair() -> tuple:
    """Get student and faculty profiles that should match highly"""
    students = get_mock_student_profiles()
    faculty = get_mock_faculty_profiles()
    return students[0], faculty[0]  # Alex + Dr. Chen (CRISPR cancer)


def get_medium_match_pair() -> tuple:
    """Get student and faculty profiles that should match moderately"""
    students = get_mock_student_profiles()
    faculty = get_mock_faculty_profiles()
    return students[1], faculty[4]  # Maria + Dr. Garcia (neuroscience, but funding issue)


def get_low_match_pair() -> tuple:
    """Get student and faculty profiles that should match poorly"""
    students = get_mock_student_profiles()
    faculty = get_mock_faculty_profiles()
    return students[2], faculty[4]  # David (genomics) + Dr. Garcia (neuroscience)


def get_computational_match_pair() -> tuple:
    """Get student and faculty profiles for computational research"""
    students = get_mock_student_profiles()
    faculty = get_mock_faculty_profiles()
    return students[2], faculty[1]  # David + Dr. Martinez (both computational)
