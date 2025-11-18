"""
Hierarchical Research Topic Classification
"""
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")


class ResearchTopicClassifier:
    """
    Hierarchical research area classification for biology
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize classifier

        Args:
            config_path: Path to configuration
        """
        self.mesh_tree = self._load_mesh_hierarchy()
        self.custom_taxonomy = self._load_custom_taxonomy()
        self.classifier_model = None

        # Load classification model if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.classifier_model = pipeline(
                    "text-classification",
                    model="allenai/scibert_scivocab_uncased",
                    top_k=None
                )
                logger.info("Loaded classification model")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}")

        logger.info("ResearchTopicClassifier initialized")

    def _load_mesh_hierarchy(self) -> Dict:
        """Load MeSH tree structure"""
        # Simplified MeSH hierarchy
        # In production, load from actual MeSH files
        return {
            'Anatomy': ['A01', 'A02'],
            'Organisms': ['B01', 'B02', 'B03'],
            'Diseases': ['C01', 'C02', 'C03'],
            'Chemicals and Drugs': ['D01', 'D02', 'D03'],
            'Analytical Techniques': ['E01', 'E02', 'E05'],
            'Psychiatry and Psychology': ['F01', 'F02'],
            'Phenomena and Processes': ['G01', 'G02', 'G03'],
            'Disciplines and Occupations': ['H01', 'H02'],
            'Anthropology': ['I01'],
            'Technology': ['J01'],
            'Humanities': ['K01'],
            'Information Science': ['L01'],
            'Named Groups': ['M01'],
            'Health Care': ['N01', 'N02'],
            'Publication Characteristics': ['V01'],
            'Geographicals': ['Z01']
        }

    def _load_custom_taxonomy(self) -> Dict:
        """Load custom biology research taxonomy"""
        return {
            'Molecular Biology': {
                'keywords': ['gene', 'DNA', 'RNA', 'protein', 'transcription', 'translation'],
                'subtopics': [
                    'Gene regulation',
                    'Protein synthesis',
                    'DNA replication',
                    'RNA processing',
                    'Epigenetics'
                ]
            },
            'Cell Biology': {
                'keywords': ['cell', 'membrane', 'organelle', 'cytoplasm', 'nucleus'],
                'subtopics': [
                    'Cell signaling',
                    'Cell cycle',
                    'Apoptosis',
                    'Cell differentiation',
                    'Cell division'
                ]
            },
            'Genetics': {
                'keywords': ['genome', 'mutation', 'inheritance', 'genetic', 'chromosome'],
                'subtopics': [
                    'Population genetics',
                    'Molecular genetics',
                    'Human genetics',
                    'Medical genetics',
                    'Evolutionary genetics'
                ]
            },
            'Neuroscience': {
                'keywords': ['neuron', 'brain', 'synapse', 'neural', 'cognitive'],
                'subtopics': [
                    'Neurodevelopment',
                    'Neurodegeneration',
                    'Synaptic plasticity',
                    'Neural circuits',
                    'Behavioral neuroscience'
                ]
            },
            'Immunology': {
                'keywords': ['immune', 'antibody', 'antigen', 'T cell', 'B cell'],
                'subtopics': [
                    'Innate immunity',
                    'Adaptive immunity',
                    'Autoimmunity',
                    'Immunotherapy',
                    'Vaccine development'
                ]
            },
            'Cancer Biology': {
                'keywords': ['cancer', 'tumor', 'oncogene', 'metastasis', 'carcinoma'],
                'subtopics': [
                    'Tumor microenvironment',
                    'Cancer genetics',
                    'Cancer immunotherapy',
                    'Tumor metabolism',
                    'Cancer stem cells'
                ]
            },
            'Developmental Biology': {
                'keywords': ['development', 'embryo', 'differentiation', 'morphogenesis'],
                'subtopics': [
                    'Embryonic development',
                    'Stem cell biology',
                    'Organogenesis',
                    'Pattern formation',
                    'Regeneration'
                ]
            },
            'Microbiology': {
                'keywords': ['bacteria', 'virus', 'microbe', 'pathogen', 'infection'],
                'subtopics': [
                    'Bacterial pathogenesis',
                    'Virology',
                    'Host-pathogen interactions',
                    'Antimicrobial resistance',
                    'Microbiome'
                ]
            },
            'Biochemistry': {
                'keywords': ['enzyme', 'metabolism', 'catalysis', 'biochemical pathway'],
                'subtopics': [
                    'Protein biochemistry',
                    'Metabolic pathways',
                    'Structural biology',
                    'Enzyme kinetics',
                    'Lipid biochemistry'
                ]
            },
            'Computational Biology': {
                'keywords': ['bioinformatics', 'genomics', 'computational', 'algorithm'],
                'subtopics': [
                    'Genomics',
                    'Transcriptomics',
                    'Proteomics',
                    'Systems biology',
                    'Machine learning in biology'
                ]
            }
        }

    def classify_research_area(
        self,
        text: str,
        threshold: float = 0.3,
        max_areas: int = 5
    ) -> Dict[str, any]:
        """
        Multi-label hierarchical classification

        Args:
            text: Research text to classify
            threshold: Minimum confidence threshold
            max_areas: Maximum number of areas to return

        Returns:
            Dictionary with classification results
        """
        # Keyword-based classification
        keyword_scores = self._keyword_based_classification(text)

        # Get top scoring areas
        sorted_areas = sorted(
            keyword_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary_area = sorted_areas[0][0] if sorted_areas else None
        secondary_areas = [
            area for area, score in sorted_areas[1:max_areas]
            if score >= threshold
        ]

        # Extract specific topics
        specific_topics = self._extract_specific_topics(text, primary_area)

        # Calculate confidence scores
        confidence_scores = dict(sorted_areas[:max_areas])

        return {
            'primary_area': primary_area,
            'secondary_areas': secondary_areas,
            'specific_topics': specific_topics,
            'confidence_scores': confidence_scores,
            'is_interdisciplinary': len(secondary_areas) >= 2
        }

    def _keyword_based_classification(self, text: str) -> Dict[str, float]:
        """Classify based on keyword matching"""
        text_lower = text.lower()
        scores = defaultdict(float)

        for area, info in self.custom_taxonomy.items():
            keywords = info['keywords']

            # Count keyword occurrences
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword) + r'\w*\b'
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                scores[area] += matches

        # Normalize scores
        max_score = max(scores.values()) if scores else 1
        normalized_scores = {
            area: score / max_score
            for area, score in scores.items()
        }

        return normalized_scores

    def _extract_specific_topics(
        self,
        text: str,
        primary_area: Optional[str]
    ) -> List[str]:
        """Extract specific topics within primary area"""
        if not primary_area or primary_area not in self.custom_taxonomy:
            return []

        specific_topics = []
        subtopics = self.custom_taxonomy[primary_area]['subtopics']

        text_lower = text.lower()

        for subtopic in subtopics:
            # Simple presence check
            if any(word.lower() in text_lower for word in subtopic.split()):
                specific_topics.append(subtopic)

        return specific_topics[:5]  # Return top 5

    def identify_interdisciplinary_research(
        self,
        profile: Dict
    ) -> Dict[str, any]:
        """
        Detect cross-disciplinary work

        Args:
            profile: Faculty/researcher profile

        Returns:
            Interdisciplinary analysis
        """
        # Combine all research text
        research_texts = []

        if 'research_summary' in profile:
            research_texts.append(profile['research_summary'])

        if 'publications' in profile:
            for pub in profile['publications'][:10]:  # Recent publications
                text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
                research_texts.append(text)

        # Classify each text
        all_classifications = []
        for text in research_texts:
            if text.strip():
                classification = self.classify_research_area(text)
                all_classifications.append(classification)

        # Aggregate areas
        all_areas = set()
        for classification in all_classifications:
            if classification['primary_area']:
                all_areas.add(classification['primary_area'])
            all_areas.update(classification['secondary_areas'])

        # Determine field combinations
        field_combinations = list(all_areas)

        # Calculate interdisciplinary score
        interdisciplinary_score = min(len(all_areas) / 3.0, 1.0)

        return {
            'is_interdisciplinary': len(all_areas) >= 2,
            'interdisciplinary_score': interdisciplinary_score,
            'field_combinations': field_combinations,
            'diversity': len(all_areas)
        }

    def classify_publications_over_time(
        self,
        publications: List[Dict]
    ) -> Dict[str, any]:
        """
        Track research area evolution over time

        Args:
            publications: List of publications with year and text

        Returns:
            Temporal classification analysis
        """
        # Sort by year
        sorted_pubs = sorted(publications, key=lambda x: x.get('year', 0))

        # Split into time periods
        mid_point = len(sorted_pubs) // 2

        early_pubs = sorted_pubs[:mid_point]
        recent_pubs = sorted_pubs[mid_point:]

        # Classify each period
        early_areas = defaultdict(int)
        recent_areas = defaultdict(int)

        for pub in early_pubs:
            text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
            classification = self.classify_research_area(text)
            if classification['primary_area']:
                early_areas[classification['primary_area']] += 1

        for pub in recent_pubs:
            text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
            classification = self.classify_research_area(text)
            if classification['primary_area']:
                recent_areas[classification['primary_area']] += 1

        # Analyze changes
        emerging_areas = [
            area for area in recent_areas
            if area not in early_areas
        ]

        declining_areas = [
            area for area in early_areas
            if area not in recent_areas
        ]

        stable_areas = [
            area for area in recent_areas
            if area in early_areas
        ]

        return {
            'emerging_areas': emerging_areas,
            'declining_areas': declining_areas,
            'stable_areas': stable_areas,
            'early_focus': dict(early_areas),
            'recent_focus': dict(recent_areas)
        }
