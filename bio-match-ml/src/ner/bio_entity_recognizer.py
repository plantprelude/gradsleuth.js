"""
Biological Entity Recognition using multiple NER models
"""
import logging
from typing import List, Dict, Optional, Set, Tuple
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")


class BioEntityRecognizer:
    """
    Extract biological entities with high precision using ensemble methods
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Bio Entity Recognizer

        Args:
            config_path: Path to configuration file
        """
        self.models = {}
        self.entity_types = [
            'GENE', 'PROTEIN', 'DISEASE', 'CHEMICAL',
            'CELL_LINE', 'CELL_TYPE', 'SPECIES', 'MUTATION'
        ]

        # Load models
        self._load_models()

        # Load knowledge bases
        self.knowledge_bases = self._load_knowledge_bases()

        logger.info("BioEntityRecognizer initialized")

    def _load_models(self):
        """Load NER models"""
        # Load scispacy if available
        if SPACY_AVAILABLE:
            try:
                self.models['scispacy'] = spacy.load("en_ner_bc5cdr_md")
                logger.info("Loaded scispacy model")
            except Exception as e:
                logger.warning(f"Could not load scispacy: {e}")

        # Load BioBERT NER if available
        if TRANSFORMERS_AVAILABLE:
            try:
                self.models['biobert_ner'] = pipeline(
                    "ner",
                    model="dmis-lab/biobert-base-cased-v1.2",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded BioBERT NER model")
            except Exception as e:
                logger.warning(f"Could not load BioBERT NER: {e}")

    def _load_knowledge_bases(self) -> Dict:
        """Load knowledge bases for entity linking"""
        # This would load actual knowledge bases
        # For now, return empty structure
        return {
            'genes': {},
            'proteins': {},
            'diseases': {},
            'chemicals': {},
            'species': {}
        }

    def extract_all_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Comprehensive entity extraction

        Args:
            text: Input text

        Returns:
            Dictionary with entity types as keys
        """
        results = {
            'genes': [],
            'proteins': [],
            'organisms': [],
            'diseases': [],
            'chemicals': [],
            'cell_lines': [],
            'techniques': [],
            'anatomical_parts': []
        }

        # Use scispacy if available
        if 'scispacy' in self.models:
            scispacy_entities = self._extract_scispacy(text)
            results = self._merge_entities(results, scispacy_entities)

        # Use BioBERT NER if available
        if 'biobert_ner' in self.models:
            biobert_entities = self._extract_biobert(text)
            results = self._merge_entities(results, biobert_entities)

        # Use rule-based extraction for techniques
        techniques = self.extract_research_methods(text)
        results['techniques'] = techniques

        # Use pattern matching for organisms
        organisms = self.extract_model_organisms(text)
        results['organisms'] = organisms

        # Deduplicate and score
        results = self._deduplicate_entities(results)

        return results

    def _extract_scispacy(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using scispacy"""
        doc = self.models['scispacy'](text)

        entities = defaultdict(list)

        for ent in doc.ents:
            entity_dict = {
                'name': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': 0.8,  # scispacy doesn't provide confidence
                'source': 'scispacy'
            }

            # Map scispacy labels to our categories
            if ent.label_ == 'GENE':
                entities['genes'].append(entity_dict)
            elif ent.label_ == 'DISEASE':
                entities['diseases'].append(entity_dict)
            elif ent.label_ == 'CHEMICAL':
                entities['chemicals'].append(entity_dict)

        return dict(entities)

    def _extract_biobert(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities using BioBERT NER"""
        try:
            ner_results = self.models['biobert_ner'](text)
        except Exception as e:
            logger.error(f"BioBERT NER error: {e}")
            return {}

        entities = defaultdict(list)

        for ent in ner_results:
            entity_dict = {
                'name': ent['word'],
                'start': ent['start'],
                'end': ent['end'],
                'confidence': ent['score'],
                'source': 'biobert'
            }

            # Map entity types
            entity_group = ent.get('entity_group', '').upper()

            if 'GENE' in entity_group or 'PROTEIN' in entity_group:
                entities['genes'].append(entity_dict)
            elif 'DISEASE' in entity_group:
                entities['diseases'].append(entity_dict)
            elif 'CHEMICAL' in entity_group or 'DRUG' in entity_group:
                entities['chemicals'].append(entity_dict)
            elif 'CELL' in entity_group:
                entities['cell_lines'].append(entity_dict)

        return dict(entities)

    def extract_research_methods(self, text: str) -> List[Dict]:
        """
        Identify experimental techniques

        Args:
            text: Input text

        Returns:
            List of technique dictionaries
        """
        techniques_patterns = {
            # Molecular biology techniques
            'PCR': r'\b(PCR|polymerase\s+chain\s+reaction|qPCR|RT-PCR)\b',
            'Western blot': r'\b(western\s+blot|immunoblot|WB)\b',
            'CRISPR': r'\b(CRISPR|Cas9|CRISPR-Cas9|gene\s+editing)\b',
            'Flow cytometry': r'\b(flow\s+cytometry|FACS|fluorescence-activated)\b',
            'Immunofluorescence': r'\b(immunofluorescence|IF|immunostaining)\b',
            'Immunohistochemistry': r'\b(immunohistochemistry|IHC)\b',

            # Sequencing techniques
            'RNA-seq': r'\b(RNA-seq|RNA\s+sequencing|transcriptome\s+sequencing)\b',
            'ChIP-seq': r'\b(ChIP-seq|ChIP\s+sequencing|chromatin\s+immunoprecipitation)\b',
            'ATAC-seq': r'\b(ATAC-seq|ATAC\s+sequencing)\b',
            'Single-cell sequencing': r'\b(single-cell\s+sequencing|scRNA-seq|single\s+cell\s+RNA-seq)\b',
            'Whole genome sequencing': r'\b(WGS|whole\s+genome\s+sequencing)\b',

            # Microscopy
            'Confocal microscopy': r'\b(confocal\s+microscopy|confocal\s+imaging)\b',
            'Electron microscopy': r'\b(electron\s+microscopy|EM|TEM|SEM)\b',
            'Live-cell imaging': r'\b(live-cell\s+imaging|time-lapse\s+microscopy)\b',
            'Super-resolution microscopy': r'\b(super-resolution|STORM|PALM|SIM)\b',

            # Computational
            'Molecular dynamics': r'\b(molecular\s+dynamics|MD\s+simulation)\b',
            'Bioinformatics': r'\b(bioinformatics|computational\s+biology)\b',
            'Machine learning': r'\b(machine\s+learning|deep\s+learning|neural\s+network)\b',

            # Cell culture
            'Cell culture': r'\b(cell\s+culture|tissue\s+culture|primary\s+culture)\b',
            'Organoid': r'\b(organoid|3D\s+culture|spheroid)\b',

            # Proteomics
            'Mass spectrometry': r'\b(mass\s+spectrometry|MS|LC-MS|proteomics)\b',
            'Co-immunoprecipitation': r'\b(co-immunoprecipitation|Co-IP|pull-down)\b',

            # Other
            'In vivo imaging': r'\b(in\s+vivo\s+imaging|animal\s+imaging)\b',
            'Electrophysiology': r'\b(electrophysiology|patch\s+clamp|voltage\s+clamp)\b'
        }

        techniques = []
        text_lower = text.lower()

        for technique_name, pattern in techniques_patterns.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                techniques.append({
                    'name': technique_name,
                    'matched_text': match.group(),
                    'position': match.start(),
                    'category': self._categorize_technique(technique_name),
                    'confidence': 0.9
                })

        # Deduplicate
        seen = set()
        unique_techniques = []
        for tech in techniques:
            key = (tech['name'], tech['position'])
            if key not in seen:
                seen.add(key)
                unique_techniques.append(tech)

        return unique_techniques

    def _categorize_technique(self, technique_name: str) -> str:
        """Categorize technique into broader category"""
        wet_lab = [
            'PCR', 'Western blot', 'CRISPR', 'Flow cytometry',
            'Immunofluorescence', 'Immunohistochemistry', 'Cell culture',
            'Organoid', 'Co-immunoprecipitation'
        ]
        sequencing = [
            'RNA-seq', 'ChIP-seq', 'ATAC-seq', 'Single-cell sequencing',
            'Whole genome sequencing'
        ]
        microscopy = [
            'Confocal microscopy', 'Electron microscopy',
            'Live-cell imaging', 'Super-resolution microscopy'
        ]
        computational = [
            'Molecular dynamics', 'Bioinformatics', 'Machine learning'
        ]

        if technique_name in wet_lab:
            return 'wet_lab'
        elif technique_name in sequencing:
            return 'sequencing'
        elif technique_name in microscopy:
            return 'microscopy'
        elif technique_name in computational:
            return 'computational'
        else:
            return 'other'

    def extract_model_organisms(self, text: str) -> List[Dict]:
        """
        Identify and normalize organism mentions

        Args:
            text: Input text

        Returns:
            List of organism dictionaries
        """
        organism_patterns = {
            'Mouse': {
                'patterns': [r'\bmice\b', r'\bmouse\b', r'\bMus musculus\b'],
                'scientific_name': 'Mus musculus',
                'taxonomy_id': '10090'
            },
            'Human': {
                'patterns': [r'\bhuman\b', r'\bHomo sapiens\b', r'\bpatient'],
                'scientific_name': 'Homo sapiens',
                'taxonomy_id': '9606'
            },
            'Rat': {
                'patterns': [r'\brat\b', r'\brats\b', r'\bRattus norvegicus\b'],
                'scientific_name': 'Rattus norvegicus',
                'taxonomy_id': '10116'
            },
            'Zebrafish': {
                'patterns': [r'\bzebrafish\b', r'\bDanio rerio\b'],
                'scientific_name': 'Danio rerio',
                'taxonomy_id': '7955'
            },
            'Fruit fly': {
                'patterns': [r'\bDrosophila\b', r'\bfruit fly\b', r'\bDrosophila melanogaster\b'],
                'scientific_name': 'Drosophila melanogaster',
                'taxonomy_id': '7227'
            },
            'C. elegans': {
                'patterns': [r'\bC\.\s*elegans\b', r'\bCaenorhabditis elegans\b', r'\bnematode\b'],
                'scientific_name': 'Caenorhabditis elegans',
                'taxonomy_id': '6239'
            },
            'Yeast': {
                'patterns': [r'\byeast\b', r'\bS\.\s*cerevisiae\b', r'\bSaccharomyces cerevisiae\b'],
                'scientific_name': 'Saccharomyces cerevisiae',
                'taxonomy_id': '4932'
            },
            'E. coli': {
                'patterns': [r'\bE\.\s*coli\b', r'\bEscherichia coli\b'],
                'scientific_name': 'Escherichia coli',
                'taxonomy_id': '562'
            }
        }

        organisms = []
        text_lower = text.lower()

        for common_name, info in organism_patterns.items():
            for pattern in info['patterns']:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    organisms.append({
                        'common_name': common_name,
                        'scientific_name': info['scientific_name'],
                        'taxonomy_id': info['taxonomy_id'],
                        'matched_text': match.group(),
                        'position': match.start(),
                        'confidence': 0.95
                    })

        # Deduplicate
        seen = set()
        unique_organisms = []
        for org in organisms:
            key = (org['common_name'], org['position'])
            if key not in seen:
                seen.add(key)
                unique_organisms.append(org)

        return unique_organisms

    def link_entities_to_knowledge_bases(
        self,
        entities: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """
        Entity linking and disambiguation

        Args:
            entities: Extracted entities

        Returns:
            Entities with knowledge base IDs
        """
        # This would perform actual linking to databases
        # For now, just add placeholder IDs

        linked_entities = {}

        for entity_type, entity_list in entities.items():
            linked_list = []
            for entity in entity_list:
                entity_copy = entity.copy()

                # Add placeholder KB IDs
                if entity_type == 'genes':
                    entity_copy['entrez_id'] = None
                    entity_copy['ensembl_id'] = None
                elif entity_type == 'proteins':
                    entity_copy['uniprot_id'] = None
                elif entity_type == 'diseases':
                    entity_copy['mesh_id'] = None
                    entity_copy['icd10_code'] = None
                elif entity_type == 'chemicals':
                    entity_copy['pubchem_id'] = None
                    entity_copy['chembl_id'] = None

                linked_list.append(entity_copy)

            linked_entities[entity_type] = linked_list

        return linked_entities

    def extract_research_focus_evolution(
        self,
        publication_list: List[Dict]
    ) -> Dict[str, any]:
        """
        Track entity changes over time in publications

        Args:
            publication_list: List of publications with year and text

        Returns:
            Dictionary describing research evolution
        """
        # Sort by year
        sorted_pubs = sorted(publication_list, key=lambda x: x.get('year', 0))

        # Extract entities from each time period
        early_entities = defaultdict(set)
        recent_entities = defaultdict(set)

        mid_point = len(sorted_pubs) // 2

        for i, pub in enumerate(sorted_pubs):
            text = f"{pub.get('title', '')} {pub.get('abstract', '')}"
            entities = self.extract_all_entities(text)

            target = early_entities if i < mid_point else recent_entities

            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    target[entity_type].add(entity['name'].lower())

        # Analyze changes
        evolution = {}

        for entity_type in ['organisms', 'techniques', 'diseases']:
            early = early_entities.get(entity_type, set())
            recent = recent_entities.get(entity_type, set())

            evolution[entity_type] = {
                'new': list(recent - early),
                'stable': list(recent & early),
                'deprecated': list(early - recent)
            }

        return evolution

    def _merge_entities(
        self,
        base: Dict[str, List[Dict]],
        new: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Merge entity dictionaries"""
        for key, values in new.items():
            if key in base:
                base[key].extend(values)
            else:
                base[key] = values
        return base

    def _deduplicate_entities(
        self,
        entities: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        """Remove duplicate entities"""
        deduplicated = {}

        for entity_type, entity_list in entities.items():
            seen = {}

            for entity in entity_list:
                name_lower = entity['name'].lower()

                if name_lower in seen:
                    # Keep the one with higher confidence
                    if entity.get('confidence', 0) > seen[name_lower].get('confidence', 0):
                        seen[name_lower] = entity
                else:
                    seen[name_lower] = entity

            deduplicated[entity_type] = list(seen.values())

        return deduplicated
