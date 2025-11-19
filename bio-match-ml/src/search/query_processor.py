"""
Query processing with intelligent expansion and entity extraction for biology domain
"""
import logging
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


# Biology-specific knowledge bases
CAREER_STAGE_KEYWORDS = {
    'young': 'assistant_professor',
    'new': 'assistant_professor',
    'junior': 'assistant_professor',
    'early career': 'assistant_professor',
    'established': 'associate_professor',
    'mid-career': 'associate_professor',
    'senior': 'full_professor',
    'emeritus': 'emeritus',
    'tenured': 'full_professor'
}

COMMON_GENE_ALIASES = {
    'p53': ['TP53', 'tumor protein 53', 'tumor protein p53'],
    'egfr': ['epidermal growth factor receptor', 'ERBB1'],
    'kras': ['KRAS', 'K-Ras', 'kirsten rat sarcoma'],
    'brca1': ['BRCA1', 'breast cancer 1'],
    'brca2': ['BRCA2', 'breast cancer 2'],
    'her2': ['ERBB2', 'HER2/neu'],
    'myc': ['c-Myc', 'MYC proto-oncogene'],
    'akt': ['AKT', 'protein kinase B', 'PKB'],
    'mtor': ['mTOR', 'mammalian target of rapamycin'],
    'nf-kb': ['NF-κB', 'nuclear factor kappa B']
}

TECHNIQUE_SYNONYMS = {
    'crispr': ['CRISPR-Cas9', 'gene editing', 'genome editing', 'CRISPR screening'],
    'rna-seq': ['RNA sequencing', 'transcriptomics', 'RNA-sequencing'],
    'pcr': ['polymerase chain reaction', 'qPCR', 'RT-PCR'],
    'western blot': ['immunoblot', 'western blotting', 'WB'],
    'flow cytometry': ['FACS', 'fluorescence-activated cell sorting'],
    'chip-seq': ['ChIP sequencing', 'chromatin immunoprecipitation sequencing'],
    'mass spectrometry': ['MS', 'mass spec', 'proteomics'],
    'confocal microscopy': ['confocal imaging', 'laser scanning microscopy'],
    'single-cell': ['single cell sequencing', 'scRNA-seq', 'single-cell analysis'],
    'cryoem': ['cryo-EM', 'cryo-electron microscopy', 'electron cryomicroscopy']
}

ORGANISM_SYNONYMS = {
    'mouse': ['Mus musculus', 'mice', 'murine'],
    'human': ['Homo sapiens', 'patient', 'clinical'],
    'rat': ['Rattus norvegicus', 'rats'],
    'zebrafish': ['Danio rerio', 'zebra fish'],
    'fly': ['Drosophila', 'Drosophila melanogaster', 'fruit fly'],
    'worm': ['C. elegans', 'Caenorhabditis elegans', 'nematode'],
    'yeast': ['S. cerevisiae', 'Saccharomyces cerevisiae']
}

FUNDING_KEYWORDS = {
    'well-funded': {'min_funding': 500000, 'has_active_funding': True},
    'funded': {'has_active_funding': True},
    'nih': {'funding_agency': 'NIH'},
    'nsf': {'funding_agency': 'NSF'},
    'r01': {'grant_type': 'R01'}
}

PRODUCTIVITY_KEYWORDS = {
    'prolific': {'min_publications': 50},
    'productive': {'min_publications': 30},
    'high impact': {'min_h_index': 30},
    'highly cited': {'min_citations': 5000}
}


@dataclass
class QueryAnalysis:
    """Data class for query analysis results"""
    original: str
    normalized: str
    entities: Dict[str, List]
    intent: str
    expansions: List[str]
    filters: Dict[str, Any]
    boost_terms: List[str]


class QueryProcessor:
    """
    Intelligent query understanding and expansion for biology domain
    """

    def __init__(self, entity_recognizer=None, topic_classifier=None):
        """
        Initialize with existing NER and classification models

        Args:
            entity_recognizer: BioEntityRecognizer instance
            topic_classifier: ResearchTopicClassifier instance
        """
        self.entity_recognizer = entity_recognizer
        self.topic_classifier = topic_classifier

        logger.info("QueryProcessor initialized")

    def process_query(self, raw_query: str) -> QueryAnalysis:
        """
        Complete query analysis pipeline

        Args:
            raw_query: User's search query

        Returns:
            QueryAnalysis object containing:
            - original: raw query
            - normalized: cleaned query
            - entities: extracted biological entities
            - intent: detected search intent
            - expansions: query variations
            - filters: implicit filters from query
            - boost_terms: important terms for boosting

        Example:
            Input: "young PI studying CRISPR in neurons at Harvard"
            Output: QueryAnalysis(
                original="young PI studying CRISPR in neurons at Harvard",
                normalized="principal investigator crispr neurons harvard",
                entities={'techniques': ['CRISPR'], 'cell_types': ['neurons']},
                intent='technique_based',
                expansions=["CRISPR cas9", "gene editing neurons", ...],
                filters={'institution': 'Harvard', 'career_stage': 'assistant_professor'},
                boost_terms=['CRISPR', 'neurons']
            )
        """
        if not raw_query or not raw_query.strip():
            logger.warning("Empty query received")
            return QueryAnalysis(
                original="",
                normalized="",
                entities={},
                intent='general',
                expansions=[],
                filters={},
                boost_terms=[]
            )

        logger.debug(f"Processing query: {raw_query}")

        # Step 1: Normalize query
        normalized = self.normalize_biology_terms(raw_query)

        # Step 2: Extract entities
        entities = {}
        if self.entity_recognizer:
            try:
                entities = self.entity_recognizer.extract_all_entities(raw_query)
                # Simplify entity structure for query analysis
                entities = {
                    k: [e['name'] if isinstance(e, dict) else e for e in v]
                    for k, v in entities.items() if v
                }
            except Exception as e:
                logger.error(f"Entity extraction error: {e}")
                entities = {}

        # Step 3: Extract implicit filters
        filters = self.extract_implicit_filters(raw_query)

        # Step 4: Detect query intent
        intent = self.detect_query_intent(raw_query, entities)

        # Step 5: Generate query expansions
        expansions = self.expand_query_scientifically(normalized, entities)

        # Step 6: Identify boost terms
        boost_terms = self._identify_boost_terms(raw_query, entities)

        result = QueryAnalysis(
            original=raw_query,
            normalized=normalized,
            entities=entities,
            intent=intent,
            expansions=expansions,
            filters=filters,
            boost_terms=boost_terms
        )

        logger.info(f"Query processed: intent={intent}, entities={len(entities)}, expansions={len(expansions)}")

        return result

    def expand_query_scientifically(self, query: str, entities: Dict) -> List[str]:
        """
        Generate scientific query expansions using domain knowledge

        Expansions include:
        - Synonyms from biology ontologies
        - Related techniques
        - Organism variants (e.g., "mouse" → "Mus musculus")
        - Protein/gene aliases (e.g., "p53" → "TP53", "tumor protein 53")

        Args:
            query: Normalized query
            entities: Extracted entities from NER

        Returns:
            List of expanded query strings

        Example:
            Input: "p53 cancer"
            Output: ["p53 cancer", "TP53 tumor", "tumor protein 53 neoplasm", ...]
        """
        expansions = [query]  # Always include original
        query_lower = query.lower()

        # Gene/Protein expansions
        for gene, aliases in COMMON_GENE_ALIASES.items():
            if gene.lower() in query_lower:
                for alias in aliases:
                    expanded = re.sub(
                        r'\b' + re.escape(gene) + r'\b',
                        alias,
                        query,
                        flags=re.IGNORECASE
                    )
                    if expanded != query:
                        expansions.append(expanded)

        # Technique expansions
        for technique, synonyms in TECHNIQUE_SYNONYMS.items():
            if technique.lower() in query_lower:
                for synonym in synonyms:
                    expanded = re.sub(
                        r'\b' + re.escape(technique) + r'\b',
                        synonym,
                        query,
                        flags=re.IGNORECASE
                    )
                    if expanded != query:
                        expansions.append(expanded)

        # Organism expansions
        for organism, synonyms in ORGANISM_SYNONYMS.items():
            if organism.lower() in query_lower:
                for synonym in synonyms:
                    expanded = re.sub(
                        r'\b' + re.escape(organism) + r'\b',
                        synonym,
                        query,
                        flags=re.IGNORECASE
                    )
                    if expanded != query:
                        expansions.append(expanded)

        # Create combinations for high-value terms
        if entities.get('techniques') and entities.get('organisms'):
            for tech in entities['techniques'][:2]:  # Top 2 techniques
                for org in entities['organisms'][:2]:  # Top 2 organisms
                    expansions.append(f"{tech} in {org}")

        # Deduplicate while preserving order
        seen = set()
        unique_expansions = []
        for exp in expansions:
            exp_lower = exp.lower()
            if exp_lower not in seen:
                seen.add(exp_lower)
                unique_expansions.append(exp)

        # Limit total expansions
        return unique_expansions[:10]

    def detect_query_intent(self, query: str, entities: Dict) -> str:
        """
        Classify the user's search intent

        Intent types:
        - 'specific_person': Looking for a known researcher (has name)
        - 'research_area': Exploring a research field (broad terms)
        - 'technique_based': Finding technique experts (methods mentioned)
        - 'collaborative': Seeking collaboration partners (interdisciplinary)
        - 'funding_based': Grant-related search (funding keywords)
        - 'organism_based': Focused on model organism

        Args:
            query: User query
            entities: Extracted entities

        Returns:
            Intent classification string
        """
        query_lower = query.lower()

        # Check for person names (capitalized words that aren't known keywords)
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        if re.search(name_pattern, query):
            return 'specific_person'

        # Check for funding keywords
        funding_terms = ['funded', 'funding', 'grant', 'nih', 'nsf', 'r01']
        if any(term in query_lower for term in funding_terms):
            return 'funding_based'

        # Check for collaboration keywords
        collab_terms = ['collaboration', 'collaborative', 'interdisciplinary', 'multidisciplinary']
        if any(term in query_lower for term in collab_terms):
            return 'collaborative'

        # Check for techniques (highest priority after person/funding)
        if entities.get('techniques') and len(entities['techniques']) > 0:
            return 'technique_based'

        # Check for organism focus
        if entities.get('organisms') and len(entities['organisms']) > 0:
            # If organism is prominent, it's organism-based
            org_count = sum(1 for org in entities['organisms'])
            if org_count >= 1 and len(query.split()) <= 5:
                return 'organism_based'

        # Default to research area
        return 'research_area'

    def extract_implicit_filters(self, query: str) -> Dict[str, Any]:
        """
        Parse filters from natural language

        Recognize patterns like:
        - "young PI" → career_stage: assistant_professor
        - "well-funded" → has_active_funding: true
        - "at MIT" → institution: MIT
        - "prolific researcher" → min_publications: high_threshold

        Args:
            query: User query

        Returns:
            Dictionary of extracted filters
        """
        filters = {}
        query_lower = query.lower()

        # Extract career stage
        for keyword, stage in CAREER_STAGE_KEYWORDS.items():
            if keyword in query_lower:
                filters['career_stage'] = stage
                break

        # Extract institution
        # Pattern: "at [Institution]" or "[Institution] university/college"
        institution_pattern = r'at\s+([A-Z][a-zA-Z\s]+?)(?:\s|$)|([A-Z][a-zA-Z]+)\s+(?:University|College|Institute)'
        matches = re.finditer(institution_pattern, query)
        for match in matches:
            institution = match.group(1) or match.group(2)
            if institution:
                filters['institution'] = institution.strip()
                break

        # Extract funding filters
        for keyword, filter_dict in FUNDING_KEYWORDS.items():
            if keyword in query_lower:
                filters.update(filter_dict)

        # Extract productivity filters
        for keyword, filter_dict in PRODUCTIVITY_KEYWORDS.items():
            if keyword in query_lower:
                filters.update(filter_dict)

        # Extract accepting students
        if 'accepting students' in query_lower or 'taking students' in query_lower:
            filters['accepting_students'] = True

        # Extract department if mentioned
        dept_pattern = r'(biology|chemistry|physics|neuroscience|genetics|immunology|biochemistry)\s+department'
        dept_match = re.search(dept_pattern, query_lower)
        if dept_match:
            filters['department'] = dept_match.group(1).capitalize()

        logger.debug(f"Extracted filters: {filters}")

        return filters

    def normalize_biology_terms(self, text: str) -> str:
        """
        Normalize biological terminology

        - Standardize organism names
        - Expand common abbreviations
        - Handle case sensitivity appropriately

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        normalized = text.strip()

        # Expand common abbreviations
        abbreviations = {
            r'\bPI\b': 'principal investigator',
            r'\bPIs\b': 'principal investigators',
            r'\bPhD\b': 'PhD',
            r'\bpostdoc\b': 'postdoctoral researcher',
            r'\blab\b': 'laboratory',
        }

        for abbr, expansion in abbreviations.items():
            normalized = re.sub(abbr, expansion, normalized, flags=re.IGNORECASE)

        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()

        return normalized

    def _identify_boost_terms(self, query: str, entities: Dict) -> List[str]:
        """
        Identify terms that should be boosted in search

        Args:
            query: Original query
            entities: Extracted entities

        Returns:
            List of terms to boost
        """
        boost_terms = []

        # Add all extracted techniques (high importance)
        if entities.get('techniques'):
            boost_terms.extend(entities['techniques'])

        # Add organisms
        if entities.get('organisms'):
            boost_terms.extend(entities['organisms'])

        # Add genes/proteins
        if entities.get('genes'):
            boost_terms.extend(entities['genes'][:5])  # Top 5

        # Add diseases
        if entities.get('diseases'):
            boost_terms.extend(entities['diseases'][:3])  # Top 3

        # Remove duplicates
        boost_terms = list(set(boost_terms))

        return boost_terms[:10]  # Limit to top 10
