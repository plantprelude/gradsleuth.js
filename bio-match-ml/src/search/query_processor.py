"""
Query Processor for intelligent query understanding and expansion

Transforms raw user queries into optimized search parameters with:
- Entity extraction (genes, techniques, organisms)
- Query expansion with biology synonyms
- Intent detection (technique-based, organism-based, etc.)
- Implicit filter extraction (career stage, institution, etc.)
- Term normalization
"""

import logging
import re
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, asdict
from collections import Counter

# Import biology knowledge
from .biology_knowledge import (
    expand_biology_term,
    expand_query_terms,
    get_career_stage,
    categorize_technique,
    normalize_organism_name,
    is_funding_keyword,
    is_technique_keyword,
    CAREER_STAGE_KEYWORDS,
    COMMON_GENE_ALIASES,
    TECHNIQUE_SYNONYMS,
    ORGANISM_MAPPINGS,
    DISEASE_SYNONYMS,
    FUNDING_KEYWORDS,
    INSTITUTION_KEYWORDS,
    PUBLICATION_KEYWORDS
)

logger = logging.getLogger(__name__)


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

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class QueryProcessor:
    """
    Intelligent query understanding and expansion for biology domain
    """

    def __init__(
        self,
        entity_recognizer=None,
        topic_classifier=None,
        use_ner: bool = True
    ):
        """
        Initialize with existing NER and classification models

        Args:
            entity_recognizer: BioEntityRecognizer instance
            topic_classifier: ResearchTopicClassifier instance (optional)
            use_ner: Whether to use NER models (set False for faster processing)
        """
        self.entity_recognizer = entity_recognizer
        self.topic_classifier = topic_classifier
        self.use_ner = use_ner

        # Load NER if requested and not provided
        if self.use_ner and self.entity_recognizer is None:
            try:
                from ..ner.bio_entity_recognizer import BioEntityRecognizer
                self.entity_recognizer = BioEntityRecognizer()
                logger.info("Loaded BioEntityRecognizer")
            except Exception as e:
                logger.warning(f"Could not load BioEntityRecognizer: {e}")
                self.use_ner = False

        # Initialize stopwords
        self.stopwords = self._load_stopwords()

        logger.info("QueryProcessor initialized")

    def _load_stopwords(self) -> Set[str]:
        """Load common stopwords"""
        try:
            import nltk
            try:
                stopwords_set = set(nltk.corpus.stopwords.words('english'))
            except LookupError:
                # Download if not available
                nltk.download('stopwords', quiet=True)
                stopwords_set = set(nltk.corpus.stopwords.words('english'))

            # Remove some words that are important in bio context
            bio_important = {'in', 'on', 'with', 'from', 'by', 'about'}
            stopwords_set -= bio_important

            return stopwords_set
        except Exception as e:
            logger.warning(f"Could not load stopwords: {e}")
            return set()

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
        logger.debug(f"Processing query: {raw_query}")

        # Step 1: Extract entities (using NER if available)
        entities = self._extract_entities(raw_query)

        # Step 2: Normalize query
        normalized = self.normalize_biology_terms(raw_query)

        # Step 3: Extract implicit filters
        filters = self.extract_implicit_filters(raw_query)

        # Step 4: Detect query intent
        intent = self.detect_query_intent(normalized, entities)

        # Step 5: Expand query scientifically
        expansions = self.expand_query_scientifically(normalized, entities)

        # Step 6: Identify boost terms (important entities)
        boost_terms = self._identify_boost_terms(entities)

        analysis = QueryAnalysis(
            original=raw_query,
            normalized=normalized,
            entities=entities,
            intent=intent,
            expansions=expansions,
            filters=filters,
            boost_terms=boost_terms
        )

        logger.debug(f"Query analysis complete: intent={intent}, {len(expansions)} expansions")
        return analysis

    def _extract_entities(self, text: str) -> Dict[str, List]:
        """
        Extract biological entities from query

        Args:
            text: Query text

        Returns:
            Dictionary of entities by type
        """
        entities = {
            'genes': [],
            'proteins': [],
            'organisms': [],
            'diseases': [],
            'chemicals': [],
            'techniques': [],
            'institutions': [],
            'cell_types': []
        }

        # Use NER if available
        if self.use_ner and self.entity_recognizer is not None:
            try:
                ner_entities = self.entity_recognizer.extract_all_entities(text)
                # Merge NER results
                for key in entities.keys():
                    if key in ner_entities:
                        entities[key] = [e['name'] for e in ner_entities[key]]
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")

        # Always use pattern matching as fallback/supplement
        # Extract techniques
        if not entities['techniques']:
            entities['techniques'] = self._extract_techniques_pattern(text)

        # Extract organisms
        if not entities['organisms']:
            entities['organisms'] = self._extract_organisms_pattern(text)

        # Extract institutions
        entities['institutions'] = self._extract_institutions(text)

        # Extract genes from known aliases
        entities['genes'].extend(self._extract_genes_pattern(text))

        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities

    def _extract_techniques_pattern(self, text: str) -> List[str]:
        """Extract techniques using pattern matching"""
        techniques = []
        text_lower = text.lower()

        for technique in TECHNIQUE_SYNONYMS.keys():
            if technique in text_lower:
                techniques.append(technique)

        return techniques

    def _extract_organisms_pattern(self, text: str) -> List[str]:
        """Extract organisms using pattern matching"""
        organisms = []
        text_lower = text.lower()

        for organism, patterns in ORGANISM_MAPPINGS.items():
            if organism in text_lower or any(p.lower() in text_lower for p in patterns):
                organisms.append(organism)

        return organisms

    def _extract_institutions(self, text: str) -> List[str]:
        """Extract institution names"""
        institutions = []
        text_lower = text.lower()

        # Common institutions
        common_institutions = [
            'harvard', 'mit', 'stanford', 'yale', 'princeton', 'berkeley',
            'caltech', 'columbia', 'penn', 'cornell', 'uchicago', 'johns hopkins',
            'duke', 'northwestern', 'washington', 'ucla', 'ucsd', 'ucsf'
        ]

        for inst in common_institutions:
            if inst in text_lower:
                institutions.append(inst.title())

        # Look for patterns like "at [Institution]" or "from [Institution]"
        at_pattern = r'(?:at|from|in)\s+([A-Z][A-Za-z\s]+?)(?:\s+(?:university|institute|college|hospital))?(?:\s|$|,)'
        matches = re.findall(at_pattern, text)
        institutions.extend(matches)

        return list(set(institutions))

    def _extract_genes_pattern(self, text: str) -> List[str]:
        """Extract genes from known aliases"""
        genes = []
        text_lower = text.lower()

        for gene in COMMON_GENE_ALIASES.keys():
            if gene in text_lower:
                genes.append(gene.upper())

        return genes

    def _identify_boost_terms(self, entities: Dict[str, List]) -> List[str]:
        """
        Identify important terms to boost in search

        Args:
            entities: Extracted entities

        Returns:
            List of terms to boost
        """
        boost_terms = []

        # Boost techniques (high importance)
        boost_terms.extend(entities.get('techniques', []))

        # Boost organisms (high importance)
        boost_terms.extend(entities.get('organisms', []))

        # Boost diseases (medium importance)
        boost_terms.extend(entities.get('diseases', []))

        # Boost genes (medium importance)
        boost_terms.extend(entities.get('genes', [])[:5])  # Limit to top 5

        return list(set(boost_terms))

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

        # Expand techniques
        for technique in entities.get('techniques', []):
            technique_expansions = expand_biology_term(technique)
            for exp in technique_expansions:
                if exp.lower() != technique.lower():
                    # Create expanded query
                    expanded = query.replace(technique, exp)
                    if expanded != query:
                        expansions.append(expanded)

        # Expand organisms
        for organism in entities.get('organisms', []):
            organism_expansions = expand_biology_term(organism)
            for exp in organism_expansions:
                if exp.lower() != organism.lower():
                    expanded = query.replace(organism, exp)
                    if expanded != query:
                        expansions.append(expanded)

        # Expand genes
        for gene in entities.get('genes', []):
            gene_expansions = expand_biology_term(gene.lower())
            for exp in gene_expansions:
                if exp.lower() != gene.lower():
                    expanded = query.replace(gene.lower(), exp)
                    if expanded != query:
                        expansions.append(expanded)

        # Expand diseases
        for disease in entities.get('diseases', []):
            disease_expansions = expand_biology_term(disease.lower())
            for exp in disease_expansions:
                if exp.lower() != disease.lower():
                    expanded = query.replace(disease.lower(), exp)
                    if expanded != query:
                        expansions.append(expanded)

        # Generate combinatorial expansions (limit to avoid explosion)
        if len(expansions) > 1 and len(expansions) < 10:
            # Try a few strategic combinations
            if entities.get('techniques') and entities.get('organisms'):
                tech_expansions = expand_biology_term(entities['techniques'][0])
                org_expansions = expand_biology_term(entities['organisms'][0])
                for tech in tech_expansions[:2]:
                    for org in org_expansions[:2]:
                        expansions.append(f"{tech} {org}")

        # Remove duplicates and very similar expansions
        expansions = list(set(expansions))

        # Limit total expansions
        if len(expansions) > 20:
            expansions = expansions[:20]

        logger.debug(f"Generated {len(expansions)} query expansions")
        return expansions

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

        # Check for specific person (has capitalized name patterns)
        name_pattern = r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'
        if re.search(name_pattern, query) and not any(word in query_lower for word in ['dr', 'professor', 'pi']):
            return 'specific_person'

        # Check for technique-based
        if entities.get('techniques') or is_technique_keyword(query):
            return 'technique_based'

        # Check for organism-based
        if entities.get('organisms'):
            return 'organism_based'

        # Check for funding-based
        if is_funding_keyword(query):
            return 'funding_based'

        # Check for collaborative intent
        collaborative_keywords = ['collaboration', 'collaborative', 'interdisciplinary', 'multidisciplinary',
                                   'cross-disciplinary', 'partner', 'joint']
        if any(kw in query_lower for kw in collaborative_keywords):
            return 'collaborative'

        # Check for disease-based
        if entities.get('diseases'):
            return 'disease_based'

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
        institutions = self._extract_institutions(query)
        if institutions:
            filters['institution'] = institutions[0]  # Use first found

        # Check for funding indicators
        if any(word in query_lower for word in ['well-funded', 'funded', 'active grants', 'nih', 'nsf']):
            filters['has_active_funding'] = True

        # Check for publication indicators
        if any(word in query_lower for word in ['prolific', 'many publications', 'highly cited']):
            filters['min_publications'] = 50  # Threshold for prolific
            filters['min_h_index'] = 20

        if 'high impact' in query_lower or 'top journal' in query_lower:
            filters['has_high_impact_pubs'] = True

        # Check for student acceptance
        if any(word in query_lower for word in ['accepting students', 'taking students', 'new students']):
            filters['accepting_students'] = True

        # Check for lab size preferences
        if 'small lab' in query_lower:
            filters['max_lab_size'] = 8
        elif 'large lab' in query_lower:
            filters['min_lab_size'] = 15

        # Geographic filters
        regions = {
            'boston': ['harvard', 'mit', 'boston'],
            'bay area': ['stanford', 'berkeley', 'ucsf'],
            'new york': ['columbia', 'cornell', 'nyu'],
            'california': ['caltech', 'ucla', 'ucsd', 'ucsf', 'berkeley']
        }

        for region, institutions in regions.items():
            if region in query_lower:
                filters['region'] = region

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
        normalized = text.lower()

        # Normalize common abbreviations
        abbreviations = {
            r'\bpi\b': 'principal investigator',
            r'\bko\b': 'knockout',
            r'\bwt\b': 'wildtype',
            r'\bngs\b': 'next generation sequencing',
            r'\bip\b': 'immunoprecipitation',
            r'\bgfp\b': 'green fluorescent protein',
            r'\brfp\b': 'red fluorescent protein',
            r'\bkd\b': 'knockdown',
            r'\boe\b': 'overexpression'
        }

        for abbrev, full_form in abbreviations.items():
            normalized = re.sub(abbrev, full_form, normalized, flags=re.IGNORECASE)

        # Normalize organism names to scientific names where appropriate
        # But keep common names for better matching
        # This is handled by entity extraction

        # Remove extra whitespace
        normalized = ' '.join(normalized.split())

        return normalized

    def suggest_query_improvements(self, query: str) -> List[str]:
        """
        Suggest better query formulations

        Args:
            query: Original query

        Returns:
            List of suggested improved queries
        """
        suggestions = []

        # If query is very short, suggest adding context
        if len(query.split()) < 3:
            suggestions.append(f"Try adding more context: e.g., '{query} cancer research'")

        # If no entities found, suggest adding specifics
        entities = self._extract_entities(query)
        if not any(entities.values()):
            suggestions.append("Try adding specific techniques, organisms, or diseases")

        # If query is too generic
        generic_terms = ['biology', 'research', 'science', 'study', 'work']
        if any(term in query.lower() for term in generic_terms) and len(query.split()) < 5:
            suggestions.append("Try being more specific about research area or techniques")

        return suggestions

    def parse_boolean_query(self, query: str) -> Dict[str, List[str]]:
        """
        Parse boolean queries (AND, OR, NOT)

        Args:
            query: Query with boolean operators

        Returns:
            Dictionary with must, should, must_not terms
        """
        # Simple boolean parsing
        result = {
            'must': [],      # AND terms
            'should': [],    # OR terms
            'must_not': []   # NOT terms
        }

        # Split by AND
        and_parts = re.split(r'\s+AND\s+', query, flags=re.IGNORECASE)

        for part in and_parts:
            # Check for NOT
            not_match = re.match(r'NOT\s+(.+)', part, flags=re.IGNORECASE)
            if not_match:
                result['must_not'].append(not_match.group(1).strip())
                continue

            # Check for OR
            or_parts = re.split(r'\s+OR\s+', part, flags=re.IGNORECASE)
            if len(or_parts) > 1:
                result['should'].extend([p.strip() for p in or_parts])
            else:
                result['must'].append(part.strip())

        return result
