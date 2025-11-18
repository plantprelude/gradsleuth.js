"""
Biology domain knowledge for query processing and search

This module contains biology-specific knowledge bases including:
- Gene aliases and synonyms
- Technique synonyms and categories
- Organism name mappings
- Disease terminology
- Career stage keywords
- Funding-related terms
"""

from typing import List, Dict, Set, Optional
import logging

logger = logging.getLogger(__name__)

# Career stage mappings
CAREER_STAGE_KEYWORDS = {
    'young': 'assistant_professor',
    'new': 'assistant_professor',
    'junior': 'assistant_professor',
    'early career': 'assistant_professor',
    'early-career': 'assistant_professor',
    'established': 'associate_professor',
    'mid career': 'associate_professor',
    'mid-career': 'associate_professor',
    'senior': 'full_professor',
    'experienced': 'full_professor',
    'emeritus': 'emeritus'
}

# Common gene aliases (top most searched genes in biology)
COMMON_GENE_ALIASES = {
    'p53': ['TP53', 'tumor protein 53', 'tumor protein p53', 'TRP53'],
    'egfr': ['epidermal growth factor receptor', 'ERBB1', 'HER1'],
    'brca1': ['breast cancer 1', 'BRCA1', 'breast cancer type 1'],
    'brca2': ['breast cancer 2', 'BRCA2', 'breast cancer type 2'],
    'myc': ['c-myc', 'MYC', 'v-myc'],
    'ras': ['KRAS', 'NRAS', 'HRAS', 'rat sarcoma'],
    'akt': ['protein kinase B', 'PKB', 'AKT1', 'AKT2'],
    'vegf': ['vascular endothelial growth factor', 'VEGFA'],
    'il6': ['interleukin 6', 'interleukin-6', 'IL-6'],
    'tnf': ['tumor necrosis factor', 'TNF-alpha', 'TNFα', 'TNFA'],
    'nf-kb': ['nuclear factor kappa B', 'NFKB', 'NF-κB'],
    'pi3k': ['phosphoinositide 3-kinase', 'PI3K', 'PI 3-kinase'],
    'mtor': ['mechanistic target of rapamycin', 'mammalian target of rapamycin'],
    'pten': ['phosphatase and tensin homolog', 'PTEN'],
    'stat3': ['signal transducer and activator of transcription 3', 'STAT3'],
    'braf': ['B-Raf', 'v-raf murine sarcoma viral oncogene homolog B1'],
    'cd4': ['cluster of differentiation 4', 'CD4'],
    'cd8': ['cluster of differentiation 8', 'CD8'],
    'her2': ['human epidermal growth factor receptor 2', 'ERBB2', 'HER2'],
    'bcl2': ['B-cell lymphoma 2', 'BCL2', 'BCL-2']
}

# Research technique synonyms (expanded list)
TECHNIQUE_SYNONYMS = {
    # Gene editing
    'crispr': ['CRISPR-Cas9', 'gene editing', 'genome editing', 'cas9', 'CRISPR/Cas9'],
    'talen': ['transcription activator-like effector nucleases', 'TALEN'],
    'zinc finger': ['zinc finger nucleases', 'ZFN'],

    # Sequencing
    'rna-seq': ['RNA sequencing', 'transcriptomics', 'RNA-sequencing', 'RNA seq'],
    'chip-seq': ['chromatin immunoprecipitation sequencing', 'ChIP sequencing', 'ChIP-seq'],
    'atac-seq': ['ATAC sequencing', 'assay for transposase-accessible chromatin'],
    'single-cell': ['single-cell sequencing', 'scRNA-seq', 'single cell RNA-seq', 'sc-seq'],
    'wgs': ['whole genome sequencing', 'WGS', 'genome sequencing'],
    'wes': ['whole exome sequencing', 'WES', 'exome sequencing'],

    # PCR and related
    'pcr': ['polymerase chain reaction', 'RT-PCR', 'qPCR', 'quantitative PCR'],
    'qpcr': ['quantitative PCR', 'real-time PCR', 'RT-qPCR'],

    # Protein methods
    'western blot': ['western blotting', 'immunoblot', 'protein blot', 'WB'],
    'elisa': ['enzyme-linked immunosorbent assay', 'ELISA'],
    'co-ip': ['co-immunoprecipitation', 'Co-IP', 'pull-down assay'],
    'mass spec': ['mass spectrometry', 'LC-MS', 'proteomics', 'MS'],

    # Cell biology
    'flow cytometry': ['FACS', 'fluorescence activated cell sorting', 'flow cytometry'],
    'immunofluorescence': ['IF', 'immunostaining', 'fluorescence microscopy'],
    'immunohistochemistry': ['IHC', 'immunohistochemistry'],
    'cell culture': ['tissue culture', 'cell culture', 'primary culture'],
    'organoid': ['organoid culture', '3D culture', 'spheroid'],

    # Microscopy
    'confocal': ['confocal microscopy', 'confocal imaging', 'CLSM'],
    'electron microscopy': ['EM', 'TEM', 'SEM', 'transmission electron microscopy'],
    'live imaging': ['live-cell imaging', 'time-lapse microscopy', 'live imaging'],
    'super-resolution': ['super-resolution microscopy', 'STORM', 'PALM', 'SIM', 'STED'],

    # Molecular biology
    'cloning': ['molecular cloning', 'gene cloning', 'DNA cloning'],
    'transfection': ['gene transfection', 'cell transfection'],
    'transformation': ['bacterial transformation', 'transformation'],

    # Computational
    'bioinformatics': ['computational biology', 'bioinformatics', 'systems biology'],
    'machine learning': ['deep learning', 'neural network', 'AI', 'artificial intelligence'],
    'molecular dynamics': ['MD simulation', 'molecular dynamics simulation'],

    # In vivo methods
    'animal model': ['mouse model', 'in vivo', 'animal study', 'preclinical'],
    'in vivo imaging': ['animal imaging', 'intravital imaging']
}

# Model organism mappings
ORGANISM_MAPPINGS = {
    'mouse': ['Mus musculus', 'mice', 'murine'],
    'rat': ['Rattus norvegicus', 'rats'],
    'human': ['Homo sapiens', 'humans', 'clinical', 'patient'],
    'fly': ['Drosophila melanogaster', 'fruit fly', 'drosophila', 'flies'],
    'worm': ['C. elegans', 'Caenorhabditis elegans', 'nematode', 'worms'],
    'zebrafish': ['Danio rerio', 'zebra fish', 'danio'],
    'yeast': ['Saccharomyces cerevisiae', 'S. cerevisiae', 'budding yeast'],
    'fission yeast': ['Schizosaccharomyces pombe', 'S. pombe'],
    'e. coli': ['Escherichia coli', 'E. coli', 'bacteria'],
    'arabidopsis': ['Arabidopsis thaliana', 'A. thaliana', 'thale cress'],
    'xenopus': ['Xenopus laevis', 'african clawed frog', 'frog'],
    'chicken': ['Gallus gallus', 'chick'],
    'pig': ['Sus scrofa', 'porcine', 'swine'],
    'monkey': ['macaque', 'rhesus', 'primate', 'non-human primate', 'NHP']
}

# Disease and condition synonyms
DISEASE_SYNONYMS = {
    'cancer': ['tumor', 'neoplasm', 'malignancy', 'carcinoma', 'oncology'],
    'alzheimers': ["alzheimer's disease", 'AD', 'dementia', 'neurodegeneration'],
    'parkinsons': ["parkinson's disease", 'PD', 'parkinsonism'],
    'diabetes': ['diabetes mellitus', 'type 2 diabetes', 'T2D', 'type 1 diabetes', 'T1D'],
    'covid': ['COVID-19', 'coronavirus', 'SARS-CoV-2', 'pandemic'],
    'heart disease': ['cardiovascular disease', 'CVD', 'cardiac disease', 'coronary artery disease'],
    'stroke': ['cerebrovascular accident', 'CVA', 'brain attack'],
    'autism': ['autism spectrum disorder', 'ASD', 'autism'],
    'depression': ['major depressive disorder', 'MDD', 'clinical depression'],
    'arthritis': ['rheumatoid arthritis', 'RA', 'osteoarthritis', 'OA'],
    'asthma': ['bronchial asthma', 'allergic asthma'],
    'inflammatory bowel disease': ['IBD', 'Crohn disease', "Crohn's", 'ulcerative colitis']
}

# Funding-related keywords
FUNDING_KEYWORDS = {
    'well-funded': ['active grants', 'strong funding', 'substantial funding'],
    'nih': ['National Institutes of Health', 'NIH', 'R01', 'R21', 'R35'],
    'nsf': ['National Science Foundation', 'NSF'],
    'dod': ['Department of Defense', 'DOD', 'DARPA'],
    'foundation': ['foundation grant', 'private foundation', 'charitable foundation'],
    'startup': ['startup funding', 'startup package', 'new investigator']
}

# Institution type keywords
INSTITUTION_KEYWORDS = {
    'ivy league': ['Harvard', 'Yale', 'Princeton', 'Columbia', 'Penn', 'Brown', 'Dartmouth', 'Cornell'],
    'r1': ['research university', 'R1 university', 'major research'],
    'medical school': ['medical school', 'school of medicine', 'med school'],
    'cancer center': ['cancer center', 'cancer institute', 'oncology center'],
    'national lab': ['national laboratory', 'DOE lab', 'LBNL', 'LANL', 'Argonne']
}

# Publication-related keywords
PUBLICATION_KEYWORDS = {
    'prolific': ['high publication rate', 'many publications', 'prolific'],
    'high impact': ['nature', 'science', 'cell', 'high impact', 'top journal'],
    'first author': ['first author', 'lead author'],
    'last author': ['last author', 'senior author', 'corresponding author'],
    'recent': ['recent publications', 'recently published', 'latest work']
}


def expand_biology_term(term: str) -> List[str]:
    """
    Expand a biology term to include synonyms

    Args:
        term: Input term (gene, technique, organism, disease)

    Returns:
        List of synonyms/expansions including original term
    """
    term_lower = term.lower().strip()
    expansions = [term]  # Always include original

    # Check all dictionaries
    knowledge_dicts = [
        COMMON_GENE_ALIASES,
        TECHNIQUE_SYNONYMS,
        ORGANISM_MAPPINGS,
        DISEASE_SYNONYMS,
        FUNDING_KEYWORDS,
        INSTITUTION_KEYWORDS,
        PUBLICATION_KEYWORDS
    ]

    for knowledge_dict in knowledge_dicts:
        if term_lower in knowledge_dict:
            expansions.extend(knowledge_dict[term_lower])
            break  # Found it, no need to check other dicts

    # Remove duplicates (case-insensitive)
    seen = set()
    unique_expansions = []
    for exp in expansions:
        exp_lower = exp.lower()
        if exp_lower not in seen:
            seen.add(exp_lower)
            unique_expansions.append(exp)

    return unique_expansions


def expand_query_terms(terms: List[str]) -> List[str]:
    """
    Expand multiple terms and return all unique expansions

    Args:
        terms: List of terms to expand

    Returns:
        Combined list of all expansions
    """
    all_expansions = []
    for term in terms:
        expansions = expand_biology_term(term)
        all_expansions.extend(expansions)

    # Remove duplicates
    return list(set(all_expansions))


def get_career_stage(keyword: str) -> Optional[str]:
    """
    Map a career stage keyword to standard category

    Args:
        keyword: Career stage keyword

    Returns:
        Standard career stage or None
    """
    keyword_lower = keyword.lower().strip()
    return CAREER_STAGE_KEYWORDS.get(keyword_lower)


def categorize_technique(technique: str) -> str:
    """
    Categorize a technique into broader category

    Args:
        technique: Technique name

    Returns:
        Category string
    """
    technique_lower = technique.lower()

    # Define categories
    categories = {
        'sequencing': ['seq', 'sequencing', 'rna-seq', 'chip-seq', 'atac-seq', 'wgs', 'wes', 'single-cell'],
        'gene_editing': ['crispr', 'talen', 'zinc finger', 'gene editing', 'genome editing'],
        'microscopy': ['microscopy', 'imaging', 'confocal', 'electron', 'super-resolution', 'live'],
        'molecular_biology': ['pcr', 'cloning', 'transfection', 'transformation', 'qpcr'],
        'protein_methods': ['western', 'elisa', 'mass spec', 'proteomics', 'co-ip'],
        'cell_biology': ['flow cytometry', 'facs', 'cell culture', 'organoid', 'immunofluorescence', 'ihc'],
        'computational': ['bioinformatics', 'machine learning', 'molecular dynamics', 'computational'],
        'in_vivo': ['animal', 'in vivo', 'mouse model', 'preclinical']
    }

    for category, keywords in categories.items():
        if any(keyword in technique_lower for keyword in keywords):
            return category

    return 'other'


def normalize_organism_name(organism: str) -> Dict[str, str]:
    """
    Normalize organism name to standard format

    Args:
        organism: Common or scientific name

    Returns:
        Dictionary with common_name, scientific_name, taxonomy_id
    """
    organism_lower = organism.lower().strip()

    # Mapping of organism data
    organism_data = {
        'mouse': {'common': 'Mouse', 'scientific': 'Mus musculus', 'taxid': '10090'},
        'rat': {'common': 'Rat', 'scientific': 'Rattus norvegicus', 'taxid': '10116'},
        'human': {'common': 'Human', 'scientific': 'Homo sapiens', 'taxid': '9606'},
        'fly': {'common': 'Fruit fly', 'scientific': 'Drosophila melanogaster', 'taxid': '7227'},
        'worm': {'common': 'C. elegans', 'scientific': 'Caenorhabditis elegans', 'taxid': '6239'},
        'zebrafish': {'common': 'Zebrafish', 'scientific': 'Danio rerio', 'taxid': '7955'},
        'yeast': {'common': 'Yeast', 'scientific': 'Saccharomyces cerevisiae', 'taxid': '4932'},
        'e. coli': {'common': 'E. coli', 'scientific': 'Escherichia coli', 'taxid': '562'},
        'arabidopsis': {'common': 'Arabidopsis', 'scientific': 'Arabidopsis thaliana', 'taxid': '3702'},
        'xenopus': {'common': 'Xenopus', 'scientific': 'Xenopus laevis', 'taxid': '8355'},
        'chicken': {'common': 'Chicken', 'scientific': 'Gallus gallus', 'taxid': '9031'},
        'pig': {'common': 'Pig', 'scientific': 'Sus scrofa', 'taxid': '9823'},
        'monkey': {'common': 'Monkey', 'scientific': 'Macaca mulatta', 'taxid': '9544'}
    }

    # Try to find match
    for key, data in organism_data.items():
        if organism_lower in ORGANISM_MAPPINGS.get(key, []) or organism_lower == key:
            return {
                'common_name': data['common'],
                'scientific_name': data['scientific'],
                'taxonomy_id': data['taxid']
            }

    # If not found, return original
    return {
        'common_name': organism,
        'scientific_name': organism,
        'taxonomy_id': 'unknown'
    }


def is_funding_keyword(text: str) -> bool:
    """
    Check if text contains funding-related keywords

    Args:
        text: Text to check

    Returns:
        True if funding-related
    """
    text_lower = text.lower()

    funding_indicators = [
        'grant', 'funding', 'funded', 'nih', 'nsf', 'r01', 'r21', 'r35',
        'foundation', 'award', 'fellowship', 'endowment'
    ]

    return any(indicator in text_lower for indicator in funding_indicators)


def is_technique_keyword(text: str) -> bool:
    """
    Check if text contains technique-related keywords

    Args:
        text: Text to check

    Returns:
        True if technique-related
    """
    text_lower = text.lower()

    # Check if any technique is mentioned
    for technique in TECHNIQUE_SYNONYMS.keys():
        if technique in text_lower:
            return True

    return False


# Export commonly used items
__all__ = [
    'CAREER_STAGE_KEYWORDS',
    'COMMON_GENE_ALIASES',
    'TECHNIQUE_SYNONYMS',
    'ORGANISM_MAPPINGS',
    'DISEASE_SYNONYMS',
    'FUNDING_KEYWORDS',
    'INSTITUTION_KEYWORDS',
    'PUBLICATION_KEYWORDS',
    'expand_biology_term',
    'expand_query_terms',
    'get_career_stage',
    'categorize_technique',
    'normalize_organism_name',
    'is_funding_keyword',
    'is_technique_keyword'
]
