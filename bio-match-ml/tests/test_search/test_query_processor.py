"""
Unit tests for QueryProcessor
"""

import pytest
from src.search.query_processor import QueryProcessor, QueryAnalysis


class TestQueryProcessor:
    """Test QueryProcessor functionality"""

    @pytest.fixture
    def processor(self):
        """Create a QueryProcessor instance without NER for faster testing"""
        return QueryProcessor(use_ner=False)

    def test_initialization(self, processor):
        """Test QueryProcessor initializes correctly"""
        assert processor is not None
        assert isinstance(processor.stopwords, set)

    def test_simple_query_processing(self, processor):
        """Test processing a simple query"""
        query = "CRISPR gene editing"
        analysis = processor.process_query(query)

        assert isinstance(analysis, QueryAnalysis)
        assert analysis.original == query
        assert len(analysis.normalized) > 0
        assert 'crispr' in analysis.normalized.lower()

    def test_technique_extraction(self, processor):
        """Test extraction of research techniques"""
        query = "Looking for faculty using CRISPR and RNA-seq"
        analysis = processor.process_query(query)

        assert 'techniques' in analysis.entities
        assert 'crispr' in [t.lower() for t in analysis.entities['techniques']]
        assert 'rna-seq' in [t.lower() for t in analysis.entities['techniques']]

    def test_organism_extraction(self, processor):
        """Test extraction of model organisms"""
        query = "studying cancer in mouse models"
        analysis = processor.process_query(query)

        assert 'organisms' in analysis.entities
        assert 'mouse' in [o.lower() for o in analysis.entities['organisms']]

    def test_intent_detection_technique_based(self, processor):
        """Test detection of technique-based intent"""
        query = "faculty using CRISPR for gene editing"
        analysis = processor.process_query(query)

        assert analysis.intent == 'technique_based'

    def test_intent_detection_organism_based(self, processor):
        """Test detection of organism-based intent"""
        query = "researchers working with zebrafish"
        analysis = processor.process_query(query)

        assert analysis.intent == 'organism_based'

    def test_intent_detection_funding_based(self, processor):
        """Test detection of funding-based intent"""
        query = "well-funded labs with active NIH grants"
        analysis = processor.process_query(query)

        assert analysis.intent == 'funding_based'

    def test_career_stage_filter(self, processor):
        """Test extraction of career stage filter"""
        query = "young PI studying neuroscience"
        analysis = processor.process_query(query)

        assert 'career_stage' in analysis.filters
        assert analysis.filters['career_stage'] == 'assistant_professor'

    def test_institution_filter(self, processor):
        """Test extraction of institution filter"""
        query = "researchers at MIT studying cancer"
        analysis = processor.process_query(query)

        assert 'institution' in analysis.filters
        assert 'MIT' in analysis.filters['institution']

    def test_funding_filter(self, processor):
        """Test extraction of funding filter"""
        query = "well-funded researchers"
        analysis = processor.process_query(query)

        assert 'has_active_funding' in analysis.filters
        assert analysis.filters['has_active_funding'] is True

    def test_publication_filter(self, processor):
        """Test extraction of publication filter"""
        query = "prolific researchers with many publications"
        analysis = processor.process_query(query)

        assert 'min_publications' in analysis.filters

    def test_query_expansion(self, processor):
        """Test scientific query expansion"""
        query = "crispr cancer"
        analysis = processor.process_query(query)

        assert len(analysis.expansions) > 1
        # Should include original and expansions
        assert query.lower() in [e.lower() for e in analysis.expansions]

    def test_boost_terms(self, processor):
        """Test identification of boost terms"""
        query = "CRISPR gene editing in mouse neurons"
        analysis = processor.process_query(query)

        assert len(analysis.boost_terms) > 0
        # Techniques and organisms should be boosted
        boost_terms_lower = [t.lower() for t in analysis.boost_terms]
        assert 'crispr' in boost_terms_lower or 'mouse' in boost_terms_lower

    def test_normalization_abbreviations(self, processor):
        """Test normalization of common abbreviations"""
        query = "PI using KO mice"
        normalized = processor.normalize_biology_terms(query)

        assert 'principal investigator' in normalized
        assert 'knockout' in normalized

    def test_complex_query(self, processor):
        """Test processing of complex multi-faceted query"""
        query = "young PI at Harvard using CRISPR-Cas9 to study cancer in mouse models"
        analysis = processor.process_query(query)

        # Should extract multiple entity types
        assert analysis.entities['techniques']
        assert analysis.entities['organisms']

        # Should detect career stage and institution
        assert 'career_stage' in analysis.filters
        assert 'institution' in analysis.filters

        # Should detect technique-based intent
        assert analysis.intent == 'technique_based'

        # Should have expansions
        assert len(analysis.expansions) > 1

    def test_gene_extraction(self, processor):
        """Test extraction of gene names"""
        query = "p53 mutation in cancer research"
        analysis = processor.process_query(query)

        # Should extract p53 as a gene
        genes_lower = [g.lower() for g in analysis.entities.get('genes', [])]
        assert 'p53' in genes_lower or 'tp53' in genes_lower

    def test_empty_query(self, processor):
        """Test handling of empty query"""
        query = ""
        analysis = processor.process_query(query)

        assert analysis.original == ""
        assert len(analysis.expansions) >= 1  # At least includes the original

    def test_query_with_no_entities(self, processor):
        """Test query with no recognizable entities"""
        query = "general biology research"
        analysis = processor.process_query(query)

        assert analysis.intent == 'research_area'
        assert analysis.normalized

    def test_to_dict_conversion(self, processor):
        """Test QueryAnalysis to_dict conversion"""
        query = "CRISPR research"
        analysis = processor.process_query(query)

        analysis_dict = analysis.to_dict()
        assert isinstance(analysis_dict, dict)
        assert 'original' in analysis_dict
        assert 'normalized' in analysis_dict
        assert 'entities' in analysis_dict
        assert 'intent' in analysis_dict

    def test_multiple_techniques(self, processor):
        """Test extraction of multiple techniques"""
        query = "using CRISPR, RNA-seq, and flow cytometry"
        analysis = processor.process_query(query)

        techniques = analysis.entities['techniques']
        assert len(techniques) >= 2

    def test_multiple_organisms(self, processor):
        """Test extraction of multiple organisms"""
        query = "comparing mouse and human models"
        analysis = processor.process_query(query)

        organisms = analysis.entities['organisms']
        assert len(organisms) >= 2

    def test_collaborative_intent(self, processor):
        """Test detection of collaborative intent"""
        query = "looking for interdisciplinary collaboration partners"
        analysis = processor.process_query(query)

        assert analysis.intent == 'collaborative'

    def test_region_filter(self, processor):
        """Test extraction of regional filters"""
        query = "researchers in the Boston area"
        analysis = processor.process_query(query)

        assert 'region' in analysis.filters or 'institution' in analysis.filters

    def test_lab_size_filter(self, processor):
        """Test extraction of lab size preferences"""
        query = "small lab environment"
        analysis = processor.process_query(query)

        assert 'max_lab_size' in analysis.filters

    def test_student_acceptance_filter(self, processor):
        """Test extraction of student acceptance filter"""
        query = "labs accepting new students"
        analysis = processor.process_query(query)

        assert 'accepting_students' in analysis.filters
        assert analysis.filters['accepting_students'] is True

    def test_query_suggestions(self, processor):
        """Test query improvement suggestions"""
        # Very short query
        suggestions = processor.suggest_query_improvements("biology")
        assert len(suggestions) > 0

        # Better query should have fewer suggestions
        suggestions2 = processor.suggest_query_improvements("CRISPR gene editing in cancer cells")
        # May or may not have suggestions, but should not error

    def test_boolean_query_parsing(self, processor):
        """Test parsing of boolean queries"""
        query = "CRISPR AND cancer NOT leukemia"
        parsed = processor.parse_boolean_query(query)

        assert 'must' in parsed
        assert 'must_not' in parsed
        assert 'should' in parsed

        assert any('crispr' in term.lower() for term in parsed['must'])
        assert any('leukemia' in term.lower() for term in parsed['must_not'])

    def test_or_query_parsing(self, processor):
        """Test parsing of OR queries"""
        query = "CRISPR OR TALEN OR zinc finger"
        parsed = processor.parse_boolean_query(query)

        assert len(parsed['should']) >= 2

    def test_case_insensitivity(self, processor):
        """Test that query processing is case-insensitive"""
        query1 = "CRISPR Research"
        query2 = "crispr research"

        analysis1 = processor.process_query(query1)
        analysis2 = processor.process_query(query2)

        # Should extract same techniques regardless of case
        assert analysis1.entities['techniques'] == analysis2.entities['techniques']

    def test_special_characters(self, processor):
        """Test handling of special characters"""
        query = "CRISPR-Cas9 & RNA-seq @ MIT"
        analysis = processor.process_query(query)

        # Should still extract entities
        assert len(analysis.entities['techniques']) > 0

    def test_expansion_limit(self, processor):
        """Test that query expansions are limited"""
        query = "CRISPR gene editing cancer research mouse models"
        analysis = processor.process_query(query)

        # Should limit expansions to avoid explosion
        assert len(analysis.expansions) <= 20

    def test_institution_extraction_patterns(self, processor):
        """Test various institution extraction patterns"""
        test_cases = [
            ("research at Stanford", "Stanford"),
            ("from MIT", "MIT"),
            ("working in Harvard", "Harvard")
        ]

        for query, expected_inst in test_cases:
            analysis = processor.process_query(query)
            if analysis.entities['institutions']:
                assert any(expected_inst.lower() in inst.lower()
                          for inst in analysis.entities['institutions'])


class TestQueryAnalysisDataclass:
    """Test QueryAnalysis dataclass"""

    def test_dataclass_creation(self):
        """Test creating QueryAnalysis instance"""
        analysis = QueryAnalysis(
            original="test query",
            normalized="test query",
            entities={},
            intent="research_area",
            expansions=["test query"],
            filters={},
            boost_terms=[]
        )

        assert analysis.original == "test query"
        assert analysis.intent == "research_area"

    def test_dataclass_to_dict(self):
        """Test to_dict method"""
        analysis = QueryAnalysis(
            original="test",
            normalized="test",
            entities={'techniques': ['CRISPR']},
            intent="technique_based",
            expansions=["test"],
            filters={'career_stage': 'assistant_professor'},
            boost_terms=['CRISPR']
        )

        analysis_dict = analysis.to_dict()
        assert isinstance(analysis_dict, dict)
        assert analysis_dict['original'] == "test"
        assert analysis_dict['entities']['techniques'] == ['CRISPR']
        assert analysis_dict['filters']['career_stage'] == 'assistant_professor'


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def processor(self):
        return QueryProcessor(use_ner=False)

    def test_very_long_query(self, processor):
        """Test handling of very long queries"""
        query = " ".join(["biology research"] * 100)
        analysis = processor.process_query(query)

        assert analysis is not None
        assert analysis.normalized

    def test_unicode_characters(self, processor):
        """Test handling of unicode characters"""
        query = "α-synuclein in Parkinson's disease"
        analysis = processor.process_query(query)

        assert analysis is not None

    def test_numbers_in_query(self, processor):
        """Test handling of numbers"""
        query = "p53 gene mutation in 50% of cancers"
        analysis = processor.process_query(query)

        assert analysis is not None

    def test_query_with_only_stopwords(self, processor):
        """Test query with only stopwords"""
        query = "the and or"
        analysis = processor.process_query(query)

        assert analysis is not None
        # Should still return a valid analysis

    def test_repeated_terms(self, processor):
        """Test query with repeated terms"""
        query = "CRISPR CRISPR CRISPR cancer cancer"
        analysis = processor.process_query(query)

        # Entities should be deduplicated
        techniques = analysis.entities['techniques']
        assert techniques.count('crispr') <= 1 or techniques.count('CRISPR') <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
