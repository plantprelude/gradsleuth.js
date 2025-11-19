"""Test query processing functionality"""
import pytest
from src.search.query_processor import QueryProcessor, QueryAnalysis


class TestQueryProcessor:

    @pytest.fixture
    def processor(self):
        return QueryProcessor()

    def test_basic_query_processing(self, processor):
        """Test basic query is processed correctly"""
        result = processor.process_query("CRISPR in neurons")

        assert isinstance(result, QueryAnalysis)
        assert result.original == "CRISPR in neurons"
        assert result.normalized
        assert result.intent is not None

    def test_empty_query(self, processor):
        """Test handling of empty or invalid queries"""
        result = processor.process_query("")
        assert result is not None
        assert result.intent == 'general'

        result = processor.process_query("   ")
        assert result is not None

    def test_implicit_filter_extraction_career_stage(self, processor):
        """Test career stage filter extraction"""
        result = processor.process_query("young PI at Harvard studying CRISPR")

        assert 'career_stage' in result.filters
        assert result.filters['career_stage'] == 'assistant_professor'

    def test_implicit_filter_extraction_institution(self, processor):
        """Test institution filter extraction"""
        result = processor.process_query("researcher at MIT working on cancer")

        assert 'institution' in result.filters
        assert 'MIT' in result.filters['institution']

    def test_implicit_filter_extraction_funding(self, processor):
        """Test funding filter extraction"""
        result = processor.process_query("well-funded lab studying genomics")

        assert 'has_active_funding' in result.filters
        assert result.filters['has_active_funding'] is True

    def test_scientific_expansion(self, processor):
        """Test query expansion with scientific synonyms"""
        expansions = processor.expand_query_scientifically(
            "p53 cancer",
            {'genes': ['p53'], 'diseases': ['cancer']}
        )

        # Should include original and expansions
        assert len(expansions) > 0
        assert "p53 cancer" in expansions
        # Should include TP53 alias
        assert any('TP53' in exp or 'tumor protein' in exp for exp in expansions)

    def test_technique_expansion(self, processor):
        """Test technique synonym expansion"""
        expansions = processor.expand_query_scientifically(
            "CRISPR screening",
            {'techniques': ['CRISPR']}
        )

        # Should expand CRISPR to related terms
        assert any('gene editing' in exp.lower() or 'cas9' in exp.lower() for exp in expansions)

    def test_intent_detection_technique_based(self, processor):
        """Test technique-based intent detection"""
        result = processor.process_query("CRISPR techniques in stem cells")
        assert result.intent == 'technique_based'

    def test_intent_detection_funding_based(self, processor):
        """Test funding-based intent detection"""
        result = processor.process_query("well-funded cancer research")
        assert result.intent == 'funding_based'

    def test_intent_detection_person_search(self, processor):
        """Test person name detection"""
        result = processor.process_query("Jane Smith biology")
        assert result.intent == 'specific_person'

    def test_intent_detection_collaborative(self, processor):
        """Test collaborative intent detection"""
        result = processor.process_query("interdisciplinary collaboration in neuroscience")
        assert result.intent == 'collaborative'

    def test_normalize_biology_terms(self, processor):
        """Test biological term normalization"""
        normalized = processor.normalize_biology_terms("young PI looking for postdoc in his lab")

        assert 'principal investigator' in normalized.lower()
        assert 'laboratory' in normalized.lower()

    def test_boost_terms_identification(self, processor):
        """Test boost term extraction"""
        result = processor.process_query("CRISPR gene editing in mouse neurons")

        assert len(result.boost_terms) > 0
        # Should boost important biological terms
        # (actual terms depend on entity recognizer, so just check we get something)

    def test_complex_query_processing(self, processor):
        """Test processing of complex multi-faceted query"""
        result = processor.process_query(
            "young PI at Stanford studying CRISPR-Cas9 gene editing in mouse neurons "
            "with active NIH funding"
        )

        # Should extract multiple filters
        assert result.filters.get('career_stage') == 'assistant_professor'
        assert 'Stanford' in result.filters.get('institution', '')
        assert result.intent in ['technique_based', 'funding_based']
        assert len(result.expansions) > 0

    def test_query_expansion_limits(self, processor):
        """Test that query expansions are limited"""
        result = processor.process_query("CRISPR in neurons")

        # Should limit expansions to reasonable number
        assert len(result.expansions) <= 10

    def test_organism_expansion(self, processor):
        """Test organism synonym expansion"""
        expansions = processor.expand_query_scientifically(
            "gene editing in mouse",
            {'organisms': ['mouse']}
        )

        # Should include scientific names
        assert any('mus musculus' in exp.lower() or 'murine' in exp for exp in expansions)
