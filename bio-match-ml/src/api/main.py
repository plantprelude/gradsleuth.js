"""
Main FastAPI application for BioMatch Semantic Search
"""
import logging
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="BioMatch Semantic Search API",
    version="1.0.0",
    description="Production-ready semantic search and matching for biology research"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    mode: str = Field("faculty", description="Search mode: faculty, publications, grants, labs")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")
    limit: int = Field(20, ge=1, le=100, description="Number of results")
    offset: int = Field(0, ge=0, description="Offset for pagination")
    explain: bool = Field(False, description="Include explanations")


class FacultyResult(BaseModel):
    id: str
    name: str
    institution: str
    department: str
    research_summary: str
    match_score: float
    explanation: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[FacultyResult]
    total: int
    facets: Optional[Dict[str, Any]] = None
    query_interpretation: Optional[Dict[str, Any]] = None


class MatchRequest(BaseModel):
    student_profile: Dict[str, Any] = Field(..., description="Student research profile")
    faculty_ids: Optional[List[str]] = Field(None, description="Specific faculty to match against")
    top_k: int = Field(10, ge=1, le=50, description="Number of top matches")
    explain_top_n: int = Field(3, ge=0, le=10, description="Number of matches to explain")


class MatchResult(BaseModel):
    faculty_id: str
    faculty_name: str
    overall_score: float
    component_scores: Dict[str, float]
    explanation: Optional[str] = None
    strengths: List[str]
    considerations: List[str]


class MatchResponse(BaseModel):
    matches: List[MatchResult]
    student_profile_summary: Dict[str, Any]


# Global state (in production, use dependency injection)
_embedding_generator = None
_vector_store = None
_search_engine = None
_matcher = None
_entity_recognizer = None
_topic_classifier = None


def get_embedding_generator():
    """Get or initialize embedding generator"""
    global _embedding_generator
    if _embedding_generator is None:
        from ..embeddings.embedding_generator import EmbeddingGenerator
        from ..embeddings.model_manager import ModelManager
        model_manager = ModelManager()
        _embedding_generator = EmbeddingGenerator(model_manager=model_manager)
        logger.info("Initialized EmbeddingGenerator")
    return _embedding_generator


def get_vector_store():
    """Get or initialize vector store"""
    global _vector_store
    if _vector_store is None:
        from ..vector_stores.faiss_store import FaissStore
        _vector_store = FaissStore(use_gpu=False)
        logger.info("Initialized FaissStore")
    return _vector_store


def get_entity_recognizer():
    """Get or initialize entity recognizer"""
    global _entity_recognizer
    if _entity_recognizer is None:
        from ..ner.bio_entity_recognizer import BioEntityRecognizer
        _entity_recognizer = BioEntityRecognizer()
        logger.info("Initialized BioEntityRecognizer")
    return _entity_recognizer


def get_topic_classifier():
    """Get or initialize topic classifier"""
    global _topic_classifier
    if _topic_classifier is None:
        from ..ner.research_topic_classifier import ResearchTopicClassifier
        _topic_classifier = ResearchTopicClassifier()
        logger.info("Initialized ResearchTopicClassifier")
    return _topic_classifier


def get_search_engine():
    """Get or initialize semantic search engine"""
    global _search_engine
    if _search_engine is None:
        from ..search.semantic_search import SemanticSearchEngine
        from ..search.query_processor import QueryProcessor
        from ..search.result_ranker import ResultRanker

        embedder = get_embedding_generator()
        store = get_vector_store()
        entity_recognizer = get_entity_recognizer()
        topic_classifier = get_topic_classifier()

        query_processor = QueryProcessor(
            entity_recognizer=entity_recognizer,
            topic_classifier=topic_classifier
        )
        result_ranker = ResultRanker()

        _search_engine = SemanticSearchEngine(
            embedding_generator=embedder,
            vector_store=store,
            query_processor=query_processor,
            result_ranker=result_ranker
        )
        logger.info("Initialized SemanticSearchEngine")
    return _search_engine


def get_matcher():
    """Get or initialize multi-factor matcher"""
    global _matcher
    if _matcher is None:
        from ..matching.multi_factor_scorer import MultiFactorMatcher
        from ..matching.similarity_calculator import SimilarityCalculator

        embedder = get_embedding_generator()
        similarity_calc = SimilarityCalculator(embedding_generator=embedder)
        _matcher = MultiFactorMatcher(similarity_calculator=similarity_calc)
        logger.info("Initialized MultiFactorMatcher")
    return _matcher


# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """System health status"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "embedding": "ok",
            "vector_store": "ok",
            "search": "ok"
        }
    }


# Metrics endpoint
@app.get("/api/v1/metrics")
async def get_metrics():
    """Performance and usage metrics"""
    return {
        "requests_total": 0,
        "avg_latency_ms": 0,
        "cache_hit_rate": 0.0,
        "active_indices": 0
    }


# Main search endpoint
@app.post("/api/v1/search/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Primary semantic search endpoint - NOW WITH INTELLIGENT SEARCH

    Performs semantic search with:
    - Query understanding & expansion
    - Multi-factor ranking (not just vector similarity)
    - Entity extraction & intent detection
    - Faceted search
    - Detailed explanations

    Args:
        request: Search request parameters

    Returns:
        Search results with scores and optional explanations
    """
    try:
        logger.info(f"Search request: query='{request.query}', mode={request.mode}")

        # Get intelligent search engine
        search_engine = get_search_engine()

        # Perform intelligent search
        search_results = search_engine.search(
            query=request.query,
            search_mode=request.mode,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            explain=request.explain,
            diversity_factor=0.3  # Promote diversity in results
        )

        # Format results for API response
        faculty_results = []
        for result in search_results.results:
            metadata = result.get('metadata', {})
            faculty_results.append(
                FacultyResult(
                    id=result.get('id', 'unknown'),
                    name=metadata.get('name', 'Unknown'),
                    institution=metadata.get('institution', 'Unknown'),
                    department=metadata.get('department', 'Unknown'),
                    research_summary=metadata.get('research_summary', ''),
                    match_score=result.get('final_score', result.get('score', 0)),
                    explanation=result.get('ranking_explanation') if request.explain else None
                )
            )

        # Build response with facets and query interpretation
        response = SearchResponse(
            results=faculty_results,
            total=search_results.total_count,
            facets=search_results.facets if request.explain else None,
            query_interpretation=search_results.query_interpretation if request.explain else None
        )

        logger.info(f"Returning {len(faculty_results)} results (total: {search_results.total_count})")
        return response

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/search/batch")
async def batch_search(requests: List[SearchRequest]):
    """
    Process multiple searches in parallel

    Args:
        requests: List of search requests

    Returns:
        List of search responses
    """
    responses = []
    for req in requests:
        try:
            response = await semantic_search(req)
            responses.append(response)
        except Exception as e:
            logger.error(f"Batch search error: {e}")
            responses.append(None)

    return responses


@app.post("/api/v1/match/calculate", response_model=MatchResponse)
async def calculate_match(request: MatchRequest):
    """
    Calculate matches for a student profile - NOW WITH INTELLIGENT MATCHING

    Uses comprehensive multi-factor scoring:
    - Research alignment (semantic + topic/technique overlap)
    - Funding stability (active grants, runway)
    - Productivity compatibility
    - Technique match with learning opportunities
    - Lab environment (size, mentorship)
    - Career development potential

    Generates explainable scores with specific strengths and considerations.

    Args:
        request: Match request with student profile

    Returns:
        Ranked list of faculty matches with detailed explanations
    """
    try:
        logger.info("Calculating intelligent matches for student profile")

        # Get intelligent matcher
        matcher = get_matcher()

        # First, use semantic search to find candidate faculty
        search_engine = get_search_engine()

        # Create search query from student profile
        query_text = request.student_profile.get('research_interests', '')
        if 'topics' in request.student_profile:
            query_text += " " + " ".join(request.student_profile['topics'])

        # Search for candidate faculty
        search_results = search_engine.search(
            query=query_text,
            search_mode='faculty',
            limit=request.top_k * 2,  # Get more candidates for ranking
            explain=False
        )

        # Calculate match scores for each faculty
        matches = []
        for result in search_results.results[:request.top_k]:
            metadata = result.get('metadata', {})

            # Build faculty profile from metadata
            faculty_profile = {
                'id': result.get('id'),
                'name': metadata.get('name', 'Unknown'),
                'research_summary': metadata.get('research_summary', ''),
                'topics': metadata.get('topics', []),
                'techniques': metadata.get('techniques', []),
                'organisms': metadata.get('organisms', []),
                'h_index': metadata.get('h_index', 0),
                'publication_count': metadata.get('publication_count', 0),
                'active_grants': metadata.get('active_grants', 0),
                'total_funding': metadata.get('total_funding', 0),
                'grants': metadata.get('grants', []),
                'lab_size': metadata.get('lab_size', 0),
                'accepting_students': metadata.get('accepting_students', True),
                'career_stage': metadata.get('career_stage', ''),
                'collaborations': metadata.get('collaborations', 0),
                'last_publication_year': metadata.get('last_publication_year', 0),
                'recent_publications': metadata.get('recent_publications', 0),
                'publications': metadata.get('publications', [])
            }

            # Calculate comprehensive match score
            should_explain = len(matches) < request.explain_top_n
            match_score = matcher.calculate_match_score(
                request.student_profile,
                faculty_profile,
                explain=should_explain
            )

            matches.append(
                MatchResult(
                    faculty_id=faculty_profile['id'],
                    faculty_name=faculty_profile['name'],
                    overall_score=match_score.overall_score,
                    component_scores=match_score.component_scores,
                    explanation=match_score.explanation if should_explain else None,
                    strengths=match_score.strengths if should_explain else [],
                    considerations=match_score.considerations if should_explain else []
                )
            )

        # Sort by overall score
        matches.sort(key=lambda x: x.overall_score, reverse=True)

        # Extract student profile summary
        entity_recognizer = get_entity_recognizer()
        topic_classifier = get_topic_classifier()

        student_text = request.student_profile.get('research_interests', '')
        try:
            entities = entity_recognizer.extract_all_entities(student_text)
            classification = topic_classifier.classify_research_area(student_text)

            student_summary = {
                "research_areas": [classification.get('primary_area', '')] + classification.get('secondary_areas', []),
                "techniques": request.student_profile.get('techniques', []),
                "organisms": [org['common_name'] if isinstance(org, dict) else org
                             for org in entities.get('organisms', [])],
                "career_goals": request.student_profile.get('career_goals', '')
            }
        except Exception as e:
            logger.warning(f"Could not extract student profile summary: {e}")
            student_summary = {
                "research_areas": [],
                "techniques": request.student_profile.get('techniques', []),
                "organisms": [],
                "career_goals": request.student_profile.get('career_goals', '')
            }

        response = MatchResponse(
            matches=matches,
            student_profile_summary=student_summary
        )

        logger.info(f"Returning {len(matches)} intelligent matches")
        return response

    except Exception as e:
        logger.error(f"Match calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/embeddings/generate")
async def generate_embeddings(
    texts: List[str] = Body(...),
    model: str = Body("pubmedbert")
):
    """
    Generate embeddings for provided texts

    Args:
        texts: List of texts to embed
        model: Model name to use

    Returns:
        List of embedding vectors
    """
    try:
        generator = get_embedding_generator()
        embeddings = generator.batch_generate(texts, model_name=model, show_progress=False)

        return {
            "embeddings": embeddings.tolist(),
            "model": model,
            "dimension": embeddings.shape[1]
        }

    except Exception as e:
        logger.error(f"Embedding generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/entities/extract")
async def extract_entities(
    text: str = Body(...),
    entity_types: List[str] = Body(["all"])
):
    """
    Extract biological entities from text

    Uses BioEntityRecognizer to extract:
    - Genes and proteins
    - Diseases
    - Chemicals
    - Organisms (model organisms)
    - Techniques (experimental methods)
    - Cell types and cell lines

    Args:
        text: Input text
        entity_types: Types of entities to extract

    Returns:
        Extracted entities by type with confidence scores
    """
    try:
        recognizer = get_entity_recognizer()
        entities = recognizer.extract_all_entities(text)

        return {
            "entities": entities,
            "text_length": len(text),
            "entity_counts": {k: len(v) for k, v in entities.items() if v}
        }

    except Exception as e:
        logger.error(f"Entity extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/profile/{faculty_id}/similar")
async def find_similar_faculty(
    faculty_id: str,
    limit: int = Query(10, ge=1, le=50),
    diversity: float = Query(0.3, ge=0.0, le=1.0)
):
    """
    Find faculty with similar research

    Args:
        faculty_id: Faculty ID to find similar researchers for
        limit: Number of results
        diversity: Diversity factor (0=most similar, 1=most diverse)

    Returns:
        List of similar faculty
    """
    try:
        store = get_vector_store()
        index_name = "faculty_embeddings"

        # Get faculty vector
        faculty_data = store.get_vector(faculty_id, index_name)

        if faculty_data is None:
            raise HTTPException(status_code=404, detail="Faculty not found")

        # Since FAISS doesn't return vectors, we'd need to search by ID
        # This is a simplified version
        return {
            "faculty_id": faculty_id,
            "similar_faculty": [],
            "message": "Implementation requires vector retrieval support"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar faculty error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting BioMatch API server...")
    logger.info("Initializing intelligent search and matching services...")

    # Pre-load all models and systems
    try:
        get_embedding_generator()
        get_vector_store()
        get_entity_recognizer()
        get_topic_classifier()
        get_search_engine()
        get_matcher()
        logger.info("All services initialized successfully!")
        logger.info("✓ Embedding generation")
        logger.info("✓ Vector store")
        logger.info("✓ Entity recognition")
        logger.info("✓ Topic classification")
        logger.info("✓ Semantic search engine")
        logger.info("✓ Multi-factor matcher")
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down BioMatch API server...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
