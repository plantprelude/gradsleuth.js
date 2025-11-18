"""
FastAPI application for the biology research matching platform.
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

from ..data_access import ResearchProfileBuilder, CompleteFacultyProfile
from ..data_collectors import PubMedFetcher, FacultyProfileScraper, NIHGrantCollector
from ..utils import setup_logger

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="BioMatch Data Collection API",
    description="API for collecting and aggregating biology research data",
    version="1.0.0"
)

# Initialize data collectors
profile_builder = ResearchProfileBuilder()
pubmed_fetcher = PubMedFetcher()
faculty_scraper = FacultyProfileScraper()
grant_collector = NIHGrantCollector()

# Job tracking
jobs = {}


# Request/Response Models
class PublicationRequest(BaseModel):
    author_name: str
    affiliation: Optional[str] = None
    years: int = 5


class FacultyScraperRequest(BaseModel):
    university: str
    department: str = "biology"


class GrantRequest(BaseModel):
    investigator_name: str
    organization: Optional[str] = None
    years: int = 10


class ProfileRequest(BaseModel):
    faculty_name: str
    institution: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[dict] = None


class ProfileUpdate(BaseModel):
    faculty_id: str
    update_type: str
    data: dict


# Helper functions
def create_job(job_type: str) -> str:
    """Create a new background job."""
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'id': job_id,
        'type': job_type,
        'status': 'pending',
        'message': 'Job created',
        'created_at': datetime.now().isoformat(),
        'completed_at': None,
        'result': None
    }
    return job_id


def update_job(job_id: str, status: str, message: str, result: dict = None):
    """Update job status."""
    if job_id in jobs:
        jobs[job_id]['status'] = status
        jobs[job_id]['message'] = message
        if status in ['completed', 'failed']:
            jobs[job_id]['completed_at'] = datetime.now().isoformat()
        if result:
            jobs[job_id]['result'] = result


# Background task functions
async def collect_publications_task(
    job_id: str,
    author_name: str,
    affiliation: Optional[str],
    years: int
):
    """Background task for collecting publications."""
    try:
        logger.info(f"Job {job_id}: Collecting publications for {author_name}")
        update_job(job_id, 'running', 'Fetching publications from PubMed')

        profile = pubmed_fetcher.fetch_author_publications(
            author_name,
            affiliation=affiliation,
            years=years
        )

        result = {
            'author_name': author_name,
            'total_publications': len(profile.publications),
            'h_index': profile.h_index,
            'total_citations': profile.total_citations,
            'publications': [pub.to_dict() for pub in profile.publications[:10]]
        }

        update_job(job_id, 'completed', 'Publications collected successfully', result)
        logger.info(f"Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id}: Error - {e}")
        update_job(job_id, 'failed', str(e))


async def scrape_faculty_task(
    job_id: str,
    university: str,
    department: str
):
    """Background task for scraping faculty."""
    try:
        logger.info(f"Job {job_id}: Scraping faculty from {university}")
        update_job(job_id, 'running', f'Scraping faculty from {university}')

        faculty_list = faculty_scraper.scrape_university(university, department)

        result = {
            'university': university,
            'department': department,
            'total_faculty': len(faculty_list),
            'faculty': [f.to_dict() for f in faculty_list]
        }

        update_job(job_id, 'completed', 'Faculty scraped successfully', result)
        logger.info(f"Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id}: Error - {e}")
        update_job(job_id, 'failed', str(e))


async def collect_grants_task(
    job_id: str,
    investigator_name: str,
    organization: Optional[str],
    years: int
):
    """Background task for collecting grants."""
    try:
        logger.info(f"Job {job_id}: Collecting grants for {investigator_name}")
        update_job(job_id, 'running', 'Fetching grants from NIH RePORTER')

        profile = grant_collector.get_investigator_funding_history(
            investigator_name,
            years=years
        )

        result = {
            'investigator_name': investigator_name,
            'total_funding': profile.total_funding,
            'active_grants': len(profile.active_grants),
            'completed_grants': len(profile.completed_grants),
            'grants': [g.to_dict() for g in profile.active_grants + profile.completed_grants]
        }

        update_job(job_id, 'completed', 'Grants collected successfully', result)
        logger.info(f"Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id}: Error - {e}")
        update_job(job_id, 'failed', str(e))


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "BioMatch Data Collection API",
        "version": "1.0.0",
        "endpoints": [
            "/collect/publications",
            "/collect/faculty/{university}",
            "/collect/grants",
            "/profile/{faculty_id}",
            "/status/{job_id}"
        ]
    }


@app.post("/collect/publications")
async def collect_publications(
    request: PublicationRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger publication data collection.

    Returns job_id for status checking.
    """
    job_id = create_job('publications')

    background_tasks.add_task(
        collect_publications_task,
        job_id,
        request.author_name,
        request.affiliation,
        request.years
    )

    return {
        "job_id": job_id,
        "message": "Publication collection started",
        "status_url": f"/status/{job_id}"
    }


@app.post("/collect/faculty/{university}")
async def scrape_faculty(
    university: str,
    department: str = Query("biology", description="Department name"),
    background_tasks: BackgroundTasks = None
):
    """
    Start faculty scraping job.
    """
    job_id = create_job('faculty_scraping')

    background_tasks.add_task(
        scrape_faculty_task,
        job_id,
        university,
        department
    )

    return {
        "job_id": job_id,
        "message": f"Faculty scraping started for {university}",
        "status_url": f"/status/{job_id}"
    }


@app.post("/collect/grants")
async def collect_grants(
    request: GrantRequest,
    background_tasks: BackgroundTasks
):
    """
    Fetch grant data for investigator.
    """
    job_id = create_job('grants')

    background_tasks.add_task(
        collect_grants_task,
        job_id,
        request.investigator_name,
        request.organization,
        request.years
    )

    return {
        "job_id": job_id,
        "message": "Grant collection started",
        "status_url": f"/status/{job_id}"
    }


@app.post("/profile/build")
async def build_profile(request: ProfileRequest):
    """
    Build complete faculty profile.
    """
    try:
        logger.info(f"Building profile for {request.faculty_name}")

        profile = profile_builder.build_complete_profile(
            request.faculty_name,
            request.institution
        )

        return {
            "message": "Profile built successfully",
            "profile": profile.to_dict()
        }

    except Exception as e:
        logger.error(f"Error building profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def check_job_status(job_id: str):
    """
    Check background job status.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs[job_id]


@app.post("/webhooks/profile-updated")
async def profile_update_webhook(update: ProfileUpdate):
    """
    Handle profile update notifications.
    """
    logger.info(f"Profile update webhook: {update.faculty_id} - {update.update_type}")

    return {
        "message": "Webhook received",
        "faculty_id": update.faculty_id,
        "update_type": update.update_type
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Get API statistics."""
    return {
        "total_jobs": len(jobs),
        "pending_jobs": len([j for j in jobs.values() if j['status'] == 'pending']),
        "running_jobs": len([j for j in jobs.values() if j['status'] == 'running']),
        "completed_jobs": len([j for j in jobs.values() if j['status'] == 'completed']),
        "failed_jobs": len([j for j in jobs.values() if j['status'] == 'failed']),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
