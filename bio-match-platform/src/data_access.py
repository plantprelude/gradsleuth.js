"""
Integration layer for combining data from all sources.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .data_collectors import PubMedFetcher, FacultyProfileScraper, NIHGrantCollector
from .models import (
    FacultyInfo,
    Publication,
    PublicationMetrics,
    Grant,
    FundingMetrics,
    ResearchTrajectory,
    NetworkGraph,
)
from .utils import setup_logger, ProgressLogger

logger = setup_logger(__name__)


@dataclass
class CompleteFacultyProfile:
    """
    Complete faculty profile combining all data sources.
    """

    # Basic Information
    personal_info: FacultyInfo

    # Research Output
    publications: List[Publication] = field(default_factory=list)
    publication_metrics: Optional[PublicationMetrics] = None

    # Funding
    grants: List[Grant] = field(default_factory=list)
    funding_metrics: Optional[FundingMetrics] = None

    # Derived Insights
    research_trajectory: Optional[ResearchTrajectory] = None
    collaboration_network: Optional[NetworkGraph] = None
    productivity_score: float = 0.0
    funding_stability_score: float = 0.0

    def to_search_index(self) -> Dict:
        """
        Prepare profile for Elasticsearch indexing.

        Returns:
            Dict: Indexable document
        """
        return {
            'faculty_id': self.personal_info.faculty_id,
            'name': self.personal_info.name,
            'institution': self.personal_info.institution,
            'department': self.personal_info.department,
            'email': self.personal_info.email,
            'research_interests': self.personal_info.research_interests,
            'research_keywords': self.personal_info.extract_keywords(),

            # Publication data
            'total_publications': len(self.publications),
            'total_citations': self.publication_metrics.total_citations if self.publication_metrics else 0,
            'h_index': self.publication_metrics.h_index if self.publication_metrics else 0,
            'recent_publications': [
                {
                    'title': pub.title,
                    'year': pub.publication_date.year,
                    'journal': pub.journal,
                }
                for pub in self.publications[:5]
            ],

            # Funding data
            'total_funding': sum(g.total_cost for g in self.grants),
            'active_grants': len([g for g in self.grants if g.is_active()]),
            'grant_titles': [g.title for g in self.grants if g.is_active()],

            # Scores
            'productivity_score': self.productivity_score,
            'funding_stability_score': self.funding_stability_score,

            # For search
            'search_text': self._build_search_text(),
        }

    def _build_search_text(self) -> str:
        """Build concatenated text for full-text search."""
        components = [
            self.personal_info.name,
            self.personal_info.research_interests,
            ' '.join(self.personal_info.extract_keywords()),
        ]

        # Add publication titles
        components.extend([pub.title for pub in self.publications[:10]])

        # Add grant titles
        components.extend([g.title for g in self.grants if g.is_active()])

        return ' '.join(components)

    def generate_summary(self) -> str:
        """
        Create human-readable profile summary.

        Returns:
            str: Profile summary
        """
        lines = [
            f"# Faculty Profile: {self.personal_info.name}",
            f"",
            f"**Institution:** {self.personal_info.institution}",
            f"**Department:** {self.personal_info.department}",
            f"",
            f"## Research Interests",
            self.personal_info.research_interests,
            f"",
        ]

        # Publication summary
        if self.publication_metrics:
            lines.extend([
                f"## Publications",
                f"- Total Publications: {self.publication_metrics.total_publications}",
                f"- Total Citations: {self.publication_metrics.total_citations}",
                f"- H-Index: {self.publication_metrics.h_index}",
                f"- i10-Index: {self.publication_metrics.i10_index}",
                f"",
            ])

        # Funding summary
        if self.funding_metrics:
            lines.extend([
                f"## Funding",
                f"- Total Funding: ${self.funding_metrics.total_funding:,.0f}",
                f"- Active Grants: {self.funding_metrics.active_grants_count}",
                f"- Funding Stability: {self.funding_metrics.success_rate:.1%}",
                f"",
            ])

        # Recent publications
        if self.publications:
            lines.append("## Recent Publications")
            for pub in self.publications[:5]:
                lines.append(
                    f"- {pub.title} ({pub.publication_date.year}). "
                    f"*{pub.journal}*"
                )
            lines.append("")

        # Active grants
        active_grants = [g for g in self.grants if g.is_active()]
        if active_grants:
            lines.append("## Active Grants")
            for grant in active_grants[:5]:
                lines.append(
                    f"- {grant.title} ({grant.activity_code}) - "
                    f"${grant.total_cost:,.0f}"
                )
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'personal_info': self.personal_info.to_dict(),
            'publications': [pub.to_dict() for pub in self.publications],
            'publication_metrics': self.publication_metrics.to_dict() if self.publication_metrics else None,
            'grants': [g.to_dict() for g in self.grants],
            'funding_metrics': self.funding_metrics.to_dict() if self.funding_metrics else None,
            'research_trajectory': self.research_trajectory.to_dict() if self.research_trajectory else None,
            'collaboration_network': self.collaboration_network.to_dict() if self.collaboration_network else None,
            'productivity_score': self.productivity_score,
            'funding_stability_score': self.funding_stability_score,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class ResearchProfileBuilder:
    """
    Combines data from all three sources to build complete faculty profiles.
    """

    def __init__(
        self,
        pubmed_api_key: Optional[str] = None,
        nih_api_key: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the profile builder.

        Args:
            pubmed_api_key: PubMed API key (optional)
            nih_api_key: NIH API key (optional)
            cache_enabled: Whether to enable caching
        """
        self.pubmed_fetcher = PubMedFetcher(
            api_key=pubmed_api_key,
            cache_enabled=cache_enabled
        )
        self.grant_collector = NIHGrantCollector(
            api_key=nih_api_key,
            cache_enabled=cache_enabled
        )
        self.faculty_scraper = FacultyProfileScraper(cache_enabled=cache_enabled)

        logger.info("ResearchProfileBuilder initialized")

    def build_complete_profile(
        self,
        faculty_name: str,
        institution: str,
        faculty_info: Optional[FacultyInfo] = None
    ) -> CompleteFacultyProfile:
        """
        Orchestrate data collection from all sources.

        Args:
            faculty_name: Faculty member name
            institution: Institution name
            faculty_info: Pre-fetched faculty info (optional)

        Returns:
            CompleteFacultyProfile: Complete profile
        """
        logger.info(f"Building complete profile for {faculty_name} at {institution}")

        # If faculty info not provided, create a basic one
        if faculty_info is None:
            import hashlib
            faculty_info = FacultyInfo(
                faculty_id=hashlib.md5(f"{institution}_{faculty_name}".encode()).hexdigest()[:16],
                name=faculty_name,
                title="Faculty",
                department="Biology",
                institution=institution,
            )

        # Fetch data from all sources in parallel
        publications = []
        pub_metrics = None
        grants = []
        funding_metrics = None

        try:
            # Fetch publications
            logger.info("Fetching publications...")
            author_profile = self.pubmed_fetcher.fetch_author_publications(
                faculty_name,
                affiliation=institution,
                years=5
            )
            publications = author_profile.publications
            pub_metrics = author_profile.calculate_metrics()

        except Exception as e:
            logger.error(f"Error fetching publications: {e}")

        try:
            # Fetch grants
            logger.info("Fetching grants...")
            funding_profile = self.grant_collector.get_investigator_funding_history(
                faculty_name,
                years=10
            )
            grants = funding_profile.active_grants + funding_profile.completed_grants

            # Calculate funding metrics
            funding_metrics = FundingMetrics(
                total_funding=funding_profile.total_funding,
                active_grants_count=len(funding_profile.active_grants),
                completed_grants_count=len(funding_profile.completed_grants),
                success_rate=funding_profile.success_rate,
                avg_grant_size=funding_profile.avg_grant_size(),
                funding_trend=funding_profile.get_funding_trend(),
            )

        except Exception as e:
            logger.error(f"Error fetching grants: {e}")

        # Calculate derived metrics
        productivity_score = self._calculate_productivity_score(
            publications,
            pub_metrics
        )

        funding_stability_score = 0.0
        if grants:
            stability_metrics = self.grant_collector.analyze_funding_stability(grants)
            funding_stability_score = stability_metrics.get('stability_score', 0.0)

        # Build research trajectory
        research_trajectory = self._build_research_trajectory(publications, grants)

        # Build collaboration network
        collaboration_network = self._build_collaboration_network(
            publications,
            grants,
            faculty_name
        )

        return CompleteFacultyProfile(
            personal_info=faculty_info,
            publications=publications,
            publication_metrics=pub_metrics,
            grants=grants,
            funding_metrics=funding_metrics,
            research_trajectory=research_trajectory,
            collaboration_network=collaboration_network,
            productivity_score=productivity_score,
            funding_stability_score=funding_stability_score,
        )

    def _calculate_productivity_score(
        self,
        publications: List[Publication],
        metrics: Optional[PublicationMetrics]
    ) -> float:
        """
        Calculate overall productivity score (0-1).

        Args:
            publications: List of publications
            metrics: Publication metrics

        Returns:
            float: Productivity score
        """
        if not metrics:
            return 0.0

        # Factors: publication count, h-index, citations
        pub_score = min(metrics.total_publications / 50, 1.0) * 0.3
        h_index_score = min(metrics.h_index / 30, 1.0) * 0.3
        citation_score = min(metrics.total_citations / 1000, 1.0) * 0.4

        return pub_score + h_index_score + citation_score

    def _build_research_trajectory(
        self,
        publications: List[Publication],
        grants: List[Grant]
    ) -> ResearchTrajectory:
        """
        Build research trajectory showing topic evolution.

        Args:
            publications: List of publications
            grants: List of grants

        Returns:
            ResearchTrajectory: Research trajectory data
        """
        from datetime import datetime

        # Combine topics by year from publications and grants
        topics_by_year = {}

        for pub in publications:
            year = pub.publication_date.year
            if year not in topics_by_year:
                topics_by_year[year] = []
            topics_by_year[year].extend(pub.extract_research_topics())

        for grant in grants:
            if grant.start_date:
                year = grant.start_date.year
                if year not in topics_by_year:
                    topics_by_year[year] = []
                topics_by_year[year].extend(grant.keywords[:5])

        # Deduplicate topics per year
        for year in topics_by_year:
            topics_by_year[year] = list(set(topics_by_year[year]))

        # Identify current focus (last 2 years)
        current_year = datetime.now().year
        current_focus = []
        for year in range(current_year - 1, current_year + 1):
            if year in topics_by_year:
                current_focus.extend(topics_by_year[year])
        current_focus = list(set(current_focus))[:10]

        # Identify emerging interests (topics in last 2 years not in previous 3)
        recent_years = set(range(current_year - 1, current_year + 1))
        older_years = set(range(current_year - 5, current_year - 1))

        recent_topics = set()
        for year in recent_years:
            if year in topics_by_year:
                recent_topics.update(topics_by_year[year])

        older_topics = set()
        for year in older_years:
            if year in topics_by_year:
                older_topics.update(topics_by_year[year])

        emerging = list(recent_topics - older_topics)[:5]

        return ResearchTrajectory(
            topics_by_year=topics_by_year,
            major_transitions=[],  # Would need more sophisticated analysis
            current_focus=current_focus,
            emerging_interests=emerging,
        )

    def _build_collaboration_network(
        self,
        publications: List[Publication],
        grants: List[Grant],
        focal_person: str
    ) -> NetworkGraph:
        """
        Build collaboration network graph.

        Args:
            publications: List of publications
            grants: List of grants
            focal_person: Name of focal person

        Returns:
            NetworkGraph: Collaboration network
        """
        nodes = [{'id': focal_person, 'type': 'faculty', 'label': focal_person}]
        edges = []
        collaborator_counts = {}

        # Extract collaborators from publications
        for pub in publications:
            for author in pub.authors:
                name = author['name']
                if name != focal_person:
                    collaborator_counts[name] = collaborator_counts.get(name, 0) + 1

        # Extract collaborators from grants
        for grant in grants:
            for co_inv in grant.co_investigators:
                name = co_inv.get('name', '')
                if name and name != focal_person:
                    collaborator_counts[name] = collaborator_counts.get(name, 0) + 1

        # Add top collaborators as nodes
        top_collaborators = sorted(
            collaborator_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]

        for name, count in top_collaborators:
            nodes.append({
                'id': name,
                'type': 'collaborator',
                'label': name,
                'collaborations': count
            })

            edges.append({
                'source': focal_person,
                'target': name,
                'weight': count
            })

        # Calculate basic network metrics
        metrics = {
            'total_collaborators': len(top_collaborators),
            'total_collaborations': sum(count for _, count in top_collaborators),
            'avg_collaborations_per_person': (
                sum(count for _, count in top_collaborators) / len(top_collaborators)
                if top_collaborators else 0
            ),
        }

        return NetworkGraph(
            nodes=nodes,
            edges=edges,
            metrics=metrics,
        )

    def bulk_profile_generation(
        self,
        faculty_list: List[Dict[str, str]],
        max_workers: int = 4
    ) -> List[CompleteFacultyProfile]:
        """
        Efficiently process multiple faculty using multiprocessing.

        Args:
            faculty_list: List of dicts with 'name' and 'institution'
            max_workers: Maximum number of parallel workers

        Returns:
            List[CompleteFacultyProfile]: List of complete profiles
        """
        logger.info(f"Bulk processing {len(faculty_list)} faculty profiles")

        profiles = []
        progress = ProgressLogger(logger, len(faculty_list), "Profile generation")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_faculty = {
                executor.submit(
                    self.build_complete_profile,
                    faculty['name'],
                    faculty['institution']
                ): faculty
                for faculty in faculty_list
            }

            # Collect results
            for future in as_completed(future_to_faculty):
                faculty = future_to_faculty[future]
                try:
                    profile = future.result()
                    profiles.append(profile)
                    progress.update()
                except Exception as e:
                    logger.error(
                        f"Error processing {faculty['name']}: {e}"
                    )

        progress.complete()
        return profiles

    def export_profiles(
        self,
        profiles: List[CompleteFacultyProfile],
        output_format: str = 'json',
        output_dir: str = 'data/processed'
    ):
        """
        Export profiles to files.

        Args:
            profiles: List of profiles
            output_format: Output format ('json', 'csv', or 'parquet')
            output_dir: Output directory
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if output_format == 'json':
            for profile in profiles:
                filename = f"{profile.personal_info.faculty_id}.json"
                filepath = output_path / filename
                with open(filepath, 'w') as f:
                    f.write(profile.to_json())

            logger.info(f"Exported {len(profiles)} profiles to {output_dir}")

        elif output_format == 'csv':
            import csv

            filename = output_path / 'faculty_profiles.csv'
            fieldnames = [
                'faculty_id', 'name', 'institution', 'department',
                'total_publications', 'h_index', 'total_citations',
                'total_funding', 'active_grants', 'productivity_score',
                'funding_stability_score'
            ]

            with open(filename, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for profile in profiles:
                    writer.writerow({
                        'faculty_id': profile.personal_info.faculty_id,
                        'name': profile.personal_info.name,
                        'institution': profile.personal_info.institution,
                        'department': profile.personal_info.department,
                        'total_publications': len(profile.publications),
                        'h_index': profile.publication_metrics.h_index if profile.publication_metrics else 0,
                        'total_citations': profile.publication_metrics.total_citations if profile.publication_metrics else 0,
                        'total_funding': sum(g.total_cost for g in profile.grants),
                        'active_grants': len([g for g in profile.grants if g.is_active()]),
                        'productivity_score': profile.productivity_score,
                        'funding_stability_score': profile.funding_stability_score,
                    })

            logger.info(f"Exported profiles to {filename}")

        else:
            logger.error(f"Unsupported output format: {output_format}")
