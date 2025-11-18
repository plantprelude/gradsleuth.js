"""
NIH grant data collector using the RePORTER API.
"""
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import requests
import json
from pathlib import Path
from collections import Counter

from ..config import APIConfig, GrantConfig, CACHE_DIR
from ..models import Grant, InvestigatorFundingProfile
from ..utils import (
    RateLimiter,
    retry_with_backoff,
    setup_logger,
    validate_nih_project_number,
    normalize_name,
    extract_name_variants,
)

logger = setup_logger(__name__)


class NIHGrantCollector:
    """
    NIH RePORTER API client for collecting grant data.

    Implements fuzzy name matching, caching, and network analysis.
    """

    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True):
        """
        Initialize NIH grant collector.

        Args:
            api_key: NIH RePORTER API key (optional)
            cache_enabled: Whether to use caching
        """
        self.api_key = api_key or APIConfig.NIH_REPORTER_API_KEY
        self.base_url = APIConfig.NIH_REPORTER_BASE_URL
        self.cache_enabled = cache_enabled
        self.cache_dir = CACHE_DIR / 'nih_grants'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiter (100 requests per hour = ~0.028 per second)
        self.rate_limiter = RateLimiter(0.028)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BiologyMatchBot/1.0 (Academic Research)',
            'Content-Type': 'application/json',
        })

    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{key}.json"

    def _get_cache(self, key: str) -> Optional[Dict]:
        """
        Get cached data if available and valid.

        Args:
            key: Cache key

        Returns:
            Cached data or None
        """
        if not self.cache_enabled:
            return None

        cache_file = self._get_cache_path(key)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)

            # Check if cache is expired (30 days)
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if (datetime.now() - cached_time).days > 30:
                return None

            return cached['data']
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None

    def _set_cache(self, key: str, data: Dict):
        """
        Cache data.

        Args:
            key: Cache key
            data: Data to cache
        """
        if not self.cache_enabled:
            return

        try:
            cache_file = self._get_cache_path(key)
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f)
        except Exception as e:
            logger.warning(f"Cache write error for {key}: {e}")

    @retry_with_backoff(max_retries=3, exceptions=(requests.RequestException,))
    def _make_request(self, endpoint: str, data: Dict) -> Dict:
        """
        Make API request with rate limiting and retries.

        Args:
            endpoint: API endpoint
            data: Request payload

        Returns:
            Response data
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.post(
                url,
                json=data,
                timeout=APIConfig.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request error: {e}")
            raise

    def search_pi_grants(
        self,
        investigator_name: str,
        organization: Optional[str] = None,
        years: int = 10
    ) -> List[Grant]:
        """
        Search for grants by principal investigator.

        Args:
            investigator_name: PI name
            organization: Organization name (optional)
            years: Number of years to look back

        Returns:
            List[Grant]: List of grants
        """
        # Generate name variants for fuzzy matching
        normalized_name = normalize_name(investigator_name)
        name_variants = extract_name_variants(normalized_name)

        # Check cache
        cache_key = f"pi_{hashlib.md5(investigator_name.encode()).hexdigest()}_{years}"
        cached = self._get_cache(cache_key)
        if cached:
            logger.info(f"Cache hit for PI: {investigator_name}")
            return [self._dict_to_grant(g) for g in cached]

        logger.info(f"Searching grants for PI: {investigator_name}")

        # Calculate fiscal year range
        current_year = datetime.now().year
        start_year = current_year - years

        # Build search criteria
        criteria = {
            "pi_names": [{"any_name": investigator_name}],
            "fiscal_years": list(range(start_year, current_year + 1)),
        }

        if organization:
            criteria["org_names"] = [organization]

        # Make request
        try:
            response = self._make_request('projects/search', {
                "criteria": criteria,
                "limit": 500,
                "offset": 0
            })

            grants = []
            if 'results' in response:
                for project_data in response['results']:
                    grant = self._parse_grant_data(project_data)
                    if grant:
                        grants.append(grant)

            logger.info(f"Found {len(grants)} grants for {investigator_name}")

            # Cache result
            self._set_cache(cache_key, [g.to_dict() for g in grants])

            return grants

        except Exception as e:
            logger.error(f"Error searching grants for {investigator_name}: {e}")
            return []

    def fetch_grant_details(self, project_number: str) -> Optional[Grant]:
        """
        Fetch detailed grant information.

        Args:
            project_number: NIH project number

        Returns:
            Grant object or None
        """
        if not validate_nih_project_number(project_number):
            logger.warning(f"Invalid project number format: {project_number}")

        # Check cache
        cache_key = f"grant_{project_number}"
        cached = self._get_cache(cache_key)
        if cached:
            logger.debug(f"Cache hit for grant: {project_number}")
            return self._dict_to_grant(cached)

        logger.info(f"Fetching grant details: {project_number}")

        try:
            response = self._make_request('projects/search', {
                "criteria": {
                    "project_nums": [project_number]
                },
                "limit": 1
            })

            if 'results' in response and len(response['results']) > 0:
                grant = self._parse_grant_data(response['results'][0])

                # Cache result
                if grant:
                    self._set_cache(cache_key, grant.to_dict())

                return grant

        except Exception as e:
            logger.error(f"Error fetching grant {project_number}: {e}")

        return None

    def _parse_grant_data(self, data: Dict) -> Optional[Grant]:
        """
        Parse grant data from API response.

        Args:
            data: API response data

        Returns:
            Grant object or None
        """
        try:
            # Extract basic information
            project_number = data.get('project_num', '')
            title = data.get('project_title', '')
            abstract = data.get('abstract_text', '')

            # PI information
            pi_name = ""
            pi_institution = ""
            if 'principal_investigators' in data and data['principal_investigators']:
                pi_data = data['principal_investigators'][0]
                pi_name = pi_data.get('full_name', '')

            if 'organization' in data:
                pi_institution = data['organization'].get('org_name', '')

            # Co-investigators
            co_investigators = []
            if 'other_investigators' in data:
                for inv in data['other_investigators']:
                    co_investigators.append({
                        'name': inv.get('full_name', ''),
                        'role': inv.get('is_pi', False) and 'Co-PI' or 'Investigator'
                    })

            # Funding information
            total_cost = data.get('award_amount', 0) or 0
            direct_cost = data.get('direct_cost_amt', 0) or 0

            # Dates
            start_date = None
            end_date = None

            if 'project_start_date' in data and data['project_start_date']:
                try:
                    start_date = datetime.fromisoformat(
                        data['project_start_date'].replace('Z', '+00:00')
                    )
                except:
                    pass

            if 'project_end_date' in data and data['project_end_date']:
                try:
                    end_date = datetime.fromisoformat(
                        data['project_end_date'].replace('Z', '+00:00')
                    )
                except:
                    pass

            # Activity code
            activity_code = data.get('activity_code', '')

            # Study section
            study_section = data.get('study_section', '')

            # Keywords/terms
            keywords = []
            if 'project_terms' in data:
                keywords = data['project_terms'][:20]  # Limit to 20

            # Publications
            publications_reported = []
            if 'publications' in data:
                for pub in data['publications'][:10]:  # Limit to 10
                    if 'pmid' in pub:
                        publications_reported.append(str(pub['pmid']))

            return Grant(
                project_number=project_number,
                title=title,
                abstract=abstract,
                pi_name=pi_name,
                pi_institution=pi_institution,
                co_investigators=co_investigators,
                total_cost=float(total_cost),
                direct_cost=float(direct_cost),
                start_date=start_date,
                end_date=end_date,
                funding_agency='NIH',
                activity_code=activity_code,
                study_section=study_section,
                keywords=keywords,
                publications_reported=publications_reported,
            )

        except Exception as e:
            logger.warning(f"Error parsing grant data: {e}")
            return None

    def _dict_to_grant(self, data: Dict) -> Grant:
        """Convert dictionary to Grant object."""
        return Grant(
            project_number=data['project_number'],
            title=data['title'],
            abstract=data['abstract'],
            pi_name=data['pi_name'],
            pi_institution=data['pi_institution'],
            co_investigators=data.get('co_investigators', []),
            total_cost=data.get('total_cost', 0),
            direct_cost=data.get('direct_cost', 0),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else None,
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            funding_agency=data.get('funding_agency', 'NIH'),
            activity_code=data.get('activity_code', ''),
            study_section=data.get('study_section', ''),
            keywords=data.get('keywords', []),
            publications_reported=data.get('publications_reported', []),
        )

    def get_investigator_funding_history(
        self,
        pi_name: str,
        years: int = 10
    ) -> InvestigatorFundingProfile:
        """
        Get complete funding history for an investigator.

        Args:
            pi_name: Principal investigator name
            years: Number of years to look back

        Returns:
            InvestigatorFundingProfile: Complete funding profile
        """
        logger.info(f"Building funding profile for: {pi_name}")

        # Get all grants
        all_grants = self.search_pi_grants(pi_name, years=years)

        # Separate active and completed grants
        active_grants = [g for g in all_grants if g.is_active()]
        completed_grants = [g for g in all_grants if not g.is_active()]

        # Calculate total funding
        total_funding = sum(g.total_cost for g in all_grants)

        # Calculate funding gaps
        funding_gaps = self._calculate_funding_gaps(all_grants)

        # Calculate collaborators
        collaborators = self._calculate_collaborators(all_grants)

        # Calculate research topics funded
        topics_funded = self._calculate_topics_funded(all_grants)

        # Calculate success rate (simplified - would need application data)
        success_rate = 0.0  # Placeholder

        return InvestigatorFundingProfile(
            investigator_name=pi_name,
            total_funding=total_funding,
            active_grants=active_grants,
            completed_grants=completed_grants,
            success_rate=success_rate,
            funding_gaps=funding_gaps,
            frequent_collaborators=collaborators,
            research_topics_funded=topics_funded,
        )

    def _calculate_funding_gaps(
        self,
        grants: List[Grant]
    ) -> List[Tuple[datetime, datetime]]:
        """
        Calculate periods without funding.

        Args:
            grants: List of grants

        Returns:
            List of (start, end) tuples for gap periods
        """
        if not grants:
            return []

        # Filter grants with valid dates
        valid_grants = [
            g for g in grants
            if g.start_date and g.end_date
        ]

        if not valid_grants:
            return []

        # Sort by start date
        sorted_grants = sorted(valid_grants, key=lambda g: g.start_date)

        gaps = []
        for i in range(len(sorted_grants) - 1):
            current_end = sorted_grants[i].end_date
            next_start = sorted_grants[i + 1].start_date

            # If there's a gap of more than 30 days
            if (next_start - current_end).days > 30:
                gaps.append((current_end, next_start))

        return gaps

    def _calculate_collaborators(
        self,
        grants: List[Grant]
    ) -> List[Tuple[str, int]]:
        """
        Calculate frequent collaborators.

        Args:
            grants: List of grants

        Returns:
            List of (name, count) tuples
        """
        collaborator_counts = Counter()

        for grant in grants:
            for co_inv in grant.co_investigators:
                name = co_inv.get('name', '')
                if name:
                    collaborator_counts[name] += 1

        return collaborator_counts.most_common(20)

    def _calculate_topics_funded(
        self,
        grants: List[Grant]
    ) -> Dict[str, float]:
        """
        Calculate research topics and total funding per topic.

        Args:
            grants: List of grants

        Returns:
            Dict mapping topic to total funding
        """
        topics = {}

        for grant in grants:
            for keyword in grant.keywords:
                if keyword not in topics:
                    topics[keyword] = 0.0
                topics[keyword] += grant.total_cost

        # Sort and return top 20
        sorted_topics = sorted(
            topics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_topics[:20])

    def analyze_funding_stability(
        self,
        grant_list: List[Grant]
    ) -> Dict[str, any]:
        """
        Analyze funding stability metrics.

        Args:
            grant_list: List of grants

        Returns:
            Dict with stability metrics
        """
        if not grant_list:
            return {
                'stability_score': 0.0,
                'avg_grant_duration': 0.0,
                'funding_continuity': 0.0,
                'gap_count': 0,
            }

        # Calculate average grant duration
        valid_grants = [
            g for g in grant_list
            if g.start_date and g.end_date
        ]

        if valid_grants:
            avg_duration = sum(
                g.grant_duration_years() for g in valid_grants
            ) / len(valid_grants)
        else:
            avg_duration = 0.0

        # Calculate gaps
        gaps = self._calculate_funding_gaps(grant_list)
        gap_count = len(gaps)

        # Calculate funding continuity
        if valid_grants:
            earliest = min(g.start_date for g in valid_grants)
            latest = max(g.end_date for g in valid_grants)
            total_span_days = (latest - earliest).days

            funded_days = sum(
                (g.end_date - g.start_date).days for g in valid_grants
            )

            continuity = funded_days / total_span_days if total_span_days > 0 else 0
        else:
            continuity = 0.0

        # Overall stability score
        stability_score = continuity * (1 - (gap_count * 0.1))
        stability_score = max(0.0, min(1.0, stability_score))

        return {
            'stability_score': stability_score,
            'avg_grant_duration': avg_duration,
            'funding_continuity': continuity,
            'gap_count': gap_count,
            'total_grants': len(grant_list),
            'active_grants': len([g for g in grant_list if g.is_active()]),
        }

    def identify_research_network(
        self,
        pi_name: str
    ) -> Dict[str, any]:
        """
        Identify research collaboration network.

        Args:
            pi_name: Principal investigator name

        Returns:
            Dict with network data (nodes and edges)
        """
        logger.info(f"Building research network for: {pi_name}")

        # Get grants
        grants = self.search_pi_grants(pi_name, years=GrantConfig.DEFAULT_YEARS_LOOKBACK)

        # Build network
        nodes = [{'id': pi_name, 'type': 'pi', 'label': pi_name}]
        edges = []
        collaborator_ids = set()

        for grant in grants:
            for co_inv in grant.co_investigators:
                name = co_inv.get('name', '')
                if name and name != pi_name:
                    if name not in collaborator_ids:
                        nodes.append({
                            'id': name,
                            'type': 'collaborator',
                            'label': name
                        })
                        collaborator_ids.add(name)

                    edges.append({
                        'source': pi_name,
                        'target': name,
                        'weight': 1,
                        'grant': grant.project_number
                    })

        return {
            'nodes': nodes,
            'edges': edges,
            'metrics': {
                'total_collaborators': len(collaborator_ids),
                'total_collaborations': len(edges),
            }
        }

    def export_to_json(self, profile: InvestigatorFundingProfile, filepath: str):
        """Export funding profile to JSON file."""
        with open(filepath, 'w') as f:
            f.write(profile.to_json())
        logger.info(f"Exported funding profile to {filepath}")
