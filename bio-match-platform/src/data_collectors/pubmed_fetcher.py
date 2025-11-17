"""
PubMed data fetcher for collecting publication information.
"""
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional
import requests
from urllib.parse import urlencode
import json
from pathlib import Path

from ..config import APIConfig, PubMedConfig, CACHE_DIR
from ..models import Publication, AuthorPublicationProfile, PublicationMetrics
from ..utils import (
    RateLimiter,
    retry_with_backoff,
    async_retry_with_backoff,
    setup_logger,
    ProgressLogger,
    validate_pmid,
)

logger = setup_logger(__name__)


class PubMedFetcher:
    """
    PubMed API wrapper for fetching publication data.

    Implements rate limiting, caching, and robust error handling.
    """

    def __init__(self, api_key: Optional[str] = None, cache_enabled: bool = True):
        """
        Initialize PubMed fetcher.

        Args:
            api_key: NCBI API key (optional, increases rate limit)
            cache_enabled: Whether to use caching
        """
        self.api_key = api_key or APIConfig.NCBI_API_KEY
        self.base_url = APIConfig.NCBI_BASE_URL
        self.rate_limiter = RateLimiter(APIConfig.NCBI_RATE_LIMIT)
        self.cache_enabled = cache_enabled
        self.cache_dir = CACHE_DIR / 'pubmed'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BiologyMatchBot/1.0 (Academic Research)'
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

            # Check if cache is expired (7 days)
            cached_time = datetime.fromisoformat(cached['timestamp'])
            if (datetime.now() - cached_time).days > 7:
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
    def _make_request(self, endpoint: str, params: Dict) -> str:
        """
        Make API request with rate limiting and retries.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Response text
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        # Add API key if available
        if self.api_key:
            params['api_key'] = self.api_key

        url = f"{self.base_url}{endpoint}"
        response = self.session.get(
            url,
            params=params,
            timeout=APIConfig.REQUEST_TIMEOUT
        )
        response.raise_for_status()

        return response.text

    def search_author(
        self,
        author_name: str,
        affiliation: Optional[str] = None,
        start_year: int = None,
        max_results: int = 100
    ) -> List[str]:
        """
        Search for publications by author.

        Args:
            author_name: Author name to search
            affiliation: Institution affiliation (optional)
            start_year: Start year for publications
            max_results: Maximum number of results

        Returns:
            List[str]: List of PubMed IDs
        """
        # Build search query
        query_parts = [f"{author_name}[Author]"]

        if affiliation:
            query_parts.append(f"{affiliation}[Affiliation]")

        if start_year:
            query_parts.append(f"{start_year}:3000[Publication Date]")

        query = " AND ".join(query_parts)

        # Check cache
        cache_key = f"search_{hash(query)}_{max_results}"
        cached = self._get_cache(cache_key)
        if cached:
            logger.info(f"Cache hit for search: {author_name}")
            return cached

        logger.info(f"Searching PubMed for: {author_name}")

        # Make request
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
        }

        try:
            response_text = self._make_request('esearch.fcgi', params)
            response_data = json.loads(response_text)

            pmids = response_data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(pmids)} publications for {author_name}")

            # Cache result
            self._set_cache(cache_key, pmids)

            return pmids

        except Exception as e:
            logger.error(f"Error searching for {author_name}: {e}")
            return []

    def fetch_article_details(self, pmid_list: List[str]) -> List[Publication]:
        """
        Fetch detailed article information for a list of PMIDs.

        Args:
            pmid_list: List of PubMed IDs

        Returns:
            List[Publication]: List of Publication objects
        """
        if not pmid_list:
            return []

        # Process in batches
        batch_size = PubMedConfig.BATCH_SIZE
        all_publications = []

        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i:i + batch_size]
            publications = self._fetch_batch_details(batch)
            all_publications.extend(publications)

        return all_publications

    def _fetch_batch_details(self, pmid_list: List[str]) -> List[Publication]:
        """
        Fetch details for a batch of PMIDs.

        Args:
            pmid_list: List of PubMed IDs

        Returns:
            List[Publication]: List of Publication objects
        """
        # Check cache
        cache_key = f"details_{'_'.join(pmid_list[:5])}"
        cached = self._get_cache(cache_key)
        if cached:
            logger.debug(f"Cache hit for batch of {len(pmid_list)} articles")
            return [self._dict_to_publication(p) for p in cached]

        logger.info(f"Fetching details for {len(pmid_list)} articles")

        params = {
            'db': 'pubmed',
            'id': ','.join(pmid_list),
            'retmode': 'xml',
        }

        try:
            response_text = self._make_request('efetch.fcgi', params)
            publications = self._parse_pubmed_xml(response_text)

            # Cache result
            self._set_cache(cache_key, [p.to_dict() for p in publications])

            return publications

        except Exception as e:
            logger.error(f"Error fetching article details: {e}")
            return []

    def _parse_pubmed_xml(self, xml_text: str) -> List[Publication]:
        """
        Parse PubMed XML response into Publication objects.

        Args:
            xml_text: XML response text

        Returns:
            List[Publication]: Parsed publications
        """
        publications = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall('.//PubmedArticle'):
                try:
                    pub = self._parse_article_xml(article)
                    if pub:
                        publications.append(pub)
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")

        return publications

    def _parse_article_xml(self, article_elem) -> Optional[Publication]:
        """
        Parse a single article XML element.

        Args:
            article_elem: XML element for article

        Returns:
            Publication object or None
        """
        try:
            # PMID
            pmid_elem = article_elem.find('.//PMID')
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text

            # Title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_parts = []
            for abstract_elem in article_elem.findall('.//AbstractText'):
                if abstract_elem.text:
                    abstract_parts.append(abstract_elem.text)
            abstract = " ".join(abstract_parts)

            # Journal
            journal_elem = article_elem.find('.//Journal/Title')
            journal = journal_elem.text if journal_elem is not None else ""

            # Publication date
            pub_date = self._parse_publication_date(article_elem)

            # Authors
            authors = self._parse_authors(article_elem)

            # Keywords
            keywords = []
            for keyword_elem in article_elem.findall('.//Keyword'):
                if keyword_elem.text:
                    keywords.append(keyword_elem.text)

            # MeSH terms
            mesh_terms = []
            for mesh_elem in article_elem.findall('.//MeshHeading/DescriptorName'):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)

            # DOI
            doi = None
            for id_elem in article_elem.findall('.//ArticleId'):
                if id_elem.get('IdType') == 'doi':
                    doi = id_elem.text
                    break

            # Grant IDs
            grant_ids = []
            for grant_elem in article_elem.findall('.//Grant/GrantID'):
                if grant_elem.text:
                    grant_ids.append(grant_elem.text)

            return Publication(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                publication_date=pub_date,
                keywords=keywords,
                mesh_terms=mesh_terms,
                doi=doi,
                grant_ids=grant_ids,
                citation_count=0,  # Would need separate API call
            )

        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None

    def _parse_publication_date(self, article_elem) -> datetime:
        """Parse publication date from article XML."""
        try:
            pub_date_elem = article_elem.find('.//PubDate')
            if pub_date_elem is None:
                return datetime.now()

            year = pub_date_elem.find('Year')
            month = pub_date_elem.find('Month')
            day = pub_date_elem.find('Day')

            year_val = int(year.text) if year is not None else datetime.now().year
            month_val = self._parse_month(month.text) if month is not None else 1
            day_val = int(day.text) if day is not None else 1

            return datetime(year_val, month_val, day_val)

        except Exception:
            return datetime.now()

    def _parse_month(self, month_str: str) -> int:
        """Convert month string to number."""
        months = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
            'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
            'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
        return months.get(month_str[:3], 1)

    def _parse_authors(self, article_elem) -> List[Dict[str, str]]:
        """Parse author information from article XML."""
        authors = []

        author_list = article_elem.findall('.//Author')
        for i, author_elem in enumerate(author_list):
            last_name = author_elem.find('LastName')
            fore_name = author_elem.find('ForeName')
            affiliation = author_elem.find('.//Affiliation')

            if last_name is not None and fore_name is not None:
                full_name = f"{fore_name.text} {last_name.text}"
                authors.append({
                    'name': full_name,
                    'affiliation': affiliation.text if affiliation is not None else "",
                    'is_first': i == 0,
                    'is_last': i == len(author_list) - 1,
                })

        return authors

    def _dict_to_publication(self, data: Dict) -> Publication:
        """Convert dictionary to Publication object."""
        return Publication(
            pmid=data['pmid'],
            title=data['title'],
            abstract=data['abstract'],
            authors=data['authors'],
            journal=data['journal'],
            publication_date=datetime.fromisoformat(data['publication_date']),
            keywords=data.get('keywords', []),
            mesh_terms=data.get('mesh_terms', []),
            doi=data.get('doi'),
            grant_ids=data.get('grant_ids', []),
            citation_count=data.get('citation_count', 0),
        )

    def fetch_author_publications(
        self,
        author_name: str,
        affiliation: Optional[str] = None,
        years: int = 5
    ) -> AuthorPublicationProfile:
        """
        Fetch complete publication profile for an author.

        Args:
            author_name: Author name
            affiliation: Institution affiliation
            years: Number of years to look back

        Returns:
            AuthorPublicationProfile: Complete author profile
        """
        logger.info(f"Fetching publication profile for {author_name}")

        # Calculate start year
        start_year = datetime.now().year - years

        # Search for publications
        pmids = self.search_author(
            author_name,
            affiliation=affiliation,
            start_year=start_year,
            max_results=PubMedConfig.DEFAULT_MAX_RESULTS
        )

        # Fetch details
        publications = self.fetch_article_details(pmids)

        # Calculate metrics
        total_citations = sum(pub.citation_count for pub in publications)

        # Research evolution
        research_evolution = self._calculate_research_evolution(publications)

        # Collaboration network
        collaboration_network = self._calculate_collaborations(publications, author_name)

        # Calculate h-index
        sorted_pubs = sorted(publications, key=lambda p: p.citation_count, reverse=True)
        h_index = 0
        for i, pub in enumerate(sorted_pubs, 1):
            if pub.citation_count >= i:
                h_index = i
            else:
                break

        return AuthorPublicationProfile(
            author_name=author_name,
            author_variants=[author_name],  # TODO: Add name variants
            affiliation=affiliation or "",
            publications=publications,
            total_citations=total_citations,
            h_index=h_index,
            research_evolution=research_evolution,
            collaboration_network=collaboration_network,
        )

    def _calculate_research_evolution(
        self,
        publications: List[Publication]
    ) -> Dict[int, List[str]]:
        """Calculate research topic evolution by year."""
        evolution = {}

        for pub in publications:
            year = pub.publication_date.year
            if year not in evolution:
                evolution[year] = []

            topics = pub.extract_research_topics()
            evolution[year].extend(topics)

        # Deduplicate topics per year
        for year in evolution:
            evolution[year] = list(set(evolution[year]))

        return evolution

    def _calculate_collaborations(
        self,
        publications: List[Publication],
        main_author: str
    ) -> Dict[str, int]:
        """Calculate collaboration network."""
        collaborations = {}

        for pub in publications:
            for author in pub.authors:
                name = author['name']
                if name != main_author:
                    collaborations[name] = collaborations.get(name, 0) + 1

        return collaborations

    async def batch_fetch_multiple_authors(
        self,
        author_list: List[Dict[str, str]]
    ) -> List[AuthorPublicationProfile]:
        """
        Efficiently fetch data for multiple authors using async.

        Args:
            author_list: List of dicts with 'name' and 'affiliation'

        Returns:
            List[AuthorPublicationProfile]: List of author profiles
        """
        logger.info(f"Fetching publications for {len(author_list)} authors")

        profiles = []
        for author_data in author_list:
            profile = self.fetch_author_publications(
                author_data['name'],
                affiliation=author_data.get('affiliation'),
                years=5
            )
            profiles.append(profile)

        return profiles

    def export_to_json(self, profile: AuthorPublicationProfile, filepath: str):
        """Export profile to JSON file."""
        with open(filepath, 'w') as f:
            f.write(profile.to_json())
        logger.info(f"Exported profile to {filepath}")

    def export_to_csv(self, publications: List[Publication], filepath: str):
        """Export publications to CSV file."""
        import csv

        with open(filepath, 'w', newline='') as f:
            if not publications:
                return

            fieldnames = [
                'pmid', 'title', 'journal', 'publication_date',
                'doi', 'citation_count'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for pub in publications:
                writer.writerow({
                    'pmid': pub.pmid,
                    'title': pub.title,
                    'journal': pub.journal,
                    'publication_date': pub.publication_date.isoformat(),
                    'doi': pub.doi or '',
                    'citation_count': pub.citation_count,
                })

        logger.info(f"Exported {len(publications)} publications to {filepath}")
