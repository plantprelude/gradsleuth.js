"""
Publication data models for research output tracking.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import re
import json


@dataclass
class Publication:
    """Represents a single research publication."""

    pmid: str
    title: str
    abstract: str
    authors: List[Dict[str, str]]  # {name, affiliation, is_first, is_last}
    journal: str
    publication_date: datetime
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    grant_ids: List[str] = field(default_factory=list)
    citation_count: int = 0

    def to_embedding_text(self) -> str:
        """
        Prepare text for embedding generation.

        Returns:
            str: Formatted text combining title, abstract, and keywords
        """
        components = [
            f"Title: {self.title}",
            f"Abstract: {self.abstract}",
        ]

        if self.keywords:
            components.append(f"Keywords: {', '.join(self.keywords)}")

        if self.mesh_terms:
            components.append(f"MeSH Terms: {', '.join(self.mesh_terms)}")

        return "\n".join(components)

    def extract_research_topics(self) -> List[str]:
        """
        Extract key research topics using regex and keyword analysis.

        Returns:
            List[str]: Identified research topics
        """
        topics = set()

        # Add keywords and MeSH terms
        topics.update(self.keywords)
        topics.update(self.mesh_terms)

        # Extract common biology research patterns from title and abstract
        text = f"{self.title} {self.abstract}".lower()

        # Common biology research areas
        patterns = [
            r'\b(cancer|oncology|tumor)\b',
            r'\b(neuroscience|neurological|brain)\b',
            r'\b(immunology|immune system)\b',
            r'\b(genetics|genomics|gene expression)\b',
            r'\b(cell biology|cellular)\b',
            r'\b(molecular biology|molecular)\b',
            r'\b(microbiology|microbiome)\b',
            r'\b(ecology|environmental)\b',
            r'\b(developmental biology)\b',
            r'\b(biochemistry|metabolism)\b',
            r'\b(structural biology|protein structure)\b',
            r'\b(stem cells?)\b',
            r'\b(crispr|gene editing)\b',
            r'\b(proteomics|transcriptomics)\b',
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                match = re.search(pattern, text)
                topics.add(match.group(1))

        return list(topics)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'pmid': self.pmid,
            'title': self.title,
            'abstract': self.abstract,
            'authors': self.authors,
            'journal': self.journal,
            'publication_date': self.publication_date.isoformat(),
            'keywords': self.keywords,
            'mesh_terms': self.mesh_terms,
            'doi': self.doi,
            'grant_ids': self.grant_ids,
            'citation_count': self.citation_count,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class PublicationMetrics:
    """Aggregated publication metrics for an author."""

    total_publications: int
    total_citations: int
    h_index: int
    i10_index: int
    first_author_count: int
    last_author_count: int
    avg_citations_per_paper: float
    publications_per_year: Dict[int, int]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_publications': self.total_publications,
            'total_citations': self.total_citations,
            'h_index': self.h_index,
            'i10_index': self.i10_index,
            'first_author_count': self.first_author_count,
            'last_author_count': self.last_author_count,
            'avg_citations_per_paper': self.avg_citations_per_paper,
            'publications_per_year': self.publications_per_year,
        }


@dataclass
class AuthorPublicationProfile:
    """Complete publication profile for an author."""

    author_name: str
    author_variants: List[str]
    affiliation: str
    publications: List[Publication]
    total_citations: int
    h_index: int
    research_evolution: Dict[int, List[str]]  # year -> topics
    collaboration_network: Dict[str, int]  # co-author -> count

    def get_recent_focus(self, months: int = 12) -> List[str]:
        """
        Identify recent research focus areas.

        Args:
            months: Number of months to look back (default 12)

        Returns:
            List[str]: Recent research topics
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=months * 30)
        recent_pubs = [
            pub for pub in self.publications
            if pub.publication_date >= cutoff_date
        ]

        topics = []
        for pub in recent_pubs:
            topics.extend(pub.extract_research_topics())
            topics.extend(pub.keywords)
            topics.extend(pub.mesh_terms)

        # Count frequency
        topic_freq = {}
        for topic in topics:
            topic_freq[topic] = topic_freq.get(topic, 0) + 1

        # Sort by frequency and return top topics
        sorted_topics = sorted(topic_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:10]]

    def calculate_metrics(self) -> PublicationMetrics:
        """Calculate comprehensive publication metrics."""
        # Sort publications by citation count
        sorted_pubs = sorted(
            self.publications,
            key=lambda p: p.citation_count,
            reverse=True
        )

        # Calculate h-index
        h_index = 0
        for i, pub in enumerate(sorted_pubs, 1):
            if pub.citation_count >= i:
                h_index = i
            else:
                break

        # Calculate i10-index (papers with at least 10 citations)
        i10_index = sum(1 for pub in self.publications if pub.citation_count >= 10)

        # Count first/last author papers
        first_author_count = sum(
            1 for pub in self.publications
            if pub.authors and pub.authors[0].get('is_first', False)
        )
        last_author_count = sum(
            1 for pub in self.publications
            if pub.authors and pub.authors[-1].get('is_last', False)
        )

        # Publications per year
        pubs_per_year = {}
        for pub in self.publications:
            year = pub.publication_date.year
            pubs_per_year[year] = pubs_per_year.get(year, 0) + 1

        # Average citations
        avg_citations = (
            self.total_citations / len(self.publications)
            if self.publications else 0
        )

        return PublicationMetrics(
            total_publications=len(self.publications),
            total_citations=self.total_citations,
            h_index=h_index,
            i10_index=i10_index,
            first_author_count=first_author_count,
            last_author_count=last_author_count,
            avg_citations_per_paper=avg_citations,
            publications_per_year=pubs_per_year,
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'author_name': self.author_name,
            'author_variants': self.author_variants,
            'affiliation': self.affiliation,
            'publications': [pub.to_dict() for pub in self.publications],
            'total_citations': self.total_citations,
            'h_index': self.h_index,
            'research_evolution': self.research_evolution,
            'collaboration_network': self.collaboration_network,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
