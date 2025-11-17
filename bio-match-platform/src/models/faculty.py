"""
Faculty data models for profile information.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


@dataclass
class FacultyInfo:
    """Basic faculty information scraped from university websites."""

    faculty_id: str
    name: str
    title: str
    department: str
    institution: str
    email: Optional[str] = None
    research_interests: str = ""
    lab_website: Optional[str] = None
    personal_website: Optional[str] = None
    education_history: List[Dict[str, str]] = field(default_factory=list)
    lab_members: List[str] = field(default_factory=list)
    recent_news: List[Dict[str, str]] = field(default_factory=list)
    office_location: Optional[str] = None
    phone: Optional[str] = None
    photo_url: Optional[str] = None
    profile_url: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'faculty_id': self.faculty_id,
            'name': self.name,
            'title': self.title,
            'department': self.department,
            'institution': self.institution,
            'email': self.email,
            'research_interests': self.research_interests,
            'lab_website': self.lab_website,
            'personal_website': self.personal_website,
            'education_history': self.education_history,
            'lab_members': self.lab_members,
            'recent_news': self.recent_news,
            'office_location': self.office_location,
            'phone': self.phone,
            'photo_url': self.photo_url,
            'profile_url': self.profile_url,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def extract_keywords(self) -> List[str]:
        """
        Extract keywords from research interests.

        Returns:
            List[str]: Extracted keywords
        """
        import re

        # Simple keyword extraction from research interests
        text = self.research_interests.lower()

        # Remove common stop words
        stop_words = {
            'the', 'and', 'of', 'in', 'to', 'a', 'is', 'for', 'on', 'with',
            'as', 'by', 'at', 'from', 'or', 'an', 'we', 'our', 'my',
        }

        # Split into words and filter
        words = re.findall(r'\b[a-z]{4,}\b', text)
        keywords = [w for w in words if w not in stop_words]

        # Return unique keywords (most frequent first)
        from collections import Counter
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(20)]


@dataclass
class ResearchTrajectory:
    """Tracks evolution of research focus over time."""

    topics_by_year: Dict[int, List[str]]
    major_transitions: List[Dict[str, any]]
    current_focus: List[str]
    emerging_interests: List[str]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'topics_by_year': self.topics_by_year,
            'major_transitions': self.major_transitions,
            'current_focus': self.current_focus,
            'emerging_interests': self.emerging_interests,
        }


@dataclass
class NetworkGraph:
    """Collaboration network representation."""

    nodes: List[Dict[str, any]]  # {id, name, type}
    edges: List[Dict[str, any]]  # {source, target, weight}
    metrics: Dict[str, float]  # centrality, clustering, etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'metrics': self.metrics,
        }


@dataclass
class FundingMetrics:
    """Aggregated funding metrics."""

    total_funding: float
    active_grants_count: int
    completed_grants_count: int
    success_rate: float
    avg_grant_size: float
    funding_trend: Dict[int, float]  # year -> total funding

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'total_funding': self.total_funding,
            'active_grants_count': self.active_grants_count,
            'completed_grants_count': self.completed_grants_count,
            'success_rate': self.success_rate,
            'avg_grant_size': self.avg_grant_size,
            'funding_trend': self.funding_trend,
        }
