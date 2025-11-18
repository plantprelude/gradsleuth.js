"""
Grant data models for NIH funding tracking.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class Grant:
    """Represents a single NIH grant."""

    project_number: str
    title: str
    abstract: str
    pi_name: str
    pi_institution: str
    co_investigators: List[Dict[str, str]] = field(default_factory=list)
    total_cost: float = 0.0
    direct_cost: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    funding_agency: str = "NIH"
    activity_code: str = ""  # R01, R21, etc.
    study_section: str = ""
    keywords: List[str] = field(default_factory=list)
    publications_reported: List[str] = field(default_factory=list)  # PMIDs

    def is_active(self) -> bool:
        """
        Check if grant is currently active.

        Returns:
            bool: True if grant is currently active
        """
        if not self.start_date or not self.end_date:
            return False

        now = datetime.now()
        return self.start_date <= now <= self.end_date

    def months_remaining(self) -> int:
        """
        Calculate remaining grant period.

        Returns:
            int: Months remaining (0 if expired or not started)
        """
        if not self.end_date:
            return 0

        now = datetime.now()
        if now > self.end_date:
            return 0

        delta = self.end_date - now
        return int(delta.days / 30)

    def yearly_budget(self) -> float:
        """
        Calculate average yearly funding.

        Returns:
            float: Average yearly budget
        """
        if not self.start_date or not self.end_date or self.total_cost == 0:
            return 0.0

        duration_years = (self.end_date - self.start_date).days / 365.25
        if duration_years <= 0:
            return 0.0

        return self.total_cost / duration_years

    def grant_duration_years(self) -> float:
        """
        Calculate total grant duration in years.

        Returns:
            float: Duration in years
        """
        if not self.start_date or not self.end_date:
            return 0.0

        return (self.end_date - self.start_date).days / 365.25

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'project_number': self.project_number,
            'title': self.title,
            'abstract': self.abstract,
            'pi_name': self.pi_name,
            'pi_institution': self.pi_institution,
            'co_investigators': self.co_investigators,
            'total_cost': self.total_cost,
            'direct_cost': self.direct_cost,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'funding_agency': self.funding_agency,
            'activity_code': self.activity_code,
            'study_section': self.study_section,
            'keywords': self.keywords,
            'publications_reported': self.publications_reported,
            'is_active': self.is_active(),
            'months_remaining': self.months_remaining(),
            'yearly_budget': self.yearly_budget(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class InvestigatorFundingProfile:
    """Complete funding profile for an investigator."""

    investigator_name: str
    total_funding: float
    active_grants: List[Grant]
    completed_grants: List[Grant]
    success_rate: float
    funding_gaps: List[Tuple[datetime, datetime]]
    frequent_collaborators: List[Tuple[str, int]]
    research_topics_funded: Dict[str, float]  # topic -> total funding

    def calculate_funding_stability(self) -> float:
        """
        Calculate funding stability score (0-1).

        A higher score indicates more consistent funding with fewer/shorter gaps.

        Returns:
            float: Stability score between 0 and 1
        """
        if not self.active_grants and not self.completed_grants:
            return 0.0

        all_grants = self.active_grants + self.completed_grants
        if not all_grants:
            return 0.0

        # Calculate total time span
        valid_grants = [g for g in all_grants if g.start_date and g.end_date]
        if not valid_grants:
            return 0.0

        earliest = min(g.start_date for g in valid_grants)
        latest = max(g.end_date for g in valid_grants)
        total_span_days = (latest - earliest).days

        if total_span_days <= 0:
            return 0.0

        # Calculate total funded days
        funded_days = sum(
            (g.end_date - g.start_date).days
            for g in valid_grants
        )

        # Calculate gap penalty
        total_gap_days = sum(
            (end - start).days for start, end in self.funding_gaps
        )

        # Stability score: ratio of funded time to total span, minus gap penalty
        funded_ratio = funded_days / total_span_days
        gap_penalty = min(total_gap_days / total_span_days, 0.5)

        return max(0.0, min(1.0, funded_ratio - gap_penalty))

    def get_funding_trend(self) -> Dict[int, float]:
        """
        Get funding trend by year.

        Returns:
            Dict[int, float]: Year -> total funding amount
        """
        trend = {}
        all_grants = self.active_grants + self.completed_grants

        for grant in all_grants:
            if not grant.start_date or not grant.end_date:
                continue

            yearly_amount = grant.yearly_budget()
            start_year = grant.start_date.year
            end_year = grant.end_date.year

            for year in range(start_year, end_year + 1):
                trend[year] = trend.get(year, 0.0) + yearly_amount

        return trend

    def get_top_collaborators(self, n: int = 10) -> List[Tuple[str, int]]:
        """
        Get top N collaborators by frequency.

        Args:
            n: Number of top collaborators to return

        Returns:
            List[Tuple[str, int]]: List of (name, count) tuples
        """
        return self.frequent_collaborators[:n]

    def has_active_funding(self) -> bool:
        """Check if investigator has any active grants."""
        return len(self.active_grants) > 0

    def avg_grant_size(self) -> float:
        """Calculate average grant size."""
        all_grants = self.active_grants + self.completed_grants
        if not all_grants:
            return 0.0

        total = sum(g.total_cost for g in all_grants)
        return total / len(all_grants)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'investigator_name': self.investigator_name,
            'total_funding': self.total_funding,
            'active_grants': [g.to_dict() for g in self.active_grants],
            'completed_grants': [g.to_dict() for g in self.completed_grants],
            'success_rate': self.success_rate,
            'funding_gaps': [
                (start.isoformat(), end.isoformat())
                for start, end in self.funding_gaps
            ],
            'frequent_collaborators': [
                {'name': name, 'count': count}
                for name, count in self.frequent_collaborators
            ],
            'research_topics_funded': self.research_topics_funded,
            'funding_stability': self.calculate_funding_stability(),
            'has_active_funding': self.has_active_funding(),
            'avg_grant_size': self.avg_grant_size(),
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
