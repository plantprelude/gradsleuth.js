"""
Data validation utilities.
"""
import re
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_email(email: str) -> bool:
    """
    Validate email address format.

    Args:
        email: Email address to validate

    Returns:
        bool: True if valid email format
    """
    if not email:
        return False

    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_pmid(pmid: str) -> bool:
    """
    Validate PubMed ID format.

    Args:
        pmid: PubMed ID to validate

    Returns:
        bool: True if valid PMID format
    """
    if not pmid:
        return False

    # PMIDs are numeric strings
    return bool(re.match(r'^\d+$', pmid))


def validate_doi(doi: str) -> bool:
    """
    Validate DOI format.

    Args:
        doi: DOI to validate

    Returns:
        bool: True if valid DOI format
    """
    if not doi:
        return False

    # DOI pattern: 10.XXXX/...
    pattern = r'^10\.\d{4,}/[-._;()/:\w]+$'
    return bool(re.match(pattern, doi))


def validate_nih_project_number(project_number: str) -> bool:
    """
    Validate NIH project number format.

    Args:
        project_number: NIH project number

    Returns:
        bool: True if valid format

    Example valid formats:
        - 5R01CA123456-05
        - R21AI098765
    """
    if not project_number:
        return False

    # NIH project numbers have specific patterns
    pattern = r'^\d?[A-Z]\d{2}[A-Z]{2}\d{6}(-\d{2})?$'
    return bool(re.match(pattern, project_number))


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        bool: True if valid URL format
    """
    if not url:
        return False

    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url, re.IGNORECASE))


def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    """
    Validate that date range is logical.

    Args:
        start_date: Start date
        end_date: End date

    Returns:
        bool: True if start is before end
    """
    if not start_date or not end_date:
        return False

    return start_date <= end_date


def validate_grant_data(grant_data: Dict[str, Any]) -> List[str]:
    """
    Validate grant data structure.

    Args:
        grant_data: Grant data dictionary

    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    required_fields = ['project_number', 'title', 'pi_name']
    for field in required_fields:
        if field not in grant_data or not grant_data[field]:
            errors.append(f"Missing required field: {field}")

    # Validate project number format
    if 'project_number' in grant_data:
        if not validate_nih_project_number(grant_data['project_number']):
            errors.append(f"Invalid project number format: {grant_data['project_number']}")

    # Validate dates
    if 'start_date' in grant_data and 'end_date' in grant_data:
        try:
            start = datetime.fromisoformat(grant_data['start_date'])
            end = datetime.fromisoformat(grant_data['end_date'])
            if not validate_date_range(start, end):
                errors.append("End date must be after start date")
        except (ValueError, TypeError):
            errors.append("Invalid date format")

    # Validate costs
    if 'total_cost' in grant_data:
        try:
            cost = float(grant_data['total_cost'])
            if cost < 0:
                errors.append("Total cost cannot be negative")
        except (ValueError, TypeError):
            errors.append("Invalid total cost value")

    return errors


def validate_publication_data(pub_data: Dict[str, Any]) -> List[str]:
    """
    Validate publication data structure.

    Args:
        pub_data: Publication data dictionary

    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    required_fields = ['pmid', 'title', 'journal']
    for field in required_fields:
        if field not in pub_data or not pub_data[field]:
            errors.append(f"Missing required field: {field}")

    # Validate PMID
    if 'pmid' in pub_data:
        if not validate_pmid(pub_data['pmid']):
            errors.append(f"Invalid PMID format: {pub_data['pmid']}")

    # Validate DOI if present
    if 'doi' in pub_data and pub_data['doi']:
        if not validate_doi(pub_data['doi']):
            errors.append(f"Invalid DOI format: {pub_data['doi']}")

    # Validate date
    if 'publication_date' in pub_data:
        try:
            datetime.fromisoformat(pub_data['publication_date'])
        except (ValueError, TypeError):
            errors.append("Invalid publication date format")

    # Validate citation count
    if 'citation_count' in pub_data:
        try:
            count = int(pub_data['citation_count'])
            if count < 0:
                errors.append("Citation count cannot be negative")
        except (ValueError, TypeError):
            errors.append("Invalid citation count value")

    return errors


def validate_faculty_data(faculty_data: Dict[str, Any]) -> List[str]:
    """
    Validate faculty data structure.

    Args:
        faculty_data: Faculty data dictionary

    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    errors = []

    # Required fields
    required_fields = ['name', 'institution', 'department']
    for field in required_fields:
        if field not in faculty_data or not faculty_data[field]:
            errors.append(f"Missing required field: {field}")

    # Validate email if present
    if 'email' in faculty_data and faculty_data['email']:
        if not validate_email(faculty_data['email']):
            errors.append(f"Invalid email format: {faculty_data['email']}")

    # Validate URLs if present
    url_fields = ['lab_website', 'personal_website', 'profile_url']
    for field in url_fields:
        if field in faculty_data and faculty_data[field]:
            if not validate_url(faculty_data[field]):
                errors.append(f"Invalid URL format for {field}: {faculty_data[field]}")

    return errors


def sanitize_text(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize text input by removing/escaping dangerous characters.

    Args:
        text: Text to sanitize
        max_length: Maximum length (truncate if exceeded)

    Returns:
        str: Sanitized text
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace('\x00', '')

    # Normalize whitespace
    text = ' '.join(text.split())

    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length] + '...'

    return text


def normalize_name(name: str) -> str:
    """
    Normalize person name for matching.

    Args:
        name: Name to normalize

    Returns:
        str: Normalized name
    """
    if not name:
        return ""

    # Remove titles
    titles = ['Dr.', 'Prof.', 'Mr.', 'Ms.', 'Mrs.', 'Ph.D.', 'M.D.']
    for title in titles:
        name = name.replace(title, '')

    # Normalize whitespace
    name = ' '.join(name.split())

    # Remove special characters
    name = re.sub(r'[^\w\s-]', '', name)

    return name.strip()


def extract_name_variants(full_name: str) -> List[str]:
    """
    Extract possible name variants for fuzzy matching.

    Args:
        full_name: Full name

    Returns:
        List[str]: List of name variants

    Example:
        "John Michael Smith" -> ["John Smith", "J Smith", "JM Smith", "Smith J", etc.]
    """
    variants = [full_name]

    parts = full_name.split()
    if len(parts) < 2:
        return variants

    # First + Last
    if len(parts) >= 2:
        variants.append(f"{parts[0]} {parts[-1]}")

    # First initial + Last
    variants.append(f"{parts[0][0]} {parts[-1]}")

    # All initials + Last
    if len(parts) >= 3:
        initials = ''.join(p[0] for p in parts[:-1])
        variants.append(f"{initials} {parts[-1]}")

    # Last + First initial
    variants.append(f"{parts[-1]} {parts[0][0]}")

    # Last, First
    variants.append(f"{parts[-1]}, {parts[0]}")

    return list(set(variants))


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    Generic validation for required fields.

    Args:
        data: Data dictionary
        required_fields: List of required field names

    Returns:
        List[str]: List of missing fields
    """
    missing = []
    for field in required_fields:
        if field not in data or data[field] is None or data[field] == '':
            missing.append(field)
    return missing


def is_valid_year(year: int) -> bool:
    """
    Validate that year is reasonable.

    Args:
        year: Year to validate

    Returns:
        bool: True if valid year
    """
    current_year = datetime.now().year
    return 1900 <= year <= current_year + 5
