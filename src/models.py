"""
Core Pydantic data schemas for the Fact Knowledge Layer.
Generic dynamic fact representation, relationship types, and document metadata.
"""

from enum import Enum
from typing import Optional, Union, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class RelationshipType(str, Enum):
    """Classification of cross-document fact relationships."""
    CORROBORATES = "CORROBORATES"
    CONTRADICTS = "CONTRADICTS"
    RECONCILABLE = "RECONCILABLE"


class Fact(BaseModel):
    """Generic dynamic representation of an extracted, grounded fact."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = Field(description="Dynamic entity name, e.g. Delhivery Limited, India, Reserve Bank of India")
    predicate: str = Field(description="Dynamic attribute name in snake_case, e.g. revenue_from_services, pin_codes_covered")
    raw_value: str = Field(description="Exact value as written in the source document, e.g. '₹8,142 Cr' or '98,135'")
    normalized_value: Optional[Union[float, str]] = Field(default=None, description="Standardized numeric or canonical text for comparison")
    value_type: str = Field(default="numeric", description="Category: numeric, currency, percentage, text, status, date")
    unit: Optional[str] = Field(default=None, description="Normalized measurement unit, e.g. 'INR crore', 'metric tonnes', 'count'")
    time_period: Optional[str] = Field(default=None, description="Temporal context, e.g. 'FY2023-24', 'Q4 FY24', 'as of March 31, 2024'")
    scope: Optional[str] = Field(default=None, description="Contextual boundary, e.g. 'consolidated', 'standalone', 'including partner agents'")
    qualifiers: Optional[str] = Field(default=None, description="Modifiers such as 'approximate', 'pro forma', 'projected'")
    comparison_key: str = Field(default="", description="Content-derived key for candidate matching, e.g. 'delhivery_limited|revenue_from_services'")
    
    # Source & Grounding
    source_document_id: str = Field(description="Stable hash/id of the source document")
    source_filename: str = Field(description="Original filename of the PDF")
    page_number: int = Field(description="1-based page number where the fact was located")
    evidence_quote: str = Field(description="Direct verbatim or near-verbatim quote from the page text supporting this fact")
    is_grounded: bool = Field(default=True, description="Whether the evidence quote was verified against the page text")
    grounding_score: float = Field(default=1.0, description="Verification confidence between 0.0 and 1.0")
    confidence: float = Field(default=1.0, description="Extraction confidence score between 0.0 and 1.0")
    extraction_notes: Optional[str] = Field(default=None, description="Any warnings or reasoning failure notes")


class CrossDocumentRelationship(BaseModel):
    """Cross-document relationship between two candidate facts."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact_a_id: str
    fact_b_id: str
    relationship_type: RelationshipType
    confidence: float = Field(default=1.0, description="Confidence in relationship classification")
    short_explanation: str = Field(description="Concise 1-2 sentence factual explanation based only on evidence")
    contextual_difference: Optional[str] = Field(default=None, description="Specific contextual factor explaining difference (time, scope, unit, etc.)")
    fact_a: Fact
    fact_b: Fact


class DocumentMetadata(BaseModel):
    """Metadata and processing status for an ingested PDF."""
    id: str
    filename: str
    page_count: int
    processed_pages: int = 0
    extracted_facts_count: int = 0
    grounded_facts_count: int = 0
    status: str = "pending"  # pending, processed, failed, partial
    warnings: List[str] = Field(default_factory=list)


class ExtractionFailureEvent(BaseModel):
    """Structured record of an extraction or reasoning failure."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_filename: str
    page_number: int
    failure_type: str  # e.g., "table_misalignment", "ungrounded_evidence", "low_confidence_match"
    what_happened: str
    why_it_happened: str
    how_handled: str
    future_improvement: str
