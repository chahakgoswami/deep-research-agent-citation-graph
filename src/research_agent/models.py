"""Core data models for the Research Agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ResearchQuery(BaseModel):
    """Represents the initial research query submitted by the user."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(..., min_length=1, description="The research question or topic.")
    max_hops: int = Field(default=3, ge=1, le=10, description="Maximum reasoning hops.")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query text must not be blank.")
        return v.strip()

    class Config:
        frozen = False


class Source(BaseModel):
    """Represents a single retrieved web source."""

    url: str = Field(..., description="URL of the source.")
    title: str = Field(default="", description="Page title.")
    snippet: str = Field(default="", description="Short excerpt from the source.")
    domain: str = Field(default="", description="Domain extracted from the URL.")
    published_at: Optional[datetime] = Field(None, description="Publication date.")
    domain_authority: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Simulated domain authority score."
    )
    citation_count: Optional[int] = Field(
        None, ge=0, description="Simulated citation count."
    )
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("domain", mode="before")
    @classmethod
    def extract_domain(cls, v: str, info: Any) -> str:
        """Auto-extract domain from URL if domain field is empty."""
        if v:
            return v
        # Try to get url from the data being validated
        try:
            url = info.data.get("url", "")
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or ""
        except Exception:
            return v

    class Config:
        frozen = False


class Hop(BaseModel):
    """Represents one reasoning hop in the multi-hop search chain."""

    hop_index: int = Field(..., ge=0, description="Zero-based hop index.")
    sub_query: str = Field(..., description="The sub-question explored in this hop.")
    sources: List[Source] = Field(default_factory=list)
    parent_query_id: str = Field(..., description="ID of the originating ResearchQuery.")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        frozen = False


class CitationNode(BaseModel):
    """A node in the citation graph, wrapping a Source with graph metadata."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: Source
    hop_index: int = Field(..., ge=0)
    grade: Optional[str] = Field(
        None, description="Letter grade assigned by SourceGrader (A-F)."
    )
    confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Grader confidence score."
    )
    claim_snippet: str = Field(
        default="", description="Key claim extracted from this source."
    )
    contradicts: List[str] = Field(
        default_factory=list,
        description="IDs of CitationNodes this node contradicts.",
    )
    supports: List[str] = Field(
        default_factory=list,
        description="IDs of CitationNodes this node supports.",
    )

    class Config:
        frozen = False
