"""Mock search backend that returns simulated results from a JSON fixture file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from research_agent.models import Source

# Default path to the fixture file (relative to this module's directory)
_DEFAULT_FIXTURE = (
    Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "mock_results.json"
)


class MockSearchBackend:
    """Simulates a search engine using a local JSON fixture file.

    The fixture file must be a JSON object where each key is a keyword
    (matched against the query by substring) and the value is a list of
    source dicts.  A ``"default"`` key is used as a fallback.

    Example fixture structure::

        {
            "climate change": [{"url": "...", "title": "...", ...}],
            "default": [{"url": "...", "title": "...", ...}]
        }
    """

    def __init__(self, fixture_path: Optional[Path] = None) -> None:
        self.fixture_path: Path = fixture_path or _DEFAULT_FIXTURE
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load (or reload) the fixture data from disk."""
        if not self.fixture_path.exists():
            raise FileNotFoundError(
                f"Mock search fixture not found: {self.fixture_path}"
            )
        with self.fixture_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(
                "Fixture must be a JSON object mapping query keywords to result lists."
            )
        self._data = raw

    def _find_results(
        self, query: str
    ) -> List[Dict[str, Any]]:
        """Return the best-matching result list for *query*."""
        query_lower = query.lower()
        for key, results in self._data.items():
            if key == "default":
                continue
            if key.lower() in query_lower or query_lower in key.lower():
                return results
        return self._data.get("default", [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Source]:
        """Return up to *top_k* :class:`Source` objects for *query*.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            A list of :class:`~research_agent.models.Source` instances.
        """
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")

        raw_results = self._find_results(query)[:top_k]
        sources: List[Source] = []
        for item in raw_results:
            # Allow fixture entries to omit optional fields gracefully
            source = Source(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                domain=item.get("domain", ""),
                published_at=item.get("published_at"),
                domain_authority=item.get("domain_authority"),
                citation_count=item.get("citation_count"),
                raw_metadata=item.get("raw_metadata", {}),
            )
            sources.append(source)
        return sources

    def available_keys(self) -> List[str]:
        """Return all keyword keys present in the fixture (excluding 'default')."""
        return [k for k in self._data if k != "default"]
