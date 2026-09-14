"""Unit tests for the MockSearchBackend."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from research_agent.backends.mock_search import MockSearchBackend
from research_agent.models import Source

# Path to the shared fixture used by all tests
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_results.json"


@pytest.fixture
def backend() -> MockSearchBackend:
    """Return a MockSearchBackend using the real fixture file."""
    return MockSearchBackend(fixture_path=FIXTURE_PATH)


class TestMockSearchBackendInit:
    def test_loads_fixture_successfully(self, backend: MockSearchBackend):
        assert backend._data is not None
        assert len(backend._data) > 0

    def test_missing_fixture_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            MockSearchBackend(fixture_path=tmp_path / "nonexistent.json")

    def test_invalid_fixture_raises(self, tmp_path: Path):
        bad_fixture = tmp_path / "bad.json"
        bad_fixture.write_text(json.dumps([1, 2, 3]))  # array, not object
        with pytest.raises(ValueError):
            MockSearchBackend(fixture_path=bad_fixture)


class TestMockSearchBackendSearch:
    def test_returns_list_of_sources(self, backend: MockSearchBackend):
        results = backend.search("climate change")
        assert isinstance(results, list)
        assert all(isinstance(r, Source) for r in results)

    def test_climate_change_query_returns_results(self, backend: MockSearchBackend):
        results = backend.search("climate change")
        assert len(results) > 0

    def test_ai_query_returns_results(self, backend: MockSearchBackend):
        results = backend.search("artificial intelligence")
        assert len(results) > 0
        urls = [r.url for r in results]
        assert any("arxiv" in url or "deepmind" in url for url in urls)

    def test_quantum_query_returns_results(self, backend: MockSearchBackend):
        results = backend.search("quantum computing")
        assert len(results) > 0

    def test_unknown_query_returns_default(self, backend: MockSearchBackend):
        results = backend.search("medieval basket weaving techniques")
        assert isinstance(results, list)
        # Should fall back to default entries
        assert len(results) > 0

    def test_top_k_limits_results(self, backend: MockSearchBackend):
        results = backend.search("climate change", top_k=1)
        assert len(results) <= 1

    def test_top_k_zero_returns_empty(self, backend: MockSearchBackend):
        results = backend.search("climate change", top_k=0)
        assert results == []

    def test_empty_query_raises(self, backend: MockSearchBackend):
        with pytest.raises(ValueError):
            backend.search("")

    def test_blank_query_raises(self, backend: MockSearchBackend):
        with pytest.raises(ValueError):
            backend.search("   ")

    def test_source_fields_populated(self, backend: MockSearchBackend):
        results = backend.search("climate change")
        first = results[0]
        assert first.url
        assert first.title
        assert first.snippet
        assert first.domain

    def test_source_domain_authority(self, backend: MockSearchBackend):
        results = backend.search("climate change")
        first = results[0]
        assert first.domain_authority is not None
        assert 0.0 <= first.domain_authority <= 100.0

    def test_source_citation_count(self, backend: MockSearchBackend):
        results = backend.search("climate change")
        first = results[0]
        assert first.citation_count is not None
        assert first.citation_count >= 0

    def test_case_insensitive_matching(self, backend: MockSearchBackend):
        results_lower = backend.search("climate change")
        results_upper = backend.search("CLIMATE CHANGE")
        assert len(results_lower) == len(results_upper)

    def test_partial_query_match(self, backend: MockSearchBackend):
        """A query containing a fixture keyword should match."""
        results = backend.search("recent research on climate change impacts")
        urls = [r.url for r in results]
        assert any("nasa" in u or "ipcc" in u for u in urls)


class TestMockSearchBackendAvailableKeys:
    def test_available_keys_excludes_default(self, backend: MockSearchBackend):
        keys = backend.available_keys()
        assert "default" not in keys

    def test_available_keys_includes_fixture_topics(self, backend: MockSearchBackend):
        keys = backend.available_keys()
        assert "climate change" in keys
        assert "artificial intelligence" in keys
        assert "quantum computing" in keys


class TestMockSearchBackendWithCustomFixture:
    def test_custom_fixture(self, tmp_path: Path):
        fixture = {
            "dogs": [
                {
                    "url": "https://dogfacts.example.com",
                    "title": "Dog Facts",
                    "snippet": "Dogs are domesticated mammals.",
                    "domain": "dogfacts.example.com",
                    "domain_authority": 55.0,
                    "citation_count": 120,
                }
            ],
            "default": [
                {
                    "url": "https://generic.example.com",
                    "title": "Generic Result",
                    "snippet": "A generic fallback result.",
                    "domain": "generic.example.com",
                }
            ],
        }
        fixture_file = tmp_path / "custom.json"
        fixture_file.write_text(json.dumps(fixture))
        b = MockSearchBackend(fixture_path=fixture_file)
        results = b.search("dogs")
        assert len(results) == 1
        assert results[0].url == "https://dogfacts.example.com"

    def test_custom_fixture_fallback(self, tmp_path: Path):
        fixture = {
            "dogs": [{"url": "https://dogfacts.example.com", "title": "Dogs", "snippet": "...", "domain": "dogfacts.example.com"}],
            "default": [{"url": "https://fallback.example.com", "title": "Fallback", "snippet": "fallback", "domain": "fallback.example.com"}],
        }
        fixture_file = tmp_path / "custom.json"
        fixture_file.write_text(json.dumps(fixture))
        b = MockSearchBackend(fixture_path=fixture_file)
        results = b.search("unrelated topic")
        assert results[0].url == "https://fallback.example.com"

    def test_source_with_null_published_at(self, tmp_path: Path):
        fixture = {
            "default": [
                {
                    "url": "https://example.com",
                    "title": "No Date",
                    "snippet": "No publication date.",
                    "domain": "example.com",
                    "published_at": None,
                    "domain_authority": None,
                    "citation_count": None,
                }
            ]
        }
        fixture_file = tmp_path / "nulls.json"
        fixture_file.write_text(json.dumps(fixture))
        b = MockSearchBackend(fixture_path=fixture_file)
        results = b.search("anything")
        assert results[0].published_at is None
        assert results[0].domain_authority is None
        assert results[0].citation_count is None
