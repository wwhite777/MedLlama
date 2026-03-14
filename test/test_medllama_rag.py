"""
Integration tests for MedLlama RAG pipeline.

Tests hybrid retrieval, RRF merging, and RAG orchestrator logic.
Runs on CPU with local Qdrant (no GPU or Docker required).
"""

import importlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────
# Test RRF merging (pure logic, no external dependencies)
# ──────────────────────────────────────────────────────

class TestRRFMerge:
    """Test Reciprocal Rank Fusion logic."""

    def setup_method(self):
        retriever_mod = importlib.import_module("src.rag.medllama-hybrid-retrieve")
        self.HybridRetriever = retriever_mod.HybridRetriever

    def test_rrf_basic(self):
        """RRF should combine results from both lists."""
        dense = [
            {"id": "1", "score": 0.9, "pmid": "A", "title": "A", "chunk_text": "A", "abstract": "", "journal": ""},
            {"id": "2", "score": 0.8, "pmid": "B", "title": "B", "chunk_text": "B", "abstract": "", "journal": ""},
        ]
        bm25 = [
            {"id": "2", "score": 0.9, "pmid": "B", "title": "B", "chunk_text": "B", "abstract": "", "journal": ""},
            {"id": "3", "score": 0.7, "pmid": "C", "title": "C", "chunk_text": "C", "abstract": "", "journal": ""},
        ]

        # Call static-like: instantiate with mocked init
        with patch.object(self.HybridRetriever, "__init__", lambda self, **kw: None):
            retriever = self.HybridRetriever()
            merged = retriever.rrf_merge(dense, bm25, k=60)

        # Doc "2" appears in both lists, should have highest RRF score
        assert merged[0]["id"] == "2"
        assert len(merged) == 3
        # All docs should have rrf_score
        for doc in merged:
            assert "rrf_score" in doc
            assert doc["rrf_score"] > 0

    def test_rrf_empty_inputs(self):
        """RRF with empty lists should return empty."""
        with patch.object(self.HybridRetriever, "__init__", lambda self, **kw: None):
            retriever = self.HybridRetriever()
            merged = retriever.rrf_merge([], [], k=60)
        assert merged == []

    def test_rrf_disjoint(self):
        """RRF with completely disjoint sets should return all docs."""
        dense = [{"id": "1", "score": 0.9, "pmid": "A", "title": "", "chunk_text": "", "abstract": "", "journal": ""}]
        bm25 = [{"id": "2", "score": 0.8, "pmid": "B", "title": "", "chunk_text": "", "abstract": "", "journal": ""}]

        with patch.object(self.HybridRetriever, "__init__", lambda self, **kw: None):
            retriever = self.HybridRetriever()
            merged = retriever.rrf_merge(dense, bm25, k=60)

        assert len(merged) == 2


# ──────────────────────────────────────────────────────
# Test RAG Orchestrator logic (no models needed)
# ──────────────────────────────────────────────────────

class TestRAGOrchestrator:
    """Test RAG orchestrator query assessment and expansion."""

    def setup_method(self):
        self.orch_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")

    def test_assess_retrieval_greeting(self):
        """Greetings should not trigger retrieval."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            assert orch.assess_retrieval_need("Hello") is False
            assert orch.assess_retrieval_need("hi there") is False
            assert orch.assess_retrieval_need("thanks!") is False

    def test_assess_retrieval_medical(self):
        """Medical queries should trigger retrieval."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            assert orch.assess_retrieval_need("What is the treatment for hypertension?") is True
            assert orch.assess_retrieval_need("Side effects of metformin in elderly patients") is True

    def test_assess_retrieval_short(self):
        """Very short queries should not trigger retrieval."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            assert orch.assess_retrieval_need("ok") is False

    def test_query_expansion(self):
        """Query expansion should generate variants."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            expanded = orch.expand_query("treatment for hypertension")
            assert len(expanded) >= 2
            assert "treatment for hypertension" in expanded

    def test_query_expansion_no_match(self):
        """Query with no matching patterns should return original only."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            expanded = orch.expand_query("describe the human heart")
            assert expanded == ["describe the human heart"]

    def test_build_messages_no_context(self):
        """Messages without RAG context should be simple."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            msgs = orch.build_messages("What is aspirin?")
            assert len(msgs) == 2
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert msgs[1]["content"] == "What is aspirin?"

    def test_build_messages_with_context(self):
        """Messages with RAG context should include literature."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            msgs = orch.build_messages("What is aspirin?", context="Aspirin is an NSAID...")
            assert len(msgs) == 2
            assert "Retrieved Medical Literature" in msgs[1]["content"]
            assert "Aspirin is an NSAID" in msgs[1]["content"]

    def test_build_context(self):
        """Build context should format documents properly."""
        docs = [
            {"pmid": "123", "title": "Test Article", "chunk_text": "Sample text", "journal": "JAMA"},
            {"pmid": "456", "title": "Another", "chunk_text": "More text", "journal": ""},
        ]
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            context = orch.build_context(docs)
            assert "PMID: 123" in context
            assert "JAMA" in context
            assert "Sample text" in context

    def test_build_context_dedup(self):
        """Build context should deduplicate by PMID."""
        docs = [
            {"pmid": "123", "title": "Same", "chunk_text": "Text 1", "journal": ""},
            {"pmid": "123", "title": "Same", "chunk_text": "Text 2", "journal": ""},
        ]
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            context = orch.build_context(docs)
            assert context.count("PMID: 123") == 1

    def test_format_sources(self):
        """Format sources should create proper source dicts."""
        docs = [
            {"pmid": "111", "title": "T1", "chunk_text": "C1", "rrf_score": 0.033},
            {"pmid": "222", "title": "T2", "chunk_text": "C2", "rerank_score": 0.85},
        ]
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            sources = orch.format_sources(docs)
            assert len(sources) == 2
            assert sources[0]["pmid"] == "111"
            assert "score" in sources[0]

    def test_assess_relevance_empty(self):
        """Empty documents should return 0 confidence."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            assert orch.assess_relevance("test query", []) == 0.0

    def test_assess_relevance_good_match(self):
        """Documents with matching keywords should get higher confidence."""
        docs = [
            {"chunk_text": "hypertension treatment blood pressure medication", "rrf_score": 0.03},
        ]
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            score = orch.assess_relevance("treatment for hypertension", docs)
            assert score > 0.3

    def test_run_no_rag(self):
        """Run with use_rag=False should skip retrieval."""
        with patch.object(self.orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = self.orch_mod.RAGOrchestrator()
            result = orch.run("What is aspirin?", use_rag=False)
            assert result["needs_retrieval"] is False
            assert result["sources"] == []
            assert result["iterations"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
