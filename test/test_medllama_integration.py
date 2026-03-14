"""
End-to-end integration tests for MedLlama.

Tests the full pipeline: query → RAG retrieval → model generation → API response.
These tests run without GPU by mocking the LLM and using local Qdrant.
"""

import importlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────────
# Test RAG → API integration (no GPU needed)
# ──────────────────────────────────────────────────────

class TestRAGIntegration:
    """Test the RAG pipeline produces proper API-compatible output."""

    def test_rag_orchestrator_output_format(self):
        """RAG output should contain messages and sources compatible with API."""
        retriever_mod = importlib.import_module("src.rag.medllama-hybrid-retrieve")
        orch_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")

        # Mock the retriever to avoid GPU
        with patch.object(retriever_mod.HybridRetriever, "__init__", lambda self, **kw: None):
            with patch.object(orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
                orch = orch_mod.RAGOrchestrator()
                orch.max_iterations = 1
                orch.confidence_threshold = 0.0
                orch.query_expansion_enabled = False

                # Mock retriever
                mock_retriever = MagicMock()
                mock_retriever.final_top_k = 3
                mock_retriever.retrieve.return_value = {
                    "query": "test",
                    "documents": [
                        {
                            "id": "1",
                            "pmid": "12345",
                            "title": "Test Article",
                            "chunk_text": "Test content about medicine",
                            "abstract": "Abstract",
                            "journal": "JAMA",
                            "rrf_score": 0.03,
                        }
                    ],
                    "retrieval_time_ms": 50.0,
                    "stats": {},
                }
                orch.retriever = mock_retriever

                result = orch.run("What is aspirin?", use_rag=True)

        # Verify output structure
        assert "messages" in result
        assert "sources" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) >= 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][1]["role"] == "user"

        # Verify sources format
        for src in result["sources"]:
            assert "pmid" in src
            assert "title" in src
            assert "text" in src
            assert "score" in src

    def test_rag_no_retrieval_for_greeting(self):
        """Greetings should bypass RAG and return simple messages."""
        orch_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")

        with patch.object(orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = orch_mod.RAGOrchestrator()
            result = orch.run("Hello!", use_rag=True)

        assert result["needs_retrieval"] is False
        assert result["sources"] == []
        assert len(result["messages"]) == 2

    def test_rag_disabled(self):
        """With use_rag=False, should skip retrieval entirely."""
        orch_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")

        with patch.object(orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = orch_mod.RAGOrchestrator()
            result = orch.run("What is diabetes?", use_rag=False)

        assert result["needs_retrieval"] is False
        assert result["iterations"] == 0


class TestModelMerge:
    """Test that the merged model checkpoint exists and is valid."""

    def test_merged_checkpoint_exists(self):
        """Merged model directory should contain required files."""
        merged_path = PROJECT_ROOT / "checkpoints" / "merged"
        if not merged_path.exists():
            pytest.skip("Merged checkpoint not found (run adapter merge first)")

        # Check for essential files
        assert (merged_path / "config.json").exists()
        assert (merged_path / "tokenizer.json").exists() or \
               (merged_path / "tokenizer_config.json").exists()

        # Check for model weights
        safetensor_files = list(merged_path.glob("*.safetensors"))
        assert len(safetensor_files) > 0, "No safetensors model files found"

    def test_tokenizer_loads(self):
        """Tokenizer should load from merged checkpoint."""
        merged_path = PROJECT_ROOT / "checkpoints" / "merged"
        if not merged_path.exists():
            pytest.skip("Merged checkpoint not found")

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(merged_path))
        assert tokenizer is not None

        # Test chat template works
        messages = [
            {"role": "system", "content": "You are a medical assistant."},
            {"role": "user", "content": "What is aspirin?"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        assert len(prompt) > 0
        assert "aspirin" in prompt


class TestAPIWiring:
    """Test API server with mocked model backend."""

    @pytest.fixture
    def mock_app(self):
        """Create app with mocked LLM."""
        api_mod = importlib.import_module("src.serving.medllama-api-serve")

        # Mock the state
        api_mod._state["model_loaded"] = True
        api_mod._state["llm_type"] = "mock"
        api_mod._state["qdrant_connected"] = False
        api_mod._state["rag_orchestrator"] = None

        # Patch _generate_response
        original_gen = api_mod._generate_response

        def mock_generate(messages, **kwargs):
            return "This is a test response about medical topics with evidence-based information."

        api_mod._generate_response = mock_generate

        yield api_mod.app

        # Restore
        api_mod._generate_response = original_gen
        api_mod._state["model_loaded"] = False

    @pytest.mark.asyncio
    async def test_chat_with_mock_model(self, mock_app):
        """Chat endpoint should use the model and return structured response."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/chat", json={
                "message": "What is hypertension?",
                "use_rag": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert "medical" in data["response"].lower() or len(data["response"]) > 0

    @pytest.mark.asyncio
    async def test_health_with_model_loaded(self, mock_app):
        """Health should report ok when model is loaded."""
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mock_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True


class TestEndToEndPipeline:
    """Test the complete query→retrieve→format→generate pipeline logic."""

    def test_full_pipeline_logic(self):
        """Simulate the full pipeline without GPU."""
        orch_mod = importlib.import_module("src.rag.medllama-rag-orchestrate")

        # Build messages as the API would
        with patch.object(orch_mod.RAGOrchestrator, "__init__", lambda self, **kw: None):
            orch = orch_mod.RAGOrchestrator()
            messages = orch.build_messages(
                "What is metformin?",
                context="Metformin is a first-line medication for type 2 diabetes."
            )

        # Verify message structure for LLM
        assert messages[0]["role"] == "system"
        assert "MedLlama" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "metformin" in messages[1]["content"].lower()
        assert "Retrieved Medical Literature" in messages[1]["content"]
        assert "type 2 diabetes" in messages[1]["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
