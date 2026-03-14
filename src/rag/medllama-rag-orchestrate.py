#!/usr/bin/env python3
"""
MedLlama Agentic RAG Orchestrator

Implements an agentic RAG pipeline that decides whether to retrieve,
expand the query, or answer directly based on confidence assessment.

The orchestrator follows this loop:
1. Assess if retrieval is needed
2. If yes: retrieve → check relevance → optionally expand query and re-retrieve
3. Build augmented prompt with retrieved context
4. Return structured output for the LLM generation step

Usage:
    from importlib import import_module
    orchestrator_mod = import_module("src.rag.medllama-rag-orchestrate")
    rag = orchestrator_mod.RAGOrchestrator(config_path="configs/medllama-rag-config.yaml")
    result = rag.run("What are the side effects of metformin?")
"""

import importlib
import re
import time
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

console = Console()

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "medllama-rag-config.yaml"
)

SYSTEM_PROMPT = (
    "You are MedLlama, a knowledgeable medical assistant trained on medical literature. "
    "You provide accurate, evidence-based medical information. When referencing specific "
    "medical facts, cite the relevant sources. Always note that your responses are for "
    "informational purposes only and should not replace professional medical advice."
)

RAG_CONTEXT_TEMPLATE = """Based on the following medical literature, answer the user's question.

=== Retrieved Medical Literature ===
{context}
=== End of Retrieved Literature ===

Instructions:
- Use the provided literature to support your answer
- Cite sources by their PMID when referencing specific facts
- If the literature doesn't contain relevant information, say so and answer based on your training
- Be precise and evidence-based"""

QUERY_EXPANSION_PATTERNS = {
    "treatment": ["therapy", "management", "intervention", "medication"],
    "diagnosis": ["diagnostic criteria", "differential diagnosis", "clinical presentation"],
    "symptoms": ["clinical manifestations", "signs and symptoms", "presentation"],
    "cause": ["etiology", "pathogenesis", "risk factors"],
    "prognosis": ["outcome", "survival", "mortality"],
    "prevention": ["prophylaxis", "risk reduction", "screening"],
    "mechanism": ["pathophysiology", "mechanism of action", "pharmacodynamics"],
    "drug": ["pharmacotherapy", "medication", "pharmaceutical"],
    "side effect": ["adverse effect", "adverse reaction", "toxicity"],
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RAGOrchestrator:
    """Agentic RAG pipeline with iterative retrieval and query expansion."""

    def __init__(
        self,
        config_path: str = str(DEFAULT_CONFIG_PATH),
        qdrant_local_path: Optional[str] = None,
        load_models: bool = True,
    ):
        self.config = load_config(config_path)
        self.agentic_cfg = self.config.get("agentic_rag", {})
        self.max_iterations = self.agentic_cfg.get("max_iterations", 3)
        self.confidence_threshold = self.agentic_cfg.get("confidence_threshold", 0.7)
        self.query_expansion_enabled = self.agentic_cfg.get("query_expansion", True)

        # Load the hybrid retriever
        retriever_mod = importlib.import_module("src.rag.medllama-hybrid-retrieve")
        self.retriever = retriever_mod.HybridRetriever(
            config_path=config_path,
            qdrant_local_path=qdrant_local_path,
            load_reranker=load_models,
            load_embedder=load_models,
        )

    def assess_retrieval_need(self, query: str) -> bool:
        """
        Determine if a query needs RAG retrieval.

        Simple heuristic: medical queries almost always benefit from retrieval.
        Skip only for greetings, meta-questions, or very short queries.
        """
        query_lower = query.lower().strip()

        # Skip retrieval for non-medical queries
        skip_patterns = [
            r"^(hi|hello|hey|thanks|thank you|bye|goodbye)",
            r"^(who are you|what are you|how are you)",
            r"^(help|menu|options|commands)",
        ]
        for pattern in skip_patterns:
            if re.match(pattern, query_lower):
                return False

        # Too short to be a real medical query
        if len(query_lower.split()) < 3:
            return False

        return True

    def expand_query(self, query: str) -> list[str]:
        """
        Generate expanded queries using medical synonym patterns.

        Returns a list of query variants for multi-query retrieval.
        """
        expanded = [query]
        query_lower = query.lower()

        for trigger, synonyms in QUERY_EXPANSION_PATTERNS.items():
            if trigger in query_lower:
                for synonym in synonyms[:2]:  # Limit expansions
                    expanded_query = query_lower.replace(trigger, synonym)
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)

        return expanded[:3]  # Max 3 variants

    def assess_relevance(self, query: str, documents: list[dict]) -> float:
        """
        Score how relevant the retrieved documents are to the query.

        Uses a combination of:
        - Retrieval scores (rerank or RRF)
        - Keyword overlap between query and documents
        """
        if not documents:
            return 0.0

        # Factor 1: Average retrieval score of top docs
        score_key = "rerank_score" if "rerank_score" in documents[0] else "rrf_score"
        avg_score = sum(d.get(score_key, 0) for d in documents) / len(documents)

        # Factor 2: Keyword overlap
        query_words = set(query.lower().split())
        stopwords = {"what", "is", "the", "a", "an", "of", "for", "in", "to", "and",
                      "how", "does", "can", "are", "do", "with", "from", "this", "that"}
        query_words -= stopwords

        if not query_words:
            return min(avg_score, 1.0)

        overlap_scores = []
        for doc in documents:
            doc_text = doc.get("chunk_text", "").lower()
            matches = sum(1 for w in query_words if w in doc_text)
            overlap_scores.append(matches / len(query_words))

        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0

        # Combine factors (rerank scores are typically 0-1 for cross-encoder)
        if score_key == "rerank_score":
            # Cross-encoder scores: sigmoid output, 0-1 range
            confidence = 0.6 * min(avg_score, 1.0) + 0.4 * avg_overlap
        else:
            # RRF scores are small (0.01-0.03 range), normalize
            normalized_rrf = min(avg_score * 30, 1.0)
            confidence = 0.5 * normalized_rrf + 0.5 * avg_overlap

        return min(confidence, 1.0)

    def build_context(self, documents: list[dict]) -> str:
        """Build context string from retrieved documents."""
        if not documents:
            return ""

        context_parts = []
        seen_pmids = set()

        for i, doc in enumerate(documents):
            pmid = doc.get("pmid", "unknown")
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            title = doc.get("title", "Untitled")
            text = doc.get("chunk_text", "")
            journal = doc.get("journal", "")

            source_line = f"[Source {i+1}] PMID: {pmid}"
            if journal:
                source_line += f" | {journal}"

            context_parts.append(
                f"{source_line}\n"
                f"Title: {title}\n"
                f"{text}\n"
            )

        return "\n".join(context_parts)

    def build_messages(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> list[dict]:
        """Build the message list for the LLM, with or without RAG context."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if context:
            user_content = RAG_CONTEXT_TEMPLATE.format(context=context) + f"\n\nQuestion: {query}"
        else:
            user_content = query

        messages.append({"role": "user", "content": user_content})
        return messages

    def format_sources(self, documents: list[dict]) -> list[dict]:
        """Format retrieved documents as source objects for the API response."""
        sources = []
        seen_pmids = set()

        for doc in documents:
            pmid = doc.get("pmid", "unknown")
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            score_key = "rerank_score" if "rerank_score" in doc else "rrf_score"
            sources.append({
                "pmid": pmid,
                "title": doc.get("title", ""),
                "text": doc.get("chunk_text", "")[:500],
                "score": round(doc.get(score_key, 0.0), 4),
            })

        return sources

    def run(
        self,
        query: str,
        use_rag: bool = True,
        top_k: Optional[int] = None,
        use_reranker: bool = True,
    ) -> dict:
        """
        Execute the agentic RAG pipeline.

        Returns:
            dict with keys:
            - messages: list of messages for LLM generation
            - sources: list of source dicts
            - retrieval_time_ms: total retrieval time
            - iterations: number of retrieval iterations
            - confidence: relevance confidence score
            - needs_retrieval: whether retrieval was performed
        """
        start_time = time.time()

        # Check if retrieval is needed
        if not use_rag or not self.assess_retrieval_need(query):
            return {
                "messages": self.build_messages(query),
                "sources": [],
                "retrieval_time_ms": 0,
                "iterations": 0,
                "confidence": 1.0,
                "needs_retrieval": False,
            }

        # Iterative retrieval loop
        all_documents = []
        best_confidence = 0.0
        iterations = 0
        queries_tried = []

        # Start with original query
        current_queries = [query]
        if self.query_expansion_enabled:
            current_queries = self.expand_query(query)

        for iteration in range(self.max_iterations):
            iterations += 1

            # Retrieve for each query variant
            for q in current_queries:
                if q in queries_tried:
                    continue
                queries_tried.append(q)

                result = self.retriever.retrieve(
                    query=q,
                    top_k=top_k,
                    use_reranker=use_reranker,
                )
                all_documents.extend(result["documents"])

            # Deduplicate by document ID
            seen_ids = set()
            unique_docs = []
            for doc in all_documents:
                doc_id = doc["id"]
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)

            # Re-sort by best score
            score_key = "rerank_score" if any("rerank_score" in d for d in unique_docs) else "rrf_score"
            unique_docs.sort(key=lambda x: x.get(score_key, 0), reverse=True)

            # Take top_k
            final_top_k = top_k or self.retriever.final_top_k
            top_docs = unique_docs[:final_top_k]

            # Assess relevance
            confidence = self.assess_relevance(query, top_docs)

            if confidence > best_confidence:
                best_confidence = confidence
                all_documents = top_docs

            # If confidence exceeds threshold, stop iterating
            if confidence >= self.confidence_threshold:
                break

            # If below threshold and more iterations allowed, try broader query
            if iteration < self.max_iterations - 1:
                # Generate a broader query
                broader = f"medical research {query}"
                current_queries = [broader]

        total_time = (time.time() - start_time) * 1000

        # Build final output
        context = self.build_context(all_documents)
        messages = self.build_messages(query, context if all_documents else None)
        sources = self.format_sources(all_documents)

        return {
            "messages": messages,
            "sources": sources,
            "retrieval_time_ms": round(total_time, 2),
            "iterations": iterations,
            "confidence": round(best_confidence, 4),
            "needs_retrieval": True,
        }


def main():
    """Test the RAG orchestrator with sample queries."""
    import argparse

    parser = argparse.ArgumentParser(description="Test MedLlama RAG Orchestrator")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--query", type=str, default="What are the side effects of metformin?")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qdrant-local", type=str, default=None)
    args = parser.parse_args()

    console.print("[bold blue]MedLlama RAG Orchestrator Test[/bold blue]\n")

    rag = RAGOrchestrator(
        config_path=args.config,
        qdrant_local_path=args.qdrant_local,
    )

    result = rag.run(
        query=args.query,
        use_rag=not args.no_rag,
        top_k=args.top_k,
    )

    console.print(f"[bold]Query:[/bold] {args.query}")
    console.print(f"[bold]Needs retrieval:[/bold] {result['needs_retrieval']}")
    console.print(f"[bold]Iterations:[/bold] {result['iterations']}")
    console.print(f"[bold]Confidence:[/bold] {result['confidence']}")
    console.print(f"[bold]Retrieval time:[/bold] {result['retrieval_time_ms']} ms")
    console.print(f"[bold]Sources:[/bold] {len(result['sources'])}")

    for i, src in enumerate(result["sources"]):
        console.print(f"\n  [{i+1}] PMID: {src['pmid']} | Score: {src['score']}")
        console.print(f"      Title: {src['title'][:100]}")
        console.print(f"      Text: {src['text'][:200]}...")

    console.print(f"\n[bold]Messages for LLM ({len(result['messages'])} messages):[/bold]")
    for msg in result["messages"]:
        role = msg["role"]
        content = msg["content"][:300]
        console.print(f"  [{role}] {content}...")


if __name__ == "__main__":
    main()
