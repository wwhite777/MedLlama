# MedLlama

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-Llama%203.1%20Community-green)
![Tests](https://img.shields.io/badge/Tests-37%20passing-brightgreen)

**A production-ready medical question-answering system combining fine-tuned LLMs with retrieval-augmented generation over PubMed literature.**

MedLlama demonstrates end-to-end ML engineering: data curation, QLoRA SFT + DPO alignment, hybrid retrieval with cross-encoder reranking, streaming FastAPI serving, and Docker containerization.

## Architecture

```
User Query
    |
    v
[FastAPI Server] ---> [BGE-M3 Embedding]
    |                        |
    |                  +-----+-----+
    |                  |           |
    |            [Dense Search] [BM25 Search]
    |                  |           |
    |                  +-----+-----+
    |                        |
    |                  [RRF Fusion]
    |                        |
    |              [Cross-Encoder Rerank]
    |                        |
    |                  Top-K Documents
    |                        |
    v                        v
[Qwen2.5-7B-Instruct + QLoRA/DPO] <--- Retrieved Context
    |
    v
Streamed Response (SSE)
```

See `result/figure/exp1/medllama-architecture.drawio.xml.gpg` for the detailed draw.io diagram.

## Key Features

- **Two-stage fine-tuning**: QLoRA SFT on 11K medical instruction examples, followed by DPO alignment on 3K preference pairs
- **Hybrid RAG retrieval**: Dense (BGE-M3) + BM25 search with Reciprocal Rank Fusion over 7.4K PubMed abstracts in Qdrant
- **Cross-encoder reranking**: BGE-reranker-v2-m3 for precision-focused document selection
- **Agentic RAG orchestration**: Iterative retrieval with query expansion and relevance-based confidence gating
- **Streaming API**: FastAPI with SSE streaming, health checks, and structured Pydantic schemas
- **Docker-ready serving**: Single-container deployment with NVIDIA CUDA base, health checks, and vLLM backend
- **Comprehensive testing**: 37 tests covering unit, integration, and API layers

## Quick Start

```bash
# Clone and install
git clone https://github.com/wjeong/medllama.git
cd medllama
pip install -e .

# Start Qdrant (local or Docker)
bash scripts/start-qdrant.sh

# Ingest PubMed abstracts and build vector store
python3 src/data_prep/medllama-pubmed-ingest.py
python3 src/rag/medllama-embedding-generate.py
python3 src/rag/medllama-qdrant-store.py

# Train (single GPU with QLoRA)
CUDA_VISIBLE_DEVICES=0 python3 src/training/medllama-sft-train.py --qlora
CUDA_VISIBLE_DEVICES=0 python3 src/training/medllama-dpo-train.py

# Merge adapter weights
python3 src/training/medllama-adapter-merge.py

# Serve
uvicorn src.serving.run:app --host 0.0.0.0 --port 8090

# Run tests
pytest test/ -v
```

## Project Structure

```
medllama/
├── configs/
│   ├── medllama-training-sft.yaml      # SFT hyperparameters
│   ├── medllama-training-dpo.yaml      # DPO hyperparameters
│   ├── medllama-rag-config.yaml        # RAG pipeline config
│   └── medllama-serving-config.yaml    # API server config
├── src/
│   ├── data_prep/
│   │   ├── medllama-sft-format.py      # Format medical QA into SFT JSONL
│   │   ├── medllama-dpo-format.py      # Generate DPO preference pairs
│   │   └── medllama-pubmed-ingest.py   # Fetch PubMed abstracts via Entrez
│   ├── training/
│   │   ├── medllama-sft-train.py       # SFT with QLoRA or FSDP
│   │   ├── medllama-dpo-train.py       # DPO alignment training
│   │   └── medllama-adapter-merge.py   # Merge LoRA adapters into base
│   ├── rag/
│   │   ├── medllama-embedding-generate.py   # BGE-M3 embedding generation
│   │   ├── medllama-qdrant-store.py         # Qdrant collection management
│   │   ├── medllama-hybrid-retrieve.py      # Dense + BM25 + RRF + reranking
│   │   └── medllama-rag-orchestrate.py      # Agentic RAG orchestrator
│   ├── serving/
│   │   ├── medllama-api-serve.py       # FastAPI application
│   │   ├── medllama-schema-define.py   # Pydantic request/response models
│   │   ├── medllama-sse-stream.py      # SSE streaming utilities
│   │   └── run.py                      # Uvicorn entry point
│   └── eval/                           # Evaluation scripts
├── docker/
│   └── Dockerfile.serve                # Production serving container
├── test/
│   ├── test_medllama_api.py            # API endpoint tests
│   ├── test_medllama_integration.py    # Integration tests
│   └── test_medllama_rag.py            # RAG pipeline tests
├── data/                               # Datasets and vector store
├── checkpoints/                        # Model checkpoints
├── result/                             # Experiment outputs
└── pyproject.toml
```

## Training

### Approach

1. **SFT (Supervised Fine-Tuning)**: QLoRA 4-bit quantization with LoRA rank 64 on Qwen2.5-7B-Instruct, trained on 11K curated medical instruction-response pairs from clinical QA datasets.

2. **DPO (Direct Preference Optimization)**: Aligns the SFT model using 3K preference pairs (chosen vs. rejected responses), improving response quality and safety without reward model training.

### Hyperparameters

| Parameter | SFT | DPO |
|---|---|---|
| Base model | Qwen2.5-7B-Instruct | SFT checkpoint |
| Quantization | NF4 (4-bit) | NF4 (4-bit) |
| LoRA rank / alpha | 64 / 128 | 64 / 128 |
| Learning rate | 2e-4 | 5e-7 |
| Epochs | 3 | 1 |
| Batch size (effective) | 16 | 16 |
| Max sequence length | 2048 | 2048 |
| Scheduler | Cosine | Cosine |
| DPO beta | -- | 0.1 |

### Training Results

| Metric | Value |
|---|---|
| SFT final loss | 1.067 |
| DPO reward accuracy | 100% |
| DPO reward margin | 10.32 |

## RAG Pipeline

The retrieval pipeline follows a multi-stage architecture:

1. **Query embedding**: Encode the query with BGE-M3 (1024-dim dense vectors)
2. **Dual retrieval**: Run dense vector search and BM25 keyword search in parallel against Qdrant (7.4K PubMed chunks)
3. **RRF fusion**: Merge results via Reciprocal Rank Fusion (k=60) to combine semantic and lexical relevance
4. **Cross-encoder reranking**: Re-score top candidates with BGE-reranker-v2-m3 for precision
5. **Context assembly**: Format top-K documents with PMID citations into the LLM prompt
6. **Agentic loop**: If relevance confidence is below threshold, expand the query and re-retrieve (up to 3 iterations)

### Corpus

- **Source**: PubMed abstracts via NCBI Entrez API
- **Size**: 7,400 abstracts, chunked into ~15K segments
- **Topics**: Clinical medicine, pharmacology, pathophysiology, diagnostics, treatment guidelines
- **Embedding model**: BAAI/bge-m3 (1024 dimensions)
- **Vector store**: Qdrant with HNSW index (m=16, ef_construct=100)

## API

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health, model status, GPU memory |
| POST | `/chat` | Chat completion with optional RAG |
| POST | `/chat/stream` | Streaming chat via Server-Sent Events |
| POST | `/retrieve` | Direct document retrieval |

### Examples

```bash
# Health check
curl http://localhost:8090/health

# Chat with RAG
curl -X POST http://localhost:8090/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the first-line treatments for type 2 diabetes?", "use_rag": true}'

# Streaming response
curl -N -X POST http://localhost:8090/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the mechanism of action of ACE inhibitors", "use_rag": true}'

# Direct retrieval
curl -X POST http://localhost:8090/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "metformin side effects", "top_k": 5, "use_reranker": true}'
```

### Docker

```bash
docker build -f docker/Dockerfile.serve -t medllama:latest .
docker run --gpus all -p 8090:8090 -v ./checkpoints/merged:/models/medllama medllama:latest
```

## Evaluation

| Benchmark | Metric | Base Model | MedLlama |
|---|---|---|---|
| MedQA (USMLE, n=200) | Accuracy | 64.0% | **65.5%** |
| RAG Retrieval (n=50) | Recall@5 | -- | **100%** |
| RAG Retrieval | Avg Latency | -- | 204 ms |
| LLM-as-Judge (n=50) | Accuracy | -- | 3.18/5 |
| LLM-as-Judge | Clarity | -- | 3.60/5 |
| LLM-as-Judge | Safety | -- | 3.30/5 |
| LLM-as-Judge | Completeness | -- | 3.02/5 |
| LLM-as-Judge | Evidence | -- | 3.04/5 |
| LLM-as-Judge | Overall | -- | **3.23/5** |

Encrypted evaluation results: `result/eval/*.json.gpg`

## Tech Stack

- **Model**: Qwen2.5-7B-Instruct (QLoRA fine-tuned)
- **Training**: PyTorch, Transformers, TRL, PEFT, BitsAndBytes, DeepSpeed/FSDP
- **RAG**: Qdrant, Sentence-Transformers (BGE-M3, BGE-reranker-v2-m3)
- **Serving**: FastAPI, vLLM, Uvicorn, SSE-Starlette
- **Data**: BioPython (PubMed Entrez), HuggingFace Datasets
- **Infra**: Docker, NVIDIA CUDA 12.1, W&B logging
- **Testing**: pytest, pytest-asyncio

## License

This project uses Qwen2.5-7B-Instruct under the [Apache 2.0 License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE). The project framework also supports Llama 3.1 models under the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE).
