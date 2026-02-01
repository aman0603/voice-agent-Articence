# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Voice RAG Pipeline                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Audio   │───▶│  Streaming  │───▶│   Intent    │───▶│ Speculative │      │
│  │  Input   │    │     ASR     │    │  Detection  │    │  Pre-fetch  │      │
│  └──────────┘    └─────────────┘    └─────────────┘    └─────────────┘      │
│       │                │                   │                  │              │
│       │          Partial Transcripts       │          Early Retrieval        │
│       │                │                   │                  │              │
│       │                ▼                   ▼                  ▼              │
│       │         ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │
│       │         │   Query     │    │   Filler    │    │   Hybrid    │       │
│       │         │  Rewriter   │    │  Generator  │    │   Search    │       │
│       │         └─────────────┘    └─────────────┘    └─────────────┘       │
│       │                │                   │                  │              │
│       │                │                   │          Dense + BM25          │
│       │                ▼                   │                  │              │
│       │         ┌─────────────┐            │                  ▼              │
│       │         │  Reranker   │◀───────────┼──────────────────┘              │
│       │         │(Cross-Enc)  │            │                                 │
│       │         └─────────────┘            │                                 │
│       │                │                   │                                 │
│       │                ▼                   ▼                                 │
│       │         ┌─────────────┐    ┌─────────────┐                          │
│       │         │  Streaming  │    │    TTS      │                          │
│       │         │     LLM     │───▶│   (Piper)   │────▶ Audio Output        │
│       │         └─────────────┘    └─────────────┘                          │
│       │                │                   ▲                                 │
│       │                │                   │                                 │
│       │                └───────────────────┘                                 │
│       │                   Voice Optimizer                                    │
│       │                                                                      │
└───────┼──────────────────────────────────────────────────────────────────────┘
        │
        └────────────────────────── Metrics Tracking
```

## Component Responsibilities

### ASR Layer (Whisper)
- **Streaming Transcription**: Emits partial transcripts every 300ms
- **VAD Integration**: Detects end-of-speech for final transcript
- **Low Latency**: Base model for speed, upgradeable to larger models

### Intent Detection
- **Keyword Matching**: Fast domain detection (power, storage, network)
- **Query Type Classification**: Troubleshooting, configuration, explanation
- **Confidence Scoring**: Triggers speculative retrieval at 60%+

### Query Rewriting
- **Pronoun Resolution**: "the second one" → "second power redundancy mode"
- **Context Tracking**: 5-turn sliding window
- **Rule + LLM Hybrid**: Fast rules first, LLM fallback

### Hybrid Retrieval
- **Dense Search**: FAISS HNSW with 384-dim embeddings
- **Sparse Search**: BM25 for exact terms, error codes
- **RRF Fusion**: Reciprocal Rank Fusion for robust merging

### Reranking
- **Cross-Encoder**: ms-marco-MiniLM-L-6-v2
- **Early Exit**: Stop at 0.9+ confidence
- **Timeout**: 150ms max, fallback to hybrid scores

### LLM (Gemini)
- **Streaming**: Token-by-token generation
- **Voice-Optimized Prompt**: Short sentences, simple language
- **Context Window**: 3000 char from top-5 documents

### Voice Optimization
- **Sentence Breaking**: Max 15 words per sentence
- **Phonetic Conversion**: RAID → raid, SFP+ → S-F-P plus
- **Simplification**: ensure → make sure, utilize → use

### TTS (Piper)
- **Local Inference**: No API latency
- **Streaming Output**: Chunk-by-chunk synthesis
- **Natural Voice**: en_US-lessac-medium

## Latency Budget

| Component | Target | Notes |
|-----------|--------|-------|
| ASR Start to First Partial | 50ms | Streaming |
| Intent Detection | 50ms | Parallel with ASR |
| Speculative Retrieval | 100ms | Based on intent |
| Query Rewriting | 50ms | Rule-based fast path |
| Hybrid Search | 100ms | Dense + BM25 parallel |
| Reranking | 150ms | Top-20 candidates |
| LLM First Token | 100ms | Gemini Flash |
| TTS First Byte | 50ms | Piper local |
| **Total TTFB** | **<800ms** | With filler: ~550ms |

## Failure Handling

- **ASR Timeout**: Use partial transcript if final not received in 5s
- **Retrieval Empty**: Return "I couldn't find information about that"
- **Reranker Timeout**: Fall back to hybrid scores
- **LLM Error**: Generic apology with retry suggestion
- **TTS Error**: Return text response only

## Caching Strategy (Bonus)

- **Query Cache**: LRU cache for repeated queries (1000 entries)
- **Embedding Cache**: Pre-computed embeddings for common terms
- **Document Cache**: Recently accessed chunks in memory
