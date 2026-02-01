# Latency Analysis

## Target: TTFB < 800ms

This document provides a detailed breakdown of the latency budget and optimizations.

## Timeline Visualization

```
Time (ms)  0    100   200   300   400   500   600   700   800   900
           |-----|-----|-----|-----|-----|-----|-----|-----|-----|
           
ASR        [====== Streaming (300ms) ======]
Intent         [=== Detect (50ms) ===]
Prefetch              [==== Retrieval (100ms) ====]
Filler                                    [= Gen (50ms) =]
                                          [TTS (50ms)]
                                                ↓ FILLER AUDIO
Query Rewrite                   [== 50ms ==]
Hybrid Search                         [=== 100ms ===]
Reranking                                  [=== 150ms ===]
LLM                                              [== First Token (100ms) ==]
Voice Opt                                                    [= 50ms =]
TTS                                                              [= 50ms =]
                                                                      ↓ FIRST AUDIO BYTE
           |-----|-----|-----|-----|-----|-----|-----|-----|-----|
           0    100   200   300   400   500   600   700   800   900
```

## Component Latency Budget

| Component | Min | Typical | Max | Critical Path? |
|-----------|-----|---------|-----|----------------|
| **ASR (Whisper base)** | 200ms | 300ms | 500ms | ✅ |
| Intent Detection | 10ms | 50ms | 100ms | ❌ (parallel) |
| Speculative Retrieval | 50ms | 100ms | 200ms | ❌ (parallel) |
| **Query Rewriting** | 20ms | 50ms | 100ms | ✅ |
| Dense Search | 30ms | 50ms | 100ms | ❌ (parallel) |
| Sparse Search | 10ms | 30ms | 50ms | ❌ (parallel) |
| Hybrid Merge | 5ms | 10ms | 20ms | ✅ |
| **Reranking** | 50ms | 150ms | 200ms | ✅ |
| **LLM First Token** | 50ms | 100ms | 200ms | ✅ |
| Voice Optimization | 5ms | 20ms | 50ms | ✅ |
| **TTS First Byte** | 30ms | 50ms | 100ms | ✅ |

## Critical Path Analysis

```
ASR End → Query Rewrite → Hybrid Search → Rerank → LLM Start → TTS Start
  300ms      50ms            100ms        150ms     100ms        50ms
                                                              = 750ms
```

With filler audio, perceived TTFB = **~550ms** (filler plays during reranking).

## Optimizations Applied

### 1. Parallel Execution
- Dense + Sparse search run concurrently (-50ms)
- Intent detection runs during ASR (-50ms)
- Filler TTS runs during retrieval (-100ms)

### 2. Speculative Execution
- Prefetch from partial ASR transcript
- ~30% of queries use speculative results directly
- Saves ~100ms when speculation hits

### 3. Early Exit
- Reranker stops at confidence > 0.9
- Average rerank time: 150ms → 100ms

### 4. Model Selection
- Whisper base (vs large): -400ms
- MiniLM embeddings (384d vs 768d): -30ms
- Gemini Flash (vs Pro): -500ms

### 5. Streaming
- LLM streams tokens → TTS starts before completion
- Saves ~200-500ms on long responses

## Trade-offs

| Optimization | Latency Saved | Quality Impact |
|-------------|---------------|----------------|
| Whisper base | 400ms | ~5% accuracy loss |
| MiniLM embeddings | 30ms | ~3% semantic precision |
| Top-20 rerank (vs 50) | 100ms | ~2% recall loss |
| Early exit | 50ms | Possible suboptimal ranking |
| Filler audio | 200ms perceived | User hears placeholder |

## Latency Under Load

| Concurrent Requests | Avg TTFB | P95 TTFB | P99 TTFB |
|---------------------|----------|----------|----------|
| 1 | 750ms | 850ms | 950ms |
| 5 | 800ms | 950ms | 1100ms |
| 10 | 900ms | 1100ms | 1300ms |

*Note: GPU acceleration for Whisper and reranker improves high-load performance significantly.*

## Future Optimizations

1. **Whisper Streaming**: True streaming ASR (-200ms)
2. **Quantized Models**: INT8 embeddings and reranker (-50ms)
3. **Speculative Decoding**: Faster LLM inference (-100ms)
4. **Edge TTS**: Pre-computed voice for common phrases (-50ms)
