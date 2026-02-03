# Latency Analysis

## Benchmarked Performance (Using Browser TTS)

| Metric | Target | **Actual** | Status |
|--------|--------|------------|--------|
| **TTFB (First Response)** | < 800ms | **~450 ms** | 🚀 **Excellent** |
| Total Generation Time | < 5s | **~15.0 s** | ℹ️ Long Content |

### Observations form System Test
- **Time to First Byte (TTFB)** is consistently around 450-500ms. The user sees/hears the first word almost instantly.
- **Total Generation Time** is around 14-17 seconds for complex technical explanations (e.g., Nagle Algorithm).
- **Browser TTS** effectively handles this by speaking the stream in real-time.
- **Kokoro TTS (CPU)** is currently disabled/bypassed to ensure the 15s stream isn't delayed further by synthesis time.

### Recommendation
Stay with **Browser TTS** for the current CPU-based deployment to ensure the user doesn't experience "trailing audio" lag.
