# Query Rewriting Examples

This document demonstrates how the query rewriting system resolves referential and conversational queries.

## Example 1: Ordinal Reference

### Conversation
```
User: "Explain the power redundancy modes."
Assistant: "There are three power redundancy modes: Grid Redundancy, Power Supply Redundancy, and No Redundancy..."

User: "What about the second one?"
```

### Query Rewriting Process
1. **Original Query**: "What about the second one?"
2. **Referential Pattern Detected**: "the second one" matches ordinal pattern
3. **Context Retrieved**:
   - Previous query: "power redundancy modes"
   - Active topic: "power"
4. **Rewritten Query**: "What is the second power redundancy mode?"

### Retrieval Impact
- Original query would retrieve random documents
- Rewritten query retrieves correct PSU redundancy documentation

---

## Example 2: Pronoun Resolution

### Conversation
```
User: "How do I check if the RAID controller is working?"
Assistant: "Check the LED indicators on the RAID controller. A solid green indicates normal operation..."

User: "What if it's blinking amber?"
```

### Query Rewriting Process
1. **Original Query**: "What if it's blinking amber?"
2. **Referential Pattern Detected**: "it's" pronoun
3. **Context Retrieved**:
   - Previous query: "RAID controller"
   - Mentioned entities: ["RAID controller", "LED"]
4. **Rewritten Query**: "What does blinking amber LED on RAID controller mean?"

---

## Example 3: Elliptical Query

### Conversation
```
User: "What's the maximum memory capacity?"
Assistant: "The server supports up to 4TB of DDR4 ECC memory..."

User: "And the processor?"
```

### Query Rewriting Process
1. **Original Query**: "And the processor?"
2. **Referential Pattern Detected**: "And" at start indicates continuation
3. **Context Retrieved**:
   - Previous query pattern: "maximum ... capacity"
4. **Rewritten Query**: "What is the maximum processor capacity?"

---

## Example 4: Complex Reference Chain

### Conversation
```
User: "Tell me about the network ports."
Assistant: "The server has 4x 1GbE and 2x 25GbE SFP28 ports..."

User: "How do I configure the faster ones?"
```

### Query Rewriting Process
1. **Original Query**: "How do I configure the faster ones?"
2. **Referential Pattern Detected**: "the faster ones"
3. **Context Retrieved**:
   - Previous topic: network ports
   - Mentioned: 1GbE, 25GbE SFP28
   - "faster" → higher speed → 25GbE
4. **Rewritten Query**: "How do I configure the 25GbE SFP28 network ports?"

---

## Latency Impact

| Scenario | Without Rewriting | With Rewriting |
|----------|-------------------|----------------|
| "the second one" | Wrong docs (400ms wasted) | Correct docs |
| "it's blinking" | Empty results | Accurate retrieval |
| Direct query | No change | No change |

## Implementation Details

### Rule-Based Fast Path (<10ms)
- Pattern matching for common references
- Topic substitution from context

### LLM Fallback (~50ms)
- Used for complex resolution
- Gemini Flash with low temperature (0.1)
- Constrained output (query only, no explanation)

### Confidence Scoring
- 1.0: No rewriting needed
- 0.8+: Rule-based rewrite succeeded
- 0.5-0.8: LLM rewrite with uncertainty
- <0.5: Original query used with warning
