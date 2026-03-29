# StreamingLLM Visualization Design

## Reference
- Paper: "Efficient Streaming Language Models with Attention Sinks", Xiao et al., ICLR 2024
- Local: research/papers/14_StreamingLLM_深度解读.md

## Key Concepts to Visualize
1. Attention sinks — initial tokens absorb disproportionate attention regardless of semantic relevance
2. Window + sink tokens — retaining a small set of initial "sink" tokens alongside a sliding window of recent tokens
3. O(L) memory — constant memory usage bounded by window size L, enabling infinite-length generation
4. Position re-mapping — reassigning position indices to remaining tokens after eviction to maintain correct positional encoding

## Visualization Sections

### Section 1: Four-Method Comparison Grid
- **Type**: Comparison grid (2x2)
- **Shows**: Four KV cache strategies: (1) Dense (full cache, O(N) memory), (2) Window-only (recent L tokens, loses sinks), (3) Sliding window + recompute (expensive), (4) StreamingLLM (sink + window, O(L) memory); each shows a schematic token bar with retained positions highlighted
- **Interaction**: Click each cell to expand and see perplexity, memory cost, and failure mode description
- **Data**: Dense: PPL baseline, O(N) memory; Window-only: PPL diverges after cache full; StreamingLLM: PPL ~baseline with O(L) memory; L=1024 typical, 4 sink tokens

### Section 2: Attention Sink Heatmap
- **Type**: Canvas heatmap
- **Shows**: Attention weight distribution across token positions for multiple layers, showing that tokens at positions 0-3 consistently receive high attention across all query positions and layers, regardless of content
- **Interaction**: Hover to see exact weights; layer selector dropdown (1-32); column highlighting to show sink token columns are consistently bright
- **Data**: LLaMA-2-7B: position 0 receives 5-20x more attention than average; effect is strongest in early and late layers; sink phenomenon is model-agnostic

### Section 3: Streaming Window Animation
- **Type**: Canvas animation
- **Shows**: Token sequence growing over time; a fixed window of L recent tokens plus 4 sink tokens are retained (colored), while tokens outside the window fade out and are evicted; the window slides rightward as new tokens are generated
- **Interaction**: Auto-play with speed control; counter shows total tokens generated vs memory used (constant); pause to inspect which tokens are in the cache at any point
- **Data**: Window size L=1024, sink count=4, total cache = 1028 tokens constant; demonstrated stable generation for 4M+ tokens

### Section 4: Perplexity Curve Chart
- **Type**: Canvas line chart
- **Shows**: Perplexity (y-axis) vs sequence length (x-axis, log scale from 1K to 4M tokens) for four methods: dense (flat baseline), window-only (spikes when cache fills), StreamingLLM (flat, slightly above dense), and recompute (flat but slow)
- **Interaction**: Toggle lines on/off; hover for exact PPL values at each sequence length; zoom into the critical transition point where window-only fails
- **Data**: LLaMA-2-7B: dense PPL=5.8, StreamingLLM PPL=5.9 (L=1024), window-only PPL explodes to 1000+ at seq > L, visible cliff at cache boundary

### Section 5: Position Re-Mapping Diagram
- **Type**: Static diagram with animation
- **Shows**: Before and after eviction: original positions [0,1,2,3,...,L-1,L,...,N] mapped to new positions [0,1,2,3, 4,5,...,L+3] after removing tokens between sinks and window; arrows show the position index reassignment
- **Interaction**: Click "Evict" to animate the middle tokens disappearing and the remaining tokens smoothly snapping to consecutive position indices
- **Data**: Sink positions: always 0-3; window positions: remapped to 4 through L+3; critical for RoPE-based models where position IDs affect attention computation

## Technical Notes
- Comparison grid uses CSS grid with card-flip animation on click for expanded details
- Attention heatmap uses Canvas with column-highlight mode for sink visualization
- Streaming animation uses Canvas with smooth easing for window slide transitions
- Mobile: comparison grid becomes a vertical accordion; perplexity chart supports horizontal scroll; animation simplifies window to 8 visible tokens
