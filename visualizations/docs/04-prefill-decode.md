# Prefill and Decode Inference Phases Visualization Design

## Reference
- Paper: General LLM inference knowledge (no single paper)
- Local: N/A (cross-cutting concept referenced in multiple papers)

## Key Concepts to Visualize
1. Compute-bound prefill — processing all prompt tokens in parallel, bottlenecked by FLOPs
2. Memory-bound decode — generating tokens one at a time, bottlenecked by memory bandwidth
3. KV Cache — storing key/value tensors from previous tokens to avoid recomputation
4. TTFT vs TBT — Time to First Token (prefill latency) vs Time Between Tokens (decode latency)
5. KV cache memory growth — linear growth with sequence length and batch size

## Visualization Sections

### Section 1: Two-Phase Cards
- **Type**: Comparison grid
- **Shows**: Two side-by-side cards: Prefill (parallel arrows for all prompt tokens, GPU compute icon) and Decode (sequential single-token arrows, memory bandwidth icon)
- **Interaction**: Hover each card to reveal bottleneck details and typical hardware utilization percentages
- **Data**: Prefill: compute utilization ~60-80%, Decode: memory bandwidth utilization ~80-90%, compute utilization ~5-10%

### Section 2: Token Stream Animation
- **Type**: JS animation
- **Shows**: A prompt entering the model as a batch (prefill burst), then output tokens appearing one by one (decode drip), with a timeline bar showing TTFT and TBT intervals
- **Interaction**: Auto-play with speed control slider; pause to inspect any token's latency breakdown
- **Data**: Example: 100-token prompt, TTFT ~200ms, then 50 decode tokens at ~30ms each (TBT)

### Section 3: KV Cache Formula
- **Type**: Static formula with interactive variables
- **Shows**: KV_memory = 2 * n_layers * d_model * seq_len * batch_size * bytes_per_param, with each variable as an adjustable input
- **Interaction**: Sliders for n_layers (1-96), d_model (768-8192), seq_len (1-128K), batch_size (1-256), precision (FP16/FP32); result updates live in GB
- **Data**: Example: LLaMA-70B (80 layers, 8192 d_model, 4K seq, batch=1, FP16) = ~2.5 GB per request

### Section 4: Cache Growth Chart
- **Type**: Canvas line chart
- **Shows**: KV cache memory on y-axis vs sequence length on x-axis, with lines for different model sizes (7B, 13B, 70B) showing linear growth
- **Interaction**: Toggle model size lines; hover for exact GB values at any sequence length
- **Data**: 7B: ~0.5 GB at 4K, 13B: ~1 GB at 4K, 70B: ~2.5 GB at 4K; all scale linearly

### Section 5: Compute vs Memory Gauge
- **Type**: Dual gauge meters
- **Shows**: Two animated gauge dials — one for compute utilization, one for memory bandwidth utilization — switching between prefill and decode phases
- **Interaction**: Toggle button between "Prefill" and "Decode" to see gauges animate to respective utilization levels
- **Data**: Prefill: compute ~70%, bandwidth ~20%; Decode: compute ~5%, bandwidth ~85%

## Technical Notes
- Token stream animation uses requestAnimationFrame for smooth 60fps playback
- KV cache formula section uses HTML inputs with JS event listeners for live calculation
- Cache growth chart uses Canvas with anti-aliased lines and tooltip overlay
- Mobile: gauges stack vertically; formula sliders become dropdowns on small screens
