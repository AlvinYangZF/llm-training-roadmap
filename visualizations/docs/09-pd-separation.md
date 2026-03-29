# Prefill-Decode Disaggregation Visualization Design

## Reference
- Paper: Prefill-Decode separation concept (referenced in Mooncake and DistServe architectures)
- Local: research/papers/02_Mooncake_深度解读.md (PD separation section)

## Key Concepts to Visualize
1. Interference problem — prefill and decode compete for GPU resources, causing latency spikes
2. Separate GPU pools — dedicating GPU groups to prefill-only or decode-only workloads
3. KV transfer — shipping computed KV cache from prefill nodes to decode nodes over network
4. SLO targets — different latency objectives: TTFT for prefill, TBT for decode

## Visualization Sections

### Section 1: Coupled vs Separated Architecture
- **Type**: Canvas diagram pair
- **Shows**: Left: coupled system where prefill and decode run on same GPU (interference shown as overlapping colored bars with conflict icons); Right: separated system with distinct prefill pool and decode pool connected by KV transfer arrow
- **Interaction**: Toggle between coupled and separated views; in coupled mode, animate interference spikes on a latency timeline
- **Data**: Coupled: TTFT P99 ~500ms with interference; Separated: TTFT P99 ~150ms, TBT P99 ~30ms

### Section 2: Resource Utilization Gantt Chart
- **Type**: Interactive Gantt chart
- **Shows**: Timeline showing GPU utilization patterns: coupled system has alternating prefill/decode with idle gaps and interference; separated system shows steady high utilization on dedicated pools
- **Interaction**: Drag timeline to scrub through request arrivals; hover bars for utilization percentage at each time slice
- **Data**: Coupled: prefill bursts cause decode stalls (30-50% decode throughput drop); Separated: prefill GPUs at ~80% compute, decode GPUs at ~85% memory bandwidth

### Section 3: KV Transfer Animation
- **Type**: Canvas animation
- **Shows**: Step-by-step flow: (1) request arrives at prefill node, (2) prefill computes KV cache, (3) KV tensors transfer over network (NVLink/RDMA), (4) decode node receives KV and begins autoregressive generation
- **Interaction**: Auto-play with step indicators; click any step to pause and see data sizes and transfer latency
- **Data**: KV size for LLaMA-70B at 2K tokens: ~1.25 GB; NVLink transfer: ~2ms; RDMA: ~5-10ms; network overhead is small vs prefill compute time

### Section 4: SLO Comparison Bars
- **Type**: Grouped bar chart
- **Shows**: P50 and P99 latencies for TTFT and TBT under coupled vs separated architectures, showing how separation dramatically improves tail latencies
- **Interaction**: Hover bars for exact latency values; toggle between P50 and P99 views; animate bar growth for visual comparison
- **Data**: Coupled TTFT P99: ~500ms, Separated TTFT P99: ~150ms; Coupled TBT P99: ~80ms, Separated TBT P99: ~30ms; throughput improvement ~1.5-2x

## Technical Notes
- Architecture diagrams use Canvas with rounded rectangles for GPU nodes and animated dashed lines for data transfer
- Gantt chart uses HTML div bars positioned absolutely within a timeline container
- KV transfer animation uses requestAnimationFrame with easing functions for data packet movement
- Mobile: Gantt chart scrolls horizontally; architecture diagrams stack vertically with simplified node layout
