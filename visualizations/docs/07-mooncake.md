# Mooncake Architecture Visualization Design

## Reference
- Paper: "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving", Moonshot AI, USENIX FAST 2025 Best Paper
- Local: research/papers/02_Mooncake_深度解读.md

## Key Concepts to Visualize
1. Conductor scheduler — central coordinator that routes requests to optimal prefill/decode nodes
2. Prefill/Decode pools — physically separated GPU clusters with independent scaling
3. 3-tier KV cache (GPU/CPU/SSD) — hierarchical storage for KV cache with different latency/capacity tradeoffs
4. Cache-aware scheduling — reusing cached KV from previous similar requests to skip redundant prefill

## Visualization Sections

### Section 1: System Architecture Diagram
- **Type**: Canvas diagram
- **Shows**: Full Mooncake architecture: Conductor at the top routing to Prefill Pool (left) and Decode Pool (right), with 3-tier KV Cache Store at the bottom spanning GPU VRAM, CPU DRAM, and NVMe SSD layers; arrows show request flow and KV transfer paths
- **Interaction**: Click each component to expand a detail tooltip showing capacity, bandwidth, and role; hover arrows to see data flow descriptions
- **Data**: Prefill pool: A100/H100 GPUs optimized for compute; Decode pool: GPUs optimized for memory bandwidth; KV store: GPU ~80GB, CPU ~512GB, SSD ~4TB per node

### Section 2: Request Lifecycle Animation
- **Type**: 4-stage sequential animation
- **Shows**: (1) Request arrives at Conductor, (2) Conductor checks KV cache for prefix match and selects prefill node, (3) Prefill node computes KV (or loads cached KV) and transfers to decode node, (4) Decode node generates tokens and streams response
- **Interaction**: Auto-play through 4 stages with progress indicator; click any stage to pause and inspect; two paths shown: cache-hit (fast, green) and cache-miss (normal, blue)
- **Data**: Cache hit path: skip prefill, TTFT reduced by ~60-80%; Cache miss: full prefill + KV transfer; typical end-to-end: 200ms TTFT, 25ms TBT

### Section 3: 3-Tier Storage Hierarchy
- **Type**: Layered pyramid diagram with flow arrows
- **Shows**: Three horizontal layers — GPU VRAM (top, smallest, fastest), CPU DRAM (middle), NVMe SSD (bottom, largest, slowest) — with eviction arrows flowing down and promotion arrows flowing up
- **Interaction**: Drag a simulated KV block up and down the tiers; each tier shows current fill level as a progress bar; hover for access latency at each level
- **Data**: GPU VRAM: ~80GB, ~2 TB/s; CPU DRAM: ~512GB, ~100 GB/s; NVMe SSD: ~4TB, ~7 GB/s; eviction policy: LRU with prefix-aware priority

### Section 4: Cache-Aware Scheduling Demo
- **Type**: Interactive simulation
- **Shows**: Multiple prefill nodes with different cached prefixes visualized as colored bars; incoming request with a known prefix highlighted; Conductor routes to the node with the longest prefix match
- **Interaction**: Click "New Request" to generate a request with random prefix overlap; watch Conductor evaluate match scores and route to optimal node; counter shows cache hit rate over time
- **Data**: Prefix matching: longest common prefix in token space; typical cache hit rate: 40-70% in production (Moonshot Kimi workload); scheduling latency: <1ms

## Technical Notes
- System architecture uses Canvas with layered z-index for depth (Conductor above pools, KV store below)
- Request lifecycle uses a state machine with CSS transitions between stages
- 3-tier hierarchy uses SVG trapezoids with gradient fills indicating capacity
- Mobile: architecture diagram simplifies to vertical flow; scheduling demo reduces to 3 nodes; tier pyramid becomes a vertical stack
