# Sparse Transformers Visualization Design

## Reference
- Paper: "Generating Long Sequences with Sparse Transformers", Child et al., OpenAI 2019
- Local: research/papers/36_Sparse_Transformers.md

## Key Concepts to Visualize
1. Dense vs sparse attention patterns — full N*N attention vs structured sparse patterns with O(N*sqrt(N)) connections
2. Local attention — each token attends to a fixed window of nearby tokens (diagonal band)
3. Strided attention — each token attends to every sqrt(N)-th token (vertical columns in the attention grid)
4. Combined coverage — local + strided patterns together cover all positions within 2 hops
5. Gradient checkpointing — trading compute for memory by recomputing activations during backward pass

## Visualization Sections

### Section 1: Dense vs Sparse N*N Grid Comparison
- **Type**: Side-by-side Canvas grids
- **Shows**: Left: full N*N dense attention grid (all cells filled, O(N^2)); Right: sparse attention grid with only selected cells filled (O(N*sqrt(N))); unfilled cells are transparent to show the sparsity
- **Interaction**: Slider to adjust N (16-256); grid updates to show how the dense grid grows quadratically while sparse grid grows sub-quadratically; counter shows total active connections for each
- **Data**: N=256: dense = 65,536 connections, sparse = ~4,096 connections (16x reduction); N=1024: dense = 1M, sparse = ~32K

### Section 2: Local Attention Diagonal Band
- **Type**: Canvas grid with highlight
- **Shows**: N*N grid where only a diagonal band of width w is filled, representing local attention where each token attends to its w nearest neighbors
- **Interaction**: Slider to adjust window width w; click any row to highlight which columns that token attends to; animation sweeps through rows to show the sliding window pattern
- **Data**: Typical w = sqrt(N); for N=256, w=16; each token sees 16 neighbors; total connections = N*w = O(N*sqrt(N))

### Section 3: Strided Attention Columns
- **Type**: Canvas grid with highlight
- **Shows**: N*N grid where every sqrt(N)-th column is filled for each row group, creating a vertical stripe pattern; tokens at stride positions act as "summary" tokens
- **Interaction**: Click any row to see which columns it attends to via the stride pattern; slider to adjust stride length; hover columns to see which rows connect to them
- **Data**: Stride = sqrt(N); for N=256, stride=16; each token attends to 16 strided positions; summary tokens are at positions 0, 16, 32, ...

### Section 4: Combined Coverage Animation
- **Type**: Canvas animation
- **Shows**: Starting from an empty N*N grid, first the local pattern fills in (diagonal band), then the strided pattern overlays (vertical stripes); final combined pattern shows full reachability within 2 hops; animation highlights a 2-hop path between any two tokens
- **Interaction**: Auto-play the build-up; click any two cells to see the 2-hop path connecting them through the combined pattern; toggle local-only, strided-only, or combined views
- **Data**: Combined: any token can reach any other token in at most 2 attention layers; local covers nearby, strided covers distant; factorized attention head split: half local, half strided

### Section 5: Complexity Slider
- **Type**: Interactive comparison chart
- **Shows**: Compute and memory complexity curves for dense O(N^2) and sparse O(N*sqrt(N)) as N increases, with a draggable vertical line showing the crossover and savings at any given N
- **Interaction**: Drag the N slider from 256 to 16,384; curves update with exact FLOP counts and memory in GB; percentage savings label updates dynamically
- **Data**: N=1K: dense 1M ops, sparse 32K ops (32x savings); N=16K: dense 256M ops, sparse 2M ops (128x savings); memory with gradient checkpointing: O(N*sqrt(N)) vs O(N^2)

## Technical Notes
- Grid visualizations use Canvas with efficient cell rendering (batch draw calls for filled/empty cells)
- Combined animation uses layered Canvas (one layer per pattern) with alpha blending for overlay
- Complexity chart uses Canvas line rendering with logarithmic y-axis for readable comparison
- Mobile: grids reduce to N=32 for performance; complexity slider is full-width; combined view uses tabs instead of overlay
