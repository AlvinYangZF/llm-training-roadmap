# FlashAttention IO-Aware Optimization Visualization Design

## Reference
- Paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", Tri Dao, Stanford 2022
- Local: research/papers/24_FlashAttention_深度解读.md

## Key Concepts to Visualize
1. SRAM vs HBM hierarchy — GPU memory has fast-but-small SRAM (20MB) and slow-but-large HBM (40-80GB)
2. Tiling — splitting Q, K, V into blocks that fit in SRAM to minimize HBM reads/writes
3. Online softmax — computing softmax incrementally per tile without materializing the full N*N attention matrix
4. IO complexity — standard attention is O(N^2) HBM accesses; FlashAttention reduces to O(N^2 d / M)

## Visualization Sections

### Section 1: Memory Hierarchy Diagram
- **Type**: Static diagram
- **Shows**: GPU chip cross-section with SRAM (on-chip, labeled ~20MB, ~19 TB/s) and HBM (off-chip, labeled ~40GB, ~1.5 TB/s), with arrows showing data movement bottleneck
- **Interaction**: Hover each memory level to see capacity, bandwidth, and latency numbers
- **Data**: A100: SRAM 20MB at 19 TB/s, HBM 40/80GB at 1.5-2.0 TB/s; ratio ~13x bandwidth difference

### Section 2: Standard vs Flash Attention Comparison
- **Type**: Side-by-side animation
- **Shows**: Left: standard attention writing full N*N matrix to HBM (red = HBM traffic); Right: FlashAttention keeping tiles in SRAM (green = on-chip), only writing final output to HBM
- **Interaction**: Auto-play both simultaneously; slider to adjust sequence length N to see diverging HBM traffic
- **Data**: Standard: 3 HBM reads/writes of O(N^2); Flash: O(N^2 d / M) total HBM accesses

### Section 3: Tiling Canvas Animation
- **Type**: Canvas animation
- **Shows**: Q matrix (rows) and K^T matrix (columns) divided into colored tiles; animation shows one tile pair loading into SRAM, computing partial attention, accumulating result, then moving to next tile pair
- **Interaction**: Step-through with Next/Previous buttons or auto-play; current active tile highlighted; SRAM usage indicator bar
- **Data**: Block size B_r = B_c = M/(4d) where M = SRAM size; typical tile = 128 rows for d=64

### Section 4: Online Softmax Formula
- **Type**: Static formula with step animation
- **Shows**: The online softmax trick: tracking running max m_i and running sum l_i, rescaling previous partial results when a new tile introduces a larger max
- **Interaction**: Click through 3 tiles to see m_i and l_i update, with the rescaling factor highlighted at each step
- **Data**: m_new = max(m_old, max(x_block)), l_new = e^(m_old - m_new) * l_old + sum(e^(x_block - m_new))

### Section 5: Performance Comparison Bars
- **Type**: Bar chart
- **Shows**: Wall-clock speedup and memory reduction of FlashAttention vs standard attention across sequence lengths (1K, 2K, 4K, 8K, 16K)
- **Interaction**: Hover bars for exact ms and GB values; toggle between speed and memory views
- **Data**: FlashAttention-2 achieves 2-4x speedup; memory from O(N^2) to O(N) — e.g., seq=4K: standard ~1GB, Flash ~64MB

## Technical Notes
- Memory hierarchy uses SVG for clean chip illustration with layered rectangles
- Tiling animation uses Canvas with 2D grid rendering and smooth tile transitions
- Performance bars use HTML div bars with CSS transitions for hover effects
- Mobile: side-by-side comparison stacks vertically; tiling animation uses swipe gestures
