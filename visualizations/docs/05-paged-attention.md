# PagedAttention / vLLM Visualization Design

## Reference
- Paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention", Kwon et al., SOSP 2023
- Local: research/papers/01_PagedAttention_vLLM_中文解读.md

## Key Concepts to Visualize
1. OS virtual memory analogy — KV cache blocks map to physical GPU memory like virtual pages to physical frames
2. Block tables — per-sequence mapping from logical block indices to physical block locations
3. Memory fragmentation — conventional systems waste 60-80% of KV cache memory due to over-allocation
4. Copy-on-Write for beam search — shared KV blocks across beams with lazy copying on divergence

## Visualization Sections

### Section 1: OS Analogy Diagram
- **Type**: Side-by-side comparison diagram
- **Shows**: Left: OS virtual memory with page table mapping virtual pages to physical frames; Right: PagedAttention with logical KV blocks mapped to physical GPU memory blocks via block table
- **Interaction**: Hover to highlight matching concepts across the analogy (virtual page = logical block, page table = block table, physical frame = physical block)
- **Data**: Block size = 16 tokens (configurable); each block stores K and V tensors for those tokens

### Section 2: Memory Waste Comparison Bars
- **Type**: Stacked bar chart
- **Shows**: Memory utilization breakdown for 3 systems: (1) static allocation (large reserved, mostly wasted), (2) dynamic contiguous (fragmentation gaps), (3) PagedAttention (near-full utilization with only last-block waste)
- **Interaction**: Hover segments for exact waste percentages; animate fill as requests arrive
- **Data**: Static: ~38% utilization; Contiguous dynamic: ~55%; PagedAttention: ~96% (only last block partially filled, <4% waste)

### Section 3: Interactive Block Table
- **Type**: Interactive step-through
- **Shows**: A grid of physical memory blocks (numbered, colored by sequence) and a block table per active sequence showing logical-to-physical mapping; tokens fill blocks incrementally
- **Interaction**: Click "Add Token" to append a token to a selected sequence, watching the block table grow; click "Remove Sequence" to free blocks back to the free list; color-coded by request
- **Data**: Example: 3 concurrent sequences, 8 physical blocks, block size = 4 tokens for visibility

### Section 4: Beam Search Copy-on-Write Animation
- **Type**: Canvas animation
- **Shows**: Beam search with 4 beams sharing common prefix blocks; when beams diverge, CoW triggers: shared block is copied only when a beam writes different tokens
- **Interaction**: Auto-play 4-step animation: (1) shared prefix, (2) fork into beams, (3) beam diverges and triggers copy, (4) independent blocks; pause at any step
- **Data**: 4 beams, 3 shared prefix blocks, CoW saves ~75% memory vs naive full copy per beam

### Section 5: Fragmentation Types
- **Type**: Comparison grid
- **Shows**: Three fragmentation patterns: (1) internal fragmentation (partially filled last block), (2) external fragmentation (scattered free blocks too small for new request), (3) reservation waste (pre-allocated max_seq_len); PagedAttention eliminates external fragmentation entirely
- **Interaction**: Click each type to see an animated example of how it occurs and how PagedAttention addresses it
- **Data**: Internal: avg waste = block_size/2 tokens per sequence; External: 0% with paging; Reservation: eliminated by on-demand allocation

## Technical Notes
- Block table visualization uses HTML table with CSS grid cells for clear block indexing
- CoW animation uses Canvas for smooth arrow drawing and block cloning effects
- Memory bars use stacked div elements with percentage labels
- Mobile: block table scrolls horizontally; CoW animation simplifies to 2 beams on small screens
