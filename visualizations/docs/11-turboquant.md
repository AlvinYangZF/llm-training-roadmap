# TurboQuant Visualization Design

## Reference
- Paper: "TurboQuant: Online KV Cache Quantization via Efficient Random Rotation and Smooth Quantization", ICLR 2026
- Local: research/papers/34_TurboQuant_深度解读.md

## Key Concepts to Visualize
1. PolarQuant + QJL two-stage pipeline — first stage rotates via random orthogonal matrix, second stage projects with Johnson-Lindenstrauss
2. Random rotation — applying a random orthogonal transformation to spread outlier magnitudes evenly across dimensions
3. Rate-distortion tradeoff — balancing quantization bitwidth against reconstruction quality
4. 6x compression — achieving ~6x KV cache size reduction with minimal quality loss

## Visualization Sections

### Section 1: Two-Stage Pipeline Diagram
- **Type**: Horizontal flow diagram
- **Shows**: Input KV tensor flowing through Stage 1 (PolarQuant: random rotation + uniform quantization) then Stage 2 (QJL: dimensionality reduction via random projection), producing compressed KV output; each stage labeled with its compression ratio contribution
- **Interaction**: Hover each stage for detail tooltip; click to expand and see the mathematical operation; animated data flow particles showing tensor size shrinking at each stage
- **Data**: Stage 1 (PolarQuant): 16-bit to 4-bit quantization (~4x); Stage 2 (QJL): further dimension reduction (~1.5x); combined: ~6x total compression

### Section 2: 2D Rotation Scatter Plot
- **Type**: Canvas scatter plot with animation
- **Shows**: 2D points representing KV values before rotation (clustered with outliers along axes) and after random orthogonal rotation (spread uniformly); the rotation visually redistributes outlier energy across all dimensions
- **Interaction**: Click "Rotate" to animate the transformation; slider to adjust rotation angle; toggle between pre/post rotation views; hover points for exact coordinates
- **Data**: Before rotation: a few dimensions have 10-50x larger magnitude (outliers); after rotation: max/min ratio drops to <3x; quantization error reduces by ~60%

### Section 3: Bit-Width vs Quality Chart
- **Type**: Canvas line chart with markers
- **Shows**: Perplexity (y-axis) vs quantization bit-width (x-axis: 2, 3, 4, 6, 8, 16 bits) for three methods: naive quantization, PolarQuant only, and full TurboQuant; TurboQuant achieves near-baseline quality at much lower bit-widths
- **Interaction**: Hover points for exact PPL values; toggle method lines; vertical reference line at each bit-width showing the gap between methods
- **Data**: At 4-bit: naive PPL=14.5, PolarQuant PPL=11.2, TurboQuant PPL=10.9 (baseline FP16 PPL=10.8); at 2-bit: naive diverges, TurboQuant PPL=12.1

### Section 4: KV Compression Bars
- **Type**: Grouped bar chart
- **Shows**: KV cache memory usage for different model sizes (7B, 13B, 70B) at different quantization levels (FP16 baseline, 8-bit, 4-bit TurboQuant), showing absolute GB savings and enabling larger batch sizes or longer contexts
- **Interaction**: Hover bars for exact GB values; toggle between "memory saved" and "batch size increase" views; animate bars growing/shrinking as quantization level changes
- **Data**: LLaMA-70B at 4K seq: FP16 = 2.5 GB, 4-bit TurboQuant = 0.42 GB (6x reduction); freed memory enables 6x larger batch or 6x longer context

### Section 5: Rate-Distortion Comparison
- **Type**: Canvas chart with Pareto frontier
- **Shows**: Rate (bits per value, x-axis) vs distortion (MSE or PPL degradation, y-axis) for multiple quantization methods: round-to-nearest, GPTQ-style, PolarQuant, TurboQuant; TurboQuant defines the Pareto frontier (best quality at each bit-rate)
- **Interaction**: Hover points for method name and exact rate-distortion values; drag a target quality line to see which bit-rate each method requires to achieve it
- **Data**: At 0.5% PPL degradation: TurboQuant needs 4 bits, PolarQuant needs 5 bits, naive needs 8 bits; Pareto frontier shows TurboQuant dominates at all bit-rates tested

## Technical Notes
- Pipeline diagram uses SVG for crisp arrows and stage boxes with animated dashed-line data flow
- Scatter plot uses Canvas with WebGL fallback for smooth rotation animation of 500+ points
- Rate-distortion chart uses Canvas with Pareto frontier curve fitted via monotone cubic interpolation
- Mobile: pipeline diagram becomes vertical flow; scatter plot reduces to 100 points; charts support pinch-zoom
