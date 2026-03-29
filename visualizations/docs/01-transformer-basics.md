# Transformer Basics Visualization Design

## Reference
- Paper: "Attention Is All You Need", Vaswani et al., NeurIPS 2017
- Local: research/papers/35_Attention_Is_All_You_Need.md

## Key Concepts to Visualize
1. Encoder-Decoder architecture — the foundational structure that replaces RNNs entirely with attention
2. Scaled Dot-Product Attention — the core QKV mechanism with sqrt(d_k) scaling
3. Multi-Head Attention — parallel attention heads capturing different relationship subspaces
4. Positional Encoding — sinusoidal functions injecting sequence order into embeddings
5. Residual connections + LayerNorm — stabilizing gradient flow through deep stacks

## Visualization Sections

### Section 1: Architecture Diagram
- **Type**: Static diagram
- **Shows**: Full encoder-decoder stack with 6 layers each, showing the flow from input embeddings through encoder self-attention, cross-attention in decoder, and final linear + softmax
- **Interaction**: Hover to highlight each sub-layer (self-attention, FFN, cross-attention)
- **Data**: N=6 layers, d_model=512, d_ff=2048, h=8 heads

### Section 2: QKV Interactive Step-Through
- **Type**: Interactive step-through
- **Shows**: How Q, K, V matrices are computed from input, dot product QK^T, scaling by sqrt(d_k), softmax, and weighted sum with V
- **Interaction**: Click "Next Step" to advance through each computation stage; values update live with example 4-token sequence
- **Data**: d_k=64, example attention scores for a 4-token input ("The cat sat down")

### Section 3: Multi-Head Fan-Out Animation
- **Type**: Canvas animation
- **Shows**: Input splitting into h=8 parallel heads, each performing independent attention, then concatenation and linear projection back to d_model
- **Interaction**: Auto-play with pause/resume; click individual heads to inspect their learned attention pattern
- **Data**: h=8, d_k=d_v=64, concat dimension = 8*64 = 512

### Section 4: Positional Encoding Heatmap
- **Type**: Canvas heatmap
- **Shows**: 2D grid of PE values with position on y-axis and dimension on x-axis, revealing the sinusoidal wave patterns across dimensions
- **Interaction**: Drag slider to change sequence length (up to 512); hover cells for exact PE(pos, 2i) and PE(pos, 2i+1) values
- **Data**: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)), d_model=512

## Technical Notes
- Architecture diagram uses pure HTML/CSS grid layout for crisp rendering and accessibility
- QKV step-through uses DOM manipulation with highlight transitions for each computation stage
- Multi-head animation and PE heatmap use Canvas for performance with many elements
- Mobile: stack the architecture diagram vertically; QKV step-through scrolls horizontally; PE heatmap supports pinch-zoom
