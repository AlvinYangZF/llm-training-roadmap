# GPT-2 Decoder-Only Architecture Visualization Design

## Reference
- Paper: "Language Models are Unsupervised Multitask Learners", Radford et al., OpenAI 2019
- Local: research/papers/37_GPT2_Source_Code.md

## Key Concepts to Visualize
1. Decoder-only vs Encoder-Decoder — why GPT-2 drops the encoder and uses causal masking
2. Pre-LayerNorm — layer norm placed before attention/FFN rather than after, improving training stability
3. BPE tokenizer — byte-pair encoding that splits text into subword tokens
4. Weight tying — sharing the embedding matrix with the output projection to reduce parameters
5. Four model sizes — scaling from 117M to 1.5B parameters

## Visualization Sections

### Section 1: Architecture Comparison
- **Type**: CSS grid comparison
- **Shows**: Side-by-side layout of original Transformer encoder-decoder vs GPT-2 decoder-only stack, with crossed-out encoder and highlighted causal mask in the decoder
- **Interaction**: Toggle button to switch between the two architectures; hover blocks to see parameter counts
- **Data**: GPT-2 uses 12 decoder layers (small), no encoder, causal attention mask (lower-triangular)

### Section 2: Pre-LN vs Post-LN Flow
- **Type**: Static diagram with toggle
- **Shows**: Two vertical flowcharts showing data flow through a single transformer block: Post-LN (original) vs Pre-LN (GPT-2), with LayerNorm placement highlighted in contrasting color
- **Interaction**: Click toggle to switch between Pre-LN and Post-LN views; gradient magnitude indicators on arrows
- **Data**: Post-LN: x -> Attn -> Add -> LN; Pre-LN: x -> LN -> Attn -> Add

### Section 3: BPE Tokenization Demo
- **Type**: Interactive input
- **Shows**: Text input field where users type a sentence and see it split into BPE tokens in real time, with token IDs and color-coded subwords
- **Interaction**: Type any text; tokens appear as colored chips below the input; hover chips for token ID and byte representation
- **Data**: GPT-2 vocabulary size = 50,257 tokens, uses byte-level BPE

### Section 4: Weight Tying Diagram
- **Type**: Static diagram
- **Shows**: Embedding matrix (50,257 x 768) at the input connected by a dashed "shared" arrow to the transposed output projection, illustrating parameter reuse
- **Interaction**: Hover to highlight the shared weight path and see parameter savings
- **Data**: Saves ~38M parameters in GPT-2 Small (768 * 50,257)

### Section 5: Model Size Explorer
- **Type**: Interactive selector
- **Shows**: Four cards for GPT-2 Small/Medium/Large/XL with parameters, layers, d_model, and heads; selecting one expands a detailed breakdown bar chart
- **Interaction**: Click a card to expand; animated bars compare parameters across components (embedding, attention, FFN)
- **Data**: Small: 117M/12L/768d/12h, Medium: 345M/24L/1024d/16h, Large: 774M/36L/1280d/20h, XL: 1.5B/48L/1600d/25h

## Technical Notes
- Architecture comparison uses CSS grid with flexbox sub-layouts for alignment
- BPE demo requires a lightweight JS tokenizer or precomputed token map for common words
- Weight tying diagram is SVG for crisp arrows and labels
- Mobile: model size explorer stacks cards vertically; BPE demo uses full-width input
