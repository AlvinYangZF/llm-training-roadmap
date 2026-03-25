# LLM Storage Innovation: KV Cache Management & Memory Optimization for Inference

*Generated: 2026-03-21 | Sources: 40+ papers | Confidence: High*

## Executive Summary

The KV cache has emerged as the central bottleneck in LLM inference and serving. As context lengths grow from thousands to millions of tokens, KV cache memory consumption dominates GPU memory usage and dictates throughput, latency, and cost. A rich ecosystem of research has developed around this challenge, spanning paged memory management (vLLM), disaggregated architectures (Mooncake, DistServe), aggressive quantization (KIVI, KVQuant), intelligent eviction (H2O, StreamingLLM), CPU/SSD offloading (FlexGen, InfiniGen), prefix caching (SGLang/RadixAttention), memory-efficient attention kernels (FlashAttention), and distributed KV cache systems (Infinite-LLM, MemServe). This report catalogs the most influential papers across eight sub-areas.

---

## 1. KV Cache Management and Paging

### 1.1 PagedAttention / vLLM
- **Title:** Efficient Memory Management for Large Language Model Serving with PagedAttention
- **Authors:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang et al.
- **Venue:** ACM SOSP 2023
- **arXiv:** [2309.06180](https://arxiv.org/abs/2309.06180)
- **Summary:** Introduces PagedAttention, inspired by OS virtual memory paging, which partitions KV cache into fixed-size blocks to eliminate fragmentation and enable flexible sharing. vLLM achieves 2-4x throughput improvement over FasterTransformer and Orca.

### 1.2 Orca (Continuous Batching)
- **Title:** Orca: A Distributed Serving System for Transformer-Based Generative Models
- **Authors:** Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim et al.
- **Venue:** USENIX OSDI 2022
- **Link:** [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
- **Summary:** Proposes iteration-level scheduling (continuous batching) which schedules at the granularity of individual iterations rather than full requests, along with selective batching. Achieves 36.9x throughput improvement on GPT-3 175B.

---

## 2. Disaggregated Serving and Storage

### 2.1 Mooncake
- **Title:** Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving
- **Authors:** Ruoyu Qin et al. (Moonshot AI)
- **Venue:** USENIX FAST 2025 (Best Paper Award)
- **arXiv:** [2407.00079](https://arxiv.org/abs/2407.00079)
- **Summary:** Separates prefill and decoding into distinct clusters and leverages underutilized CPU, DRAM, and SSD resources for a disaggregated KV cache. A KVCache-centric scheduler balances throughput and SLOs, achieving up to 525% throughput increase in long-context scenarios.

### 2.2 DistServe
- **Title:** DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving
- **Authors:** Yinmin Zhong, Shengyu Liu et al. (Peking University, UC San Diego)
- **Venue:** USENIX OSDI 2024
- **arXiv:** [2401.09670](https://arxiv.org/abs/2401.09670)
- **Summary:** Disaggregates prefill and decoding phases onto separate GPU resources to optimize goodput (throughput meeting latency SLOs). Demonstrates that the two phases have fundamentally different resource requirements.

### 2.3 Splitwise
- **Title:** Splitwise: Efficient Generative LLM Inference Using Phase Splitting
- **Authors:** Pratyush Patel, Esha Choukse et al. (Microsoft Research)
- **Venue:** ACM/IEEE ISCA 2024
- **arXiv:** [2311.18677](https://arxiv.org/abs/2311.18677)
- **Summary:** Splits prefill and decode phases onto heterogeneous hardware optimized for each phase. Achieves 1.4x higher throughput at 20% lower cost, or 2.35x throughput at same cost/power budget.

### 2.4 TetriInfer
- **Title:** Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads
- **Authors:** Cunchen Hu, Heyang Huang et al.
- **arXiv:** [2401.11181](https://arxiv.org/abs/2401.11181)
- **Summary:** Schedules prefill and decode requests like Tetris blocks, using an LLM-based length prediction model to speculate decode request resource usage. Runs prefill in fixed-size computation units with dedicated instances.

---

## 3. KV Cache Compression and Quantization

### 3.1 KIVI
- **Title:** KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
- **Authors:** Zirui Liu, Jiayi Yuan et al.
- **Venue:** ICML 2024
- **arXiv:** [2402.02750](https://arxiv.org/abs/2402.02750)
- **Summary:** Discovers that key cache should be quantized per-channel while value cache per-token. Achieves 2-bit KV cache with 2.6x less peak memory, enabling 4x larger batch sizes and 2.35-3.47x throughput improvement.

### 3.2 KVQuant
- **Title:** KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization
- **Authors:** Coleman Hooper, Sehoon Kim et al. (UC Berkeley)
- **Venue:** NeurIPS 2024
- **arXiv:** [2401.18079](https://arxiv.org/abs/2401.18079)
- **Summary:** Introduces per-channel key quantization, pre-RoPE key quantization, non-uniform quantization, and dense-and-sparse quantization. Achieves <0.1 perplexity degradation at 3-bit, enabling 10M context on 8 A100 GPUs.

### 3.3 Coupled Quantization (CQ)
- **Title:** KV Cache is 1 Bit Per Channel: Efficient Large Language Model Inference with Coupled Quantization
- **Authors:** Tianyi Zhang et al.
- **Venue:** NeurIPS 2024
- **arXiv:** [2405.03917](https://arxiv.org/abs/2405.03917)
- **Summary:** Couples multiple KV channels together for quantization, exploiting their interdependence (joint entropy grows slower than sum of marginal entropies). Achieves 1-bit per channel with 1.4-3.5x throughput improvement.

### 3.4 CacheGen
- **Title:** CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving
- **Authors:** Yuhan Liu, Hanchen Li et al. (University of Chicago)
- **Venue:** ACM SIGCOMM 2024
- **arXiv:** [2310.07240](https://arxiv.org/abs/2310.07240)
- **Summary:** Custom tensor encoder leveraging KV cache distributional properties for streaming-friendly compression. Reduces KV cache size by 3.5-4.3x and total context-loading delay by 3.2-3.7x.

### 3.5 PALU
- **Title:** Palu: Compressing KV-Cache with Low-Rank Projection
- **Authors:** Chi-Chih Chang et al.
- **Venue:** ICLR 2025
- **arXiv:** [2407.21118](https://arxiv.org/abs/2407.21118)
- **Summary:** Decomposes linear layers into low-rank matrices, caches compressed intermediate states, and reconstructs full KV on-the-fly. Compresses KV cache by 50% with up to 2.91x speedup when combined with quantization.

### 3.6 MiniCache
- **Title:** MiniCache: KV Cache Compression in Depth Dimension for Large Language Models
- **Authors:** Akide Liu et al.
- **Venue:** NeurIPS 2024
- **arXiv:** [2405.14366](https://arxiv.org/abs/2405.14366)
- **Summary:** Exploits high similarity between KV cache states of adjacent layers in the depth dimension (an overlooked axis). Disentangles states into magnitude and direction for interpolation. Achieves up to 5.02x compression with 4-bit quantization.

---

## 4. KV Cache Eviction and Token Selection

### 4.1 H2O (Heavy-Hitter Oracle)
- **Title:** H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- **Authors:** Zhenyu Zhang, Ying Sheng et al.
- **Venue:** NeurIPS 2023
- **arXiv:** [2306.14048](https://arxiv.org/abs/2306.14048)
- **Summary:** Observes that LLMs are >95% sparse at inference time. Formulates KV cache eviction as a dynamic submodular problem, retaining a balance of recent and "heavy hitter" tokens. Achieves up to 29x throughput improvement with 20% heavy hitters.

### 4.2 Scissorhands
- **Title:** Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time
- **Authors:** Zichang Liu, Aditya Desai et al.
- **Venue:** NeurIPS 2023
- **arXiv:** [2305.17118](https://arxiv.org/abs/2305.17118)
- **Summary:** Proposes the "persistence of importance" hypothesis: pivotal tokens that had substantial influence at one step will significantly influence future generations. Reduces KV cache by up to 5x (20x with 4-bit quantization) without finetuning.

### 4.3 StreamingLLM
- **Title:** Efficient Streaming Language Models with Attention Sinks
- **Authors:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (MIT, Meta AI)
- **Venue:** ICLR 2024
- **arXiv:** [2309.17453](https://arxiv.org/abs/2309.17453)
- **Summary:** Discovers the "attention sink" phenomenon -- initial tokens receive disproportionately high attention scores even when not semantically important. Retaining these sink tokens plus a sliding window enables infinite-length streaming with up to 22.2x speedup.

### 4.4 SnapKV
- **Title:** SnapKV: LLM Knows What You are Looking for Before Generation
- **Authors:** Yuhong Li, Yingbing Huang et al.
- **Venue:** arXiv 2024
- **arXiv:** [2404.14469](https://arxiv.org/abs/2404.14469)
- **Summary:** Uses an observation window at the end of prompts to identify per-head important KV positions with a pooling mechanism. Achieves 3.6x generation speed increase and 8.2x memory efficiency improvement at 16K tokens.

### 4.5 PyramidInfer
- **Title:** PyramidInfer: Pyramid KV Cache Compression for High-throughput LLM Inference
- **Authors:** Dongjie Yang, Xiaodong Han et al.
- **Venue:** ACL 2024 Findings
- **arXiv:** [2405.12532](https://arxiv.org/abs/2405.12532)
- **Summary:** Observes that the number of crucial KV pairs decreases layer by layer. Applies progressively more aggressive compression in deeper layers (pyramid shape). Achieves 2.2x throughput with 54% GPU memory reduction.

### 4.6 Quest
- **Title:** Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- **Authors:** Jiaming Tang, Yilong Zhao et al. (MIT Han Lab)
- **Venue:** ICML 2024
- **arXiv:** [2406.10774](https://arxiv.org/abs/2406.10774)
- **Summary:** Tracks min/max key values per KV cache page and estimates page criticality using query vectors, making sparsity query-aware rather than static. Achieves 2.23x self-attention speedup and 7.03x inference latency reduction.

---

## 5. Offloading KV Cache to CPU/SSD/Remote Storage

### 5.1 FlexGen
- **Title:** FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU
- **Authors:** Ying Sheng, Lianmin Zheng, Binhang Yuan et al.
- **Venue:** ICML 2023
- **arXiv:** [2303.06865](https://arxiv.org/abs/2303.06865)
- **Summary:** Aggregates memory from GPU, CPU, and disk, using linear programming to find optimal tensor placement and access patterns. First system to achieve 1 token/s generation for OPT-175B on a single 16GB GPU, with 100x higher max throughput.

### 5.2 InfiniGen
- **Title:** InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
- **Authors:** Wonbeom Lee et al. (Seoul National University)
- **Venue:** USENIX OSDI 2024
- **arXiv:** [2406.19707](https://arxiv.org/abs/2406.19707)
- **Summary:** Speculatively identifies essential KV cache entries by performing minimal "rehearsal" with current-layer inputs and next-layer weights, then prefetches only those entries from host memory. Achieves 3x improvement over prior offloading methods with better accuracy.

### 5.3 KVSwap
- **Title:** KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference
- **arXiv:** [2511.11907](https://arxiv.org/abs/2511.11907)
- **Summary:** Framework for extended-context inference on resource-constrained devices, building on PagedAttention for hierarchical GPU-CPU-disk KV management with diverse storage type support.

---

## 6. Prefix Caching and KV Cache Sharing

### 6.1 SGLang / RadixAttention
- **Title:** SGLang: Efficient Execution of Structured Language Model Programs
- **Authors:** Lianmin Zheng, Liangsheng Yin et al. (UC Berkeley, Stanford)
- **Venue:** arXiv 2024
- **arXiv:** [2312.07104](https://arxiv.org/abs/2312.07104)
- **Summary:** Introduces RadixAttention, which maintains KV caches in a radix tree for automatic reuse across requests sharing common prefixes. Enables efficient prefix search, insertion, and LRU eviction, achieving up to 5x higher throughput.

### 6.2 CachedAttention / AttentionStore
- **Title:** Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention
- **Authors:** Bin Gao et al.
- **Venue:** USENIX ATC 2024
- **arXiv:** [2403.19708](https://arxiv.org/abs/2403.19708)
- **Summary:** Saves KV caches to a hierarchical store (AttentionStore) across GPU/CPU/disk between conversation turns. Uses layer-wise pre-loading, asynchronous saving, and decoupled positional encoding. Reduces TTFT by up to 87% and prefill cost by 7.8x.

### 6.3 LMCache
- **Title:** LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference
- **Authors:** (LMCache team)
- **arXiv:** [2510.09665](https://arxiv.org/abs/2510.09665)
- **Summary:** An open-source KV caching layer that extracts, stores, and shares KV caches across vLLM/SGLang engine instances via GPU, CPU DRAM, and local disk tiers. Supports reuse of KV caches for repeated content (not just prefixes). Achieves 3-10x latency reduction and up to 15x throughput improvement.

---

## 7. Memory-Efficient Attention Mechanisms

### 7.1 FlashAttention
- **Title:** FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Authors:** Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, Christopher Re (Stanford, U. Buffalo)
- **Venue:** NeurIPS 2022
- **arXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)
- **Summary:** IO-aware exact attention using tiling to minimize HBM reads/writes, computing attention in SRAM blocks. Reduces memory from O(N^2) to O(N) with 10-20x memory savings at long sequences. Foundational work enabling long-context LLMs.

### 7.2 FlashAttention-2
- **Title:** FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
- **Authors:** Tri Dao
- **Venue:** ICLR 2024
- **arXiv:** [2307.08691](https://arxiv.org/abs/2307.08691)
- **Summary:** Improves GPU utilization through better work partitioning between thread blocks and warps, reducing non-matmul FLOPs. Achieves 2x speedup over FlashAttention, reaching 50-73% of theoretical max FLOPs/s on A100.

### 7.3 FlashInfer
- **Title:** FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving
- **Authors:** Zihao Ye et al.
- **arXiv:** [2501.01005](https://arxiv.org/abs/2501.01005)
- **Summary:** A kernel library extending FlashAttention templates to support sparse attention, variable-length sequences, and diverse attention variants (MQA, GQA, MLA). Integrated into SGLang, vLLM, and MLC-Engine as production attention backend.

### 7.4 Multi-Query Attention (MQA)
- **Title:** Fast Transformer Decoding: One Write-Head is All You Need
- **Authors:** Noam Shazeer (Google)
- **Venue:** arXiv 2019
- **arXiv:** [1911.02150](https://arxiv.org/abs/1911.02150)
- **Summary:** Shares key and value heads across all attention heads, greatly reducing KV cache size and memory bandwidth requirements during incremental decoding. Enables 10-100x smaller KV storage with minor quality degradation.

### 7.5 Grouped-Query Attention (GQA)
- **Title:** GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Authors:** Joshua Ainslie, James Lee-Thorp et al. (Google)
- **Venue:** EMNLP 2023
- **arXiv:** [2305.13245](https://arxiv.org/abs/2305.13245)
- **Summary:** Proposes an intermediate between MHA and MQA: groups of query heads share key/value heads. Provides a recipe for uptraining existing MHA checkpoints to GQA using only 5% of original pre-training compute. Adopted by Llama 2, Llama 3, and many modern LLMs.

---

## 8. Distributed KV Cache Systems

### 8.1 Infinite-LLM / DistAttention
- **Title:** Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache
- **Authors:** Bin Lin, Chen Zhang et al.
- **arXiv:** [2401.02669](https://arxiv.org/abs/2401.02669)
- **Summary:** Introduces DistAttention, which segments KV cache into rBlocks distributed across GPUs and CPUs throughout a data center. Dynamically manages distributed memory, achieving 1.35-3.4x throughput and supporting up to 2M token contexts on 32 GPUs.

### 8.2 MemServe
- **Title:** MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool
- **Authors:** Cunchen Hu, Heyang Huang et al.
- **arXiv:** [2406.17565](https://arxiv.org/abs/2406.17565)
- **Summary:** Introduces MemPool, an elastic memory pool managing distributed CPU DRAM and GPU HBM, with a global prompt tree-based locality-aware scheduler. First system to combine context caching with disaggregated inference.

### 8.3 DéjàVu
- **Title:** DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving
- **Authors:** Foteini Strati, Sara Mcallister et al. (Microsoft Research, ETH Zurich)
- **Venue:** ICML 2024
- **arXiv:** [2403.01876](https://arxiv.org/abs/2403.01876)
- **Summary:** A KV cache streaming library enabling prompt-token disaggregation, microbatch swapping for GPU memory management, and state replication for fault tolerance. Achieves 1.88-2x higher throughput on OPT-66B/BLOOM-176B.

### 8.4 TraCT (CXL-based)
- **Title:** TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale
- **Authors:** (Multiple authors)
- **arXiv:** [2512.18194](https://arxiv.org/abs/2512.18194)
- **Summary:** Uses CXL shared memory as network-free KV transfer substrate and rack-wide prefix-aware KV cache, eliminating the RDMA hop. Reduces TTFT by up to 9.8x and P99 latency by 6.2x compared to RDMA-based baselines.

---

## 9. Other Notable LLM Storage/Memory Innovations

### 9.1 Exploring CXL-based KV Cache Storage
- **Authors:** Yupeng Tang et al.
- **Venue:** NeurIPS 2024 Workshop (ML for Systems)
- **Link:** [NeurIPS 2024](https://neurips.cc/virtual/2024/103619)
- **Summary:** Explores using CXL memory for KV cache storage, showing comparable latency/bandwidth to CPU-GPU interconnect. Reduces GPU requirements by up to 87% with 7.5x higher GPU utilization for prefill.

### 9.2 Online Scheduling for LLM Inference with KV Cache Constraints
- **arXiv:** [2502.07115](https://arxiv.org/abs/2502.07115)
- **Summary:** Provides theoretical foundations for LLM inference scheduling under KV cache memory constraints, with polynomial-time optimal algorithms for semi-online settings and constant-regret algorithms for fully online settings.

---

## Key Takeaways

1. **PagedAttention (vLLM) is foundational** -- virtually all modern LLM serving systems build on its paged memory management for KV cache, drawing from OS virtual memory concepts.

2. **Disaggregation is the production trend** -- separating prefill and decode (Mooncake, DistServe, Splitwise) is now standard in production systems (NVIDIA Dynamo, llm-d, SGLang, vLLM all support it).

3. **Compression and quantization are complementary** -- KIVI (2-bit), KVQuant (3-bit), and CQ (1-bit per channel) show that extreme KV cache quantization is viable, and these compose well with eviction and offloading.

4. **Eviction policies are maturing** -- from H2O's heavy-hitter approach to StreamingLLM's attention sinks to SnapKV's observation-window selection, the field is converging on query-aware, per-head token importance.

5. **Multi-tier storage is essential for long context** -- as context windows grow to millions of tokens, hierarchical GPU-CPU-SSD-CXL-remote storage for KV cache is necessary, with intelligent prefetching (InfiniGen) to hide latency.

6. **Attention mechanism design directly impacts storage** -- MQA and GQA reduce KV cache size at the architecture level, while FlashAttention reduces memory footprint at the kernel level. These are "upstream" solutions.

7. **CXL is an emerging frontier** -- CXL shared memory offers a potential paradigm shift by providing high-bandwidth, low-latency shared KV cache access without network overhead.

---

## Sources

1. [PagedAttention/vLLM - arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
2. [Mooncake - arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
3. [DistServe - arXiv:2401.09670](https://arxiv.org/abs/2401.09670)
4. [Splitwise - arXiv:2311.18677](https://arxiv.org/abs/2311.18677)
5. [KIVI - arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
6. [KVQuant - arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
7. [CQ - arXiv:2405.03917](https://arxiv.org/abs/2405.03917)
8. [CacheGen - arXiv:2310.07240](https://arxiv.org/abs/2310.07240)
9. [PALU - arXiv:2407.21118](https://arxiv.org/abs/2407.21118)
10. [MiniCache - arXiv:2405.14366](https://arxiv.org/abs/2405.14366)
11. [H2O - arXiv:2306.14048](https://arxiv.org/abs/2306.14048)
12. [Scissorhands - arXiv:2305.17118](https://arxiv.org/abs/2305.17118)
13. [StreamingLLM - arXiv:2309.17453](https://arxiv.org/abs/2309.17453)
14. [SnapKV - arXiv:2404.14469](https://arxiv.org/abs/2404.14469)
15. [PyramidInfer - arXiv:2405.12532](https://arxiv.org/abs/2405.12532)
16. [Quest - arXiv:2406.10774](https://arxiv.org/abs/2406.10774)
17. [FlexGen - arXiv:2303.06865](https://arxiv.org/abs/2303.06865)
18. [InfiniGen - arXiv:2406.19707](https://arxiv.org/abs/2406.19707)
19. [SGLang/RadixAttention - arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
20. [CachedAttention - arXiv:2403.19708](https://arxiv.org/abs/2403.19708)
21. [LMCache - arXiv:2510.09665](https://arxiv.org/abs/2510.09665)
22. [FlashAttention - arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
23. [FlashAttention-2 - arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
24. [FlashInfer - arXiv:2501.01005](https://arxiv.org/abs/2501.01005)
25. [MQA - arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
26. [GQA - arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
27. [Infinite-LLM - arXiv:2401.02669](https://arxiv.org/abs/2401.02669)
28. [MemServe - arXiv:2406.17565](https://arxiv.org/abs/2406.17565)
29. [DéjàVu - arXiv:2403.01876](https://arxiv.org/abs/2403.01876)
30. [TraCT - arXiv:2512.18194](https://arxiv.org/abs/2512.18194)
31. [Orca - OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/yu)
32. [TetriInfer - arXiv:2401.11181](https://arxiv.org/abs/2401.11181)
33. [KVSwap - arXiv:2511.11907](https://arxiv.org/abs/2511.11907)

## Methodology

Searched 20+ queries across web sources covering KV cache paging, disaggregated serving, compression/quantization, eviction policies, offloading, prefix caching, memory-efficient attention, and distributed KV cache systems. Analyzed 33 primary papers spanning 2019-2025.
