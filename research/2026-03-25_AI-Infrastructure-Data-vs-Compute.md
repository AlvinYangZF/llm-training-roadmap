# The Future of AI Infrastructure: Data vs Computing Power — Which Is More Important?

**Comprehensive Research Report | March 25, 2026**

---

## Executive Summary (Chinese) / 中文执行摘要

**AI基础设施的未来：数据与算力，哪个更重要？**

本报告深入研究了AI基础设施中数据与算力之间的核心博弈。主要发现如下：

**算力方面：** 2026年全球超大规模数据中心资本支出预计超过6600亿美元，其中约75%（4500亿美元）直接用于AI基础设施。NVIDIA市值达4.2万亿美元，数据中心芯片市场份额81%。GPU路线图从H100到B200再到Rubin（R200，2026年Q2出货），每代性能提升2-3倍。然而，传统的"规模扩张"时代正在终结——Ilya Sutskever和Yann LeCun均认为单纯增加算力已无法带来突破性进展。

**数据方面：** Epoch AI预测到2028年高质量互联网文本数据将被耗尽，部分预测认为2026年即可能出现短缺。合成数据正成为关键替代方案——Gartner预计2026年AI项目中75%的数据将是合成数据。微软的SynthLLM、NVIDIA的Nemotron、HuggingFace的Cosmopedia等项目正在推动合成数据规模化。但"模型坍塌"风险意味着人类数据仍不可替代。

**核心转变：** AI行业正经历三大范式转移：(1) 从预训练扩展转向后训练优化（RLHF/DPO/GRPO），DeepSeek-R1证明了强化学习可带来显著推理能力提升；(2) 从训练算力转向推理算力——2026年推理工作负载占AI计算的约2/3，占AI云基础设施支出的55%；(3) 从密集模型转向稀疏专家混合模型（MoE），在更低计算成本下实现同等质量。

**地缘政治维度：** 美国芯片出口管制迫使中国走向效率创新之路，DeepSeek在H800芯片上实现了与美国顶尖模型相当的性能。中国在制造业数据和超级应用（微信）生态方面具有独特优势。

**核心结论：** 数据与算力并非二选一，而是正在走向融合。短期内（2026-2027），算力投资仍将主导；中期（2028-2030），数据质量和效率将成为差异化的关键因素。最终赢家将是那些能够在两个维度上同时优化的组织——用更智能的方式处理数据，用更高效的架构利用算力。

---

## Table of Contents

1. [Scaling Laws & The Compute Argument](#1-scaling-laws--the-compute-argument)
2. [The Data Wall & The Data Argument](#2-the-data-wall--the-data-argument)
3. [Post-Training Revolution](#3-post-training-revolution)
4. [Expert Opinions & Industry Positions](#4-expert-opinions--industry-positions)
5. [Investment & Market Dynamics (2025-2026)](#5-investment--market-dynamics-2025-2026)
6. [Emerging Paradigms That Change the Equation](#6-emerging-paradigms-that-change-the-equation)
7. [Geopolitical Dimension: China vs US](#7-geopolitical-dimension-china-vs-us)
8. [Synthesis & Conclusions](#8-synthesis--conclusions)

---

## 1. Scaling Laws & The Compute Argument

### 1.1 The Evolution of Scaling Laws

The debate over compute vs. data is fundamentally grounded in scaling laws -- the empirical relationships governing how model performance improves with increased resources.

**Kaplan Scaling Laws (OpenAI, 2020):**
The original OpenAI scaling laws (Kaplan et al.) suggested that as pre-training compute budget increases, model size should be scaled more aggressively than data. Specifically, for a 10x increase in training budget, the optimal strategy was to scale model size by ~5.5x while only increasing data by ~1.8x. This led to the era of "parameter maximalism" that produced GPT-3 (175B parameters trained on 300B tokens).

**Chinchilla Scaling Laws (DeepMind, 2022):**
DeepMind's Chinchilla paper fundamentally challenged Kaplan's assumptions, demonstrating that many existing models (including GPT-3) were significantly over-parameterized and under-trained. The key finding: for a fixed compute budget, optimal performance requires a near-proportional relationship between parameters and training tokens. The Chinchilla-optimal ratio was approximately 20 tokens per parameter.

**Post-Chinchilla Reality (2023-2026):**
In practice, the industry has moved far beyond Chinchilla-optimal ratios:

| Model | Parameters | Training Tokens | Tokens/Parameter | vs. Chinchilla |
|-------|-----------|----------------|-------------------|----------------|
| Chinchilla | 70B | 1.4T | 20:1 | 1x |
| Llama 2 70B | 70B | 2T | ~29:1 | 1.4x |
| Llama 3 70B | 70B | 15T | ~200:1 | 10x |
| Llama 3 8B | 8B | 15T | ~1,875:1 | 94x |

**Key insight:** Research through 2025 has shown that training smaller models for much longer (far beyond Chinchilla-optimal) can yield better cost-efficiency when accounting for inference demand. Loss continues to decrease at token-to-parameter ratios up to 10,000:1, though with diminishing per-token improvements.

*Source: [LLM Scaling Laws: Analysis from AI Researchers in 2026](https://aimultiple.com/llm-scaling-laws); [Scaling Laws for LLMs: From GPT-3 to o3](https://cameronrwolfe.substack.com/p/llm-scaling-laws)*

### 1.2 The Compute Investment Tsunami

The sheer scale of capital being deployed toward AI compute in 2025-2026 is unprecedented in technology history:

**2026 Hyperscaler Capex Commitments:**

| Company | 2026 Capex (Projected) | YoY Change |
|---------|----------------------|------------|
| Amazon (AWS) | ~$200B | Massive increase |
| Alphabet (Google) | $175-185B | ~75% increase |
| Meta | $115-135B | ~2x increase |
| Microsoft | ~$120B+ | Significant increase |
| Oracle | ~$50B | Major expansion |
| **Total (Big 5)** | **$660-690B** | **~36% vs 2025** |

Of this total, approximately 75% (~$450B) is directly tied to AI infrastructure -- servers, GPUs, data centers, and supporting equipment. The 2025 baseline was already ~$380-400B collectively.

**The financing strain is real:** Amazon is projected to have negative free cash flow of almost $17 billion in 2026 (per Morgan Stanley estimates). Hyperscalers are increasingly leaning on debt markets to bridge the gap between rising AI capex and internal free cash flow.

*Source: [Hyperscaler capex > $600B in 2026](https://techblog.comsoc.org/2025/12/22/hyperscaler-capex-600-bn-in-2026-a-36-increase-over-2025-while-global-spending-on-cloud-infrastructure-services-skyrockets/); [AI Capex 2026: The $690B Infrastructure Sprint](https://futurumgroup.com/insights/ai-capex-2026-the-690b-infrastructure-sprint/); [CNBC: Tech AI spending approaches $700B in 2026](https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html)*

### 1.3 GPU/TPU Roadmap: The Hardware Trajectory

NVIDIA's roadmap shows a relentless cadence of generational improvements:

**H100 (Hopper) -- 2023-2024 (Legacy Leader):**
- 80 billion transistors, TSMC 4nm
- Defined the modern AI training era
- Prices stabilizing as Blackwell ships in volume

**B200/B300 (Blackwell) -- 2024-2025 (Current Generation):**
- B200 shipping in volume from Q1 2025
- B300 (Blackwell Ultra) shipping Q4 2025
- ~1.5x Blackwell FP4 compute, up to 288GB HBM3E per GPU

**R200 (Rubin) -- 2026 (Next Generation):**
- Shipping Q2 2026
- First NVIDIA part with HBM4 memory + NVLink 6
- Up to 22 TB/s memory bandwidth per GPU (2.75x B200/B300)
- Major architectural "tick"

**Beyond 2026:**
- VR200 (Vera Rubin / Rubin Ultra) -- Q2 2027, expected 2-3x Rubin performance
- Feynman architecture -- 2028+

**Critical constraint:** Supply chain lags marketing slides by 6-12 months. Both Blackwell and Rubin rely on cutting-edge chip-on-wafer-on-substrate (CoWoS) packaging and HBM stacks that remain capacity-constrained.

*Source: [NVIDIA Rubin at GTC 2026](https://blog.barrack.ai/nvidia-rubin-specs-architecture-2026/); [Tom's Hardware: NVIDIA announces Rubin GPUs in 2026](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after); [NextPlatform: NVIDIA Draws GPU System Roadmap Out To 2028](https://www.nextplatform.com/2025/03/19/nvidia-draws-gpu-system-roadmap-out-to-2028/)*

### 1.4 Diminishing Returns on Compute Scaling

There is growing evidence that pure compute scaling is hitting a wall:

- **Benchmark saturation:** Compute grows 10-100x while accuracy improvements barely move on many standard benchmarks.
- **Compute-efficient frontier (CEF):** As models approach this theoretical limit of resource efficiency with current architectures, performance improvements slow dramatically.
- **Researchers studying advanced reasoning systems in 2025 found that adding more computational steps no longer delivered proportionate improvements** -- the returns from scaling are slowing while costs accelerate.
- **A December 2025 arXiv paper** ("The AI Scaling Wall of Diminishing Returns") formally characterized these diminishing returns.

**Contrasting view:** Despite scaling concerns, the industry continues to invest massively. Deloitte's 2026 predictions argue that "AI's next phase will likely demand more computational power, not less" -- but the compute will shift from pure pre-training scaling to inference, post-training, and agent workloads.

**Epoch AI's analysis** ("Can AI Scaling Continue Through 2030?") suggests that while data and algorithmic efficiency may constrain pure LLM pre-training scaling, the total demand for compute will continue to grow through new paradigms.

*Source: [Is There a Wall? Evidence-Based Analysis of Diminishing Returns](https://medium.com/@adnanmasood/is-there-a-wall-34d02dfd85f3); [The AI Scaling Wall of Diminishing Returns](https://arxiv.org/abs/2512.20264); [Epoch AI: Can AI Scaling Continue Through 2030?](https://epoch.ai/blog/can-ai-scaling-continue-through-2030)*

---

## 2. The Data Wall & The Data Argument

### 2.1 "We've Run Out of Internet Data" -- Evidence Assessment

**The case that data is running out:**
- Epoch AI predicts that all high-quality textual data on the internet will be exhausted by **2028**, with machine learning datasets potentially depleting "high-quality language data" as early as **2026**.
- The total stock of publicly available, high-quality text on the internet is estimated at roughly 10-20 trillion tokens.
- Major models are already training on significant fractions of this total: Llama 3 used 15 trillion tokens.

**The case that data is NOT running out:**
- This is only true for **high-quality English text data**. Multimodal data (video, audio, images, sensor data) represents orders of magnitude more potential training signal.
- Private and proprietary data (enterprise documents, industrial telemetry, medical records) remains largely untapped.
- Techniques like curriculum learning, data mixing, and multi-epoch training with careful deduplication extend the effective data supply.
- Low-resource languages and specialized domains offer frontier data.

**Verdict:** The "data wall" is real for *text-only pre-training on public internet data* but is not an absolute constraint when considering the full data landscape.

*Source: [The AI Industry Faces 'Data Wall' Challenge](https://news.aibase.com/news/10757); [Microsoft Research: SynthLLM Breaking the AI Data Wall](https://www.microsoft.com/en-us/research/articles/synthllm-breaking-the-ai-data-wall-with-scalable-synthetic-data/); [World Economic Forum: AI Training Data Is Running Low](https://www.weforum.org/stories/2025/12/data-ai-training-synthetic/)*

### 2.2 Synthetic Data: Can AI Generate Its Own Training Data?

Synthetic data has emerged as the primary strategy for scaling beyond the data wall:

**Key Projects and Results:**

| Project | Creator | Scale | Key Finding |
|---------|---------|-------|-------------|
| **Cosmopedia** | HuggingFace | 25B tokens, 30M+ files | Largest open synthetic dataset; <1% duplicate content rate |
| **SynthLLM** | Microsoft Research | Variable | Confirmed scaling laws hold for synthetic data |
| **Nemotron-CC** | NVIDIA | 10T+ tokens | Open datasets with synthetic subset; competitive with curated human data |
| **Phi-3** | Microsoft | 3.8B params | Demonstrated small models trained largely on synthetic data rival much larger models |
| **BeyondWeb** | Datology AI | Trillion-scale | Outperformed Cosmopedia by 5.1pp and Nemotron-CC by 2.6pp on 14 benchmarks |

**Gartner projection:** Synthetic data will make up roughly **75% of data used in AI projects by 2026**.

**The "model collapse" risk:** Training AI on AI-generated data can produce progressively lower-quality outputs over generations. This phenomenon, well-documented in 2024-2025 research, means synthetic data strategies must be carefully architected with human data anchoring.

**The diversity challenge:** The key ingredient for effective synthetic datasets is maximizing diversity. Cosmopedia addressed this by generating over 30 million unique prompts covering hundreds of subjects.

*Source: [Cosmopedia: Large-Scale Synthetic Data for Pre-Training](https://huggingface.co/blog/cosmopedia); [AI Training in 2026: Anchoring Synthetic Data in Human Truth](https://invisibletech.ai/blog/ai-training-in-2026-anchoring-synthetic-data-in-human-truth); [Synthetic Data Explosion: How 2026 Reduces Data Costs by 70%](https://www.cogentinfo.com/resources/synthetic-data-explosion-how-2026-reduces-data-costs-by-70)*

### 2.3 Data Quality > Data Quantity

The LIMA paper (Meta, 2023) demonstrated a provocative finding: fine-tuning a 65B LLaMA model on just **1,000 carefully curated examples** produced results competitive with models fine-tuned on 50,000+ noisy examples. This finding has been repeatedly validated and extended:

- **Instruction quality** during fine-tuning matters far more than instruction quantity.
- **Data curation and filtering** during pre-training can yield 2-5x effective data efficiency gains.
- **NVIDIA's Nemotron-CC** specifically focuses on a "high-quality synthetic subset" that outperforms much larger unfiltered datasets.
- **Datology AI's BeyondWeb** showed that careful data composition at the trillion-token scale still yields meaningful quality improvements.

**Implication:** The future of data advantage lies not in having more data, but in having better data -- better curated, more diverse, more accurately labeled, and more specifically relevant to target domains.

### 2.4 Proprietary Data Moats

In the age of foundation models, proprietary data remains one of the few defensible competitive advantages:

**Bloomberg:** Invested ~$10M in creating BloombergGPT, trained on 40+ years of proprietary financial data (363B tokens of financial documents). This domain-specific training created capabilities that general-purpose models struggled to match on financial tasks.

**Medical/Healthcare:** Access to de-identified patient records, clinical trial data, and medical imaging datasets creates moats for companies like Tempus, Flatiron Health, and hospital systems partnering with AI companies.

**Legal:** Thomson Reuters (Westlaw), LexisNexis, and specialized legal AI companies leverage proprietary case law databases and legal document archives.

**However, data moats are eroding faster than expected.** Foundation models can often achieve 80-90% of domain-specific performance through general training plus prompting, reducing the marginal value of proprietary data. The remaining 10-20% gap is where proprietary data still matters most -- and for regulated industries, this gap is critical.

*Source: [Data Moats in the AI Era: What Actually Survives](https://fergusonanalytics.com/blog/ai-data-moats/); [Bloomberg's $10M Data Experiment](https://medium.com/@arjun_shah/bloombergs-10m-data-experiment-8c552ca5c212)*

### 2.5 Multimodal Data as the Next Frontier

While text data may be approaching exhaustion, multimodal data represents a vast untapped frontier:

- **Video:** YouTube alone hosts 800M+ videos. Training on video enables understanding of physical dynamics, temporal reasoning, and embodied knowledge.
- **Audio/Speech:** Podcasts, call center recordings, and multilingual speech provide rich signal.
- **Sensor Data:** IoT devices, autonomous vehicles, manufacturing sensors, and satellite imagery generate petabytes daily.
- **Robotics Data:** Physical interaction data from robots is critical for embodied AI and is extremely scarce -- making it perhaps the most valuable data type going forward.

The shift to multimodal training simultaneously increases both compute and data requirements, creating a dual bottleneck that favors organizations with advantages in both dimensions.

---

## 3. Post-Training Revolution

### 3.1 The Evolving Post-Training Stack

The standard recipe of "pre-train on trillions of tokens, then RLHF with human preferences" is now considered obsolete. Every major model released in the past year uses a different, more sophisticated post-training stack:

**Modern post-training pipeline (2025-2026):**

1. **Supervised Fine-Tuning (SFT):** Instruction following using curated demonstration data
2. **Preference Optimization (DPO/SimPO/KTO):** Alignment using preference pairs, replacing or augmenting PPO-based RLHF
3. **Reinforcement Learning with Verifiable Rewards (RLVR/GRPO/DAPO):** Reasoning enhancement using automated verification rather than human labelers

**The key shift:** From human-labeled rewards to automated verification and self-play. This is critical because human preference data is scarce, expensive (often $50-200/hour for expert annotators), and creates a scaling bottleneck. Automated verification (e.g., checking if a math answer is correct, if code compiles and passes tests) removes this bottleneck for domains where correctness can be verified.

*Source: [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026); [LLM Post-Training: A Deep Dive](https://arxiv.org/html/2502.21321v2)*

### 3.2 Test-Time Compute: The Paradigm Shift

Perhaps the most significant development in 2024-2025 was the emergence of **test-time compute scaling** -- the idea that you can improve model outputs by spending more compute at inference time rather than (or in addition to) training time.

**OpenAI's o1/o3 Series:**
- First models to demonstrate systematic inference-time scaling
- Performance improves with the length of the chain-of-thought reasoning process
- Achieved breakthrough results in mathematics, coding, and scientific reasoning
- The insight: "the more tokens a model generates, the better its response"

**DeepSeek-R1:**
- Trained via large-scale RL without supervised fine-tuning (DeepSeek-R1-Zero)
- AIME 2024 pass@1 score increased from 15.6% to **71.0%**, matching OpenAI o1-0912
- Demonstrated that powerful reasoning behaviors emerge naturally from pure RL training
- Key innovation: RLVR (RL with Verifiable Rewards) using deterministic correctness labels

**Implication for Data vs. Compute:** Test-time compute fundamentally reshapes the equation. Instead of needing ever-more training data, models can "think harder" at inference time. This shifts the bottleneck from training data to:
1. **Verification data** -- problems with checkable answers for RLVR training
2. **Inference compute** -- GPUs deployed for serving, not training
3. **Algorithm design** -- the architecture and training procedure that enable effective reasoning

*Source: [DeepSeek-R1 Paper Explained](https://aipapersacademy.com/deepseek-r1/); [How to Train LLMs to "Think" (o1 & DeepSeek-R1)](https://towardsdatascience.com/how-to-train-llms-to-think-o1-deepseek-r1/); [DeepSeek-R1: Incentivizing Reasoning via RL](https://arxiv.org/html/2501.12948v1)*

### 3.3 Does Post-Training Data Matter More Than Pre-Training Data?

The evidence suggests a nuanced answer:

- **Pre-training data** provides the foundational knowledge and general capabilities. Without sufficient pre-training, no amount of post-training can compensate.
- **Post-training data** has an outsized impact per token. The LIMA finding (1,000 high-quality examples rivaling 50,000 noisy ones) demonstrates that the marginal value of carefully curated post-training data far exceeds that of pre-training data.
- **The sweet spot is shifting:** As pre-training approaches diminishing returns (due to data exhaustion and compute scaling limits), the marginal gains from post-training innovation are accelerating.

**Forward prediction:** By 2027, the competitive differentiation between frontier models will be primarily determined by post-training methodology (what data, what reward signals, what RL algorithms) rather than pre-training scale.

---

## 4. Expert Opinions & Industry Positions

### 4.1 Ilya Sutskever: "The Age of Scaling Is Over"

Sutskever, co-founder of OpenAI and now leading Safe Superintelligence Inc. (SSI), has undergone a significant evolution in his views:

- **2020-2023:** A primary architect and evangelist of the scaling paradigm. His conviction that "scale is all you need" drove OpenAI's strategy from GPT-2 through GPT-4.
- **2025:** Publicly stated that the "age of scaling" (roughly 2020-2025) is ending. The period when big data and compute almost guaranteed progress is giving way to the **"age of research"** where new fundamental ideas are needed.
- **Key quote:** He notes that the main driver of improvement -- high-quality human-generated text -- is limited, and that reaching AGI will require new breakthroughs beyond scaling existing approaches.
- **Current position:** Founded SSI specifically to pursue safety-focused superintelligence, implying he believes the path forward requires qualitatively different approaches.

*Source: [Ilya Sutskever: AI's Bottleneck Is Ideas, Not Compute](https://www.calcalistech.com/ctechnews/article/h1fudk7z11x); [What Ilya Sutskever Thinks About AI Progress in 2025](https://bytepawn.com/ai-ilya-dwarkesh-satya-altman-2025.html)*

### 4.2 Sam Altman: "Compute Is the Most Important Resource"

Altman remains the most bullish voice on continued compute scaling:

- **"The Gentle Singularity" blog post (mid-2025):** Predicted "We already know how to build AGI" and firmly believes the Scaling Law is far from reaching its ceiling.
- **Vision:** The cost of intelligence will approach zero with automated electricity production.
- **Strategy:** Still believes in another few big rounds of scaling (GPT-5 to GPT-6) with major payoffs.
- **Investment posture:** OpenAI has raised over $40B (including SoftBank's landmark investment) to fund massive compute infrastructure.

**The tension:** Altman's optimism coexists with his acknowledgment that new work is needed on safety and agents, suggesting even he recognizes that raw scaling alone is insufficient.

### 4.3 Yann LeCun: "LLMs Are a Dead End"

LeCun (Meta's former Chief AI Scientist, 2024 Turing Award co-recipient) represents the most radical departure:

- **Core thesis:** LLMs "can't understand the physical world," "don't have persistent memory," and "can't really reason."
- **Proposed alternative:** World models that learn from interaction with the environment, closer to how animals and humans learn -- what he calls "JEPA" (Joint Embedding Predictive Architecture).
- **November 2025:** Left Meta to found a new company focused on "Advanced Machine Intelligence" built on these principles.
- **Implication for data vs. compute:** LeCun's position suggests that neither more data nor more compute within the current paradigm will suffice. Instead, we need fundamentally different architectures that are more data-efficient.

*Source: [Yann LeCun: Large Models a "Dead End"](https://eu.36kr.com/en/p/3571987975018880); [Ilya Sutskever, Yann LeCun and the End of "Just Add GPUs"](https://www.abzglobal.net/web-development-blog/ilya-sutskever-yann-lecun-and-the-end-of-just-add-gpus)*

### 4.4 Dario Amodei: Cautious Optimism with Scaling Concerns

Anthropic's CEO threads a careful line:

- **Revenue trajectory:** $0 (2023) to $100M (2023) to $1B (2024) to $9-10B (2025) -- proving the commercial viability of frontier models.
- **Scaling caution:** Warned that some companies are **"YOLO-ing"** capital by pouring billions into massive, multi-year infrastructure bets.
- **Key concern:** If the industry hits a "scaling wall" where more compute no longer produces dramatically better models, these investments could become stranded.
- **Quote:** "Data centers take 1-2 years to build, and decisions about 2027 compute needs are being made right now. If he buys too much compute, he might not get enough revenue to pay for it, and in the extreme case, there's the risk of going bankrupt."
- **Position on the spectrum:** Believes we are "near the end of the exponential" but not yet at the wall.

*Source: [Dario Amodei -- "We Are Near the End of the Exponential"](https://www.dwarkesh.com/p/dario-amodei-2); [The Scaling Paradox](https://www.forethought.org/research/the-scaling-paradox)*

### 4.5 Andrej Karpathy: Pragmatic Evolution

Karpathy, co-founder of OpenAI and former Tesla AI lead:

- **2025 paradigm shifts:** Outlined fast inference engines, model distillation, real-time agents, neural GPUs, and the rise of high-quality open models.
- **"Slopacolypse" prediction for 2026:** Warned that AI-generated low-quality content will flood GitHub, Substack, arXiv, X/Instagram, and generally all digital media -- directly impacting the quality of future training data.
- **Implication:** If the internet fills with AI-generated content, the value of curated human data increases dramatically.

*Source: [Andrej Karpathy Predicts "Slopacolypse" in 2026](https://cybernews.com/ai-news/andrej-karpathy-slopacolypse/)*

### 4.6 The Expert Consensus Map

| Expert | Compute Still Scales? | Data Is the Bottleneck? | New Paradigm Needed? |
|--------|----------------------|------------------------|---------------------|
| Sam Altman | Yes (strongly) | No (for now) | Not yet |
| Dario Amodei | Yes (but slowing) | Partially | Soon |
| Ilya Sutskever | No (era ending) | Yes | Yes (now) |
| Yann LeCun | No (wrong paradigm) | Yes (wrong type) | Yes (fundamentally) |
| Andrej Karpathy | Partially | Yes (quality crisis) | Evolving |

---

## 5. Investment & Market Dynamics (2025-2026)

### 5.1 NVIDIA: The Undisputed Compute King

**Market position as of March 2026:**
- **Market cap:** $4.238 trillion -- the world's most valuable company
- **Became the first $5 trillion company** (briefly) in February 2026
- **Data center chip market share:** 81% by revenue (IDC)

**Financial performance:**
- **Q4 FY2026 revenue:** $68 billion (73% YoY growth, accelerating from Q3)
- **Full-year FY2026 data center revenue:** $194 billion (68% YoY growth)
- **Scale:** Data center business grew nearly 13x since ChatGPT's emergence in fiscal 2023
- **FY2026 total revenue trajectory:** Approaching ~$500 billion annualized

**Analyst outlook:** Some analysts project NVIDIA could reach $6 trillion market cap in 2026, with longer-term projections suggesting potential for $20 trillion by 2030.

*Source: [How NVIDIA Became the First $5 Trillion Company](https://www.cnn.com/2026/02/07/business/nvidia-trillion-valuation-ai-chips-vis); [NVIDIA Q4 FY2026 Earnings](https://futurumgroup.com/insights/nvidia-q4-fy-2026-earnings-highlight-durable-ai-infrastructure-demand/); [NVIDIA Financial Results Q3 FY2026](https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-third-quarter-fiscal-2026)*

### 5.2 Data Infrastructure Companies

**Databricks:**
- **Valuation:** $134 billion (Series L, completed February 2026)
- **Funding:** Raised $5 billion + $2 billion in debt capacity
- **Revenue:** $5.4 billion ARR (65% YoY growth), with free cash flow positive
- **AI revenue:** $1.4 billion annualized from AI products
- **CEO ambition:** Ali Ghodsi publicly targeting $1 trillion valuation

**Snowflake:**
- **FY2026 revenue:** $4.68 billion (29% YoY growth)
- **Product revenue:** $4.72 billion annually
- **RPO:** $9.77 billion (42% YoY growth)
- **AI-specific revenue:** Crossed $100M run rate tied directly to AI data
- **460+ customers** spending more than $1M annually
- **Net revenue retention:** 126%

**Scale AI:**
- **Valuation:** $29 billion (after Meta's $14.3 billion investment for 49% stake, June 2025)
- **Revenue:** $870M in 2024, $1.5B annualized run rate by year-end 2024, projections exceeding $2B in 2025
- **Strategic position:** Critical data labeling and curation infrastructure for frontier model training

*Source: [Databricks Raises $4B at $134B Valuation](https://techcrunch.com/2025/12/16/databricks-raises-4b-at-134b-valuation-as-its-ai-business-heats-up/); [Snowflake Financial Results FY2025](https://www.snowflake.com/en/news/press-releases/snowflake-reports-financial-results-for-the-fourth-quarter-and-full-year-of-fiscal-2025/); [Scale AI Valuation](https://www.premieralts.com/companies/scale-ai/valuation)*

### 5.3 AI Infrastructure Spending: Compute vs. Data Breakdown

**Where the money goes:**

The overwhelming majority of AI infrastructure spending goes to compute:
- **~75% of hyperscaler AI capex** goes to GPUs, servers, and data center construction
- **~15-20%** goes to networking, storage, and cooling infrastructure
- **~5-10%** goes to data acquisition, labeling, and data infrastructure

However, the data ecosystem is growing faster from a lower base:
- Data labeling market: $3.5B in 2025, projected to reach $8B+ by 2028
- Data infrastructure (Databricks, Snowflake, etc.): Combined $10B+ revenue in 2025
- Synthetic data market: Projected to grow from $1.5B to $5B+ by 2028

### 5.4 VC Investment Trends

**AI captured an extraordinary share of global venture funding in 2025:**
- AI companies captured **61% of global VC investment** in 2025, totaling **$258.7 billion** of the $427.1 billion invested globally
- This represents a **75%+ YoY increase** from the $114 billion invested in AI in 2024
- **58% of AI funding** was in megarounds of $500M or more

**Largest rounds:**
- SoftBank: $40B into OpenAI (one of the biggest private investments ever)
- ICONIQ Capital: $13B into Anthropic
- Meta: $14.3B into Scale AI

**2026 projection:** AI startups expected to attract ~33% of total VC funding, with enterprise VCs predicting increased AI budgets concentrated among fewer vendors.

**Infrastructure preference:** "When there's a gold rush, invest in picks and shovels" remains the dominant thesis, with compute infrastructure companies attracting the largest checks.

*Source: [6 Charts That Show The Big AI Funding Trends Of 2025](https://news.crunchbase.com/ai/big-funding-trends-charts-eoy-2025/); [VC Outlook for 2026: 5 Key Trends](https://corpgov.law.harvard.edu/2025/12/23/venture-capital-outlook-for-2026-5-key-trends/); [2026 US VC Outlook](https://www.sganalytics.com/blog/2026-us-vc-outlook/)*

---

## 6. Emerging Paradigms That Change the Equation

### 6.1 Mixture of Experts (MoE): More Quality, Less Compute

MoE architectures fundamentally change the compute equation by selectively activating only the most relevant "expert" sub-networks for each input:

- **DeepSeek-V2/V3:** Demonstrated MoE achieving frontier performance with dramatically fewer active parameters per inference step
- **Mixtral 8x7B:** Proved that sparse MoE models can compete with dense models 3-4x their active parameter count
- **Edge-optimized variants:** EdgeMoE, LocMoE, and JetMoE adapt sparse expert routing for resource-constrained environments

**Impact:** MoE reduces the compute required for inference by 2-4x while maintaining quality, potentially reducing the total compute investment needed for deployment at scale. This weakens the "compute is everything" argument by showing that architectural innovation can substitute for raw compute.

*Source: [NVIDIA: Applying Mixture of Experts in LLM Architectures](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/); [What Is Mixture of Experts (MoE)?](https://zilliz.com/learn/what-is-mixture-of-experts)*

### 6.2 Retrieval-Augmented Generation (RAG): External Data at Inference

RAG represents a paradigm where data is consumed at inference time rather than encoded during training:

- **Core insight:** Instead of training a model on all possible knowledge, retrieve relevant information from external databases at query time.
- **Compact vector databases and efficient retrieval algorithms** now enable on-device storage and rapid retrieval with minimal computational overhead.
- **ExpertRAG** combines MoE routing with RAG for query-adaptive retrieval that better matches semantic demands.

**Impact on the debate:** RAG shifts the balance toward data (specifically, well-organized, accessible data) as a runtime resource rather than a training-time resource. Organizations with better data infrastructure for RAG can achieve superior results with smaller models.

### 6.3 On-Device AI / Edge Inference

The push toward edge AI is accelerating:

- **Apple Intelligence:** Integrating LLM capabilities on-device across iPhone, iPad, and Mac.
- **Qualcomm:** Snapdragon 8 Elite and subsequent chips bring significant NPU capability to mobile devices.
- **Key driver:** Privacy requirements, latency needs, and connectivity constraints demand local inference.

**2025-2026 inflection point:** Edge AI deployment is becoming mainstream, with the industry recognizing that not all AI needs to run in massive data centers. This creates demand for highly efficient, small models -- shifting the competition toward data efficiency and algorithmic innovation rather than raw compute.

*Source: [AI in 2026: Enabling Smarter Systems at the Edge](https://www.edn.com/ai-in-2026-enabling-smarter-more-responsive-systems-at-the-edge/); [The Power of Small: Edge AI Predictions for 2026](https://www.dell.com/en-us/blog/the-power-of-small-edge-ai-predictions-for-2026/); [Edge AI Infrastructure Reaches Inflection Point](https://siliconangle.com/2026/03/20/edge-ai-infrastructure-reaches-real-world-inflection-point-nvidiagtcai/)*

### 6.4 Energy Constraints as the New Bottleneck

Energy is emerging as the most binding constraint on AI infrastructure expansion:

**Current consumption:**
- U.S. data centers consumed **183 TWh** of electricity in 2024 (4%+ of total U.S. consumption)
- AI-optimized servers alone: 93 TWh in 2025, projected to rise to **432 TWh by 2030** (nearly 5x)
- Global data center electricity: 448 TWh in 2025, projected to reach **980 TWh by 2030**

**Power source mix for AI data centers:**
- **Natural gas:** Continues to supply the largest near-term share; expanding by ~175 TWh for data centers
- **Nuclear:** Contributing comparable additional generation, with SMRs expected online ~2030; Three Mile Island and Duane Arnold being revived
- **Renewables:** Expected to meet 50% of global growth in data center demand

**Implication:** Energy constraints may become the ultimate arbiter of the data vs. compute debate. If electricity supply cannot keep pace with compute demand, the industry will be forced toward more data-efficient and compute-efficient approaches -- favoring better data, better algorithms, and more efficient architectures over brute-force scaling.

*Source: [Pew Research: US Data Centers' Energy Use](https://www.pewresearch.org/short-reads/2025/10/24/what-we-know-about-energy-use-at-us-data-centers-amid-the-ai-boom/); [IEA: Energy Demand from AI](https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai); [Gartner: Electricity Demand for Data Centers to Double by 2030](https://www.gartner.com/en/newsroom/press-releases/2025-11-17-gartner-says-electricity-demand-for-data-centers-to-grow-16-percent-in-2025-and-double-by-2030)*

### 6.5 Inference Cost > Training Cost: The New Economics

The economics of AI are fundamentally shifting from training-dominated to inference-dominated:

**Key statistics:**
- Inference workloads account for roughly **two-thirds (66%)** of all AI compute in 2026, up from one-third in 2023
- The AI inference market now accounts for **55% of AI cloud infrastructure spending** in 2026 -- surpassing training for the first time
- Inference accounts for **80-90%** of total compute dollars over a model's production lifecycle
- Inference costs dropped from $20 to $0.07 per million tokens (Stanford AI Index, 2025) -- a **280x reduction** since November 2022

**Market projections:**
- Inference-optimized chip market: >$50 billion in 2026
- By 2030: ~70% of all data center demand from AI inferencing applications
- Barclays estimates 70% of AI compute demand from inference by 2026

**Implication:** The shift to inference-dominated economics means that data quality (which improves inference quality) and architectural efficiency (which reduces inference cost) become more important than raw training compute. Companies optimizing for inference efficiency gain a structural advantage.

*Source: [AI Inference vs Training Infrastructure: Economics Diverging](https://introl.com/blog/ai-inference-vs-training-infrastructure-economics-diverging); [Training vs. Inference: The $300B AI Shift](https://www.tonygraysonvet.com/post/ai-training-vs-inference); [Deloitte: Why AI Demands More Compute](https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2026/compute-power-ai.html)*

---

## 7. Geopolitical Dimension: China vs US

### 7.1 US Chip Export Controls: Impact and Effectiveness

**The policy landscape:**
- The US has implemented multiple rounds of chip export controls targeting China's AI capabilities, restricting access to advanced GPUs (H100 and above) and chip-making equipment.
- NVIDIA designed restricted variants (H800, A800) with reduced interconnect bandwidth for the Chinese market, but even these have been further restricted.
- By late 2025, the policy landscape was evolving with some debate about rolling back certain controls.

**DeepSeek founder's perspective:** Liang Wenfeng stated in a 2024 interview: *"Money has never been the problem for us; bans on shipments of advanced chips are the problem."*

**Effectiveness assessment:**
- Controls have **slowed but not stopped** China's AI progress
- China has shifted toward efficiency-first innovation (see DeepSeek section below)
- Huawei has developed the Ascend 910B as a domestic GPU alternative, though it lags NVIDIA by 1-2 generations
- As of October 2025, China has **overtaken the US in monthly open-source AI model downloads**, driven by more customizable tools and licensing strategies

*Source: [CSIS: DeepSeek, Huawei, Export Controls](https://www.csis.org/analysis/deepseek-huawei-export-controls-and-future-us-china-ai-race); [US-China AI Chip War: The 2026 Geopolitical Fracture](https://chinatechscope.com/nvidia-deepseek-ai-chip-war/); [Deep Dive: Export Controls and the AI Race](https://research.contrary.com/report/drawing-geopolitical-boundaries)*

### 7.2 China's Data Advantage

China possesses unique data advantages that partially offset its compute disadvantage:

**Platform scale:**
- **WeChat:** Reaches hundreds of millions of users daily, serving as an unparalleled real-world sandbox for AI deployment. Tencent's Yuanbao AI assistant grew rapidly due to WeChat integration.
- **Super-apps:** WeChat, Alipay, Douyin, and others generate massive volumes of real-world user interaction data across retail, logistics, customer service, and daily life.

**Manufacturing AI:**
- China's industrial sector generates unique proprietary data from the world's largest manufacturing base
- Chinese engineers are pioneering **"small data AI"** -- solutions that deliver high accuracy with minimal initial samples, enabling rapid deployment on factory floors
- This "policy-guided, application-first model" is building a deep moat in industrial intelligence

**Population-scale data:**
- 1.4 billion people generating behavioral, commercial, and communication data
- National-scale surveillance and social credit systems (controversial but data-rich)
- Healthcare data from massive population base

*Source: [China's AI in 2025: Progress, Players and Parity](https://www.venturousgroup.com/resources/chinas-ai-in-2025-progress-players-and-parity/); [China's Small Data AI Gains Edge in Manufacturing](https://www.prnewswire.com/news-releases/chinas-small-data-ai-gains-edge-in-manufacturing-as-industry-experts-debate-us-china-ai-competition-302668811.html)*

### 7.3 DeepSeek's Efficiency Innovations Under Compute Constraints

DeepSeek represents the most striking example of how compute constraints can drive innovation:

**Technical innovations:**
- **MoE architecture:** Selectively activates only the most relevant neural network components, achieving competitive performance with dramatically less compute.
- **Hardware-level optimization:** Programmed "20 of the 132 processing units on each H800 specifically to manage cross-chip communications" -- working at a lower programming level than NVIDIA's CUDA abstraction layer.
- **RL-first training:** DeepSeek-R1-Zero demonstrated that powerful reasoning can emerge from pure RL training without SFT, potentially reducing the need for expensive human-labeled data.

**Results:**
- DeepSeek-R1 competitive with OpenAI o1 on reasoning benchmarks
- Achieved at a fraction of the compute cost of US competitors
- Democratized access to capable AI models worldwide

**Strategic implication:** DeepSeek proved that necessity breeds innovation. Compute constraints forced Chinese researchers to develop more efficient architectures and algorithms -- advances that may actually be more sustainable than the brute-force scaling approach.

### 7.4 Different Strategic Priorities

| Dimension | United States | China |
|-----------|--------------|-------|
| **Primary advantage** | Compute (advanced GPUs, data centers) | Data (population, platforms, manufacturing) |
| **Strategic approach** | Scale-first, infrastructure-heavy | Efficiency-first, application-focused |
| **Key companies** | OpenAI, Anthropic, Google, Meta | DeepSeek, Alibaba, Baidu, Tencent |
| **AI investment** | ~$400-450B capex (2026) | ~$70B data center investment (Goldman Sachs) |
| **Military integration** | DARPA programs, defense contractors | PLA integration, autonomous weapons |
| **Open-source strategy** | Mixed (Meta open, OpenAI closed) | Increasingly open (DeepSeek, Qwen) |
| **Regulatory approach** | Light-touch, export controls | State-guided, data governance laws |

**Goldman Sachs estimate:** China's AI providers are expected to invest $70 billion in data centers, including overseas expansion -- significant but still a fraction of US hyperscaler spending.

*Source: [Goldman Sachs: China's AI Providers to Invest $70B](https://www.goldmansachs.com/insights/articles/chinas-ai-providers-expected-to-invest-70-billion-dollars-in-data-centers-amid-overseas-expansion); [EU ISS: Challenging US Dominance](https://www.iss.europa.eu/publications/briefs/challenging-us-dominance-chinas-deepseek-model-and-pluralisation-ai-development)*

---

## 8. Synthesis & Conclusions

### 8.1 The Core Finding: It's Not Either/Or -- It's Both, Evolving

The question "data vs. compute -- which is more important?" is increasingly the wrong question. The evidence points to a more nuanced reality:

**Short-term (2026-2027): Compute still dominates investment**
- $660-690B in hyperscaler capex reflects the industry's revealed preference
- NVIDIA's $4.2T market cap vs. Databricks' $134B valuation (~30:1 ratio) reflects compute's current premium
- New GPU generations (Rubin, Feynman) continue to deliver meaningful performance gains
- Inference compute demand is growing faster than training compute demand

**Medium-term (2027-2029): Data quality becomes the differentiator**
- Pre-training data exhaustion forces the shift to synthetic data, multimodal data, and proprietary data
- Post-training methodology (which is fundamentally about data curation and reward design) determines model quality more than scale
- RAG and retrieval-based approaches make data organization a competitive advantage
- The "slopacolypse" degrades internet data quality, increasing the premium on curated human data

**Long-term (2029+): Neither may matter in the current paradigm**
- If Sutskever and LeCun are right, fundamental architectural innovation will be needed
- World models, embodied learning, and new training paradigms may change what "data" and "compute" even mean
- Energy constraints may force a hard reset on scaling strategies regardless

### 8.2 The Emerging Consensus

Despite significant disagreements on specifics, a consensus is forming around several points:

1. **Pure pre-training scaling has diminishing returns.** The era of "just make the model bigger and train on more data" is ending.

2. **Compute demand is shifting, not shrinking.** The total demand for compute continues to grow -- but it's moving from pre-training to inference, post-training, and agent workloads.

3. **Data quality trumps data quantity.** The marginal value of an additional petabyte of noisy internet text is approaching zero; the marginal value of a thousand expert-curated examples is higher than ever.

4. **Efficiency is the new scaling.** DeepSeek's success, MoE architectures, and test-time compute all show that doing more with less is as powerful as doing more with more.

5. **The bottleneck is moving downstream.** The binding constraints are increasingly energy supply, inference economics, and the ability to convert AI capability into real-world applications -- not training data or training compute per se.

### 8.3 Strategic Implications

**For technology companies:**
- Invest in data infrastructure and curation capabilities alongside compute
- Build proprietary data assets in specific domains
- Optimize for inference efficiency, not just training capability
- Develop energy-efficient architectures and secure power sources

**For investors:**
- The compute "picks and shovels" thesis (NVIDIA, etc.) remains valid short-term but watch for efficiency-driven disruption
- Data infrastructure companies (Databricks, Snowflake, Scale AI) represent the growing importance of data
- Energy and power infrastructure may be the next critical bottleneck to invest in
- Efficiency innovators (like DeepSeek-style companies) could disrupt established players

**For policymakers:**
- Export controls on compute push adversaries toward efficiency innovations that may be more broadly beneficial
- Data governance and privacy regulations increasingly intersect with AI competitiveness
- Energy infrastructure investment is now a national security priority tied to AI leadership
- Open-source AI policy has geopolitical implications

### 8.4 The Final Verdict

**Neither data nor compute alone is sufficient. The winners will be those who optimize across all three dimensions: compute efficiency, data quality, and algorithmic innovation.**

The era of "throw more GPUs at the problem" is ending. The era of "throw more data at the problem" never truly existed (data quality always mattered more than quantity). What's emerging is an era where the most impactful advances come from clever combinations of:

- **Smart compute allocation** (test-time compute, MoE, efficient architectures)
- **High-quality, domain-specific data** (curated, proprietary, multimodal)
- **Novel training paradigms** (RLVR, self-play, world models)
- **Systems engineering** (inference optimization, RAG, edge deployment)

The $690 billion being spent on AI infrastructure in 2026 is not wasted -- but the companies that will extract the most value from that investment will be those that complement compute with superior data strategies and algorithmic innovation.

---

## Sources & References

### Scaling Laws & Compute
- [LLM Scaling Laws: Analysis from AI Researchers in 2026](https://aimultiple.com/llm-scaling-laws)
- [Scaling Laws for LLMs: From GPT-3 to o3](https://cameronrwolfe.substack.com/p/llm-scaling-laws)
- [A Brief History of LLM Scaling Laws](https://www.jonvet.com/blog/llm-scaling-in-2025)
- [The AI Scaling Wall of Diminishing Returns](https://arxiv.org/abs/2512.20264)
- [Epoch AI: Can AI Scaling Continue Through 2030?](https://epoch.ai/blog/can-ai-scaling-continue-through-2030)

### Infrastructure Investment
- [Hyperscaler Capex > $600B in 2026 (IEEE)](https://techblog.comsoc.org/2025/12/22/hyperscaler-capex-600-bn-in-2026-a-36-increase-over-2025-while-global-spending-on-cloud-infrastructure-services-skyrockets/)
- [AI Capex 2026: The $690B Infrastructure Sprint (Futurum)](https://futurumgroup.com/insights/ai-capex-2026-the-690b-infrastructure-sprint/)
- [CNBC: Tech AI Spending Approaches $700B in 2026](https://www.cnbc.com/2026/02/06/google-microsoft-meta-amazon-ai-cash.html)
- [TechCrunch: Billion-Dollar Infrastructure Deals Powering the AI Boom](https://techcrunch.com/2026/02/28/billion-dollar-infrastructure-deals-ai-boom-data-centers-openai-oracle-nvidia-microsoft-google-meta/)

### GPU Roadmap
- [NVIDIA Rubin at GTC 2026 (Barrack AI)](https://blog.barrack.ai/nvidia-rubin-specs-architecture-2026/)
- [Tom's Hardware: NVIDIA Announces Rubin GPUs in 2026](https://www.tomshardware.com/pc-components/gpus/nvidia-announces-rubin-gpus-in-2026-rubin-ultra-in-2027-feynam-after)
- [NextPlatform: NVIDIA Draws GPU System Roadmap to 2028](https://www.nextplatform.com/2025/03/19/nvidia-draws-gpu-system-roadmap-out-to-2028/)

### Data Wall & Synthetic Data
- [Microsoft Research: SynthLLM Breaking the AI Data Wall](https://www.microsoft.com/en-us/research/articles/synthllm-breaking-the-ai-data-wall-with-scalable-synthetic-data/)
- [NextPlatform: Can Synthetic Data Help Scale AI's Data Wall?](https://www.nextplatform.com/2025/01/16/can-synthetic-data-help-us-scale-ais-data-wall/)
- [World Economic Forum: AI Training Data Is Running Low](https://www.weforum.org/stories/2025/12/data-ai-training-synthetic/)
- [AI Training in 2026: Anchoring Synthetic Data in Human Truth](https://invisibletech.ai/blog/ai-training-in-2026-anchoring-synthetic-data-in-human-truth)
- [HuggingFace: Cosmopedia](https://huggingface.co/blog/cosmopedia)

### Post-Training & Reasoning
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [DeepSeek-R1: Incentivizing Reasoning via RL](https://arxiv.org/html/2501.12948v1)
- [How to Train LLMs to "Think" (Towards Data Science)](https://towardsdatascience.com/how-to-train-llms-to-think-o1-deepseek-r1/)

### Expert Views
- [Ilya Sutskever: AI's Bottleneck Is Ideas, Not Compute](https://www.calcalistech.com/ctechnews/article/h1fudk7z11x)
- [Dario Amodei -- "We Are Near the End of the Exponential"](https://www.dwarkesh.com/p/dario-amodei-2)
- [Yann LeCun: Large Models a "Dead End"](https://eu.36kr.com/en/p/3571987975018880)
- [Andrej Karpathy Predicts "Slopacolypse" in 2026](https://cybernews.com/ai-news/andrej-karpathy-slopacolypse/)

### Market & Investment
- [NVIDIA Q4 FY2026 Earnings (Futurum)](https://futurumgroup.com/insights/nvidia-q4-fy-2026-earnings-highlight-durable-ai-infrastructure-demand/)
- [Databricks Raises $4B at $134B Valuation (TechCrunch)](https://techcrunch.com/2025/12/16/databricks-raises-4b-at-134b-valuation-as-its-ai-business-heats-up/)
- [Snowflake FY2025 Financial Results](https://www.snowflake.com/en/news/press-releases/snowflake-reports-financial-results-for-the-fourth-quarter-and-full-year-of-fiscal-2025/)
- [Crunchbase: Big AI Funding Trends of 2025](https://news.crunchbase.com/ai/big-funding-trends-charts-eoy-2025/)
- [Harvard Law: VC Outlook for 2026](https://corpgov.law.harvard.edu/2025/12/23/venture-capital-outlook-for-2026-5-key-trends/)

### Energy & Infrastructure
- [IEA: Energy Demand from AI](https://www.iea.org/reports/energy-and-ai/energy-demand-from-ai)
- [Pew Research: US Data Centers' Energy Use](https://www.pewresearch.org/short-reads/2025/10/24/what-we-know-about-energy-use-at-us-data-centers-amid-the-ai-boom/)
- [Gartner: Data Center Electricity Demand](https://www.gartner.com/en/newsroom/press-releases/2025-11-17-gartner-says-electricity-demand-for-data-centers-to-grow-16-percent-in-2025-and-double-by-2030)

### Inference Economics
- [AI Inference vs Training Infrastructure Economics (Introl)](https://introl.com/blog/ai-inference-vs-training-infrastructure-economics-diverging)
- [Deloitte: Why AI Demands More Compute](https://www.deloitte.com/us/en/insights/industry/technology/technology-media-and-telecom-predictions/2026/compute-power-ai.html)

### Geopolitics
- [CSIS: DeepSeek, Huawei, Export Controls](https://www.csis.org/analysis/deepseek-huawei-export-controls-and-future-us-china-ai-race)
- [Goldman Sachs: China's AI Data Center Investment](https://www.goldmansachs.com/insights/articles/chinas-ai-providers-expected-to-invest-70-billion-dollars-in-data-centers-amid-overseas-expansion)
- [EU ISS: Challenging US Dominance](https://www.iss.europa.eu/publications/briefs/challenging-us-dominance-chinas-deepseek-model-and-pluralisation-ai-development)
- [ChinaTechScope: US-China AI Chip War 2026](https://chinatechscope.com/nvidia-deepseek-ai-chip-war/)

### Edge AI & MoE
- [NVIDIA: Applying MoE in LLM Architectures](https://developer.nvidia.com/blog/applying-mixture-of-experts-in-llm-architectures/)
- [Edge AI Infrastructure at Inflection Point (SiliconANGLE)](https://siliconangle.com/2026/03/20/edge-ai-infrastructure-reaches-real-world-inflection-point-nvidiagtcai/)
- [Dell: Edge AI Predictions for 2026](https://www.dell.com/en-us/blog/the-power-of-small-edge-ai-predictions-for-2026/)

---

*Report generated: March 25, 2026*
*Research methodology: Multi-source web research with 2025-2026 sources prioritized*
