# FlagOS-AI Deep Analysis Report | FlagOS-AI 深度分析报告

**Date:** 2026-03-26
**Organization:** https://github.com/flagos-ai
**Website:** https://flagos.io | https://docs.flagos.io

---

## 中文摘要 (Chinese Executive Summary)

### FlagOS 是什么？

FlagOS 是由**北京智源人工智能研究院 (BAAI)** 主导、联合十余家国内外芯片公司、系统厂商、算法软件企业、非营利组织及科研机构共建的**统一开源 AI 系统软件栈**。其核心使命是：**"一次开发，多芯运行"（Develop Once, Run Anywhere）**，消除不同 AI 芯片软件栈之间的壁垒，大幅降低迁移和维护成本。

### 核心数据

| 指标 | 数值 |
|------|------|
| 公开仓库数 | 36 |
| 总 Star 数 | ~2,779 |
| 总 Fork 数 | ~814 |
| 组织关注者 | 121 |
| 主要编程语言 | Python, C++, MLIR |
| 开源协议 | Apache 2.0 (多数), MIT (FlagTree) |
| 成立时间 | 2025年11月（GitHub组织） |

### 核心组件

1. **FlagGems** (931 stars) - 基于 Triton 的高性能通用算子库，支持 10+ 硬件后端
2. **FlagScale** (498 stars) - 大模型全生命周期统一工具包（训练 / 推理 / RLHF）
3. **FlagPerf** (363 stars) - AI 芯片一体化评测引擎
4. **FlagAttention** (289 stars) - Triton 实现的内存高效注意力算子
5. **FlagTree** (229 stars) - 统一多芯 AI 编译器（Triton 分支，支持 15+ 后端）
6. **FlagCX** (183 stars) - 跨芯片统一通信库（支持 11 种 CCL 后端）
7. **KernelGen** (33 stars) - AI 驱动的 Triton 内核自动生成平台

### 支持的芯片厂商

NVIDIA, AMD, 华为昇腾, 天数智芯 (ILUVATAR), 摩尔线程, 寒武纪, 沐曦, 海光信息, 昆仑芯, 清微智能, 安谋科技, 燧原, 曦望芯科等 **15+ 家 AI 芯片厂商**。

### 技术创新亮点

- **统一编译器 FlagTree**：基于 Triton 的多后端编译方案，单仓库支持多芯片
- **跨芯片通信 FlagCX**：首创 device-buffer IPC 和 device-buffer RDMA，实现异构芯片间高效通信
- **AI 驱动内核工程 KernelGen 2.0**：MCP + AI Agent 全自动内核开发、优化、测试
- **插件化上游适配**：vLLM-plugin-FL、Megatron-LM-FL、TransformerEngine-FL、verl-FL 等通过插件方式扩展主流框架

### 战略意义

FlagOS 是中国 AI 芯片生态 "去碎片化" 的重要基础设施项目。在 NVIDIA 替代需求日益增长的背景下，FlagOS 为国产芯片提供了统一的软件栈抽象层，使上层应用开发者无需为每款芯片重写代码。该项目已获得 CCF 开源发展技术委员会支持，并启动了总奖金 200 万元人民币的 "FlagOS Open Computing Global Challenge" 全球开发者竞赛。

---

## Detailed English Analysis

---

## 1. Project Overview

### What is FlagOS?

FlagOS is a **unified, open-source AI system software stack** designed to bridge the fragmentation between diverse AI chip ecosystems. It provides a complete abstraction layer spanning compilers, operators, communication libraries, training frameworks, and inference engines -- enabling developers to write code once and deploy across multiple AI accelerators.

**Tagline:** *"A Unified, Open-Source AI System Software Stack"*

### Who Created It?

FlagOS is led by the **Beijing Academy of Artificial Intelligence (BAAI / 北京智源人工智能研究院)**, one of China's most prominent AI research institutions. BAAI is the same organization behind:
- The **Aquila** series of language models
- The **FlagOpen** open-source ecosystem
- The **BGE** embedding models (widely used globally)

The project is co-built by **10+ domestic and international organizations**, including chip companies, system manufacturers, algorithm/software entities, non-profit organizations, and research institutions. The community wiki is hosted at `flagos-wiki.baai.ac.cn`.

### Key Metrics (as of 2026-03-26)

| Metric | Value |
|--------|-------|
| Public repositories | 36 |
| Total stars (all repos) | **2,779** |
| Total forks (all repos) | **814** |
| Organization followers | 121 |
| GitHub org created | November 8, 2025 |
| Contact | contact@flagos.io |
| Website | https://flagos.io |
| Documentation | https://docs.flagos.io |
| Community contact | qgzhu@flagos.io |
| Social: Discord | https://discord.com/invite/ubqGuFMTNE |
| Social: X/Twitter | @FlagOS_Official |
| Social: YouTube | @FlagOS_Official |
| Social: LinkedIn | /company/flagos-community |

### License

The vast majority of FlagOS repositories use the **Apache License 2.0**. FlagTree (the compiler) uses **MIT License** (inherited from the Triton fork). Documentation uses **CC0-1.0**.

---

## 2. Architecture & Technical Innovation

### System Architecture

FlagOS is structured as a layered software stack that sits between AI models/frameworks (top) and diverse AI hardware (bottom):

```
┌─────────────────────────────────────────────────┐
│  AI Models (DeepSeek, Qwen, LLaMA, Grok, etc.) │
├─────────────────────────────────────────────────┤
│  Framework Plugins (FL variants)                │
│  Megatron-LM-FL | vllm-plugin-FL | verl-FL     │
│  TransformerEngine-FL | sglang-FL               │
├─────────────────────────────────────────────────┤
│  FlagScale (Unified Training/Inference/RL CLI)  │
├──────────────┬──────────────┬───────────────────┤
│  FlagGems    │   FlagCX     │   KernelGen       │
│  (Operators) │   (Comms)    │   (AI Kernel Gen) │
├──────────────┴──────────────┴───────────────────┤
│  FlagTree (Unified Multi-Backend Compiler)      │
├─────────────────────────────────────────────────┤
│  FlagBLAS | FlagDNN | FlagFFT | FlagSparse      │
│  FlagTensor | FlagAudio                          │
├─────────────────────────────────────────────────┤
│  Hardware: NVIDIA, AMD, Ascend, ILUVATAR,       │
│  Moore Threads, Cambricon, MetaX, HYGON, KLX,   │
│  Tsingmicro, ARM China, Enflame, Sunrise, etc.  │
└─────────────────────────────────────────────────┘
```

### Key Technical Innovations

#### 1. FlagTree: Unified Multi-Backend Triton Compiler

FlagTree is a fork of OpenAI's Triton compiler extended to support **15+ AI chip backends** within a single repository. Each backend resides in protected branches corresponding to different Triton versions (3.0 through 3.6). Key innovations:

- **TLE (Triton Language Extensions):** TLE-Lite, TLE-Struct (GPU/DSA), TLE-Raw -- mechanisms for hardware-specific optimizations while maintaining portability
- **HINTS:** Compiler hints for shared memory, async DMA, and hardware-specific optimizations
- **FLIR (FlagTree IR):** A shared middle-layer IR (forked from Microsoft's triton-shared) enabling common transformations before backend-specific lowering
- **Multi-framework support:** Works with both PyTorch and PaddlePaddle
- **Source-free installation:** Pre-built wheels for all supported backends

Supported backends: NVIDIA, AMD, x86 CPU, ILUVATAR, Moore Threads, KLX, MetaX, HYGON, Ascend, Cambricon, Tsingmicro, ARM China (AIPU), Enflame, Sunrise.

#### 2. FlagCX: Cross-Chip Heterogeneous Communication

FlagCX is not just a wrapper -- it introduces original technologies:

- **Device-buffer IPC:** Inter-process communication directly between device memory buffers across different chip types
- **Device-buffer RDMA:** Remote direct memory access between heterogeneous devices
- **Heterogeneous collective communication:** AllReduce, AllGather, etc., across mixed GPU clusters (e.g., NVIDIA + Ascend)
- **11 CCL backend integrations:** NCCL, IXCCL, CNCL, MCCL, XCCL, DUCCL, HCCL, MUSACCL, RCCL, TCCL, ECCL
- **3 host-side backends:** Bootstrap, Gloo, MPI
- **Framework integration:** PyTorch and PaddlePaddle distributed backends

This is particularly significant for organizations running mixed hardware clusters.

#### 3. KernelGen 2.0: AI-Native Kernel Engineering

An AI-powered platform for automatic Triton kernel development:

- **Agentic workflow:** Uses LLM agents (not fixed pipelines) for iterative kernel generation
- **MCP-based automation:** Model Context Protocol integration for IDE and agent connectivity
- **Full lifecycle:** Generate -> Optimize -> Test -> Integrate (including automatic PR generation)
- **IDE integration:** Claude Code, VS Code, OpenClaw support
- **Web platform:** https://kernelgen.flagos.io

#### 4. Plugin Architecture for Upstream Frameworks

Rather than maintaining full forks of major frameworks, FlagOS uses a **plugin-based approach**:

| Task | FlagOS Plugin | Upstream Project |
|------|--------------|------------------|
| Training | Megatron-LM-FL, TransformerEngine-FL | NVIDIA Megatron-LM, TransformerEngine |
| Reinforcement Learning | verl-FL | veRL |
| Inference | vllm-plugin-FL | vLLM |
| Inference | sglang-FL | SGLang |

This allows FlagOS to stay in sync with upstream while adding multi-chip support.

### Comparison to Alternatives

| Feature | FlagOS | vLLM | TensorRT-LLM | SGLang |
|---------|--------|------|---------------|--------|
| Primary focus | Full-stack multi-chip | Inference serving | NVIDIA inference | Inference serving |
| Multi-chip support | 15+ backends | Limited | NVIDIA only | Limited |
| Training support | Yes (Megatron-based) | No | No | No |
| RLHF support | Yes (verl-FL) | No | No | No |
| Compiler layer | Yes (FlagTree) | No | Yes (TRT) | No |
| Communication library | Yes (FlagCX) | No | No | No |
| Operator library | Yes (FlagGems) | No | No | No |
| Open-source | Fully | Yes | Partially | Yes |

**FlagOS is not a competitor to vLLM or SGLang** -- it is a lower-level infrastructure that *enables* those frameworks to run on diverse hardware. vllm-plugin-FL and sglang-FL are plugins that extend these frameworks using FlagOS's multi-chip backend.

---

## 3. Complete Repository Catalog

### Core Components

| Repository | Stars | Forks | Language | Description |
|-----------|-------|-------|----------|-------------|
| **FlagGems** | 931 | 296 | Python | High-performance Triton operator library for LLMs |
| **FlagScale** | 498 | 148 | Python | Unified toolkit for large model training/inference/RL |
| **FlagPerf** | 363 | 117 | Python | AI chip benchmarking engine (30+ models, 80+ cases) |
| **FlagAttention** | 289 | 19 | Python | Memory-efficient attention operators in Triton |
| **FlagTree** | 229 | 47 | C++ | Unified multi-backend Triton compiler |
| **FlagCX** | 183 | 52 | C++ | Cross-chip communication library |
| **awesome-LLM-driven-kernel-generation** | 153 | 9 | - | Survey of LLM-driven kernel generation |
| **KernelGen** | 33 | 7 | Python | AI-powered kernel auto-generation platform |
| **libtriton_jit** | 32 | 12 | C++ | Triton JIT runtime and FFI provider in C++ |

### Framework Plugins

| Repository | Stars | Forks | Description |
|-----------|-------|-------|-------------|
| **vllm-plugin-FL** | 28 | 36 | vLLM plugin for multi-chip backend |
| **flir** | 9 | 4 | FlagTree IR (shared middle-layer for Triton) |
| **FlagOS-Robo** | 6 | 7 | End-to-end toolkit for embodied AI |
| **TransformerEngine-FL** | 3 | 16 | TransformerEngine plugin for multi-chip |
| **vllm-FL** | 3 | 2 | Full vLLM fork for multi-chip inference |
| **Megatron-LM-FL** | 2 | 15 | Megatron-LM plugin for multi-chip training |
| **sglang-FL** | 0 | 0 | SGLang fork for multi-chip inference |
| **verl-FL** | 0 | 3 | veRL fork for multi-chip RLHF |

### Math/Compute Libraries (New, March 2026)

| Repository | Stars | Forks | Description |
|-----------|-------|-------|-------------|
| **FlagBLAS** | 2 | 0 | Basic Linear Algebra Subroutines |
| **FlagDNN** | 2 | 0 | Deep Neural Network primitives |
| **FlagTensor** | 0 | 0 | Tensor operations library |
| **FlagFFT** | 0 | 0 | Fast Fourier Transform library |
| **FlagSparse** | 0 | 0 | Sparse matrix operations |
| **FlagAudio** | 0 | 0 | Audio processing library |

### Infrastructure & Community

| Repository | Stars | Forks | Description |
|-----------|-------|-------|-------------|
| **community** | 5 | 2 | Governance, contribution guide, CoC |
| **FlagRelease** | 3 | 2 | Automatic model migration and release |
| **skills** | 2 | 7 | FlagOS skills for deployment, eval, tuning |
| **FlagOps** | 0 | 3 | Shared CI/CD logic |
| **build-infra** | 0 | 6 | Build infrastructure |
| **docs** | 1 | 3 | FlagOS documentation |
| **EasyOfUse** | 1 | 0 | Usability issue tracker |
| **llvm-project** | 1 | 0 | Fork of LLVM for FLIR support |
| **triton-cpu** | 0 | 0 | Experimental CPU backend for Triton |
| **flagtree_mlir** | 0 | 0 | MLIR components for FlagTree |
| **.github** | 0 | 0 | Org profile |
| **stone** | 0 | 0 | Test repo for sync |
| **test-sync** | 0 | 0 | Test repo |

### Technology Stack

- **Primary languages:** Python (~60%), C++ (~25%), MLIR, Shell, CMake
- **Core dependencies:** Triton, PyTorch, LLVM/MLIR
- **Frameworks extended:** Megatron-LM, vLLM, SGLang, TransformerEngine, veRL, PaddlePaddle
- **Build systems:** CMake, pip/setuptools, Docker
- **CI/CD:** GitHub Actions (extensive per-backend CI)
- **Documentation:** MkDocs, Sphinx (ReadTheDocs-style)

---

## 4. Use Cases & Target Users

### Primary Target Users

1. **AI Chip Vendors (Downstream)**
   - Integrate their hardware into the Triton/FlagTree ecosystem
   - Get automatic compatibility with FlagGems operators, FlagScale models, etc.
   - Example: ILUVATAR, Moore Threads, Cambricon, Enflame, Tsingmicro, and others have all contributed backend integrations

2. **Cloud Providers & Data Centers**
   - Run heterogeneous GPU clusters with mixed hardware
   - FlagCX enables cross-chip communication in mixed clusters
   - FlagScale provides unified CLI for training/inference across chips

3. **Enterprise AI Teams**
   - Reduce vendor lock-in to NVIDIA
   - Deploy same models across different hardware without code changes
   - FlagPerf for hardware procurement evaluation

4. **Research Institutions**
   - Train/serve large models on diverse hardware
   - FlagOS-Robo for embodied AI research
   - KernelGen for rapid kernel development

5. **AI Framework Developers**
   - Use FlagOS as a backend to add multi-chip support to their frameworks

### Supported Models (via FlagScale)

**Training:** DeepSeek-V3, Qwen2/2.5/3, Qwen2.5-VL, QwQ, LLaMA2/3/3.1, LLaVA-OneVision, LLaVA1.5, Mixtral, RWKV, Aquila

**Inference:** DeepSeek-V3, DeepSeek-R1, Qwen2.5, Qwen3, Qwen2.5-VL, Qwen3-Omni, QwQ, Grok2, Kimi-K2

### Deployment Scenarios

- Single-chip training/inference (development)
- Multi-GPU single-node (standard)
- Multi-node distributed training
- **Heterogeneous multi-chip clusters** (unique to FlagOS)
- Embodied AI (FlagOS-Robo)

---

## 5. Code Quality & Implementation

### Repository Activity (Last 30 Days)

| Repository | Commits (30d) | Contributors | Closed PRs (total) |
|-----------|--------------|--------------|---------------------|
| FlagGems | ~100 | 123 | ~1,581 |
| FlagScale | ~31 | 17 | ~1,064 |
| FlagTree | ~22 | 283 | ~434 |
| FlagCX | ~35 | 34 | ~389 |
| FlagPerf | ~0 | 80 | - |
| KernelGen | ~18 | 11 | - |

**FlagGems** is by far the most active repository with ~100 commits in the last 30 days and the largest number of closed PRs.

**FlagTree** has the most contributors (283), which makes sense as it involves many chip vendor teams each contributing their backend.

### Release Cadence

| Repository | Latest Release | Date |
|-----------|----------------|------|
| FlagGems | v5.0.0 | 2026-03-26 (today!) |
| FlagScale | v1.0.0 | 2026-03-26 (today!) |
| FlagTree | v0.4.0 | 2026-01-08 |
| FlagCX | v0.11.0 | 2026-03-26 (today!) |

FlagGems v5.0.0 and FlagScale v1.0.0 were released today (2026-03-26), indicating a coordinated major release milestone.

### CI/CD Setup

All major repositories have extensive GitHub Actions workflows:

- **FlagGems:** Unit tests, linter, coverage, backend tests, docs build, release wheel builds, Copilot code review
- **FlagScale:** Multi-backend test runners (including Huawei, MetaX), format checks, SBOM generation
- **FlagTree:** Per-backend build-and-test workflows (AIPU, Ascend, Enflame, Tsingmicro, etc.), integration tests, code scanning
- **FlagCX:** Debian/RPM package building, container-based testing, torch API tests, unit tests, pre-commit checks

### Documentation Quality

- **FlagGems:** Full documentation site at https://flagos-ai.github.io/FlagGems/ with getting started, usage, features, and contribution guides. Both English and Chinese READMEs.
- **FlagScale:** Getting started guide, changelog, model-specific YAML configs as examples
- **FlagTree:** Detailed build instructions per backend, wiki pages for extensions (TLE, HINTS, FLIR)
- **FlagCX:** Getting started, environment variables, user guides for PyTorch and Paddle
- **FlagPerf:** Extensive Chinese documentation with detailed benchmarking methodology
- **Overall docs site:** https://docs.flagos.io with unified navigation across components

### Code Organization

Repositories follow consistent patterns:
- Clean separation between core logic and backend-specific code
- Configuration-driven approach (YAML configs in FlagScale)
- Plugin-based architecture to minimize fork divergence from upstream
- Bilingual documentation (English + Chinese)

---

## 6. Ecosystem & Community

### Organizational Backing

- **Lead:** Beijing Academy of Artificial Intelligence (BAAI)
- **Co-built by:** 10+ organizations including chip companies, system manufacturers, research institutions
- **Supported by:** CCF Open Source Development Technology Committee (ODTC)
- **Community contact:** WeChat groups, Discord, email (contact@flagos.io)
- **Wiki:** https://flagos-wiki.baai.ac.cn/

### Chip Vendor Partners (Contributing Backends)

| Vendor | Chinese Name | Components |
|--------|-------------|------------|
| NVIDIA | - | FlagTree, FlagCX (NCCL), FlagGems |
| AMD | - | FlagTree, FlagCX (RCCL) |
| Huawei Ascend | 华为昇腾 | FlagTree, FlagCX (HCCL), FlagScale |
| ILUVATAR | 天数智芯 | FlagTree, FlagCX (IXCCL) |
| Moore Threads | 摩尔线程 | FlagTree, FlagCX (MUSACCL) |
| Cambricon | 寒武纪 | FlagTree, FlagCX (CNCL) |
| MetaX | 沐曦 | FlagTree, FlagCX (MCCL), FlagScale |
| HYGON | 海光信息 | FlagTree |
| KLX/Kunlunxin | 昆仑芯 | FlagTree, FlagCX (XCCL) |
| Tsingmicro | 清微智能 | FlagTree, FlagCX (TCCL) |
| ARM China | 安谋科技 | FlagTree |
| Enflame | 燧原 | FlagTree, FlagCX (ECCL) |
| Sunrise | 曦望芯科 | FlagTree |
| DU (Baidu) | 百度昆仑 | FlagCX (DUCCL) |

### Developer Competition

**FlagOS Open Computing Global Challenge** (co-hosted by FlagOS Community, BAAI, CCF ODTC):
- Total prize pool: **2,000,000 RMB** (~$275,000 USD)
- Three tracks: Operator Development, Large Model Inference Optimization, Automatic Data Annotation
- Season 1: Registration opened January 9, 2026; results announced early June 2026
- Platform: DoraHacks

### Model Distribution

FlagOS releases adapted models through:
- **ModelScope:** https://modelscope.cn/organization/FlagRelease
- **Hugging Face:** https://huggingface.co/FlagRelease/models
- **WiseModel:** https://www.wisemodel.cn/models/FlagRelease/

### Community Channels

- WeChat official account: 智源FlagOpen
- WeChat developer groups (QR codes in repo READMEs)
- Discord: https://discord.com/invite/ubqGuFMTNE
- GitHub Discussions (coming soon)
- Email: contact@flagos.io

### Growth Trajectory

The organization was created on **November 8, 2025** (less than 5 months ago) but has already:
- Accumulated **2,779 stars** and **814 forks**
- Attracted **283 unique contributors** to FlagTree alone
- Integrated **15+ AI chip backends**
- Released coordinated v1.0.0 / v5.0.0 milestones
- Launched a $275K hackathon

Note: Several repositories (FlagPerf, FlagScale, FlagGems, FlagAttention) predate the org creation -- they were migrated from the earlier `FlagOpen` organization on GitHub.

### Roadmap & Future Direction

Based on recent activity:
1. **Expanding math libraries:** FlagBLAS, FlagDNN, FlagFFT, FlagSparse, FlagTensor, FlagAudio (all created March 2026)
2. **Embodied AI:** FlagOS-Robo for robotics workloads
3. **AI-native kernel development:** KernelGen 2.0 with MCP/Agent integration
4. **More inference frameworks:** sglang-FL recently added (March 2026)
5. **Continued backend expansion:** New chip vendor integrations on a near-monthly cadence
6. **FlagOS Skills:** Reusable skills for model deployment, HW adaptation, training, inference, eval, kernel dev, and performance tuning

---

## 7. Strategic Assessment

### Strengths

1. **Unique positioning:** No other open-source project provides this level of multi-chip abstraction across the full AI stack (compiler + operators + communication + training + inference)
2. **Strong institutional backing:** BAAI has significant credibility, resources, and industry relationships
3. **Ecosystem network effects:** 15+ chip vendors actively contributing creates a moat
4. **Pragmatic design:** Plugin approach to upstream frameworks (vLLM, Megatron) minimizes maintenance burden
5. **Active development:** Multiple releases on the same day, 100+ commits/month in core repos
6. **Bilingual community:** Chinese and English documentation enables global reach

### Weaknesses / Risks

1. **Early stage:** Most components are < 6 months old in this organization; ecosystem maturity is still developing
2. **NVIDIA baseline still dominant:** While multi-chip is the goal, NVIDIA remains the primary tested backend
3. **Community breadth vs depth:** 283 contributors to FlagTree is impressive, but many are from vendor teams contributing narrow backend code
4. **Documentation gaps:** Some newer repos (FlagBLAS, FlagDNN, etc.) have no README or description
5. **Global awareness:** Still relatively unknown outside the Chinese AI hardware ecosystem

### Competitive Landscape

FlagOS occupies a unique niche. It is **not** competing directly with:
- **vLLM / SGLang:** FlagOS extends these with multi-chip plugins
- **TensorRT-LLM:** NVIDIA-specific; FlagOS is hardware-agnostic
- **Triton:** FlagTree is a Triton extension, not a replacement
- **ROCm / oneAPI:** These are single-vendor stacks; FlagOS is vendor-neutral

The closest analog would be **Intel oneAPI** in concept (unified programming across hardware), but FlagOS is open-source, multi-vendor, and AI-focused.

---

## Sources

- [FlagOS GitHub Organization](https://github.com/flagos-ai)
- [FlagOS Official Website](https://flagos.io)
- [FlagOS Documentation](https://docs.flagos.io)
- [FlagOS Community Wiki](https://flagos-wiki.baai.ac.cn/)
- [BAAI Innovation Center](https://www.baai.ac.cn/en/system)
- [FlagOS Open Computing Global Challenge](https://dorahacks.io/hackathon/flagos-open-computing/detail)
- [FlagOS Open Computing Challenge on CompeteHub](https://www.competehub.dev/en/competitions/dorahacksflagos-open-computing)
- [FlagScale on Gitee](https://gitee.com/flagos-ai/FlagScale)
- [FlagCX Announcement (AIBase)](https://www.aibase.com/news/14326)
- [Beijing Academy of Artificial Intelligence (Wikipedia)](https://en.wikipedia.org/wiki/Beijing_Academy_of_Artificial_Intelligence)
- [FlagOpen GitHub (predecessor org)](https://github.com/FlagOpen)
- [KernelGen Web Platform](https://kernelgen.flagos.io)
- [KernelGen MCP on ModelScope](https://www.modelscope.cn/mcp/servers/flagos-ai/FlagOS_KernelGen)
