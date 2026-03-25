# Storage Industry Deep Dive V2: Server Architecture, Memory Hierarchy & Storage Evolution

**Date:** 2026-03-16
**Version:** 2.0 — Expert-reviewed and fact-checked by 5 specialist agents (HBM/HBF, CXL/Interconnect, SSD/NAND, ASIC/Server Architecture, Emerging Memory)

---

## Executive Summary

The storage and memory industry is undergoing a once-in-a-decade transformation driven by AI infrastructure demands. The traditional CPU-DRAM-SSD-HDD hierarchy is being disrupted by new tiers (CXL-attached memory, HBM, HBF, GDDR7, MRDIMM) and new architectural paradigms (disaggregated/composable memory, near-data processing, silicon photonics). Key themes:

1. **AI is the gravitational force** reshaping every layer of the memory/storage stack
2. **CXL 2.0 shipping in volume; CXL 3.1 initial deployment mid-2026**, enabling memory pooling and disaggregation
3. **HBM4 enters mass production** (2-3.3 TB/s per stack, 2048-bit interface); **HBM4E** targeting 2027
4. **GDDR7** emerges as the cost-optimal AI inference memory (Nvidia Rubin CPX: 128 GB GDDR7)
5. **HBF (High Bandwidth Flash)** — on-interposer stacked NAND, first samples H2 2026
6. **MRDIMM** doubles DDR5 bandwidth per slot; **DDR6** spec ratified, mass adoption 2027
7. **SSDs cross into the capacity tier** — 122 TB shipping from multiple vendors, 245 TB on roadmap
8. **Advanced packaging (CoWoS) is the #1 supply bottleneck** — fully booked through 2026
9. **Memory supply crisis**: AI consuming ~40% of global DRAM, prices doubling YoY
10. **New interconnects**: UALink 1.0 (GPU scale-up), NVLink 6.0 (3.6 TB/s/GPU), PCIe 7.0 finalized

---

## 1. The New Memory/Storage Hierarchy (2026 View)

The traditional 3-tier model (DRAM -> SSD -> HDD) has expanded to 8+ tiers:

```
Tier 0:   HBM4/HBM4E       | On-package, 2-3.3 TB/s BW      | GPU/accelerator-attached
Tier 0.5: GDDR7             | 128-192 GB/s per device         | Cost-optimal inference GPUs
Tier 0.5: HBF (on-interposer)| 400-800 GB/s, ~1.6 TB          | Stacked NAND alongside HBM
Tier 1:   MRDIMM (DDR5)     | 8800 MT/s, doubling BW          | CPU-attached main memory
Tier 1.5: LPDDR5X/LPDDR6    | Up to 480 GB/CPU, efficient     | Arm server CPUs (Grace/Vera)
Tier 2:   CXL Memory        | 200-500 ns, pooled/shared       | Rack-scale fabric-attached
Tier 3:   NVMe SSD          | PCIe 5.0/6.0, up to 28 GB/s    | Local or fabric-attached
Tier 4:   QLC Capacity SSD  | 122-245 TB per drive            | Replacing HDD capacity tier
Tier 5:   HDD / Tape / DNA  | Archival, cold storage          | Shrinking role (HDD), emerging (DNA)
```

### Key Insight: The "Memory Wall" Is Being Attacked From All Sides
- **From above**: HBM4 delivers 2-3.3 TB/s per stack; HBF adds TB-scale capacity on the interposer
- **From below**: CXL brings pooled memory at 200-500 ns; NVMe SSDs at 28 GB/s (PCIe 6.0)
- **In between**: MRDIMM doubles main memory bandwidth; GDDR7 fills the inference cost gap

---

## 2. HBM (High Bandwidth Memory) — The AI Compute Tier

### HBM4 (2026 Mass Production)

| Spec | HBM3E (current) | HBM4 (2026) | HBM4E (2027) |
|------|-----------------|-------------|--------------|
| Interface Width | 1024-bit | **2048-bit** | 2048-bit |
| Channels | 16 | **32** (2 pseudo-ch each) | 32 |
| Bandwidth/stack | ~1.2 TB/s | **2-3.3 TB/s** | **~3.25 TB/s** |
| Capacity/stack | 36-48 GB | 36-64 GB (12-16 Hi) | 48-64 GB+ |
| Stack height | 8-12 layers | 12-16 layers | 16 layers |
| Bonding | MR-MUF | **MR-MUF** (not hybrid bonding) | MR-MUF / hybrid |
| Base die | Memory process | **Logic process (4-5nm)** | **Custom logic (2nm)** |

**Key correction from V1:** Current HBM4 production uses MR-MUF (Mass Reflow Molded Underfill), NOT hybrid bonding. Hybrid bonding is a future direction (HBM5+).

### HBM4E — Critical Addition (Missing from V1)
- All three DRAM makers targeting **development completion H1 2026, mass production 2027**
- Samsung targeting per-pin speeds **above 13 Gbps**, peak **3.25 TB/s per stack**
- HBM4E projected at **~40% of total HBM demand in 2027** (TrendForce)
- Samsung moving custom HBM4E logic die to **2nm foundry process**
- Primary adopter: Nvidia **Rubin Ultra** — **1 TB HBM4E** (16 stacks), ~32 TB/s, ~100 PFLOPs FP4

### Logic-in-Base Die — A Paradigm Shift
- Base die manufactured using **advanced foundry logic (4-5nm)**, not traditional memory processes
- Samsung operates **two separate HBM teams**: standard HBM and **customer-specific custom HBM** with bespoke logic
- Custom base die designs being developed for **Nvidia, Google, and Meta**
- Transforms HBM from commodity into **semi-custom product** — pricing, design cycles, and vendor lock-in implications

### GDDR7 — The Inference Memory Tier (Missing from V1)
| Spec | Value |
|------|-------|
| Signaling | **PAM3** (50% more data/clock vs PAM2) |
| Data rates | 32 Gbps initial, up to **48 Gbps** (SK hynix, ISSCC 2026) |
| Per-device BW | 128-192 GB/s |
| Mass production | Started Q4 2024 |
| Key AI product | Nvidia **Rubin CPX** inference GPU: **128 GB GDDR7** (no HBM) |

GDDR7 is positioned as the **cost-optimal memory for AI inference** — lower $/GB than HBM, higher bandwidth than DDR5. Consumer: Nvidia RTX 5090 (32 GB GDDR7, ~1.8 TB/s).

### AI Accelerator Memory Landscape (2026)

| Accelerator | Memory Type | Capacity | Bandwidth | Status |
|------------|------------|----------|-----------|--------|
| **Nvidia Rubin** | HBM4 | 288 GB | 22 TB/s | Production Q1 2026 |
| **Nvidia Rubin Ultra** | HBM4E | 384 GB (1 TB system) | ~32 TB/s | 2027 |
| **Nvidia Rubin CPX** | GDDR7 | 128 GB | — | Inference-optimized |
| **AMD MI350** | HBM3e | 288 GB | ~8 TB/s | Shipping 2025 |
| **AMD MI400 (MI455X)** | HBM4 | 432 GB | 19.6 TB/s | Q3 2026 |
| **Google TPU v7 (Ironwood)** | HBM3e | 192 GB | 7.4 TB/s | GA late 2025 |
| **AWS Trainium 3** | HBM3e | 144 GB | 4.9 TB/s | 2026 |
| **Microsoft Maia 100** | HBM2e | 64 GB | 1.8 TB/s | Shipping |
| **Meta MTIA 400** | HBM | 288 GB | 9.2 Tb/s | Testing complete |
| **Intel Gaudi 3** | HBM2e | 128 GB | 3.7 TB/s | Shipping |

**Notable:** Intel **cancelled Falcon Shores** (Jan 2025). Successor **Jaguar Shores** pivots to rack-scale system. Meta releases new chips on a **6-month cadence** — fastest in industry.

### Market Dynamics
- **SK hynix** holds 62% market share (Q2 2025), began HBM4 mass production **February 2026**
- Samsung slightly later due to **1c DRAM yield challenges**; reports up to 3.3 TB/s per stack
- HBM demand grew **130% YoY in 2025**, projected **70%+ YoY in 2026**
- Over **90% of production pre-booked** by hyperscalers through 2026
- OpenAI Stargate: **900,000 DRAM wafers/month** (~40% of global output) — treat with caution, likely total buildout equivalents
- HBM consumes 3x the wafer area per bit vs. standard DRAM, cannibalizing DDR5 supply
- 16-layer HBM requires wafers thinned to **~30 um** (vs. ~50 um for 12-layer)
- **JEDEC spHBM4**: Reduced pin count variant under development for higher capacity configs

**Sources:**
- [JEDEC HBM4 Standard JESD270-4](https://www.jedec.org/news/pressreleases/jedec-and-industry-leaders-collaborate-release-jesd270-4-hbm4-standard)
- [SK hynix 2026 Market Outlook](https://news.skhynix.com/2026-market-outlook-focus-on-the-hbm-led-memory-supercycle/)
- [Samsung HBM4 Mass Production](https://videocardz.com/newz/samsung-begins-hbm4-mass-production-and-customer-shipments-up-to-3-3-tb-s-per-stack)
- [NVIDIA Vera Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [AMD Instinct GPU Roadmap](https://ir.amd.com/news-events/press-releases/detail/1201/)
- [Google TPU v7 Ironwood](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [AWS Trainium 3](https://www.hpcwire.com/2025/12/02/aws-brings-the-trainium3-chip-to-market/)
- [Meta MTIA Roadmap](https://ai.meta.com/blog/next-generation-meta-training-inference-accelerator-AI-MTIA/)
- [Microsoft Maia 100](https://techcommunity.microsoft.com/blog/azureinfrastructureblog/inside-maia-100/)

---

## 3. CXL (Compute Express Link) — The Memory Fabric Revolution

### CXL Version Timeline

| Version | PHY Layer | Key Feature | Status (March 2026) |
|---------|----------|-------------|---------------------|
| CXL 1.1 | PCIe 5.0 | Type 1/2/3 devices | Shipping |
| CXL 2.0 | PCIe 5.0 | Memory pooling, switching | **Volume shipping** |
| CXL 3.1 | PCIe 6.1 | Fabric, shared memory, GFAM | **Initial deployment mid-2026; ramp late 2026-2027** |
| CXL 4.0 | PCIe 7.0 | 128 GT/s, bundled ports (1.5 TB/s), x2 width, 4 retimers | Spec released Nov 18, 2025 (SC25) |

**Correction from V1:** CXL 3.1 is NOT in "broad deployment." CXL 2.0 is what's shipping in volume. CXL 3.1 products (Samsung CMM-D 3.1, Montage MXC CXL 3.1) begin shipping mid-2026 at earliest.

### Shipping CXL Products (2026)

| Vendor | Product | Spec | Capacity | Bandwidth | Status |
|--------|---------|------|----------|-----------|--------|
| Samsung | CMM-D 1.0 | CXL 2.0 | 128 GB | — | Mass production |
| Samsung | CMM-D 2.0 | CXL 2.0 | 128-256 GB | 36 GB/s | Samples |
| Samsung | CMM-D 3.1 | CXL 3.1 | up to 1 TB | 72 GB/s | Target Q4 2025 |
| Samsung | CMM-B | CXL 2.0/3.x | 2 TB (8 modules) | 60 GB/s, 596 ns | Box form factor |
| Samsung | CMM-H | CXL | DRAM+NAND hybrid | — | CXL-based HBF alternative |
| Micron | CZ120 | CXL 2.0 | 128-256 GB | — | **Volume production**, E3.S |
| Micron | CZ122 | CXL 2.0 | — | — | Qualification samples |
| SK hynix | CMM-DDR5 | CXL 2.0 | 96 GB | — | Montage controllers |
| SK hynix | CMM-Ax | CXL+PIM | — | — | PIM-enabled expansion |
| Astera Labs | Leo | CXL 2.0 | — | — | Deployed on **Microsoft Azure M-series** |
| Montage | MXC Gen1/Gen2 | CXL 2.0 | — | — | Mass produced |
| Montage | M88MX6852 | CXL 3.1 | — | PCIe 6.2, 64 GT/s x8 | Sampling |

### CXL Switch Ecosystem (Missing from V1)

| Vendor | Product | Spec | Notes |
|--------|---------|------|-------|
| **XConn/Marvell** | Apollo 1 | CXL 2.0 | First CXL switch shipped |
| **XConn/Marvell** | Apollo 2 | CXL 3.1, PCIe 6.2 | 64-260 lanes. **Marvell acquired XConn for $540M** (Jan 2026) |
| Microchip | Switchtec | CXL 2.0 | PCIe/CXL combined |
| Astera Labs | Scorpio X-Series | CXL fabric | 10+ engagements, $10B TAM estimate |

### Global Fabric Attached Memory (GFAM) — Missing from V1
- Introduced in CXL 3.0: a GFAM device is accessible by **up to 4,095 nodes** via port-based routing
- Provides cache-coherent shared memory across a fabric
- CXL 3.1 adds **Global Integrated Memory (GIM)** for host-to-host communication
- Concept directly descended from **Gen-Z** (dissolved into CXL Jan 2022)
- Hardware expected with CXL 3.1 platforms late 2026-2027

### CXL Security (Missing from V1)
- **IDE** (Integrity & Data Encryption): CXL 2.0+, flit-level AES-GCM with 256-bit keys
- **SPDM**: Device authentication; DMTF + CXL + PCI-SIG announced **post-quantum cryptography (PQC)** support (CNSA 2.0)
- **TSP** (CXL 3.1): Trusted execution environment for confidential computing with shared CXL memory

### Why CXL Changes Everything
1. **Memory Tiering**: Hot data in local DRAM (~100 ns), warm data on CXL (~200-500 ns), cold data on SSD (~100 us)
2. **Memory Pooling**: Shared CXL memory pool improves utilization ~50%, reduces TCO 15-20%
3. **AI Use Cases**: KV cache offloading (200-500x lower latency than SSD), RAG working sets, multi-TB model parameter hosting
4. **Power Efficiency**: CXL drops memory bandwidth cost from ~1W/GB/s to **0.75-0.83W/GB/s** (17-25% improvement)
5. **Disaggregated Architecture**: Rack-scale memory fabrics with shared, composable resources

### CXL Market
- 2026 CXL Type 3 market: **~$2.1B** (Yole Group), growing to **~$16B by 2028**
- Primary adopters: hyperscalers (AWS, Google, Meta, Microsoft) and AI infrastructure

### Notable CXL Research
- **ASPLOS 2025**: "Melody" — CXL characterization across 265 workloads, 4 devices, 7 latency levels, 5 CPU platforms
- **ASPLOS 2025**: "CENT" — CXL+PIM GPU-free LLM inference: 2.3x throughput vs GPUs, 5.2x tokens/dollar
- **ASPLOS 2025**: "Toleo" — 168 GB CXL smart memory freshness guarantee for 28 TB expanded pool
- **OSDI 2024**: "Nomad" — non-exclusive memory tiering via transactional page migration
- **HPCA 2025**: "SkyByte" — memory-semantic CXL-SSD with adaptive page migration for GC latency
- **HPCA 2026**: "REDIT" — redirection-enabled memory-side directory for CXL fabric
- **FAST 2026**: "MOST" — mirror-optimized storage tiering with CXL awareness
- **ACM TACO 2025**: "ShieldCXL" — practical obliviousness support for CXL side-channel defense

**Sources:**
- [CXL 4.0 Spec Release (Nov 18, 2025)](https://www.businesswire.com/news/home/20251118275848/en/)
- [Samsung CMM-D Family](https://semiconductor.samsung.com/cxl-memory/cmm-d/)
- [Micron CZ120](https://www.micron.com/products/memory/cxl-memory)
- [Astera Labs Leo on Azure](https://www.asteralabs.com/news/astera-labs-leo-cxl-smart-memory-controllers-on-microsoft-azure/)
- [Marvell XConn Acquisition ($540M)](https://www.datacenterdynamics.com/en/news/marvell-acquires-pcie-and-cxl-switch-provider-xconn-technologies-for-540m/)
- [Montage CXL 3.1 MXC](https://finance.yahoo.com/news/montage-technology-introduces-cxl-3-150000451.html)
- [CXL Consortium: GFAM](https://www.servethehome.com/compute-express-link-cxl-3-0-is-the-exciting-building-block-for-disaggregation/)
- [DMTF SPDM PQC](https://www.dmtf.org/news/pr/25252525/)
- [Yole CXL Market $2.1B](https://cm.asiae.co.kr/en/article/2025122521051393782)

---

## 4. Interconnect Ecosystem (New Section)

### UALink (Ultra Accelerator Link) — GPU Scale-Up Fabric

| Spec | Value |
|------|-------|
| Version | **UALink 200G 1.0** (April 2025) |
| Speed | 200 GT/s per lane |
| Bandwidth | 800 GB/s per connection (4 lanes) |
| Scale | Up to **1,024 accelerators per pod** |
| Members | 75+ (AMD, Broadcom, Google, Intel, Meta, Microsoft — **Nvidia absent**) |

- AMD + Astera Labs switch chip: tape-out Q1 2026, mass production **Q4 2026**
- **Upscale AI** "SkyHammer" ASIC: $300M raised, unicorn valuation
- UALink is **complementary to CXL** — GPU-to-GPU scale-up vs. CPU-to-memory fabric

### NVLink 6.0 / NVSwitch 6.0

| Spec | NVLink 5.0 (Blackwell) | NVLink 6.0 (Vera Rubin) |
|------|----------------------|------------------------|
| BW per GPU | 1.8 TB/s | **3.6 TB/s** (2x) |
| SerDes | — | 224G, doubled lanes |
| vs. PCIe 6 | 7x | **14x** |

**Vera Rubin NVL72**: 72 GPUs + 36 Vera CPUs, 20.7 TB HBM4, 54 TB LPDDR5x, 9 NVSwitch 6 blades, **260 TB/s total aggregate bandwidth**. Production 2026.

### Ultra Ethernet Consortium (UEC)
- **UE 1.0 Specification** released June 2025 (100+ members)
- Optimizes Ethernet for AI/HPC scale-out networking
- 2026 priorities: Programmable Congestion Management (PCM), Congestion Signaling (CSIG)

### PCIe Roadmap

| Version | Speed | BW (x16 bidi) | Status |
|---------|-------|---------------|--------|
| PCIe 5.0 | 32 GT/s | 128 GB/s | Volume shipping |
| PCIe 6.0 | 64 GT/s | 256 GB/s | Enterprise shipping |
| **PCIe 7.0** | **128 GT/s** | **512 GB/s** | **Spec released June 2025**; silicon 2027-2028 |
| PCIe 8.0 | 256 GT/s | ~1 TB/s | Exploration phase |

### Gen-Z Legacy
Gen-Z Consortium **dissolved into CXL in January 2022** (~80% member overlap). Gen-Z's fabric-level concepts (multi-switch topologies, port-based routing, GFAM) were absorbed into CXL 3.0.

**Sources:**
- [UALink 1.0 Spec (April 2025)](https://ualinkconsortium.org/wp-content/uploads/2025/04/UALink-1.0-Specification-Overview_FINAL-1.pdf)
- [Upscale AI SkyHammer](https://www.hpcwire.com/2025/12/02/upscale-ai-eyes-late-2026-for-scale-up-ualink-switch/)
- [NVIDIA NVLink 6.0](https://www.nvidia.com/en-us/data-center/nvlink/)
- [UEC Spec 1.0](https://www.linuxfoundation.org/press/uec-launches-spec-1.0)
- [PCIe 7.0 Final Spec](https://finance.yahoo.com/news/pci-sig-releases-pcie-7-183000437.html)

---

## 5. DRAM & Main Memory Evolution

### MRDIMM (Multiplexed Rank DIMM)

| Gen | Speed | Bandwidth/Channel | Timeline |
|-----|-------|-------------------|----------|
| Gen1 | DDR5-8800 | ~70 GB/s | **Shipping 2025** (Xeon 6900P, 1DPC only) |
| Gen2 | DDR5-12800 | ~102 GB/s | **2026-2027** |
| Gen3 | DDR5-17600 | ~140 GB/s | ~2030 (overlaps DDR6 speeds) |

- Doubles bandwidth per DIMM slot vs. standard RDIMM
- Capacity: Up to 256 GB per module; ~$100/module premium over RDIMM
- **Intel-only feature**: Xeon 6900P supports MRDIMM; **AMD EPYC Turin does NOT support MRDIMM** (DDR5-6400 max, CXL 2.0 only)

### LPDDR5X in Servers
- Nvidia **Grace CPU Superchip** (2x Grace CPUs): up to **960 GB LPDDR5X**, ~1 TB/s bandwidth
- **Single Grace CPU**: 480 GB max, 500 GB/s bandwidth, 32-channel LPDDR5X
- **Correction from V1**: 960 GB requires the Superchip (two CPUs), not a single CPU

### DDR6 — Next Generation (Missing from V1)

| Milestone | Timeline |
|-----------|----------|
| LPDDR6 standard (JESD209-6) | Published July 2025 |
| DDR6 Spec 1.0 ratification | Q2 2025 |
| High-end devices | Late 2026 |
| **Mass adoption** | **2027** |

- Base speed 8,800 MT/s, max 17,600 MT/s
- LPDDR6: four 24-bit sub-channels (lower latency, higher concurrency)

### DRAM Technology Scaling (Missing from V1)

| Node | ~Feature Size | EUV Layers | Status (March 2026) |
|------|-------------|------------|---------------------|
| 1-alpha (1a) | ~14-15 nm | 0-1 | Volume production (all 3 vendors) |
| 1-beta (1b) | ~12-13 nm | 2-4 | Volume production |
| **1-gamma (1c)** | ~10-11 nm | 5-6 | **Micron sampling (first, Feb 2025); SK hynix & Samsung ramping** |
| 1-delta | ~9-10 nm | 7+ (High-NA EUV) | R&D, ~2027-2028 |

- **SK hynix**: First to install **High-NA EUV** (ASML EXE:5000); plans to double EUV fleet to ~40 units by 2027
- **Samsung**: Unveiled **InGaO channel material** for sub-10nm DRAM (Dec 2025)
- **3D DRAM**: Samsung 4F-squared VCT presented at ISSCC 2026; mainstream production **~2033-2034**
- **IGZO capacitor-less DRAM** (2T0C): IMEC/SK hynix research published in Nature Reviews Electrical Engineering (2025)

### DRAM Supply Crisis (2026)
- Samsung & SK hynix raising server DRAM prices **60-70% QoQ** in Q1 2026
- Server DDR5 projected to **double YoY** by late 2026
- DDR4 production falling to ~20% of 2025 levels
- Root cause: fabrication capacity diverted to HBM (3x wafer area per bit)
- Server LPDDR5X lead times: **26-39 weeks**

**Sources:**
- [Rambus: DDR5 MRDIMMs Explained](https://www.rambus.com/blogs/ask-the-experts-ddr5-mrdimms/)
- [Micron DDR5 MRDIMM](https://www.micron.com/products/memory/dram-modules/mrdimm)
- [Micron 1-gamma DRAM](https://investors.micron.com/news-releases/news-release-details/micron-announces-shipment-1g-1-gamma-dram)
- [SK hynix High-NA EUV](https://www.trendforce.com/news/2025/09/03/)
- [Samsung InGaO](https://www.trendforce.com/news/2025/12/17/)
- [IGZO DRAM - Nature Reviews](https://www.nature.com/articles/s44287-025-00162-w)
- [TrendForce: Memory Makers Prioritize Server](https://www.trendforce.com/presscenter/news/20260105-12860.html)

---

## 6. SSD & NAND Technology — The Capacity Tier Crossover

### 2026: SSDs Displace HDDs in Data Centers

| Metric | Current State (March 2026) |
|--------|---------------------------|
| Max capacity | **122 TB shipping** (Solidigm D5-P5336, Micron 6600 ION, SK hynix) |
| 245 TB roadmap | Solidigm (end 2026), Kioxia LC9 (PCIe 5.0), Samsung (PCIe 6.0 target 2027) |
| PCIe 6.0 SSDs | **Micron 9650 in mass production** (28 GB/s read, 14 GB/s write, 5.5M IOPS) |
| Form factors | EDSFF **rapidly growing** — E1.S for density (32/1U), E3.S for capacity/performance |
| Cost/TB | QLC approaching HDD economics for read-heavy workloads |

### NAND Layer Count Landscape (Corrected)

| Vendor | Generation | Layers | Status (March 2026) |
|--------|-----------|--------|---------------------|
| **Kioxia/SanDisk** | BiCS10 | **332** | Mass production accelerated to 2026 |
| **SK hynix** | 9th gen | **321** (QLC) | Mass production H1 2026 (PC SSDs first) |
| **Samsung** | V9 | **286** | **Delayed** to H1 2026 (was H2 2025) |
| **Samsung** | V10 | **400+** (TLC) | Unveiled ISSCC 2025; production target TBD |
| **Micron** | G9 | **276** | Volume production shipping |

**Correction from V1:** Samsung is at 286 layers (V9), NOT "300+". Kioxia BiCS10 at 332 layers actually leads.

### Key SSD Technology Trends

1. **PCIe 6.0 SSDs**: **Micron 9650** is the world's first PCIe Gen6 SSD in mass production (28 GB/s read). Samsung PM1763 targeting early 2026. Consumer PCIe 6.0 SSDs not expected until **~2030**.

2. **NVMe FDP Replaces ZNS** (Correction from V1): NVMe Flexible Data Placement (FDP, TP4146) is **overtaking ZNS** as the preferred data placement mechanism. FDP retains SSD L2P mapping/GC control while reducing write amplification. ZNS adoption was limited to a few hyperscalers. Samsung and Meta actively adopting FDP.

3. **PLC NAND (5-bit)** (Missing from V1): Solidigm demonstrated first PLC SSD (FMS 2022). Kioxia+WD initiated pilot production ($2.3B investment). SK hynix **Multi-Site Cell (MSC)** approach at IEDM 2025: splits cells, **20x faster reads** vs non-MSC PLC. No commercial products yet — target 2027-2029.

4. **NAND Scaling Challenges at 300+ Layers**: Cell current degradation from increased string resistance; HAR etch at >100:1 aspect ratio; tier bending/collapse (major yield killer); capacitive cell-to-cell interference. Industry targeting 500-1000 layers via multi-tier bonding.

5. **CXL-SSD / Memory-Semantic SSDs**: Samsung Memory-Semantic CXL-SSD: DRAM cache + NAND + CXL interface, **20x improvement** in random read. Samsung CMM-B: 2 TB, 60 GB/s, 596 ns latency.

6. **NVMe-oF (NVMe over Fabrics)**: At 5-20% market penetration (Gartner 2024).
   - **NVMe/TCP**: Growing fastest — simple, broad compatibility
   - **NVMe/RoCE (RDMA)**: Ultra-low latency for HPC
   - **FC-NVMe**: Upgrade path for existing FC SANs

7. **Computational Storage — Market Stalled** (Correction from V1): NGD Systems effectively **defunct**. Samsung SmartSSD "all but disappeared." ScaleFlux pivoted to compression-enabled SSDs. New entrant: **XCENA MX1** at FMS 2025 (PCIe 6.0 + CXL 3.2).

### Notable Research (FAST 2026)
- **"SolidAttention"** — SSD-based LLM serving (**Best Paper + Distinguished Artifact Award**)
- **"Fast Cloud Storage for AI Jobs"** — Grouped I/O API for AI workloads
- **"Cylon"** — CXL-SSD emulation framework
- **"DOGI"** — Data placement optimization
- **"CETOFS"** — Disaggregated NVMe filesystem
- **"LESS"** — Advanced erasure codes
- **"DRBoost"** — Degraded read optimization

### AI Storage Software Landscape
- **MinIO**: Open-source repo **archived and read-only** by early 2026. Pivoted to commercial AIStor.
- **Ceph Tentacle** (v20.2.0, Nov 2025): 1 EB deployed globally. **FastEC** improves small I/O 2-3x.
- **WEKA NeuralMesh, VAST Data, DDN, NetApp, Pure Storage**: All competing for AI storage
- Object storage + data lakehouse (Iceberg/Delta Lake/Hudi) is the dominant AI training data architecture

**Sources:**
- [Solidigm D5-P5336 122 TB Shipping](https://www.servethehome.com/solidigm-d5-p5336-122-88tb-nvme-ssd-launched-shipping-in-q1-2025/)
- [Micron 9650 PCIe Gen6 Mass Production](https://www.tomshardware.com/pc-components/ssds/worlds-first-pcie-6-0-ssd-enters-mass-production-with-28gb-s-speeds)
- [Kioxia BiCS10 332 Layers](https://www.tomshardware.com/pc-components/ssds/kioxias-next-gen-3d-nand-production-gets-expedited-to-2026)
- [Samsung FDP Blog](https://semiconductor.samsung.com/news-events/tech-blog/nvme-fdp-a-promising-new-ssd-data-placement-approach/)
- [SK hynix Split-Cell PLC](https://blocksandfiles.com/2026/01/15/sk-hynix-developing-split-cell-5-bit-flash/)
- [Samsung Memory-Semantic CXL-SSD](https://www.tomshardware.com/news/samsung-memory-semantic-cxl-ssd-brings-20x-performance-uplift)
- [FAST 2026 Technical Sessions](https://www.usenix.org/conference/fast26/technical-sessions)
- [Ceph Tentacle / MinIO 2026](https://sixe.eu/news/ceph-minio-2026-storage-guide)

---

## 7. HBF (High Bandwidth Flash) — The On-Interposer Memory Tier

### What Is HBF?
- Uses **TSV** to stack multiple NAND flash layers — same approach as HBM but with flash
- Designed to sit **on the GPU interposer alongside HBM** (NOT a separate fabric-attached tier)
- **Correction from V1**: HBF belongs at Tier 0.5 (co-located with HBM), not Tier 4

### Performance Targets

| Metric | HBM4 | HBF (target) | NVMe SSD |
|--------|-------|---------------|----------|
| Bandwidth/device | ~2-3.3 TB/s | **400-800 GB/s** | ~14-28 GB/s |
| Capacity/device | 48-64 GB | **~1.6 TB** | 4-122 TB |
| Latency | ~ns-class | ~us-class | ~10-100 us |

### Industry Status

| Company | Status | Timeline | Notes |
|---------|--------|----------|-------|
| SanDisk | First samples | **H2 2026** | MoU with SK hynix (Aug 2025) |
| SK hynix | IEEE paper published | **Feb 2026** | HBM+HBF hybrid on single interposer, **2.69x perf/watt** |
| Samsung | CMM-H (CXL hybrid) | **2027-2028** | CXL-based DRAM+NAND alternative approach |

### Standardization
- **HBF is NOT a JEDEC standard** (correction from V1) — currently a bilateral SanDisk/SK hynix effort
- Samsung's **CMM-H** (CXL Memory Module - Hybrid) represents a competing CXL-based approach

**Sources:**
- [SanDisk HBF Collaboration](https://www.sandisk.com/company/newsroom/press-releases/2025/2025-08-06-sandisk-to-collaborate-with-sk-hynix-to-drive-standardization-of-high-bandwidth-flash-memory-technology)
- [TrendForce: HBF Battleground](https://www.trendforce.com/news/2025/11/11/news-sk-hynix-samsung-and-sandisk-bet-on-hbf-the-next-battleground-in-memory-sector/)
- [SK hynix HBM+HBF IEEE Paper (Feb 2026)](https://blocksandfiles.com/2025/11/27/stacked-layers-of-stacked-layers-hbf-capacity-and-complexity/)

---

## 8. SCM & Emerging Memory Technologies

### Post-Optane Landscape

| Technology | Latency | Persistence | Status |
|-----------|---------|-------------|--------|
| CXL Type 3 DRAM | 200-500 ns | No (volatile) | **Volume shipping 2025-2026** |
| **SanDisk 3D Matrix Memory** | DRAM-like | Yes | **Most credible post-Optane SCM** |
| Numemory NM101 | Microsecond-class | Yes | Early-stage, **unverified claims** |
| NVDIMM-P | ~150-300 ns | Yes | Standard defined, **no commercial products** |

### SanDisk 3D Matrix Memory (Missing from V1 — Critical Addition)
- Spun off from Western Digital (Feb 2025)
- Technology: **OTS (Ovonic Threshold Switch) selector-only memory**, developed with **IMEC** under **Project Neo**
- Claims: DRAM-like performance, 4x capacity, half the cost of DRAM, CXL-compatible
- IMEC OTS-only devices: >10^8 endurance cycles, 10 ns read/write, <15 uA write current
- Won **$35M USAF AFRL award** (Project ANGSTRM) for radiation-hardened memory
- Sampling 32 Gbit and 64 Gbit chips
- **Most credible post-Optane SCM effort** with real IMEC fab partnership

### Corrections from V1
- **Numemory NM101**: Claims are **unverified** — no independent benchmarks, no major OEM design wins, no third-party validation. Uses microsecond-class latency (faster than NAND but NOT DRAM-like). NAND interface constrains real-world throughput.
- **NVDIMM-P**: No confirmed Granite Rapids support. With Optane gone, it's effectively a **standard without a product**. CXL has superseded its use cases.

### Emerging Memory Technologies (Missing from V1)

| Technology | Key Players | Status | Use Case |
|-----------|------------|--------|----------|
| **MRAM** | Everspin ($48.3M 2025 rev, 238 design wins), Samsung eMRAM (8nm by 2026) | **Commercially shipping** | SSD metadata protection, write caching |
| **ReRAM** | Weebit Nano (licensed to TI, onsemi), $756M market 2026 | **Entering commercialization** | Embedded NVM, analog compute-in-memory |
| **FeRAM/FeFET** | FMC (Dresden, EUR 100M raised), Samsung/TSMC | Legacy shipping; next-gen 2026-2027 | SoC/AI accelerator embedded memory |
| **PCM** | SanDisk 3D Matrix (see above), IBM Zurich | Post-Optane revival | SCM, neuromorphic/analog AI computing |
| **ULTRARAM** | Quinas Technology (Lancaster spinout) | Very early (8-15 years out) | III-V compound semiconductor, unlimited endurance |

### PIM (Processing-in-Memory)
- **Samsung LPDDR5X-PIM**: Announced Feb 2026 for on-device AI
- **SK hynix GDDR6-AiM**: Deployed in real-world LLM inference
- **LPDDR6-PIM**: Samsung + SK hynix jointly standardizing through JEDEC (spec expected 2026)
- **UPMEM**: First commercially available general-purpose PIM — DDR4 DIMMs with 8 DPUs per chip
- TrendForce (March 2026): Samsung and SK hynix exploring PIM architectures that could "challenge NVIDIA"

**Sources:**
- [SanDisk 3D Matrix Memory](https://www.techradar.com/pro/sandisks-revolutionary-new-memory-promises-dram-like-performance-4x-capacity-at-half-the-price)
- [IMEC OTS-only Memories](https://www.imec-int.com/en/articles/promise-ots-only-memories-next-gen-compute-system-architectures-0)
- [Everspin PERSYST MRAM](https://www.businesswire.com/news/home/20251118226253/en/Everspin-Expands-PERSYST-MRAM-Family)
- [Weebit Nano-TI ReRAM Deal](https://finance.yahoo.com/news/why-weebit-nano-asx-wbt-130949451.html)
- [FMC Ferroelectric Memory](https://marklapedus.substack.com/p/next-gen-ferroelectric-memory-still)
- [SIGARCH: Persistent Memory](https://www.sigarch.org/persistent-memory-a-new-hope/)

---

## 9. Advanced Packaging — The #1 Supply Bottleneck (New Section)

### CoWoS (Chip-on-Wafer-on-Substrate)

| Feature | CoWoS-S | CoWoS-L |
|---------|---------|---------|
| Interposer | Monolithic silicon | Segmented with LSI bridges |
| Max area | ~2700 mm^2 | >3000 mm^2 |
| Key users | Nvidia H100, AMD MI300 | **Nvidia Blackwell/Rubin, AMD MI400** |
| Status | Mature, proven | **Fully booked through 2026** |

- TSMC monthly CoWoS capacity: 75-80K wafers (2025) -> **120-130K target (end 2026)**
- **Still sold out** through 2026 — confirmed by TSMC CEO
- New facilities: AP6 (Zhunan), AP7 (Chiayi), AP8 (Tainan)
- CoWoS capacity directly constrains HBM packaging availability

### Intel EMIB & Foveros
- **EMIB**: Tiny silicon bridges embedded in substrate (no full interposer needed)
- **EMIB-T**: Evolution adding TSVs for HBM stacking without full interposers
- **Foveros**: 3D die stacking with direct copper bonding
- Intel demonstrated packaging with **16 compute tiles (18A/14A)** and up to **24 HBM sites**
- Apple and Qualcomm reportedly seeking Intel's EMIB expertise
- Intel scaling EMIB +30% and Foveros +150% capacity

### UCIe (Universal Chiplet Interconnect Express)
- **UCIe 3.0** released 2025: 48-64 GT/s data rates
- Co-developed by AMD, Arm, ASE, Google, Intel, Meta, Microsoft, Qualcomm, Samsung, TSMC
- Enables memory disaggregation — memory chiplets placed separately from GPU at low latency
- Intel presented "On-Package Memory with UCIe" at Hot Interconnects 2025

### Silicon Photonics for Memory Fabric
- **Lightmatter Passage M1000** (March 2025): 114 Tbps optical bandwidth
- **Ayar Labs**: 8 optical engines on single substrate, >100 Tbps
- **Celestial AI** (acquired by Marvell for **$3.25B**, early 2026): Photonic Fabric, 16 Tbps per chiplet, 25x bandwidth at 10x lower latency
- **Nvidia Spectrum-X Photonics** (H2 2026): 409.6 Tb/s co-packaged optics

**Sources:**
- [TSMC CoWoS](https://3dfabric.tsmc.com/english/dedicatedFoundry/technology/cowos.htm)
- [CoWoS Fully Booked](https://www.trendforce.com/news/2025/12/08/)
- [Intel 24 HBM Sites](https://wccftech.com/intel-next-level-advanced-packaging-capabilities/)
- [UCIe 2025 Year in Review](https://www.uciexpress.org/post/ucie-consortium-s-2025-year-in-review)
- [Marvell Celestial AI $3.25B](https://investor.marvell.com/news-events/press-releases/detail/1005/)
- [Lightmatter Passage M1000](https://lightmatter.co/press-release/lightmatter-unveils-passage-m1000/)

---

## 10. Server Architecture Evolution — Composable & Disaggregated

### The 2026 Server Architecture Blueprint

```
┌─────────────────────────────────────────────────────────────────┐
│                    GPU / Accelerator                            │
│                    HBM4 (on-package, 2-3.3 TB/s/stack)         │
│                    + HBF (on-interposer, 400-800 GB/s)         │
│                    + GDDR7 (inference GPUs, 128-192 GB/s)      │
├──────────────────────┬──────────────────────────────────────────┤
│  Intel Xeon 6900P    │  AMD EPYC Turin        │ Nvidia Grace   │
│  MRDIMM DDR5-8800   │  DDR5-6400 (no MRDIMM) │ LPDDR5X 480GB  │
│  12-ch, 1DPC         │  12-ch, CXL 2.0 only   │ 32-ch, 500GB/s │
├──────────────────────┴──────────────────────────────────────────┤
│   CXL 2.0 Fabric (shipping) / CXL 3.1 Fabric (mid-2026)       │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐              │
│   │ CXL Memory │  │ CXL Switch │  │ CXL-SSD    │              │
│   │ Pool       │  │ (Marvell   │  │ (Samsung   │              │
│   │ 200-500ns  │  │  Apollo 2) │  │  20x perf) │              │
│   └────────────┘  └────────────┘  └────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│   PCIe 5.0/6.0 / NVMe-oF (TCP/RDMA/FC-NVMe)                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │ NVMe SSDs   │  │ QLC 122-245 │  │ Archive     │          │
│   │ E1.S / E3.S │  │ TB Capacity │  │ Tape / DNA  │          │
│   │ PCIe 5.0/6.0│  │ SSDs        │  │             │          │
│   └─────────────┘  └─────────────┘  └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Power & Cooling (Missing from V1)
- Current AI rack power: ~270 kW shipping, ~480 kW expected soon; **1 MW designs unveiled at OCP 2025**
- Nvidia Rubin systems: **~600 kW per rack**; AI rack cost: **$3.9M** (vs $500K traditional)
- HBM per-stack power: ~75W; with 8 stacks = ~600W for memory alone per GPU (**40-50% of GPU power**)
- **Direct-to-chip liquid cooling is mandatory** for HBM4-class accelerators
- 2026 adoption: ~22% of new facilities using liquid cooling (70% liquid / 30% air split)

### Composable Infrastructure Vendors
Beyond Liqid and GigaIO: One Stop Systems, Dolphin ICS, Seagate CMA (CXL-based, Redfish API), HPE Synergy, Dell PowerEdge MX, Cisco UCS. Market growing ~17% CAGR through 2031.

**Sources:**
- [AMD EPYC Turin 12ch DDR5-6400](https://www.storagereview.com/review/amd-epyc-turin-review-192-cores-of-zen-5)
- [NVIDIA Grace Superchip 480GB/CPU](https://developer.nvidia.com/blog/nvidia-grace-cpu-superchip-architecture-in-depth/)
- [High-Density Racks OCP 2025](https://introl.com/blog/high-density-racks-100kw-ai-data-center-ocp-2025)
- [Liquid Cooling 2026](https://www.kad8.com/server/data-center-liquid-cooling-for-ai-workloads-2026/)

---

## 11. Archival Tier — Tape, Cold SSD, DNA Storage

| Technology | Cost/TB | Density | Status |
|-----------|---------|---------|--------|
| **LTO Tape** | Lowest | Standard | Undisputed archival leader through 2030+ |
| **Cold QLC SSD** | Medium | 122-245 TB/drive | "Landing zone" between active and tape |
| **DNA Storage** | ~$3,500/MB (current) | 2.2 PB/gram | **Atlas Eon 100**: first commercial service (Dec 2025) |

- **Atlas Data Storage** (Twist Bioscience spinout, May 2025): **$155M seed round**
- Atlas Eon 100: 60 PB in 60 cubic inches (1,000x LTO-10 density)
- Target: 13 TB in a single drop of water by 2026
- Not cost-viable for general use until 2030+

**Sources:**
- [Atlas Data Storage $155M](https://www.businesswire.com/news/home/20250505213784/en/)
- [Atlas Terabyte Target](https://www.techradar.com/pro/after-nearly-10-years-twist-bioscience-spin-off-plans-terabyte-scale-dna-storage-in-2026)

---

## 12. Top Conferences & Key Industry Events

### Must-Watch Conferences

| Conference | Focus | 2026 Dates |
|-----------|-------|-----------|
| **USENIX FAST** | File & storage systems | Feb 24-26, 2026 (Santa Clara) |
| **HPCA** | High-performance computer architecture | Feb 2026 |
| **ASPLOS** | Architecture + PL + OS | March-April 2026 |
| **ISCA** | Computer architecture | June 2026 |
| **NVMW** | Non-volatile memories workshop | March 9-10, 2026 (UCSC) |
| **FMS** | Future of Memory & Storage | Aug 2026 |
| **Hot Chips** | Chip architecture | Aug 2026 |
| **MICRO** | Microarchitecture | Oct-Nov 2026 |
| **OCP Global Summit** | Open compute hardware | Oct 2026 |
| **SC26** | Supercomputing | Nov 2026 |
| **IEEE MSST** | Mass storage systems | 2026 (returning to full-week) |

### FMS 2025 Key Announcements
- **NVMe 2.3** specification introduced
- **PCIe Gen 8** presented by PCI-SIG
- **UALink 200G 1.0** specification
- **UCIe 3.0** specification
- **Kioxia LC9 245 TB** — Best of Show
- **Micron, Kioxia, SanDisk** all showed 245-256 TB SSDs
- **XCENA MX1** — Computational Memory (PCIe 6.0 + CXL 3.2)
- **NEO Semiconductor X-HBM**: 10x bandwidth, 90% reduction in cost/height/power

### Hot Chips 2025 Highlights
- **Google Ironwood TPU v7**: 192 GB HBM3e, 7.4 TB/s, 9,216-chip superpods
- **d-Matrix 3D DRAM**: Claims 100x bandwidth vs standard HBM4
- **IBM Universal Memory Interface**: SerDes at 38.4 Gbps/lane, any DRAM generation
- **Marvell**: Dense SRAM (17x BW density), custom HBM, Arm cores in CXL controllers
- **Huawei UB-Mesh**: Unified mesh fabric up to 10 Tbps/chip
- **Lightmatter Passage M1000**: 3D photonic interconnect

### OCP Global Summit 2025 Highlights
- **AMD Helios**: Rack-scale AI, 50% more memory than Vera Rubin claim
- **Samsung**: CXL solutions for agentic AI, HBM4, Custom HBM
- **Seagate CMA**: CXL-based Composable Memory Appliance
- **Arm FCSA**: Foundation Chiplet System Architecture
- **ESUN**: Ethernet for Scale-Up Networking (AMD, Arista, Broadcom, Cisco, Meta, Microsoft, Nvidia, OpenAI)
- **1 MW rack designs** unveiled

### Landmark Papers (2024-2026)

**CXL & Memory:**
- "Melody" (ASPLOS 2025), "CENT" (ASPLOS 2025), "Toleo" (ASPLOS 2025)
- "Nomad" (OSDI 2024), "REDIT" (HPCA 2026), "ShieldCXL" (ACM TACO 2025)
- "SkyByte" (HPCA 2025), "MOST" (FAST 2026)

**SSD & Storage:**
- "SolidAttention" (FAST 2026 — **Best Paper + Distinguished Artifact**)
- "Cylon" (FAST 2026), "CETOFS" (FAST 2026), "DOGI" (FAST 2026)
- "LIA" (ISCA 2025)

**AI Accelerators:**
- "MTIA v2" (ISCA 2025) — Meta's inference accelerator architecture

**Emerging Memory:**
- "Breaking the Memory Wall" (Frontiers in Science, 2025)
- IGZO capacitor-less DRAM (Nature Reviews Electrical Engineering, 2025)
- SK hynix HBM+HBF hybrid architecture (IEEE, Feb 2026)

---

## 13. Trend Summary & Outlook

### What's Happening Now (March 2026)

| Technology | Maturity | Impact |
|-----------|----------|--------|
| HBM4 | Mass production (SK hynix Feb 2026, Samsung) | 2-3.3 TB/s per stack, 32 channels |
| CXL 2.0 | Volume shipping | Memory expansion deployed (Azure, etc.) |
| CXL 3.1 | Initial products mid-2026 | Fabric, GFAM, shared memory begins |
| MRDIMM Gen1 | Shipping (Xeon 6900P) | Doubles DDR5 bandwidth (Intel-only) |
| 122 TB QLC SSDs | Shipping (Solidigm, Micron, SK hynix) | Multi-vendor capacity tier |
| PCIe 6.0 SSDs | Mass production (Micron 9650) | 28 GB/s read speeds |
| GDDR7 | Mass production | Cost-optimal AI inference memory |
| UALink 1.0 | Spec released, silicon in design | GPU scale-up alternative to NVLink |
| CoWoS | Fully booked, expanding | #1 supply bottleneck for AI chips |

### What's Coming (2027-2028)

| Technology | Expected | Impact |
|-----------|----------|--------|
| HBM4E | Mass production 2027 | 3.25 TB/s, Rubin Ultra 1 TB |
| HBF samples | H2 2026 - 2027 | On-interposer NAND, 400-800 GB/s |
| CXL 4.0 products | 2028+ | 128 GT/s, 1.5 TB/s bundled ports |
| MRDIMM Gen2 | 2026-2027 | DDR5-12800, ~102 GB/s/channel |
| DDR6 | Mass adoption 2027 | 8,800-17,600 MT/s |
| LPDDR6 | Late 2026 | 4 sub-channels, lower latency |
| 245-512 TB SSDs | 2026-2028 | Petabyte-scale all-flash racks |
| PCIe 7.0 silicon | 2027-2028 | 128 GT/s, 512 GB/s x16 |
| UALink hardware | Q4 2026 | AMD+Astera switch, Upscale AI |
| SanDisk 3D Matrix | Sampling | Most credible post-Optane SCM |
| NVLink 6.0 | 2026 (Vera Rubin) | 3.6 TB/s/GPU |
| Silicon photonics | 2026-2028 | Lightmatter, Celestial AI/Marvell |

### Long-Term Horizon (2029+)

| Technology | Timeline | Impact |
|-----------|----------|--------|
| HBM5/HBM5E | 2029-2031 | Hybrid bonding, 3D-on-3D stacking |
| 3D DRAM | ~2033-2034 | Samsung 4F-Square VCT, IGZO 2T0C |
| PLC NAND (5-bit) | 2027-2029 | 25% more bits/cell vs QLC |
| 500-1000 layer NAND | 2028-2030 | Multi-tier bonded stacks |
| DNA storage (viable) | 2030+ | 2.2 PB/gram archival |
| CXL over optical | 2027-2028 | Multi-rack memory fabric |
| Quantum cryogenic memory | 2030+ | SureCore 22FDX SRAM at 4K |

### The Big Picture

The memory/storage hierarchy is **expanding from 3 tiers to 8+**, driven by AI's insatiable demand for bandwidth and capacity. Three forces are reshaping the landscape:

1. **CXL is the unifying memory fabric** — pooling, sharing, and disaggregating memory at rack scale, with GFAM enabling 4,095-node shared memory
2. **Advanced packaging is the bottleneck** — CoWoS capacity, not silicon, is the constraint. Intel EMIB, UCIe, and silicon photonics are emerging alternatives
3. **The interconnect wars are heating up** — NVLink 6.0 (Nvidia proprietary) vs. UALink (open consortium, Nvidia absent) vs. CXL (CPU-memory) vs. UEC (scale-out Ethernet). The winner determines data center architecture for a decade.

The industry is moving from **monolithic servers with fixed memory** to **composable, disaggregated architectures** where memory, storage, and compute are independently scalable, dynamically allocated resources connected by multi-protocol fabrics.

---

## Appendix: V1 -> V2 Changelog

Major corrections applied:
1. HBM4 bonding: MR-MUF, not hybrid bonding
2. Samsung NAND: 286 layers (V9), not "300+"
3. PCIe 6.0 SSD leader: Micron 9650 (mass production), not Samsung
4. CXL 3.1: "Initial deployment mid-2026", not "broad deployment"
5. ZNS: FDP is overtaking ZNS, not "ZNS dominates"
6. E3.S: "Rapidly growing", not "dominant"
7. EPYC Turin: Does NOT support MRDIMM (Intel-only)
8. Grace CPU: 480 GB per CPU, 960 GB requires Superchip (2 CPUs)
9. Computational storage: Market stalled (NGD defunct, SmartSSD disappeared)
10. Numemory: Unverified claims; microsecond latency, not DRAM-like
11. NVDIMM-P: No commercial products, superseded by CXL
12. HBF tier placement: On-interposer (Tier 0.5), not fabric-attached (Tier 4)

Major sections added:
1. HBM4E roadmap (3.25 TB/s, 2027)
2. GDDR7 (PAM3, 48 Gbps, inference-optimized)
3. AI accelerator memory landscape (Rubin, MI400, TPU v7, Trainium 3, Maia 100, MTIA)
4. Interconnect ecosystem (UALink, NVLink 6.0, UEC, PCIe 7.0, Gen-Z legacy)
5. CXL switches, GFAM, security (IDE/SPDM/TSP)
6. DRAM scaling (1a/1b/1c nodes, EUV, 3D DRAM)
7. DDR6 roadmap
8. Advanced packaging (CoWoS bottleneck, EMIB, UCIe, silicon photonics)
9. PLC NAND, NVMe FDP, CXL-SSD
10. Emerging memories (MRAM, ReRAM, FeRAM, SanDisk 3D Matrix)
11. Power & cooling (rack budgets, liquid cooling, CXL efficiency)
12. DNA storage (Atlas Eon 100)
13. PIM expansion (Samsung, SK hynix, UPMEM)
14. Conference coverage (FMS 2025, Hot Chips 2025, OCP 2025, SC25)

---

*Research compiled from web sources, conference proceedings, industry reports, and expert agent review. March 2026.*
