// ==================== LLM Algorithm Visualization — i18n System ==================== //

const I18N_KEY = 'llm-viz-lang';
const SUPPORTED_LANGS = ['en', 'zh-CN', 'zh-TW'];
const LANG_LABELS = { 'en': 'English', 'zh-CN': '简体中文', 'zh-TW': '繁體中文' };

// Global translation registry
const _i18n = { 'en': {}, 'zh-CN': {}, 'zh-TW': {} };

function getLang() {
  return localStorage.getItem(I18N_KEY) || 'en';
}

function setLang(lang) {
  if (SUPPORTED_LANGS.indexOf(lang) === -1) return;
  localStorage.setItem(I18N_KEY, lang);
  applyI18n();
  // Update html lang attribute
  var htmlLang = lang === 'zh-CN' ? 'zh-Hans' : lang === 'zh-TW' ? 'zh-Hant' : 'en';
  document.documentElement.setAttribute('lang', htmlLang);
  // Update selector UI
  var sel = document.getElementById('langSelect');
  if (sel) sel.value = lang;
}

// Register translations (merges into global registry)
function i18nRegister(data) {
  SUPPORTED_LANGS.forEach(function(lang) {
    if (data[lang]) {
      var keys = Object.keys(data[lang]);
      for (var i = 0; i < keys.length; i++) {
        _i18n[lang][keys[i]] = data[lang][keys[i]];
      }
    }
  });
}

// Get translated string
function t(key) {
  var lang = getLang();
  return _i18n[lang] && _i18n[lang][key] || _i18n['en'] && _i18n['en'][key] || key;
}

// Apply translations to all elements with data-i18n attribute
function applyI18n() {
  var elements = document.querySelectorAll('[data-i18n]');
  for (var i = 0; i < elements.length; i++) {
    var el = elements[i];
    var key = el.getAttribute('data-i18n');
    var val = t(key);
    if (val !== key) {
      if (el.hasAttribute('data-i18n-html')) {
        el.innerHTML = val;
      } else {
        el.textContent = val;
      }
    }
  }
  // Update page title
  var titleEl = document.querySelector('title[data-i18n]');
  if (titleEl) {
    document.title = t(titleEl.getAttribute('data-i18n'));
  }
  // Rebuild nav sidebar with translated strings
  rebuildSidebarI18n();
}

// Rebuild sidebar text with translations
function rebuildSidebarI18n() {
  var sidebar = document.getElementById('siteSidebar');
  if (!sidebar) return;
  // Update category headers
  sidebar.querySelectorAll('.nav-category').forEach(function(el) {
    var key = el.getAttribute('data-i18n');
    if (key) el.textContent = t(key);
  });
  // Update link text (topic titles)
  sidebar.querySelectorAll('a').forEach(function(link) {
    var href = link.getAttribute('href') || '';
    var m = href.match(/(\d{2})-/);
    if (!m) return;
    var num = m[1];
    var titleKey = 'nav.topic.' + num;
    var translated = t(titleKey);
    if (translated !== titleKey) {
      // Keep the nav-num span, replace text after it
      var numSpan = link.querySelector('.nav-num');
      var progressSpan = link.querySelector('.sidebar-progress');
      link.textContent = '';
      if (numSpan) link.appendChild(numSpan);
      link.appendChild(document.createTextNode(translated));
      if (progressSpan) link.appendChild(progressSpan);
    }
  });
  // Update nav brand
  var brand = document.querySelector('.nav-brand');
  if (brand) brand.textContent = t('nav.brand');
}

// ==================== Common Translations ==================== //
i18nRegister({
  'en': {
    // Nav
    'nav.brand': 'LLM Algorithm Visualizations',
    'nav.cat.prerequisites': 'Prerequisites',
    'nav.cat.foundations': '2017–2019 Foundations',
    'nav.cat.compute': '2022 Compute Optimization',
    'nav.cat.memory': '2023 Memory Management',
    'nav.cat.advanced': '2024–2026 Advanced',
    'nav.cat.rag': 'RAG & Retrieval',
    'nav.topic.00': 'Linear Algebra Basics',
    'nav.topic.01': 'Transformer Basics',
    'nav.topic.02': 'GPT-2 Architecture',
    'nav.topic.03': 'Sparse Transformers',
    'nav.topic.04': 'Prefill & Decode',
    'nav.topic.05': 'FlashAttention',
    'nav.topic.06': 'PagedAttention',
    'nav.topic.07': 'H2O',
    'nav.topic.08': 'StreamingLLM',
    'nav.topic.09': 'PD Separation',
    'nav.topic.10': 'Mooncake',
    'nav.topic.11': 'TurboQuant',
    'nav.topic.12': 'Dense Retrieval',
    'nav.topic.13': 'RAG Pipeline',
    'nav.topic.14': 'Adaptive RAG',
    'nav.topic.15': 'GraphRAG',
    'nav.topic.16': 'RAG at Scale',
    'nav.cat.vectorsearch': 'Vector Search',
    'nav.topic.17': 'DiskANN',
    'nav.topic.18': 'AiSAQ',
    'nav.topic.19': 'HNSW',
    'nav.cat.retrievaleng': 'Retrieval Engineering',
    'nav.topic.20': 'Hybrid Search',
    'nav.topic.21': 'KG Construction',
    'nav.topic.22': 'CAG vs RAG',
    // Progress
    'progress.visited': 'Pages Visited',
    'progress.games': 'Games Completed',
    'progress.reset': 'Reset Progress',
    // Common buttons
    'btn.prev': '\u2190 Prev',
    'btn.next': 'Next \u2192',
    'btn.play': 'Play',
    'btn.pause': 'Pause',
    'btn.reset': 'Reset',
    'btn.randomize': 'Randomize',
    'btn.nextRound': 'Next Round \u2192',
    'btn.playAgain': 'Play Again',
    'btn.animate': 'Animate',
    // Breadcrumb
    'breadcrumb.home': 'Home',
    // Footer
    'footer.text': 'LLM Algorithm Visualization Series \u2014 Built for interactive learning',
    // Index page
    'index.title': 'LLM Algorithm Visualizations',
    'index.pageTitle': 'LLM Algorithm Visualizations',
    'index.subtitle': 'Interactive visual guides from Transformer fundamentals to production serving innovations. Learn how modern large language models work under the hood.',
    'index.viewGrid': 'Grid',
    'index.viewGraph': 'Knowledge Graph',
    'index.section.prerequisites': 'Prerequisites',
    'index.00.title': 'Linear Algebra Basics',
    'index.00.desc': 'Vectors, dot products, matrix multiplication, softmax — the math building blocks behind every transformer layer.',
    'index.section.foundations': '2017\u20132019 Foundations',
    'index.section.compute': '2022 Compute Optimization',
    'index.section.memory': '2023 Memory Management',
    'index.section.advanced': '2024\u20132026 Advanced',
    'index.section.rag': 'RAG & Retrieval',
    'index.12.title': 'Dense Retrieval',
    'index.12.desc': 'DPR & ColBERT \u2014 dense vectors beat BM25 by +19%. Late interaction with token-level MaxSim matching.',
    'index.13.title': 'RAG Pipeline',
    'index.13.desc': 'Retrieve-then-generate paradigm & HyDE query enhancement. Ground LLM outputs in external knowledge.',
    'index.14.title': 'Adaptive RAG',
    'index.14.desc': 'Self-RAG reflection tokens, CRAG corrective retrieval, FLARE active mid-generation retrieval.',
    'index.15.title': 'GraphRAG',
    'index.15.desc': 'Knowledge graph communities for corpus-level reasoning. 72-83% comprehensiveness over vector RAG.',
    'index.16.title': 'RAG at Scale',
    'index.16.desc': 'RETRO: 7.5B + retrieval matches 175B GPT-3. RAG evolution from Naive to Modular paradigm.',
    // Index topic cards
    'index.01.title': 'Transformer Basics',
    'index.01.desc': 'Attention Is All You Need \u2014 the architecture that started it all. Interactive QKV attention walkthrough.',
    'index.02.title': 'GPT-2 Architecture',
    'index.02.desc': 'Decoder-only Transformer, Pre-LayerNorm, BPE tokenization, and weight tying explained visually.',
    'index.03.title': 'Sparse Transformers',
    'index.03.desc': 'O(N\u221AN) attention via local + strided patterns. Process sequences up to 65K tokens.',
    'index.04.title': 'Prefill & Decode',
    'index.04.desc': 'The two-phase inference pipeline \u2014 compute-bound prefill and memory-bound autoregressive decoding.',
    'index.05.title': 'FlashAttention',
    'index.05.desc': 'IO-aware attention with tiling and online softmax. Never materialize the N\u00D7N matrix in HBM.',
    'index.06.title': 'PagedAttention / vLLM',
    'index.06.desc': 'OS-inspired KV cache paging with block tables. Boost throughput 2-4\u00D7 by eliminating memory fragmentation.',
    'index.07.title': 'H2O \u2014 Heavy-Hitter Oracle',
    'index.07.desc': 'Exploit 95%+ attention sparsity. Keep only heavy-hitter tokens for 20\u00D7 KV cache compression.',
    'index.08.title': 'StreamingLLM',
    'index.08.desc': 'Attention sinks discovery \u2014 constant memory streaming by keeping initial sink tokens + recent window.',
    'index.09.title': 'PD Separation',
    'index.09.desc': 'Disaggregate prefill and decode into independent GPU pools for optimal resource utilization.',
    'index.10.title': 'Mooncake Architecture',
    'index.10.desc': 'KV-cache-centric disaggregated serving with distributed 3-tier cache pool. USENIX FAST 2025 Best Paper.',
    'index.11.title': 'TurboQuant',
    'index.11.desc': 'Training-free KV cache quantization via random rotation + QJL. 6\u00D7 compression, zero accuracy loss.',
    // Graph
    'graph.legend.foundation': 'Foundation',
    'graph.legend.compute': 'Compute',
    'graph.legend.memory': 'Memory',
    'graph.legend.system': 'System',
    'graph.legend.depends': 'Depends on',
    'graph.tooltip.click': 'Click to open',
    'graph.help': 'Drag nodes to rearrange \u00B7 Scroll to zoom \u00B7 Click node to navigate',
    'footer.series': 'LLM Algorithm Visualization Series',
  },
  'zh-CN': {
    // Nav
    'nav.brand': '大语言模型算法可视化',
    'nav.cat.prerequisites': '前置知识',
    'nav.cat.foundations': '2017–2019 基础架构',
    'nav.cat.compute': '2022 计算优化',
    'nav.cat.memory': '2023 内存管理',
    'nav.cat.advanced': '2024–2026 前沿技术',
    'nav.cat.rag': 'RAG 与检索',
    'nav.topic.00': '线性代数基础',
    'nav.topic.01': 'Transformer 基础',
    'nav.topic.02': 'GPT-2 架构',
    'nav.topic.03': '稀疏 Transformer',
    'nav.topic.04': '预填充与解码',
    'nav.topic.05': 'FlashAttention',
    'nav.topic.06': 'PagedAttention',
    'nav.topic.07': 'H2O',
    'nav.topic.08': 'StreamingLLM',
    'nav.topic.09': 'PD 分离',
    'nav.topic.10': 'Mooncake',
    'nav.topic.11': 'TurboQuant',
    'nav.topic.12': '稠密检索',
    'nav.topic.13': 'RAG 流水线',
    'nav.topic.14': '自适应 RAG',
    'nav.topic.15': 'GraphRAG',
    'nav.topic.16': '大规模 RAG',
    'nav.cat.vectorsearch': '向量搜索',
    'nav.topic.17': 'DiskANN',
    'nav.topic.18': 'AiSAQ',
    'nav.topic.19': 'HNSW',
    'nav.cat.retrievaleng': '检索工程',
    'nav.topic.20': '混合搜索',
    'nav.topic.21': '知识图谱构建',
    'nav.topic.22': 'CAG vs RAG',
    // Progress
    'progress.visited': '已访问页面',
    'progress.games': '已完成游戏',
    'progress.reset': '重置进度',
    // Common buttons
    'btn.prev': '\u2190 上一步',
    'btn.next': '下一步 \u2192',
    'btn.play': '播放',
    'btn.pause': '暂停',
    'btn.reset': '重置',
    'btn.randomize': '随机生成',
    'btn.nextRound': '下一轮 \u2192',
    'btn.playAgain': '再玩一次',
    'btn.animate': '播放动画',
    // Breadcrumb
    'breadcrumb.home': '首页',
    // Footer
    'footer.text': '大语言模型算法可视化系列 \u2014 为交互式学习而构建',
    // Index page
    'index.title': '大语言模型算法可视化',
    'index.pageTitle': '大语言模型算法可视化',
    'index.subtitle': '从 Transformer 基础到生产级推理优化的交互式可视化教程。深入了解现代大语言模型的底层工作原理。',
    'index.viewGrid': '网格',
    'index.viewGraph': '知识图谱',
    'index.section.prerequisites': '前置知识',
    'index.00.title': '线性代数基础',
    'index.00.desc': '向量、点积、矩阵乘法、softmax——每个 Transformer 层背后的数学基石。',
    'index.section.foundations': '2017\u20132019 基础架构',
    'index.section.compute': '2022 计算优化',
    'index.section.memory': '2023 内存管理',
    'index.section.advanced': '2024\u20132026 前沿技术',
    'index.section.rag': 'RAG 与检索',
    'index.12.title': '稠密检索',
    'index.12.desc': 'DPR 和 ColBERT \u2014 稠密向量比 BM25 高 +19%。Token 级 MaxSim 后交互匹配。',
    'index.13.title': 'RAG 流水线',
    'index.13.desc': '检索-然后-生成范式与 HyDE 查询增强。将 LLM 输出扎根于外部知识。',
    'index.14.title': '自适应 RAG',
    'index.14.desc': 'Self-RAG 反思 token、CRAG 纠正检索、FLARE 生成中主动检索。',
    'index.15.title': 'GraphRAG',
    'index.15.desc': '知识图谱社区用于语料级推理。比向量 RAG 综合性高 72-83%。',
    'index.16.title': '大规模 RAG',
    'index.16.desc': 'RETRO：7.5B + 检索匹配 175B GPT-3。RAG 从朴素到模块化范式的演进。',
    // Index topic cards
    'index.01.title': 'Transformer 基础',
    'index.01.desc': 'Attention Is All You Need \u2014 开启一切的架构。交互式 QKV 注意力机制详解。',
    'index.02.title': 'GPT-2 架构',
    'index.02.desc': '仅解码器 Transformer、Pre-LayerNorm、BPE 分词和权重绑定的可视化讲解。',
    'index.03.title': '稀疏 Transformer',
    'index.03.desc': '通过局部 + 跨步模式实现 O(N\u221AN) 注意力。处理长达 65K token 的序列。',
    'index.04.title': '预填充与解码',
    'index.04.desc': '两阶段推理流水线 \u2014 计算密集型预填充和内存密集型自回归解码。',
    'index.05.title': 'FlashAttention',
    'index.05.desc': 'IO 感知的分块注意力与在线 softmax。从不在 HBM 中实体化 N\u00D7N 矩阵。',
    'index.06.title': 'PagedAttention / vLLM',
    'index.06.desc': '受操作系统启发的 KV 缓存分页与块表。通过消除内存碎片提升 2-4 倍吞吐量。',
    'index.07.title': 'H2O \u2014 重击者预言机',
    'index.07.desc': '利用 95%+ 注意力稀疏性。仅保留重击者 token 实现 20 倍 KV 缓存压缩。',
    'index.08.title': 'StreamingLLM',
    'index.08.desc': '注意力汇聚点发现 \u2014 通过保留初始汇聚 token + 近期窗口实现恒定内存流式推理。',
    'index.09.title': 'PD 分离',
    'index.09.desc': '将预填充和解码分离到独立的 GPU 池中，实现最优资源利用率。',
    'index.10.title': 'Mooncake 架构',
    'index.10.desc': '以 KV 缓存为中心的分离式推理服务，配备分布式三级缓存池。USENIX FAST 2025 最佳论文。',
    'index.11.title': 'TurboQuant',
    'index.11.desc': '免训练 KV 缓存量化，通过随机旋转 + QJL 实现。6 倍压缩，零精度损失。',
    // Graph
    'graph.legend.foundation': '基础架构',
    'graph.legend.compute': '计算优化',
    'graph.legend.memory': '内存管理',
    'graph.legend.system': '系统架构',
    'graph.legend.depends': '依赖于',
    'graph.tooltip.click': '点击打开',
    'graph.help': '拖动节点重新排列 \u00B7 滚轮缩放 \u00B7 点击节点导航',
    'footer.series': '大语言模型算法可视化系列',
  },
  'zh-TW': {
    // Nav
    'nav.brand': '大語言模型演算法視覺化',
    'nav.cat.prerequisites': '前置知識',
    'nav.cat.foundations': '2017–2019 基礎架構',
    'nav.cat.compute': '2022 計算優化',
    'nav.cat.memory': '2023 記憶體管理',
    'nav.cat.advanced': '2024–2026 前沿技術',
    'nav.cat.rag': 'RAG 與檢索',
    'nav.topic.00': '線性代數基礎',
    'nav.topic.01': 'Transformer 基礎',
    'nav.topic.02': 'GPT-2 架構',
    'nav.topic.03': '稀疏 Transformer',
    'nav.topic.04': '預填充與解碼',
    'nav.topic.05': 'FlashAttention',
    'nav.topic.06': 'PagedAttention',
    'nav.topic.07': 'H2O',
    'nav.topic.08': 'StreamingLLM',
    'nav.topic.09': 'PD 分離',
    'nav.topic.10': 'Mooncake',
    'nav.topic.11': 'TurboQuant',
    'nav.topic.12': '稠密檢索',
    'nav.topic.13': 'RAG 流水線',
    'nav.topic.14': '自適應 RAG',
    'nav.topic.15': 'GraphRAG',
    'nav.topic.16': '大規模 RAG',
    'nav.cat.vectorsearch': '向量搜尋',
    'nav.topic.17': 'DiskANN',
    'nav.topic.18': 'AiSAQ',
    'nav.topic.19': 'HNSW',
    'nav.cat.retrievaleng': '檢索工程',
    'nav.topic.20': '混合搜尋',
    'nav.topic.21': '知識圖譜構建',
    'nav.topic.22': 'CAG vs RAG',
    // Progress
    'progress.visited': '已造訪頁面',
    'progress.games': '已完成遊戲',
    'progress.reset': '重設進度',
    // Common buttons
    'btn.prev': '\u2190 上一步',
    'btn.next': '下一步 \u2192',
    'btn.play': '播放',
    'btn.pause': '暫停',
    'btn.reset': '重設',
    'btn.randomize': '隨機產生',
    'btn.nextRound': '下一輪 \u2192',
    'btn.playAgain': '再玩一次',
    'btn.animate': '播放動畫',
    // Breadcrumb
    'breadcrumb.home': '首頁',
    // Footer
    'footer.text': '大語言模型演算法視覺化系列 \u2014 為互動式學習而打造',
    // Index page
    'index.title': '大語言模型演算法視覺化',
    'index.pageTitle': '大語言模型演算法視覺化',
    'index.subtitle': '從 Transformer 基礎到生產級推理優化的互動式視覺化教學。深入了解現代大語言模型的底層運作原理。',
    'index.viewGrid': '網格',
    'index.viewGraph': '知識圖譜',
    'index.section.prerequisites': '前置知識',
    'index.00.title': '線性代數基礎',
    'index.00.desc': '向量、點積、矩陣乘法、softmax——每個 Transformer 層背後的數學基石。',
    'index.section.foundations': '2017\u20132019 基礎架構',
    'index.section.compute': '2022 計算優化',
    'index.section.memory': '2023 記憶體管理',
    'index.section.advanced': '2024\u20132026 前沿技術',
    'index.section.rag': 'RAG 與檢索',
    'index.12.title': '稠密檢索',
    'index.12.desc': 'DPR 和 ColBERT \u2014 稠密向量比 BM25 高 +19%。Token 級 MaxSim 後交互匹配。',
    'index.13.title': 'RAG 流水線',
    'index.13.desc': '檢索-然後-生成範式與 HyDE 查詢增強。將 LLM 輸出紮根於外部知識。',
    'index.14.title': '自適應 RAG',
    'index.14.desc': 'Self-RAG 反思 token、CRAG 糾正檢索、FLARE 生成中主動檢索。',
    'index.15.title': 'GraphRAG',
    'index.15.desc': '知識圖譜社群用於語料級推理。比向量 RAG 綜合性高 72-83%。',
    'index.16.title': '大規模 RAG',
    'index.16.desc': 'RETRO：7.5B + 檢索匹配 175B GPT-3。RAG 從樸素到模組化範式的演進。',
    // Index topic cards
    'index.01.title': 'Transformer 基礎',
    'index.01.desc': 'Attention Is All You Need \u2014 開啟一切的架構。互動式 QKV 注意力機制詳解。',
    'index.02.title': 'GPT-2 架構',
    'index.02.desc': '僅解碼器 Transformer、Pre-LayerNorm、BPE 分詞和權重綁定的視覺化講解。',
    'index.03.title': '稀疏 Transformer',
    'index.03.desc': '透過局部 + 跨步模式實現 O(N\u221AN) 注意力。處理長達 65K token 的序列。',
    'index.04.title': '預填充與解碼',
    'index.04.desc': '兩階段推理管線 \u2014 計算密集型預填充和記憶體密集型自迴歸解碼。',
    'index.05.title': 'FlashAttention',
    'index.05.desc': 'IO 感知的分塊注意力與線上 softmax。從不在 HBM 中實體化 N\u00D7N 矩陣。',
    'index.06.title': 'PagedAttention / vLLM',
    'index.06.desc': '受作業系統啟發的 KV 快取分頁與區塊表。透過消除記憶體碎片提升 2-4 倍吞吐量。',
    'index.07.title': 'H2O \u2014 重擊者預言機',
    'index.07.desc': '利用 95%+ 注意力稀疏性。僅保留重擊者 token 實現 20 倍 KV 快取壓縮。',
    'index.08.title': 'StreamingLLM',
    'index.08.desc': '注意力匯聚點發現 \u2014 透過保留初始匯聚 token + 近期視窗實現恆定記憶體串流推理。',
    'index.09.title': 'PD 分離',
    'index.09.desc': '將預填充和解碼分離到獨立的 GPU 池中，實現最佳資源利用率。',
    'index.10.title': 'Mooncake 架構',
    'index.10.desc': '以 KV 快取為中心的分離式推理服務，配備分散式三級快取池。USENIX FAST 2025 最佳論文。',
    'index.11.title': 'TurboQuant',
    'index.11.desc': '免訓練 KV 快取量化，透過隨機旋轉 + QJL 實現。6 倍壓縮，零精度損失。',
    // Graph
    'graph.legend.foundation': '基礎架構',
    'graph.legend.compute': '計算優化',
    'graph.legend.memory': '記憶體管理',
    'graph.legend.system': '系統架構',
    'graph.legend.depends': '依賴於',
    'graph.tooltip.click': '點擊開啟',
    'graph.help': '拖曳節點重新排列 \u00B7 滾輪縮放 \u00B7 點擊節點導覽',
    'footer.series': '大語言模型演算法視覺化系列',
  }
});

// Initialize i18n on DOM ready
document.addEventListener('DOMContentLoaded', function() {
  // Set initial lang attribute
  var lang = getLang();
  var htmlLang = lang === 'zh-CN' ? 'zh-Hans' : lang === 'zh-TW' ? 'zh-Hant' : 'en';
  document.documentElement.setAttribute('lang', htmlLang);

  // Apply translations after a short delay to let nav.js build the DOM first
  setTimeout(function() {
    applyI18n();
  }, 50);
});
