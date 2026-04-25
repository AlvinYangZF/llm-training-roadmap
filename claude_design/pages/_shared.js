/* ═════════════════════════════════ SHARED PAGE SCRIPT ═════════════════════════════════ */
/* Topic registry — single source of truth. Kept in sync with index.html. */
const TOPICS = [
  {id:'00', slug:'linear-algebra',    yr:'math', cat:'foundation', t:'Linear Algebra Basics',   d:'Vectors, dot products, matmul, softmax — the math behind every attention head.'},
  {id:'01', slug:'transformer-basics',yr:'2017', cat:'foundation', t:'Transformer Basics',      d:'"Attention Is All You Need" — an interactive QKV walkthrough.'},
  {id:'02', slug:'gpt2',              yr:'2019', cat:'foundation', t:'GPT-2 Architecture',      d:'Decoder-only Transformer, Pre-LN, BPE tokenization, weight tying — visually.'},
  {id:'03', slug:'sparse-transformer',yr:'2019', cat:'foundation', t:'Sparse Transformers',     d:'O(N√N) attention via local + strided patterns.'},
  {id:'04', slug:'prefill-decode',    yr:'core', cat:'compute',    t:'Prefill & Decode',        d:'The two-phase inference pipeline.'},
  {id:'05', slug:'flashattention',    yr:'2022', cat:'compute',    t:'FlashAttention',          d:'IO-aware attention with tiling + online softmax.'},
  {id:'06', slug:'paged-attention',   yr:'2023', cat:'memory',     t:'PagedAttention / vLLM',   d:'OS-inspired KV cache paging with block tables.'},
  {id:'07', slug:'h2o',               yr:'2023', cat:'memory',     t:'H2O — Heavy-Hitter Oracle', d:'20× KV cache compression by keeping heavy hitters.'},
  {id:'08', slug:'streaming-llm',     yr:'2024', cat:'system',     t:'StreamingLLM',            d:'Attention sinks + sliding window for constant-memory streaming.'},
  {id:'09', slug:'pd-separation',     yr:'2025', cat:'system',     t:'PD Separation',           d:'Disaggregated prefill/decode.'},
  {id:'10', slug:'mooncake',          yr:'2025', cat:'system',     t:'Mooncake Architecture',   d:'KV-cache-centric disaggregated serving.'},
  {id:'11', slug:'kv-cache',          yr:'core', cat:'memory',     t:'KV Cache',                d:'Why decoding gets cheaper after the first token.'},
  {id:'12', slug:'dense-retrieval',   yr:'2020', cat:'foundation', t:'Dense Retrieval',         d:'DPR & ColBERT.'},
  {id:'13', slug:'rag-pipeline',      yr:'2020', cat:'foundation', t:'RAG Pipeline',            d:'Retrieve-then-generate + HyDE.'},
  {id:'14', slug:'adaptive-rag',      yr:'2024', cat:'compute',    t:'Adaptive RAG',            d:'Self-RAG, CRAG, FLARE.'},
  {id:'15', slug:'bm25',              yr:'1994', cat:'system',     t:'BM25',                    d:'The lexical baseline that still wins.'},
  {id:'16', slug:'rag-at-scale',      yr:'2022', cat:'system',     t:'RAG at Scale',            d:'RETRO and taxonomy.'},
  {id:'17', slug:'diskann',           yr:'2019', cat:'system',     t:'DiskANN',                 d:'Vamana graph + SSD index.'},
  {id:'18', slug:'aisaq',             yr:'2024', cat:'system',     t:'AiSAQ',                   d:'DRAM-free ANN.'},
  {id:'19', slug:'hnsw',              yr:'2016', cat:'foundation', t:'HNSW',                    d:'Hierarchical Navigable Small World.'},
  {id:'20', slug:'hybrid-search',     yr:'2024', cat:'compute',    t:'Hybrid Search',           d:'BM25 + Dense + Reranking.'},
  {id:'21', slug:'kg-construction',   yr:'2024', cat:'compute',    t:'KG Construction',         d:'LightRAG, HippoRAG.'},
  {id:'22', slug:'cag-vs-rag',        yr:'2025', cat:'system',     t:'CAG vs RAG',              d:'Cache-Augmented Generation.'},
  {id:'23', slug:'embeddings',        yr:'core', cat:'foundation', t:'Embeddings',              d:'Text to vectors.'},
  {id:'24', slug:'tokenization',      yr:'core', cat:'foundation', t:'Tokenization',            d:'BPE, SentencePiece.'},
  {id:'25', slug:'training',          yr:'core', cat:'compute',    t:'Training & Loss',         d:'Cross-entropy loss, backprop.'},
  {id:'26', slug:'mqa-gqa',           yr:'2023', cat:'memory',     t:'MQA / GQA',               d:'Sharing KV heads to shrink the cache.'},
  {id:'27', slug:'lora',              yr:'2021', cat:'compute',    t:'LoRA',                    d:'Low-Rank Adaptation.'},
  {id:'28', slug:'decoding',          yr:'core', cat:'foundation', t:'Decoding',                d:'Temperature, top-k, top-p, beam.'},
];
const CAT_LABEL = { foundation:'Foundations', compute:'Compute', memory:'Memory & Serving', system:'System & RAG' };

/* Theme */
function initTheme() {
  const btn = document.getElementById('themeBtn');
  function set(t) {
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem('llm-viz-theme', t);
    if (btn) btn.textContent = t === 'dark' ? '◑ light' : '◐ dark';
  }
  set(localStorage.getItem('llm-viz-theme') || 'light');
  if (btn) btn.addEventListener('click', e => {
    e.preventDefault();
    const cur = document.documentElement.getAttribute('data-theme');
    set(cur === 'dark' ? 'light' : 'dark');
    window.dispatchEvent(new CustomEvent('themechange'));
  });
}

/* Mark page visited */
function markVisited(id) {
  const v = JSON.parse(localStorage.getItem('llm-viz-visited') || '[]');
  if (!v.includes(id)) {
    v.push(id);
    localStorage.setItem('llm-viz-visited', JSON.stringify(v));
  }
}

/* Read progress */
function initReadBar() {
  const bar = document.querySelector('.read-bar .fill');
  if (!bar) return;
  function tick() {
    const h = document.documentElement;
    const pct = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
    bar.style.width = Math.min(100, Math.max(0, pct * 100)) + '%';
  }
  window.addEventListener('scroll', tick, { passive: true });
  tick();
}

/* TOC highlight */
function initTocHighlight() {
  const links = document.querySelectorAll('.rail .toc a');
  if (!links.length) return;
  const sections = Array.from(links)
    .map(a => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);
  function tick() {
    let idx = 0;
    const y = window.scrollY + 120;
    sections.forEach((s, i) => { if (s.offsetTop <= y) idx = i; });
    links.forEach((a, i) => a.classList.toggle('active', i === idx));
  }
  window.addEventListener('scroll', tick, { passive: true });
  tick();
}

/* Smooth TOC clicks */
function initTocClicks() {
  document.querySelectorAll('.rail .toc a').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const target = document.querySelector(a.getAttribute('href'));
      if (target) window.scrollTo({ top: target.offsetTop - 80, behavior: 'smooth' });
    });
  });
}

/* Hero layout tweak */
const HERO_TWEAK_DEFAULT = 'eyebrow'; // inner pages default to eyebrow layout
function initHeroTweak() {
  const heroEl = document.querySelector('.phero');
  if (!heroEl) return;
  const saved = localStorage.getItem('llm-viz-page-hero') || HERO_TWEAK_DEFAULT;
  heroEl.setAttribute('data-layout', saved);
  document.querySelectorAll('#twHeroOpts .tw-opt').forEach(o => {
    o.classList.toggle('active', o.getAttribute('data-layout') === saved);
    o.addEventListener('click', () => {
      const layout = o.getAttribute('data-layout');
      heroEl.setAttribute('data-layout', layout);
      localStorage.setItem('llm-viz-page-hero', layout);
      document.querySelectorAll('#twHeroOpts .tw-opt').forEach(x =>
        x.classList.toggle('active', x.getAttribute('data-layout') === layout));
      window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { pageHeroLayout: layout } }, '*');
    });
  });
  const tweaksEl = document.getElementById('tweaks');
  const closeBtn = document.getElementById('twClose');
  if (closeBtn) closeBtn.addEventListener('click', () => tweaksEl.classList.remove('on'));
  window.addEventListener('message', e => {
    if (!e.data || !e.data.type) return;
    if (e.data.type === '__activate_edit_mode')   tweaksEl && tweaksEl.classList.add('on');
    if (e.data.type === '__deactivate_edit_mode') tweaksEl && tweaksEl.classList.remove('on');
  });
  window.parent.postMessage({ type: '__edit_mode_available' }, '*');
}

/* Build related links from edges (imported if present) */
function initRelated(currentId, relatedIds) {
  const el = document.querySelector('.rail .related');
  if (!el || !relatedIds) return;
  el.innerHTML = relatedIds.map(id => {
    const t = TOPICS.find(x => x.id === id);
    if (!t) return '';
    return `<a href="${t.id}-${t.slug}.html"><span>${t.id} ${t.t}</span><span class="yr">${t.yr}</span></a>`;
  }).join('');
}

/* Pager — prev/next by roadmap order */
function initPager(currentId) {
  const i = TOPICS.findIndex(t => t.id === currentId);
  const prev = i > 0 ? TOPICS[i-1] : null;
  const next = i < TOPICS.length - 1 ? TOPICS[i+1] : null;
  const pEl = document.querySelector('.pager');
  if (!pEl) return;
  pEl.innerHTML = `
    ${prev ? `<a href="${prev.id}-${prev.slug}.html">
      <div class="p-lab">← previous</div>
      <div class="p-num">${prev.id} · ${prev.yr}</div>
      <div class="p-ttl">${prev.t}</div>
    </a>` : `<span></span>`}
    ${next ? `<a href="${next.id}-${next.slug}.html" class="next">
      <div class="p-lab">next →</div>
      <div class="p-num">${next.id} · ${next.yr}</div>
      <div class="p-ttl">${next.t}</div>
    </a>` : `<span></span>`}
  `;
}

function initPage({ currentId, related }) {
  initTheme();
  markVisited(currentId);
  initReadBar();
  initTocHighlight();
  initTocClicks();
  initHeroTweak();
  initRelated(currentId, related);
  initPager(currentId);
}
window.LLMViz = { TOPICS, CAT_LABEL, initPage };
