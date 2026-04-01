// ==================== LLM Algorithm Visualization — Navigation ==================== //

const NAV_TOPICS = [
  { cat: 'Prerequisites', items: [
    { num: '00', title: 'Linear Algebra Basics', file: '00-linear-algebra.html', year: 'Math' },
  ]},
  { cat: '2017–2019 Foundations', items: [
    { num: '01', title: 'Transformer Basics', file: '01-transformer-basics.html', year: '2017' },
    { num: '02', title: 'GPT-2 Architecture', file: '02-gpt2-architecture.html', year: '2019' },
    { num: '03', title: 'Sparse Transformers', file: '03-sparse-transformers.html', year: '2019' },
  ]},
  { cat: '2022 Compute Optimization', items: [
    { num: '04', title: 'Prefill & Decode', file: '04-prefill-decode.html', year: 'Core' },
    { num: '05', title: 'FlashAttention', file: '05-flash-attention.html', year: '2022' },
  ]},
  { cat: '2023 Memory Management', items: [
    { num: '06', title: 'PagedAttention', file: '06-paged-attention.html', year: '2023' },
    { num: '07', title: 'H2O', file: '07-h2o.html', year: '2023' },
  ]},
  { cat: '2024–2026 Advanced', items: [
    { num: '08', title: 'StreamingLLM', file: '08-streaming-llm.html', year: '2024' },
    { num: '09', title: 'PD Separation', file: '09-pd-separation.html', year: '2025' },
    { num: '10', title: 'Mooncake', file: '10-mooncake.html', year: '2025' },
    { num: '11', title: 'TurboQuant', file: '11-turboquant.html', year: '2026' },
  ]},
  { cat: 'RAG & Retrieval', items: [
    { num: '12', title: 'Dense Retrieval', file: '12-dense-retrieval.html', year: '2020' },
    { num: '13', title: 'RAG Pipeline', file: '13-rag-pipeline.html', year: '2020' },
    { num: '14', title: 'Adaptive RAG', file: '14-adaptive-rag.html', year: '2024' },
    { num: '15', title: 'GraphRAG', file: '15-graph-rag.html', year: '2024' },
    { num: '16', title: 'RAG at Scale', file: '16-rag-at-scale.html', year: '2022' },
  ]},
];

// Flat ordered list for prev/next navigation (chronological)
const NAV_ORDERED = [
  '00-linear-algebra.html',
  '01-transformer-basics.html',
  '02-gpt2-architecture.html',
  '03-sparse-transformers.html',
  '04-prefill-decode.html',
  '05-flash-attention.html',
  '06-paged-attention.html',
  '07-h2o.html',
  '08-streaming-llm.html',
  '09-pd-separation.html',
  '10-mooncake.html',
  '11-turboquant.html',
  '12-dense-retrieval.html',
  '13-rag-pipeline.html',
  '14-adaptive-rag.html',
  '15-graph-rag.html',
  '16-rag-at-scale.html',
];

function getCurrentPage() {
  const path = window.location.pathname;
  const filename = path.split('/').pop();
  return filename;
}

function getBasePath() {
  const path = window.location.pathname;
  if (path.includes('/pages/')) return '../';
  return './';
}

function buildNav() {
  const currentPage = getCurrentPage();
  const basePath = getBasePath();
  const isIndex = currentPage === 'index.html' || currentPage === '' || !currentPage;

  // Top nav bar
  const nav = document.createElement('nav');
  nav.className = 'site-nav';
  const currentLang = typeof getLang === 'function' ? getLang() : 'en';
  nav.innerHTML = `
    <a href="${basePath}index.html" class="nav-brand">LLM Algorithm Visualizations</a>
    <div class="nav-right">
      <div class="lang-selector">
        <select id="langSelect" onchange="setLang(this.value)" aria-label="Language">
          <option value="en" ${currentLang === 'en' ? 'selected' : ''}>English</option>
          <option value="zh-CN" ${currentLang === 'zh-CN' ? 'selected' : ''}>简体中文</option>
          <option value="zh-TW" ${currentLang === 'zh-TW' ? 'selected' : ''}>繁體中文</option>
        </select>
      </div>
      <button id="themeToggleBtn" class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme"></button>
      <button class="nav-toggle" onclick="toggleSidebar()" aria-label="Menu">&#9776;</button>
    </div>
  `;
  document.body.prepend(nav);

  // Sidebar
  const sidebar = document.createElement('aside');
  sidebar.className = 'site-sidebar';
  sidebar.id = 'siteSidebar';

  // Category i18n keys
  const catI18nKeys = [
    'nav.cat.prerequisites',
    'nav.cat.foundations',
    'nav.cat.compute',
    'nav.cat.memory',
    'nav.cat.advanced',
    'nav.cat.rag',
  ];

  let html = '';
  NAV_TOPICS.forEach((group, gi) => {
    const catKey = catI18nKeys[gi] || '';
    html += `<div class="nav-category" data-i18n="${catKey}">${group.cat}</div>`;
    group.items.forEach(item => {
      const href = `${basePath}pages/${item.file}`;
      const active = currentPage === item.file ? ' active' : '';
      html += `<a href="${href}" class="${active}"><span class="nav-num">${item.num}</span>${item.title}</a>`;
    });
  });

  sidebar.innerHTML = html;
  document.body.insertBefore(sidebar, document.body.children[1]);

  // Wrap existing content in main-content div if not already
  if (!document.querySelector('.main-content')) {
    const main = document.createElement('div');
    main.className = 'main-content';
    while (document.body.children.length > 2) {
      main.appendChild(document.body.children[2]);
    }
    document.body.appendChild(main);
  }
}

function toggleSidebar() {
  const sidebar = document.getElementById('siteSidebar');
  sidebar.classList.toggle('open');
}

// Close sidebar on click outside (mobile)
document.addEventListener('click', function(e) {
  const sidebar = document.getElementById('siteSidebar');
  const toggle = document.querySelector('.nav-toggle');
  if (sidebar && sidebar.classList.contains('open') &&
      !sidebar.contains(e.target) && !toggle.contains(e.target)) {
    sidebar.classList.remove('open');
  }
});

// Build prev/next navigation for topic pages
function buildPageNav() {
  const currentPage = getCurrentPage();
  const basePath = getBasePath();
  const idx = NAV_ORDERED.indexOf(currentPage);
  if (idx === -1) return;

  // Find title for a page file
  function getTitle(file) {
    for (const group of NAV_TOPICS) {
      for (const item of group.items) {
        if (item.file === file) return `${item.num}. ${item.title}`;
      }
    }
    return '';
  }

  const navDiv = document.createElement('div');
  navDiv.className = 'page-nav';

  if (idx > 0) {
    const prev = NAV_ORDERED[idx - 1];
    navDiv.innerHTML += `<a href="${basePath}pages/${prev}" class="prev-link">&larr; ${getTitle(prev)}</a>`;
  } else {
    navDiv.innerHTML += '<span></span>';
  }

  if (idx < NAV_ORDERED.length - 1) {
    const next = NAV_ORDERED[idx + 1];
    navDiv.innerHTML += `<a href="${basePath}pages/${next}" class="next-link">${getTitle(next)} &rarr;</a>`;
  } else {
    navDiv.innerHTML += '<span></span>';
  }

  // Insert before footer or at end of main-content
  const main = document.querySelector('.main-content');
  const footer = main ? main.querySelector('footer') : document.querySelector('footer');
  if (footer) {
    footer.parentNode.insertBefore(navDiv, footer);
  } else if (main) {
    main.appendChild(navDiv);
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  buildNav();
  buildPageNav();
});
