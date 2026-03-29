// ==================== LLM Algorithm Visualization — Navigation ==================== //

const NAV_TOPICS = [
  { cat: 'Foundation', items: [
    { num: '01', title: 'Transformer Basics', file: '01-transformer-basics.html' },
    { num: '02', title: 'GPT-2 Architecture', file: '02-gpt2-architecture.html' },
    { num: '03', title: 'Prefill & Decode', file: '03-prefill-decode.html' },
  ]},
  { cat: 'Compute Optimization', items: [
    { num: '04', title: 'FlashAttention', file: '04-flash-attention.html' },
    { num: '10', title: 'Sparse Transformers', file: '10-sparse-transformers.html' },
  ]},
  { cat: 'Memory Management', items: [
    { num: '05', title: 'PagedAttention', file: '05-paged-attention.html' },
    { num: '08', title: 'H2O', file: '08-h2o.html' },
    { num: '09', title: 'StreamingLLM', file: '09-streaming-llm.html' },
    { num: '11', title: 'TurboQuant', file: '11-turboquant.html' },
  ]},
  { cat: 'System Architecture', items: [
    { num: '06', title: 'PD Separation', file: '06-pd-separation.html' },
    { num: '07', title: 'Mooncake', file: '07-mooncake.html' },
  ]},
];

// Flat ordered list for prev/next navigation
const NAV_ORDERED = [
  '01-transformer-basics.html',
  '02-gpt2-architecture.html',
  '03-prefill-decode.html',
  '04-flash-attention.html',
  '05-paged-attention.html',
  '06-pd-separation.html',
  '07-mooncake.html',
  '08-h2o.html',
  '09-streaming-llm.html',
  '10-sparse-transformers.html',
  '11-turboquant.html',
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
  nav.innerHTML = `
    <a href="${basePath}index.html" class="nav-brand">LLM Algorithm Visualizations</a>
    <button class="nav-toggle" onclick="toggleSidebar()" aria-label="Menu">&#9776;</button>
  `;
  document.body.prepend(nav);

  // Sidebar
  const sidebar = document.createElement('aside');
  sidebar.className = 'site-sidebar';
  sidebar.id = 'siteSidebar';

  let html = '';
  NAV_TOPICS.forEach(group => {
    html += `<div class="nav-category">${group.cat}</div>`;
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
