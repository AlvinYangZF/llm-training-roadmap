// ==================== LLM Algorithm Visualization — Theme System ==================== //

const THEME_KEY = 'llm-viz-theme';

// Theme-aware color palettes for canvas drawing
const THEME_COLORS = {
  light: {
    bg:       '#f8fafc',
    surface:  '#ffffff',
    border:   '#cbd5e1',
    text:     '#1e293b',
    textDim:  '#64748b',
    accent:   '#5b5bd6',
    accent2:  '#0d9488',
    accent3:  '#e11d48',
    accent4:  '#ea580c',
    gridLine: '#e2e8f0',
    cellBg:   '#f1f5f9',
    highlight: 'rgba(91,91,214,0.15)',
    highlight2: 'rgba(13,148,136,0.15)',
    highlight3: 'rgba(225,29,72,0.12)',
    highlight4: 'rgba(234,88,12,0.12)',
    canvasBg: '#f1f5f9',
  },
  dark: {
    bg:       '#0f1117',
    surface:  '#1a1d2e',
    border:   '#2a2d3e',
    text:     '#cdd6f4',
    textDim:  '#6c7086',
    accent:   '#6c63ff',
    accent2:  '#00c9a7',
    accent3:  '#f7768e',
    accent4:  '#ff9e64',
    gridLine: '#2a2d3e',
    cellBg:   '#1a1d2e',
    highlight: 'rgba(108,99,255,0.35)',
    highlight2: 'rgba(0,201,167,0.35)',
    highlight3: 'rgba(247,118,142,0.25)',
    highlight4: 'rgba(255,158,100,0.25)',
    canvasBg: 'rgba(0,0,0,0.3)',
  }
};

function getTheme() {
  return localStorage.getItem(THEME_KEY) || 'light';
}

function setTheme(theme) {
  if (theme !== 'light' && theme !== 'dark') return;
  localStorage.setItem(THEME_KEY, theme);
  applyTheme();
}

function toggleTheme() {
  setTheme(getTheme() === 'dark' ? 'light' : 'dark');
}

// Get the current theme's canvas color palette
function getColors() {
  return THEME_COLORS[getTheme()];
}

function applyTheme() {
  var theme = getTheme();
  document.documentElement.setAttribute('data-theme', theme);

  // Update toggle button icon
  var btn = document.getElementById('themeToggleBtn');
  if (btn) {
    btn.innerHTML = theme === 'dark'
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
    btn.title = theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
  }

  // Dispatch event so pages can re-render canvases
  window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
}

// Initialize theme on load (run immediately, not on DOMContentLoaded, to prevent flash)
(function() {
  var theme = getTheme();
  document.documentElement.setAttribute('data-theme', theme);
})();

document.addEventListener('DOMContentLoaded', function() {
  applyTheme();
});
