// ==================== LLM Algorithm Visualization — Shared Animation Utilities ==================== //

// ==================== Reading Progress Bar ==================== //
(function() {
  function initReadingProgress() {
    // Only on topic pages, not index
    var path = window.location.pathname;
    if (path.indexOf('/pages/') === -1) return;

    var bar = document.createElement('div');
    bar.id = 'readingProgress';
    bar.style.cssText = 'position:fixed;top:0;left:0;height:2px;width:0;z-index:2000;pointer-events:none;transition:width 0.1s linear;background:linear-gradient(90deg,var(--accent),var(--accent2));';
    document.body.appendChild(bar);

    function updateProgress() {
      var scrollTop = window.scrollY || document.documentElement.scrollTop;
      var docHeight = document.documentElement.scrollHeight - window.innerHeight;
      var pct = docHeight > 0 ? Math.min(100, (scrollTop / docHeight) * 100) : 0;
      bar.style.width = pct + '%';
    }

    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  document.addEventListener('DOMContentLoaded', initReadingProgress);
})();

/**
 * Setup a canvas with proper DPR scaling
 * @param {HTMLCanvasElement} canvas
 * @returns {{ ctx: CanvasRenderingContext2D, W: number, H: number }}
 */
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, W: canvas.clientWidth, H: canvas.clientHeight };
}

/**
 * Draw a rounded rectangle
 */
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

/**
 * Animate speedup/comparison bars
 * @param {string} selector - CSS selector for .speedup-fill elements
 */
function animateBars(selector) {
  const bars = document.querySelectorAll(selector || '.speedup-fill');
  bars.forEach(bar => {
    bar.style.width = '0';
    setTimeout(() => {
      bar.style.width = bar.dataset.width + '%';
    }, 100);
  });
}

/**
 * Easing function — ease out cubic
 */
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Easing function — ease in out cubic
 */
function easeInOutCubic(t) {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Simple animation loop helper
 * @param {Function} drawFn - Called each frame with (progress: 0..1)
 * @param {number} duration - Duration in ms
 * @param {Function} [onComplete] - Called when animation finishes
 * @returns {number} requestAnimationFrame ID
 */
function animate(drawFn, duration, onComplete) {
  const start = performance.now();
  let rafId;
  function tick(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    drawFn(progress);
    if (progress < 1) {
      rafId = requestAnimationFrame(tick);
    } else if (onComplete) {
      onComplete();
    }
  }
  rafId = requestAnimationFrame(tick);
  return rafId;
}

// ==================== Matrix Math Utilities ==================== //

function matMul(A, B) {
  const m = A.length, nn = A[0].length, p = B[0].length;
  const C = [];
  for (let i = 0; i < m; i++) {
    C[i] = [];
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let k = 0; k < nn; k++) s += A[i][k] * B[k][j];
      C[i][j] = Math.round(s * 100) / 100;
    }
  }
  return C;
}

function transpose(M) {
  const r = M.length, c = M[0].length;
  const T = [];
  for (let j = 0; j < c; j++) {
    T[j] = [];
    for (let i = 0; i < r; i++) T[j][i] = M[i][j];
  }
  return T;
}

function softmaxRows(M) {
  return M.map(row => {
    const max = Math.max(...row);
    const exps = row.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => Math.round(v / sum * 100) / 100);
  });
}
