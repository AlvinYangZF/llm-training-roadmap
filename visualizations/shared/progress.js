// ==================== LLM Algorithm Visualization — Progress Tracker ==================== //

const PROGRESS_KEY = 'llm-viz-progress';
const TOTAL_PAGES = 17;

function getProgress() {
  try {
    return JSON.parse(localStorage.getItem(PROGRESS_KEY)) || { visited: [], gamesCompleted: [] };
  } catch (e) { return { visited: [], gamesCompleted: [] }; }
}

function saveProgress(data) {
  localStorage.setItem(PROGRESS_KEY, JSON.stringify(data));
}

function markVisited(pageId) {
  var p = getProgress();
  if (p.visited.indexOf(pageId) === -1) {
    p.visited.push(pageId);
    saveProgress(p);
  }
  updateProgressUI();
}

function markGameCompleted(pageId) {
  var p = getProgress();
  if (p.gamesCompleted.indexOf(pageId) === -1) {
    p.gamesCompleted.push(pageId);
    saveProgress(p);
  }
  updateProgressUI();
}

function isVisited(pageId) { return getProgress().visited.indexOf(pageId) !== -1; }
function isGameCompleted(pageId) { return getProgress().gamesCompleted.indexOf(pageId) !== -1; }
function getVisitedCount() { return getProgress().visited.length; }
function getGamesCompletedCount() { return getProgress().gamesCompleted.length; }

function resetAllProgress() {
  localStorage.removeItem(PROGRESS_KEY);
  updateProgressUI();
}

// Auto-mark current page as visited on load
function autoMarkVisited() {
  var page = window.location.pathname.split('/').pop();
  var match = page.match(/^(\d{2})/);
  if (match) markVisited(match[1]);
}

// Update any progress UI elements on the page
function updateProgressUI() {
  var p = getProgress();
  var visitedCount = p.visited.length;
  var gamesCount = p.gamesCompleted.length;

  // Progress bar on index page
  var bar = document.getElementById('progressFill');
  if (bar) {
    bar.style.width = Math.round((visitedCount / TOTAL_PAGES) * 100) + '%';
  }
  var visitedEl = document.getElementById('progressVisited');
  if (visitedEl) visitedEl.textContent = visitedCount;

  var gamesEl = document.getElementById('progressGames');
  if (gamesEl) gamesEl.textContent = gamesCount;

  var pctEl = document.getElementById('progressPct');
  if (pctEl) pctEl.textContent = Math.round((visitedCount / TOTAL_PAGES) * 100) + '%';

  // Update topic cards with visited/completed indicators
  document.querySelectorAll('.topic-card').forEach(function(card) {
    var href = card.getAttribute('href') || '';
    var m = href.match(/(\d{2})-/);
    if (!m) return;
    var id = m[1];
    var indicator = card.querySelector('.progress-indicator');
    if (!indicator) {
      indicator = document.createElement('span');
      indicator.className = 'progress-indicator';
      card.appendChild(indicator);
    }
    if (p.gamesCompleted.indexOf(id) !== -1) {
      indicator.textContent = '\u2713';
      indicator.className = 'progress-indicator completed';
      indicator.title = 'Game completed';
    } else if (p.visited.indexOf(id) !== -1) {
      indicator.textContent = '\u25CF';
      indicator.className = 'progress-indicator visited';
      indicator.title = 'Visited';
    } else {
      indicator.textContent = '';
      indicator.className = 'progress-indicator';
    }
  });

  // Update sidebar links with indicators
  document.querySelectorAll('.site-sidebar a').forEach(function(link) {
    var href = link.getAttribute('href') || '';
    var m = href.match(/(\d{2})-/);
    if (!m) return;
    var id = m[1];
    var dot = link.querySelector('.sidebar-progress');
    if (!dot) {
      dot = document.createElement('span');
      dot.className = 'sidebar-progress';
      dot.style.cssText = 'margin-left:auto;font-size:0.7rem;';
      link.appendChild(dot);
    }
    if (p.gamesCompleted.indexOf(id) !== -1) {
      dot.textContent = '\u2713';
      dot.style.color = 'var(--accent2)';
    } else if (p.visited.indexOf(id) !== -1) {
      dot.textContent = '\u25CF';
      dot.style.color = 'var(--accent)';
    } else {
      dot.textContent = '';
    }
  });
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', function() {
  autoMarkVisited();
  updateProgressUI();
});
