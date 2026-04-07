// ==================== Concept Intro Component ==================== //
// Renders a standardized concept overview block at the top of each topic page.
// Usage: renderConceptIntro({ badge, title, problem, solution, insight, formula, stats, flow })
// All text keys go through t() for i18n support.

// Register common labels used by the concept intro component
i18nRegister({
  'en': {
    'ci.label.problem': 'PROBLEM',
    'ci.label.solution': 'SOLUTION'
  },
  'zh-CN': {
    'ci.label.problem': '\u75DB\u70B9',
    'ci.label.solution': '\u89E3\u6CD5'
  },
  'zh-TW': {
    'ci.label.problem': '\u75DB\u9EDE',
    'ci.label.solution': '\u89E3\u6CD5'
  }
});

function renderConceptIntro(cfg) {
  var container = document.querySelector('.container');
  if (!container) return;

  var card = document.createElement('div');
  card.className = 'card concept-intro';

  // Badge
  var badge = document.createElement('span');
  badge.className = 'badge concept-intro-badge';
  badge.setAttribute('data-i18n', cfg.badge);
  badge.textContent = t(cfg.badge);
  card.appendChild(badge);

  // Title
  var h2 = document.createElement('h2');
  h2.className = 'concept-intro-title';
  h2.setAttribute('data-i18n', cfg.title);
  h2.setAttribute('data-i18n-html', '1');
  h2.innerHTML = t(cfg.title);
  card.appendChild(h2);

  // Problem / Solution split
  var ps = document.createElement('div');
  ps.className = 'concept-intro-ps';

  var probBox = document.createElement('div');
  probBox.className = 'concept-intro-box concept-intro-problem';
  var probLabel = document.createElement('div');
  probLabel.className = 'concept-intro-label';
  probLabel.setAttribute('data-i18n', 'ci.label.problem');
  probLabel.textContent = t('ci.label.problem');
  var probText = document.createElement('p');
  probText.setAttribute('data-i18n', cfg.problem);
  probText.setAttribute('data-i18n-html', '1');
  probText.innerHTML = t(cfg.problem);
  probBox.appendChild(probLabel);
  probBox.appendChild(probText);

  var arrow = document.createElement('div');
  arrow.className = 'concept-intro-arrow';
  arrow.innerHTML = '&rarr;';

  var solBox = document.createElement('div');
  solBox.className = 'concept-intro-box concept-intro-solution';
  var solLabel = document.createElement('div');
  solLabel.className = 'concept-intro-label';
  solLabel.setAttribute('data-i18n', 'ci.label.solution');
  solLabel.textContent = t('ci.label.solution');
  var solText = document.createElement('p');
  solText.setAttribute('data-i18n', cfg.solution);
  solText.setAttribute('data-i18n-html', '1');
  solText.innerHTML = t(cfg.solution);
  solBox.appendChild(solLabel);
  solBox.appendChild(solText);

  ps.appendChild(probBox);
  ps.appendChild(arrow);
  ps.appendChild(solBox);
  card.appendChild(ps);

  // Insight bar
  if (cfg.insight) {
    var insight = document.createElement('div');
    insight.className = 'concept-intro-insight';
    insight.setAttribute('data-i18n', cfg.insight);
    insight.setAttribute('data-i18n-html', '1');
    insight.innerHTML = t(cfg.insight);
    card.appendChild(insight);
  }

  // Flow SVG diagram (inline)
  if (cfg.flow) {
    var flowWrap = document.createElement('div');
    flowWrap.className = 'concept-intro-flow';
    renderFlowDiagram(flowWrap, cfg.flow);
    card.appendChild(flowWrap);
  }

  // Formula
  if (cfg.formula) {
    var formula = document.createElement('div');
    formula.className = 'concept-intro-formula';
    formula.textContent = cfg.formula;
    card.appendChild(formula);
  }

  // Stats row
  if (cfg.stats && cfg.stats.length) {
    var statsRow = document.createElement('div');
    statsRow.className = 'concept-intro-stats';
    for (var i = 0; i < cfg.stats.length; i++) {
      var s = cfg.stats[i];
      var pill = document.createElement('div');
      pill.className = 'concept-intro-stat';
      var v = document.createElement('div');
      v.className = 'concept-intro-stat-val';
      v.textContent = s.value;
      var l = document.createElement('div');
      l.className = 'concept-intro-stat-label';
      l.setAttribute('data-i18n', s.label);
      l.textContent = t(s.label);
      pill.appendChild(v);
      pill.appendChild(l);
      statsRow.appendChild(pill);
    }
    card.appendChild(statsRow);
  }

  // Insert as first child of container
  container.insertBefore(card, container.firstChild);
}

// Renders a horizontal flow diagram as inline SVG
function renderFlowDiagram(wrap, steps) {
  var stepW = 110, stepH = 36, arrowW = 32, pad = 12;
  var totalW = steps.length * stepW + (steps.length - 1) * arrowW + pad * 2;
  var totalH = stepH + pad * 2;

  var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 ' + totalW + ' ' + totalH);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', totalH);
  svg.style.display = 'block';
  svg.style.maxWidth = totalW + 'px';
  svg.style.margin = '0 auto';

  // Marker for arrowheads
  var defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
  var marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
  marker.setAttribute('id', 'ci-arrow');
  marker.setAttribute('viewBox', '0 0 10 10');
  marker.setAttribute('refX', '8');
  marker.setAttribute('refY', '5');
  marker.setAttribute('markerWidth', '6');
  marker.setAttribute('markerHeight', '6');
  marker.setAttribute('orient', 'auto');
  var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', 'M 0 0 L 10 5 L 0 10 z');
  path.setAttribute('fill', 'var(--text-dim)');
  marker.appendChild(path);
  defs.appendChild(marker);
  svg.appendChild(defs);

  var colors = ['var(--accent)', 'var(--accent2)', 'var(--accent4)', 'var(--accent)', 'var(--accent2)', 'var(--accent4)', 'var(--accent3)'];

  for (var i = 0; i < steps.length; i++) {
    var x = pad + i * (stepW + arrowW);
    var y = pad;
    var color = colors[i % colors.length];

    // Rounded rect
    var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x);
    rect.setAttribute('y', y);
    rect.setAttribute('width', stepW);
    rect.setAttribute('height', stepH);
    rect.setAttribute('rx', '8');
    rect.setAttribute('fill', 'var(--surface)');
    rect.setAttribute('stroke', color);
    rect.setAttribute('stroke-width', '1.5');
    svg.appendChild(rect);

    // Text label
    var text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', x + stepW / 2);
    text.setAttribute('y', y + stepH / 2 + 1);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'central');
    text.setAttribute('fill', color);
    text.setAttribute('font-size', '11');
    text.setAttribute('font-weight', '600');
    text.setAttribute('font-family', 'system-ui, sans-serif');
    text.textContent = t(steps[i]);
    svg.appendChild(text);

    // Arrow to next step
    if (i < steps.length - 1) {
      var lineX1 = x + stepW + 4;
      var lineX2 = x + stepW + arrowW - 4;
      var lineY = y + stepH / 2;
      var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', lineX1);
      line.setAttribute('y1', lineY);
      line.setAttribute('x2', lineX2);
      line.setAttribute('y2', lineY);
      line.setAttribute('stroke', 'var(--text-dim)');
      line.setAttribute('stroke-width', '1.5');
      line.setAttribute('marker-end', 'url(#ci-arrow)');
      svg.appendChild(line);
    }
  }

  wrap.appendChild(svg);
}
