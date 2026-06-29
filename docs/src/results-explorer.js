/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */
// ---------------------------------------------------------------------------
// results-explorer.js -- tiny, self-contained results browser.
//
// Reads assets/best_complete_predictions.csv and picks the three best subjects
// OVERALL (lowest mean word error across all their sentences, social-media
// sentences excluded). It then pages through the sentences those same three
// subjects share, showing the true text plus each fixed subject's LLM decoding
// with a word-level diff (correct words blue, wrong words red). Reports
// per-subject Word accuracy (100% - error).
//
// Embed anywhere with:
//   <div id="results-explorer"></div>
//   <script type="module" src="./src/results-explorer.js"></script>
// or call mountResultsExplorer(el, { csvUrl }) on your own element.
//
// All markup lives inside the mount element and all CSS is scoped under .rx, so
// it drops into the demo pages without touching their styles.
// ---------------------------------------------------------------------------

const TOP_N = 3;
const DEFAULT_CSV = './assets/best_complete_predictions.csv';

const CSS = `
.rx { --rx-ink:#15171c; --rx-faint:#8a909c; --rx-line:#e6e8ec; --rx-good:#3b6cff; --rx-bad:#d6443c;
  --rx-mono:"SF Mono",ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
  color:var(--rx-ink); width:100%; max-width:760px; margin:0 auto; }
.rx table { width:100%; border-collapse:collapse; }
.rx td { padding:10px 4px; vertical-align:baseline; border-bottom:1px solid var(--rx-line); }
.rx tr:last-child td { border-bottom:0; }
.rx .rx-lab { width:52px; color:var(--rx-faint); font-size:12px; white-space:nowrap; }
.rx .rx-true td { border-bottom:1px solid var(--rx-ink); }
.rx .rx-true .rx-txt { font-weight:600; }
.rx .rx-txt { font-family:var(--rx-mono); font-size:14px; }
.rx .rx-met { float:right; font-family:var(--rx-mono); font-size:12px; color:var(--rx-faint);
  white-space:nowrap; padding-left:14px; }
.rx .rx-head td { border-bottom:0; padding-bottom:0; }
.rx .rx-met-head { text-transform:uppercase; letter-spacing:.04em; font-size:11px; font-weight:400; }
.rx .rx-true .rx-lab { font-family:var(--rx-mono); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
.rx .rx-ok { color:var(--rx-good); }
.rx .rx-no { color:var(--rx-bad); }
.rx .rx-nav { float:right; display:inline-flex; gap:6px; }
.rx .rx-nav button { width:28px; height:28px; padding:0; border:1px solid var(--rx-line);
  background:#fff; border-radius:7px; cursor:pointer; font-size:13px; color:var(--rx-ink);
  display:grid; place-items:center; line-height:1; }
.rx .rx-nav button:hover { border-color:var(--rx-ink); }
.rx .rx-status { color:var(--rx-faint); font-size:14px; }
`;

function injectCSS() {
  if (document.getElementById('rx-style')) return;
  const s = document.createElement('style');
  s.id = 'rx-style';
  s.textContent = CSS;
  document.head.appendChild(s);
}

// minimal RFC-4180 CSV parser (handles quoted fields w/ commas and "" escapes)
function parseCSV(text) {
  const rows = []; let row = [], field = '', i = 0, q = false; const n = text.length;
  while (i < n) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i += 2; continue; } q = false; i++; continue; }
      field += c; i++; continue;
    }
    if (c === '"') { q = true; i++; continue; }
    if (c === ',') { row.push(field); field = ''; i++; continue; }
    if (c === '\r') { i++; continue; }
    if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; i++; continue; }
    field += c; i++;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}

function buildGroups(rows) {
  const h = rows[0].map(s => s.trim());
  const ci = {
    s: h.indexOf('subject'), t: h.indexOf('true_text'), p: h.indexOf('llm_pred'),
    c: h.indexOf('llm_cer'), w: h.indexOf('llm_wer'),
  };
  const need = Math.max(ci.c, ci.w);

  // 1) Read every (subject, sentence) prediction, excluding social-media.
  const recs = [];
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r]; if (!row || row.length <= need) continue;
    const text = (row[ci.t] || '').trim(); if (!text) continue;
    if (/social media/i.test(text)) continue;       // exclude social-media sentences
    const cer = parseFloat(row[ci.c]);
    const wer = parseFloat(row[ci.w]);
    recs.push({
      subject: (row[ci.s] || '').trim(), text, pred: (row[ci.p] || '').trim(),
      cer: Number.isFinite(cer) ? cer : 1,
      wer: Number.isFinite(wer) ? wer : 1,
    });
  }

  // 2) Rank subjects OVERALL by mean word error; keep the three best.
  const perSubject = new Map();   // subject -> { sum, n }
  for (const rec of recs) {
    const agg = perSubject.get(rec.subject) || { sum: 0, n: 0 };
    agg.sum += rec.wer; agg.n += 1;
    perSubject.set(rec.subject, agg);
  }
  const bestSubjects = [...perSubject.entries()]
    .map(([subject, { sum, n }]) => ({ subject, meanWer: sum / n }))
    .sort((a, b) => a.meanWer - b.meanWer)
    .slice(0, TOP_N)
    .map(d => d.subject);
  const rank = new Map(bestSubjects.map((s, i) => [s, i]));   // for stable order

  // 3) Group those subjects' rows by sentence; keep sentences all three share.
  const map = new Map();
  for (const rec of recs) {
    if (!rank.has(rec.subject)) continue;
    (map.get(rec.text) || map.set(rec.text, []).get(rec.text)).push(rec);
  }
  const groups = [...map.entries()]
    .filter(([, subs]) => new Set(subs.map(s => s.subject)).size === TOP_N)
    .map(([text, subs]) => {
      subs.sort((a, b) => rank.get(a.subject) - rank.get(b.subject));
      return { text, subs: subs.slice(0, TOP_N) };
    });

  // Shuffle so Prev/Next steps through a varied order (not alphabetical).
  for (let i = groups.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [groups[i], groups[j]] = [groups[j], groups[i]];
  }
  return groups;
}

const esc = s => s.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

const normWord = w => w.toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');

// word-level diff vs. truth (LCS over words): correct words blue, wrong words red
function diff(truth, pred) {
  const aTok = truth.split(/(\s+)/), bTok = pred.split(/(\s+)/);
  const aWords = [], bWords = [], bIdx = [];
  for (const t of aTok) if (t.trim()) aWords.push(normWord(t));
  bTok.forEach((t, k) => { if (t.trim()) { bWords.push(normWord(t)); bIdx.push(k); } });
  const m = aWords.length, n = bWords.length;
  const dp = Array.from({ length: m + 1 }, () => new Uint16Array(n + 1));
  for (let i = m - 1; i >= 0; i--) for (let j = n - 1; j >= 0; j--)
    dp[i][j] = aWords[i] === bWords[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const matched = new Set();   // token indices of correctly-matched predicted words
  let i = 0, j = 0;
  while (i < m && j < n) {
    if (aWords[i] === bWords[j]) { matched.add(bIdx[j]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
    else j++;
  }
  const out = bTok.map((t, k) => {
    if (!t.trim()) return esc(t);
    return `<span class="${matched.has(k) ? 'rx-ok' : 'rx-no'}">${esc(t)}</span>`;
  }).join('');
  return out || esc(pred);
}

const NAV =
  '<span class="rx-nav">' +
  '<button data-nav="-1" title="Previous (\u2190)" aria-label="Previous">\u2190</button>' +
  '<button data-nav="1" title="Next (\u2192)" aria-label="Next">\u2192</button>' +
  '<button data-nav="rand" title="Random (R)" aria-label="Random">\uD83C\uDFB2</button>' +
  '</span>';

const pct = x => (100 - x * 100).toFixed(0);   // error rate -> accuracy %

export function mountResultsExplorer(el, opts = {}) {
  if (!el) return;
  injectCSS();
  el.classList.add('rx');
  const csvUrl = opts.csvUrl || DEFAULT_CSV;
  el.innerHTML = '<p class="rx-status">Loading\u2026</p>';

  let groups = [], idx = 0;

  function render() {
    if (!groups.length) return;
    idx = (idx % groups.length + groups.length) % groups.length;
    const g = groups[idx];
    const subs = g.subs.map(s =>
      `<tr><td class="rx-lab">S${esc(s.subject).replace(/^S0?/, '')}</td>` +
      `<td class="rx-txt"><span class="rx-met">${pct(s.wer)}%</span>` +
      `${diff(g.text, s.pred)}</td></tr>`
    ).join('');
    el.innerHTML =
      `<table>` +
      `<tr class="rx-head"><td class="rx-lab"></td>` +
      `<td class="rx-txt">${NAV}</td></tr>` +
      `<tr class="rx-true"><td class="rx-lab">true</td>` +
      `<td class="rx-txt"><span class="rx-met rx-met-head">word accuracy</span>${esc(g.text)}</td></tr>${subs}</table>`;
  }

  const go = d => { idx += d; render(); };
  const rand = () => {
    if (groups.length < 2) return render();
    let r; do { r = Math.floor(Math.random() * groups.length); } while (r === idx);
    idx = r; render();
  };

  el.addEventListener('click', e => {
    const btn = e.target.closest('button[data-nav]'); if (!btn) return;
    const v = btn.dataset.nav;
    v === 'rand' ? rand() : go(parseInt(v, 10));
  });
  // keyboard shortcuts (ignored while typing in an input)
  document.addEventListener('keydown', e => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') return;
    if (e.key === 'ArrowLeft') go(-1);
    else if (e.key === 'ArrowRight') go(1);
    else if (e.key === 'r' || e.key === 'R') rand();
  });

  fetch(csvUrl, { cache: 'no-cache' })
    .then(res => { if (!res.ok) throw new Error('CSV ' + res.status); return res.text(); })
    .then(text => {
      groups = buildGroups(parseCSV(text));
      groups.length ? render() : (el.innerHTML = '<p class="rx-status">No rows.</p>');
    })
    .catch(err => { el.innerHTML = `<p class="rx-status">Error: ${esc(err.message)}</p>`; });
}

// Auto-mount if a default container is present (standalone page or simple embed).
const auto = document.getElementById('results-explorer');
if (auto) mountResultsExplorer(auto);
