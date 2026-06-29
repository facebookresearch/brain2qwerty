/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */
// ---------------------------------------------------------------------------
// demo.js -- page interactions: scroll-reveal and the architecture ablation
// walkthrough.
//
// The interactive true/typed/decoded sentence explorer that used to live here
// is archived in archive/explorer.js (markup in archive/explorer.html). The
// diff helpers below (alignFlags, renderColored) stayed because the ablation
// walkthrough still needs them.
// ---------------------------------------------------------------------------
import { ABLATION } from './sentences.js';

// --- Scroll reveal ---------------------------------------------------------
const reveals = document.querySelectorAll('[data-reveal]');
if ('IntersectionObserver' in window && reveals.length) {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }
  }, { rootMargin: '0px 0px -10% 0px', threshold: 0.08 });
  reveals.forEach((el) => io.observe(el));
} else {
  reveals.forEach((el) => el.classList.add('in'));
}

// --- Animated figures: load (and thus play) their one-shot SVG only on view --
// The SVG auto-plays its reveal on load, so deferring the src until the figure
// is onscreen makes the animation start exactly then (and only once).
const animFigs = document.querySelectorAll('img.anim-on-view[data-anim-src]');
if (animFigs.length) {
  const load = (img) => { img.src = img.dataset.animSrc; };
  if ('IntersectionObserver' in window) {
    const aio = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) { load(e.target); aio.unobserve(e.target); }
      }
    }, { rootMargin: '0px 0px -15% 0px', threshold: 0.25 });
    animFigs.forEach((img) => aio.observe(img));
  } else {
    animFigs.forEach(load);
  }
}

// ---------------------------------------------------------------------------
// Diff helper: align a hypothesis string to a reference via edit-distance
// backtrace, returning per-hypothesis-character match flags. Used both to color
// the decoded output against the truth and to underline the participant's typos.
// ---------------------------------------------------------------------------
function alignFlags(hyp, ref) {
  const n = hyp.length, m = ref.length;
  const dp = Array.from({ length: n + 1 }, () => new Int32Array(m + 1));
  for (let i = 0; i <= n; i++) dp[i][0] = i;
  for (let j = 0; j <= m; j++) dp[0][j] = j;
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const cost = hyp[i - 1] === ref[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  // Backtrace -> for each hyp char, did it survive as a match?
  const flags = new Array(n).fill(false);
  let i = n, j = m;
  while (i > 0 && j > 0) {
    const cost = hyp[i - 1] === ref[j - 1] ? 0 : 1;
    if (dp[i][j] === dp[i - 1][j - 1] + cost) {
      flags[i - 1] = cost === 0;
      i--; j--;
    } else if (dp[i][j] === dp[i - 1][j] + 1) {
      flags[i - 1] = false; i--;     // insertion in hyp
    } else {
      j--;                            // deletion from ref
    }
  }
  while (i > 0) { flags[i - 1] = false; i--; }
  return flags;
}

function renderColored(text, ref, cls) {
  const flags = alignFlags(text, ref);
  const frag = document.createDocumentFragment();
  for (let k = 0; k < text.length; k++) {
    const span = document.createElement('span');
    span.className = 'c ' + (flags[k] ? 'good' : cls);
    span.textContent = text[k] === ' ' ? '\u00a0' : text[k];
    frag.appendChild(span);
  }
  return frag;
}

// ---------------------------------------------------------------------------
// Architecture ablation walkthrough
// ---------------------------------------------------------------------------
function renderAblation() {
  const host = document.getElementById('ablation-rows');
  if (!host) return;
  // Truth row first (reference).
  const truthRow = document.createElement('div');
  truthRow.className = 'ab-row';
  truthRow.innerHTML = '<div class="ab-stage">Read</div>';
  const truthText = document.createElement('div');
  truthText.className = 'ab-text';
  truthText.textContent = ABLATION.truth;
  truthRow.appendChild(truthText);
  host.appendChild(truthRow);

  for (const stage of ABLATION.stages) {
    const row = document.createElement('div');
    row.className = 'ab-row' + (stage.final ? ' final' : '');
    const label = document.createElement('div');
    label.className = 'ab-stage';
    label.textContent = stage.label;
    const text = document.createElement('div');
    text.className = 'ab-text';
    text.appendChild(renderColored(stage.text, ABLATION.truth, 'bad'));
    row.appendChild(label);
    row.appendChild(text);
    host.appendChild(row);
  }
}
renderAblation();
