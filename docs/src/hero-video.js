/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */
// ---------------------------------------------------------------------------
// hero-video.js -- lightweight, video-driven hero (no WebGL / no Three.js).
//
// Plays the two clips baked by tools/capture-hero.mjs in place of the live
// 5-pass line-art + Gaussian-splat pipeline (src/hero-v2.js):
//   zoom.mp4      scroll-scrubbed camera journey (scrubP 0 wide MEG -> 1 close
//                 brain). currentTime is driven directly by scroll progress.
//   activity.mp4  the cortical activation, looping on its own clock, cross-faded
//                 in over the close-up (its frame matches zoom.mp4's last frame,
//                 so the handoff is seamless).
//
// The scroll choreography (heroScrollProgress / brainStartScrub) and overlay
// fades are mirrored 1:1 from hero-v2.js so the two demos feel identical -- but
// the hero now loads instantly and costs ~nothing to run.
// ---------------------------------------------------------------------------

const BASE = './assets/hero-video/';
// ROCK variant reads the rock manifest (per-frame motor-cortex track + the
// rocking activity clip); it inherits zoom/handoff/etc. from the base manifest.
const MANIFEST_URL = BASE + 'manifest_rock.json';

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
// THREE.MathUtils.smoothstep signature: (value, edge0, edge1).
const smoothstep = (x, edge0, edge1) => {
  const t = clamp((x - edge0) / (edge1 - edge0 || 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
};

// --- Page order (mirrors hero-v2.js) ---------------------------------------
const START_MODE = new URLSearchParams(location.search).get('start')
  || (document.body && document.body.dataset.heroStart) || 'meg';
const startOnBrain = START_MODE === 'brain';

// Scroll-choreography constants. scrubP: 0 = wide MEG, 1 = close on the cortex.
//  - ZOOM0: a tiny dwell on the close brain so the very first scroll/keypress
//    visibly moves the scene (avoids the old "nothing happens" feel).
//  - ZOOM1: scroll fraction at which the camera reaches the wide MEG view.
//  - ZOOM_DEFLATE: portion of the zoom window spent on the in-place deflate
//    (scrubP 1 -> MORPH_START). Set so BOTH segments share the same slope, i.e.
//    the whole brain->MEG scrub is one smooth, gradual ramp (no sudden jump).
const MORPH_START = 0.45;
const ZOOM0 = 0.04, ZOOM1 = 0.92;
// matching-slope split: deflate covers (1 - MORPH_START) of the scrubP range.
const ZOOM_DEFLATE = 1 - MORPH_START;
function brainStartScrub(p) {
  if (p <= ZOOM0) return 1;
  const z = clamp((p - ZOOM0) / (ZOOM1 - ZOOM0), 0, 1);
  if (z < ZOOM_DEFLATE) return 1 - (z / ZOOM_DEFLATE) * (1 - MORPH_START);
  return MORPH_START * (1 - (z - ZOOM_DEFLATE) / (1 - ZOOM_DEFLATE));
}

// Must match the @media breakpoint in demo-v2.css where the phone hero framing
// (#hero-activity / #hero-zoom object-position) is applied.
const PHONE_MQ = '(max-width: 820px)';
let isPhone = window.matchMedia(PHONE_MQ).matches;
window.addEventListener('resize', () => {
  isPhone = window.matchMedia(PHONE_MQ).matches;
}, { passive: true });

// --- DOM --------------------------------------------------------------------
const stage = document.getElementById('hero-stage');
const loadingEl = document.getElementById('hero-loading');
const heroEl = document.querySelector('.hero');
const stickyEl = document.querySelector('.hero-sticky');
const overlayEl = document.querySelector('.hero-overlay');
const scrollHintEl = document.querySelector('.scroll-hint');
const heroAnimEl = document.getElementById('hero-anim');
const heroKbdEl = document.getElementById('hero-kbd');
// The keyboard + decoded-sentence overlays are positioned entirely in CSS now
// (two fixed modes: desktop right-half stack, phone vertical stack). This JS
// only fades their opacity in at the brain close-up (see apply()).

function heroScrollProgress() {
  const range = (heroEl ? heroEl.offsetHeight : window.innerHeight * 2) - window.innerHeight;
  const y = window.scrollY || window.pageYOffset || 0;
  return clamp(y / Math.max(1, range), 0, 1);
}

// --- Build the two stacked <video> layers ----------------------------------
function makeVideo(src, { loop, poster }) {
  const v = document.createElement('video');
  v.src = src;
  v.muted = true;
  v.defaultMuted = true;
  v.playsInline = true;
  v.setAttribute('playsinline', '');
  v.setAttribute('muted', '');
  v.setAttribute('aria-hidden', 'true');
  v.preload = 'auto';
  v.loop = !!loop;
  if (poster) v.poster = poster;
  Object.assign(v.style, {
    position: 'absolute', inset: '0', width: '100%', height: '100%',
    objectFit: 'cover', objectPosition: 'center', pointerEvents: 'none',
  });
  return v;
}

// Both videos live INSIDE #hero-stage, which the page paints below the HTML
// overlays (title/GIF/timer/scroll-hint) that follow it in the DOM. We must NOT
// give the videos a positive z-index: that would lift them out of the stage's
// paint order and on top of those overlays (which are z-index:auto), hiding the
// title + GIF whenever the activity clip is opaque at the close-up. Keep both at
// z-index 0; the activity clip sits above the zoom clip purely by DOM order.
const zoomVideo = makeVideo(BASE + 'zoom.mp4', {
  loop: false, poster: BASE + (startOnBrain ? 'poster_close.jpg' : 'poster_wide.jpg'),
});
zoomVideo.style.zIndex = '0';
zoomVideo.id = 'hero-zoom';        // wide MEG scene (scroll-scrubbed)
// ROCK variant: the activity clip is baked WITH the brain rocking in 3D (about
// the vertical axis) by tools/capture-hero-rock.mjs, so the rotation is real
// pixels, not a CSS sway. The motor-cortex callout is driven per-frame from
// manifest_rock.json (the vertex moves with the brain).
const activityVideo = makeVideo(BASE + 'activity_rock.mp4', { loop: true, poster: BASE + 'poster_close.jpg' });
activityVideo.style.zIndex = '0';
activityVideo.id = 'hero-activity';   // close-up brain (looping)
activityVideo.style.opacity = '0';
activityVideo.style.transition = 'none';

// Mount under the loader so the "Loading…" text stays on top until first frame.
if (stage && loadingEl) {
  stage.insertBefore(zoomVideo, loadingEl);
  stage.insertBefore(activityVideo, loadingEl);
} else if (stage) {
  stage.appendChild(zoomVideo);
  stage.appendChild(activityVideo);
}

// --- "Motor cortex" callout (static, positioned from the manifest) ---------
let cortexAnnotEl = null;
if (stickyEl) {
  cortexAnnotEl = document.createElement('div');
  cortexAnnotEl.className = 'cortex-annot';
  cortexAnnotEl.setAttribute('aria-hidden', 'true');
  cortexAnnotEl.innerHTML =
    '<span class="cortex-annot-dot"></span>'
    + '<span class="cortex-annot-line"></span>'
    + '<span class="cortex-annot-label">'
    + '<span class="cortex-annot-name">Motor cortex</span>'
    + '<span class="cortex-annot-sub"><span class="cortex-annot-time" id="cortex-time">0&thinsp;ms</span> relative to keystroke</span>'
    + '</span>';
  cortexAnnotEl.style.opacity = '0';
  stickyEl.appendChild(cortexAnnotEl);
}
// The peri-keystroke time now lives inside the "Motor cortex" callout.
const cortexTimeEl = document.getElementById('cortex-time');

// --- State ------------------------------------------------------------------
let manifest = null;
let zoomDur = 5;        // seconds; overwritten from the manifest
let actDur = 3;
let fps = 30;
// Crossfade the activity (close-up) clip in over a NARROW window right at the top
// of the scrub (scrubP ~1). The zoom clip's brain only matches the activity pose
// in its final frames; fading earlier (the old 0.85->0.985) blended a smaller,
// receded zoom-brain with the fixed close-up -> a visible "double brain" ghost.
let handoff = { closeStart: 0.965, closeEnd: 0.995 };
let motorCortex = null; // { x, y } normalised 0..1 (fallback / handoff pose)
let motorCortexFrames = null; // [{ x, y } | null] in lock-step with activity_rock.mp4
let actFrames = 0;      // number of baked rock frames (for index sampling)
let actTimes = null;    // { tWin0, tWinSpan, tLoopStart, tLoopEnd }
let lastSeek = -1;
let heroVisible = true;
// Intrinsic aspect of the baked clips (manifest.output). Callout positions are
// normalised in THIS space, but the videos are shown with object-fit:cover, so
// we must map normalised coords through the same cover crop -- not just scale by
// the stage size (that drifts the callouts whenever the stage aspect differs).
let videoAspect = 1440 / 900;

// Map a point normalised in the video's intrinsic frame (0..1) to stage pixels,
// accounting for object-fit: cover (the clip is scaled to FILL the stage and the
// overflowing axis is cropped equally on both sides).
function coverMap(nx, ny) {
  const sw = stage.clientWidth, sh = stage.clientHeight;
  const stageAspect = sw / sh;
  let drawW, drawH;
  if (stageAspect > videoAspect) {
    // Stage wider than the clip: clip fills width, top/bottom cropped.
    drawW = sw; drawH = sw / videoAspect;
  } else {
    // Stage taller: clip fills height, left/right cropped.
    drawH = sh; drawW = sh * videoAspect;
  }
  const offX = (sw - drawW) / 2;
  const offY = (sh - drawH) / 2;
  return { x: offX + nx * drawW, y: offY + ny * drawH };
}

// Keep a callout's dot (and thus its leader + label) inside the visible stage on
// phones, where object-fit:cover crops the clip and pushes some normalised points
// off-screen. Generous margins leave room for the leader line + label.
function clampToStage(p) {
  const sw = stage.clientWidth, sh = stage.clientHeight;
  if ((window.innerWidth || document.documentElement.clientWidth) > 640) return p;
  const mx = 92, my = 96;   // room for leader + label
  return {
    x: Math.max(mx, Math.min(sw - mx, p.x)),
    y: Math.max(my, Math.min(sh - 24, p.y)),
  };
}

// --- Wide-view scene callouts (MEG scanner / Brain / QWERTY keyboard) --------
// Built from manifest.sceneCallouts (normalised positions captured at the wide
// MEG view). Same markup/CSS classes as the live hero so styling + the staggered
// `is-in` slide-in match. Shown only while the wide MEG is in frame.
let sceneCallouts = []; // [{ x, y, el }]
function buildSceneCallouts(list) {
  if (!stickyEl || !Array.isArray(list)) return;
  for (const c of list) {
    const el = document.createElement('div');
    el.className = `scene-callout dir-${c.dir || 'left'}`;
    el.setAttribute('aria-hidden', 'true');
    el.innerHTML = '<span class="sc-dot"></span><span class="sc-line"></span>'
      + `<span class="sc-label">${c.label || ''}</span>`;
    el.style.transitionDelay = `${c.delay || 0}s`;
    stickyEl.appendChild(el);
    sceneCallouts.push({ x: c.x, y: c.y, el });
  }
}
function updateSceneCallouts(reveal) {
  if (!sceneCallouts.length || !stage) return;
  const show = reveal > 0.4;
  for (const c of sceneCallouts) {
    if (show) {
      const p = clampToStage(coverMap(c.x, c.y));
      c.el.style.left = `${p.x}px`;
      c.el.style.top = `${p.y}px`;
    }
    c.el.classList.toggle('is-in', show);
  }
}

function seekZoom(scrubP) {
  if (!Number.isFinite(zoomDur) || zoomDur <= 0) return;
  if (zoomVideo.readyState < 1) return; // no metadata yet -> can't seek
  // Clamp a hair inside the clip so the last frame still resolves.
  const t = clamp(scrubP, 0, 1) * (zoomDur - 1 / (fps * 2));
  lastSeek = t;
  // Compare against the video's ACTUAL time (not a remembered target) so a seek
  // that didn't land yet (e.g. before the clip was seekable) self-heals next
  // frame, while landed seeks within ~half a frame are left alone.
  if (Math.abs(t - zoomVideo.currentTime) < 1 / (fps * 2)) return;
  try { zoomVideo.currentTime = t; } catch {}
}

function updateTimer() {
  // The readout is now part of the "Motor cortex" callout; only update it while
  // that callout is visible.
  if (!cortexTimeEl || !actTimes || !cortexAnnotEl || cortexAnnotEl.style.opacity === '0') return;
  const { tLoopStart, tLoopEnd } = actTimes;
  if (tLoopStart == null || !actDur) return;
  const frac = actDur > 0 ? ((activityVideo.currentTime % actDur) / actDur) : 0;
  const tMs = tLoopStart + (tLoopEnd - tLoopStart) * frac;
  const sign = tMs > 0.5 ? '+' : tMs < -0.5 ? '\u2212' : '';
  cortexTimeEl.innerHTML = `${sign}${Math.abs(Math.round(tMs))}\u200ams`;
}

// Sample the per-frame motor-cortex position for the activity clip's CURRENT
// time. The brain rocks in the baked pixels, so the vertex moves frame-to-frame;
// manifest_rock.json stores one normalised { x, y } per baked frame. We pick the
// nearest baked frame from the looping clock and cover-map it to stage pixels.
function currentMotorCortex() {
  if (motorCortexFrames && actFrames > 0 && actDur > 0) {
    const frac = (activityVideo.currentTime % actDur) / actDur; // 0..1 around the loop
    // Floor + clamp (NOT round + wrap): rounding near the loop seam (frac -> 1)
    // wrapped the index back to frame 0, whose baked motor-cortex value is an
    // outlier (captured before the rock settled), so the dot flicked far to the
    // right over the keyboard for one frame. Clamping keeps it on the in-loop
    // frames the whole way through.
    let i = Math.floor(frac * actFrames);
    if (i < 0) i = 0;
    if (i >= actFrames) i = actFrames - 1;
    const mc = motorCortexFrames[i];
    if (mc) return mc;
  }
  return motorCortex; // fallback (e.g. before the track loads)
}

function updateCortexAnnot(closeup) {
  if (!cortexAnnotEl || !stage) return;
  if (closeup <= 0.02) { if (cortexAnnotEl.style.opacity !== '0') cortexAnnotEl.style.opacity = '0'; return; }
  const mc = currentMotorCortex();
  if (!mc) return;
  const p = clampToStage(coverMap(mc.x, mc.y));
  cortexAnnotEl.style.left = `${p.x}px`;
  cortexAnnotEl.style.top = `${p.y}px`;
  cortexAnnotEl.style.opacity = String(closeup);
}

function updateHeroTexture() {
  if (stage) stage.style.setProperty('--hero-tex-y', `${-(window.scrollY || window.pageYOffset || 0)}px`);
}

// PHONE only: glide the wide-MEG zoom clip's framing so it matches the activated
// brain (#hero-activity) at the close-up and a centred wide scene when pulled out.
// frac: scrubP (1 = close brain, 0 = wide MEG). Must mirror the #hero-activity
// values in demo-v2.css (object-position 35%, scale 1.45, translateY -8%).
const ACT_OBJ_X = 35, ACT_SCALE = 1.45, ACT_TY = -8;   // close-up (frac = 1)
const WIDE_OBJ_X = 50, WIDE_SCALE = 1, WIDE_TY = 0;     // wide MEG (frac = 0)
function applyPhoneZoomFraming(frac) {
  const f = clamp(frac, 0, 1);
  const ox = WIDE_OBJ_X + (ACT_OBJ_X - WIDE_OBJ_X) * f;
  const sc = WIDE_SCALE + (ACT_SCALE - WIDE_SCALE) * f;
  const ty = WIDE_TY + (ACT_TY - WIDE_TY) * f;
  zoomVideo.style.setProperty('object-position', ox.toFixed(2) + '% center', 'important');
  zoomVideo.style.setProperty('transform', `scale(${sc.toFixed(3)}) translateY(${ty.toFixed(2)}%)`, 'important');
  zoomVideo.style.setProperty('transform-origin', '50% 50%', 'important');
}

// Apply scroll progress to the videos + overlays (mirrors hero-v2 fades).
function apply(p) {
  const scrubP = startOnBrain ? brainStartScrub(p) : p;
  seekZoom(scrubP);

  const closeup = smoothstep(scrubP, handoff.closeStart, handoff.closeEnd);
  zoomVideo.style.opacity = '1';
  activityVideo.style.opacity = String(closeup);
  // PHONE: make the zoom clip's framing converge to the activated brain's framing
  // at the close-up so the two clips align perfectly at the handoff (no jump / no
  // size mismatch), then glide back to a centred, unscaled wide MEG scene.
  //  - #hero-activity (CSS) is fixed at: object-position 35%, scale 1.45, ty -8%.
  //  - #hero-zoom glides along scrubP: 1 = match activity, 0 = wide MEG (50%,
  //    scale 1, ty 0). Driving it by scrubP (not `closeup`) keeps it continuous.
  if (isPhone) applyPhoneZoomFraming(scrubP);
  else if (zoomVideo.style.transform) {
    // Left over from a phone viewport before a resize -> clear so desktop uses
    // the plain centred, unscaled framing.
    zoomVideo.style.removeProperty('object-position');
    zoomVideo.style.removeProperty('transform');
    zoomVideo.style.removeProperty('transform-origin');
  }
  // Save work when the activation layer is fully hidden.
  if (!prefersReducedMotion) {
    if (closeup > 0.01 && activityVideo.paused && heroVisible) activityVideo.play().catch(() => {});
    else if (closeup <= 0.01 && !activityVideo.paused) activityVideo.pause();
  }

  // Title block (overlay) fades only at the very end, as the hero scrolls away.
  const heroOut = smoothstep(p, 0.95, 1.0);
  if (overlayEl) overlayEl.style.opacity = String(1 - heroOut);
  // The "Scroll to read more" pill sits at the bottom of the pinned hero, so it
  // collides with the next section's text long before the hero is fully scrolled
  // out. Fade it early -- once the reader has clearly started scrolling (and the
  // wide MEG view is coming in) it has done its job.
  if (scrollHintEl) scrollHintEl.style.opacity = String(1 - smoothstep(p, 0.35, 0.55));
  // Keyboard / decoded sentence / motor-cortex callout belong to the close-up
  // brain: the moment the camera starts pulling back toward the MEG (scrubP drops
  // below the close-up plateau) they should fade out together with the brain.
  // Tie them to `closeup` (1 = locked on the cortex, 0 = receding to MEG) so they
  // disappear as soon as the brain begins moving, not later at MORPH_START.
  const ov = closeup;
  if (heroAnimEl) heroAnimEl.style.opacity = String(ov);
  if (heroKbdEl) heroKbdEl.style.opacity = String(ov);
  updateCortexAnnot(ov);

  // Wide-view callouts: appear as the camera reaches the wide MEG (mirrors
  // hero-v2 sceneReveal). Brain-first reveals near the end of the scroll.
  const sceneReveal = startOnBrain
    ? smoothstep(p, 0.82, 0.93)
    : 1 - smoothstep(p, 0.05, 0.18);
  updateSceneCallouts(sceneReveal);
}

// --- Render loop (just reads scroll + nudges the videos -- no GPU work) -----
let _running = true;
function frame() {
  requestAnimationFrame(frame);
  if (!heroVisible || document.hidden) return;
  apply(heroScrollProgress());
  updateTimer();
}

// --- Load manifest, then go -------------------------------------------------
async function boot() {
  try {
    const r = await fetch(MANIFEST_URL);
    if (r.ok) {
      manifest = await r.json();
      fps = manifest.fps || fps;
      zoomDur = (manifest.zoom && manifest.zoom.duration) || zoomDur;
      actDur = (manifest.activity && manifest.activity.duration) || actDur;
      handoff = manifest.handoff || handoff;
      motorCortex = manifest.motorCortex || null;
      // ROCK: per-frame motor-cortex track + frame count, so the callout follows
      // the brain as it turns in the baked pixels.
      motorCortexFrames = Array.isArray(manifest.motorCortexFrames) ? manifest.motorCortexFrames : null;
      actFrames = (manifest.activity && manifest.activity.frames) || (motorCortexFrames ? motorCortexFrames.length : 0);
      // Frame 0's baked motor-cortex value can be an outlier (captured before
      // the rock settled) -- it sits far right, near the keyboard. If it
      // deviates sharply from frame 1, snap it to frame 1 so the callout never
      // jumps there at the loop start/seam.
      if (motorCortexFrames && motorCortexFrames.length > 1) {
        const a = motorCortexFrames[0], b = motorCortexFrames[1];
        if (a && b && Math.hypot(a.x - b.x, a.y - b.y) > 0.05) {
          motorCortexFrames[0] = { x: b.x, y: b.y };
        }
      }
      actTimes = manifest.activityTimes || null;
      // Intrinsic clip aspect, for the object-fit:cover callout mapping.
      const out = manifest.output || manifest.capture;
      if (out && out.width && out.height) videoAspect = out.width / out.height;
      buildSceneCallouts(manifest.sceneCallouts || []);
    }
  } catch (e) {
    console.error('hero-video: could not load manifest', e);
  }

  // Hide the loader once the scrubbed clip can paint a frame. Mobile browsers can
  // be stingy about firing `loadeddata` for a video that is never play()ed, so we
  // also listen to `loadedmetadata` / `canplay` and keep a safety timeout.
  let loaderHidden = false;
  const hideLoader = () => {
    if (loaderHidden) return;
    loaderHidden = true;
    if (loadingEl) loadingEl.style.display = 'none';
  };
  if (zoomVideo.readyState >= 2) hideLoader();
  for (const ev of ['loadedmetadata', 'loadeddata', 'canplay']) {
    zoomVideo.addEventListener(ev, hideLoader, { once: true });
  }
  setTimeout(hideLoader, 4000);
  zoomVideo.addEventListener('error', () => {
    if (loadingEl) loadingEl.textContent = 'Could not load the hero video.';
  }, { once: true });

  // The zoom clip is never play()ed -- it is scrubbed frame-by-frame via
  // currentTime on scroll. TOUCH browsers (Android Chrome / iOS Safari) refuse to
  // decode + paint a paused, seek-only <video> until it has actually played once,
  // so the scanner stayed blank/black while scrolling. Prime it with a one-shot
  // muted play() -> pause() on the first touch, then snap back to the scroll
  // position. We restore currentTime so the brief play never leaves a stale
  // frame, and we DON'T do this on desktop (mouse), where seek-only scrubbing
  // already works -- a stray play() there would fight the scroll seeking.
  let zoomPrimed = false;
  function primeZoom() {
    if (zoomPrimed) return;
    zoomPrimed = true;
    const at = zoomVideo.currentTime;
    const settle = () => {
      try { zoomVideo.pause(); } catch {}
      try { zoomVideo.currentTime = at; } catch {}
      lastSeek = -1;            // force the render loop to re-seek to the scroll pos
    };
    let p;
    try { p = zoomVideo.play(); } catch { zoomPrimed = false; return; }
    if (p && typeof p.then === 'function') p.then(settle).catch(() => { zoomPrimed = false; });
    else settle();
  }
  window.addEventListener('touchstart', primeZoom, { passive: true, once: true });

  // The activity loop runs on its own clock (unless reduced motion).
  if (!prefersReducedMotion) {
    activityVideo.autoplay = true;
    activityVideo.play().catch(() => {});
  }

  updateHeroTexture();
  apply(heroScrollProgress());
  requestAnimationFrame(frame);
}

// --- Wiring -----------------------------------------------------------------
window.addEventListener('scroll', () => { updateHeroTexture(); apply(heroScrollProgress()); }, { passive: true });
window.addEventListener('resize', () => { lastSeek = -1; apply(heroScrollProgress()); }, { passive: true });

if (heroEl && 'IntersectionObserver' in window) {
  new IntersectionObserver((entries) => {
    heroVisible = entries.some((e) => e.isIntersecting);
    if (!heroVisible && !activityVideo.paused) activityVideo.pause();
  }, { threshold: 0 }).observe(heroEl);
}
document.addEventListener('visibilitychange', () => {
  if (document.hidden && !activityVideo.paused) activityVideo.pause();
});

// Custom smooth scroll for the "scroll to read more" button. On the VIDEO page
// this also scrubs zoom.mp4 (currentTime seek) every frame, so an over-slow
// programmatic scroll spreads the pull-back across hundreds of tiny seeks and
// stutters. Keep it only mildly slower than default and drive the video seek
// synchronously in the same step so the frame matches the scroll position.
let _scrollAnim = null;
function cancelScrollAnim() {
  if (_scrollAnim) { cancelAnimationFrame(_scrollAnim); _scrollAnim = null; }
  document.documentElement.style.scrollBehavior = '';
}
function smoothScrollTo(top) {
  const startY = window.scrollY || window.pageYOffset || 0;
  const dist = top - startY;
  cancelScrollAnim();
  if (Math.abs(dist) < 2) { window.scrollTo(0, top); return; }
  const vh = window.innerHeight || 800;
  const speed = (1.1 * vh) / 5700 / 1.5;      // px per ms (~1.5x slower than default)
  const duration = Math.min(20000, Math.max(1100, Math.abs(dist) / speed));
  document.documentElement.style.scrollBehavior = 'auto'; // don't double-animate
  const t0 = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 3); // easeOutCubic
  function step(now) {
    const t = Math.min(1, (now - t0) / duration);
    window.scrollTo(0, startY + dist * ease(t));
    apply(heroScrollProgress());              // keep the video frame in lock-step
    if (t < 1) { _scrollAnim = requestAnimationFrame(step); }
    else { _scrollAnim = null; document.documentElement.style.scrollBehavior = ''; }
  }
  _scrollAnim = requestAnimationFrame(step);
}
window.addEventListener('wheel', cancelScrollAnim, { passive: true });
window.addEventListener('touchstart', cancelScrollAnim, { passive: true });
window.addEventListener('keydown', (e) => {
  if (['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End', ' '].includes(e.key)) cancelScrollAnim();
});

// --- "Next" button: advance one beat per click (mirrors hero-v2.js) --------
function goNext() {
  const range = (heroEl ? heroEl.offsetHeight : window.innerHeight * 2) - window.innerHeight;
  const p = heroScrollProgress();
  if (p < 0.45) { smoothScrollTo(0.92 * range); return; }
  const first = document.querySelector('.section');
  if (first) { smoothScrollTo(first.getBoundingClientRect().top + (window.scrollY || window.pageYOffset || 0)); }
  else smoothScrollTo(range + window.innerHeight);
}
if (scrollHintEl) {
  scrollHintEl.addEventListener('click', goNext);
  if (scrollHintEl.tagName !== 'BUTTON') {
    scrollHintEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); goNext(); }
    });
  }
}

boot();

// --- Section dot scroller (scrollspy) --------------------------------------
// Ported verbatim from hero-v2.js: lights the dot for the current section.
(function initDotNav() {
  const nav = document.querySelector('.dot-nav');
  if (!nav) return;
  const dots = Array.from(nav.querySelectorAll('.dot'));
  const entries = dots
    .map((dot) => ({ dot, sec: document.getElementById(dot.getAttribute('href').slice(1)) }))
    .filter((e) => e.sec);
  if (!entries.length) return;

  let activeDot = null;
  function setActive(dot) {
    if (dot === activeDot) return;
    activeDot = dot;
    for (const d of dots) d.classList.toggle('is-active', d === dot);
  }
  function updateActive() {
    const line = window.innerHeight * 0.38;
    let current = entries[0];
    for (const e of entries) {
      if (e.sec.getBoundingClientRect().top <= line) current = e;
    }
    setActive(current.dot);
  }
  window.addEventListener('scroll', updateActive, { passive: true });
  window.addEventListener('resize', updateActive, { passive: true });
  window.addEventListener('hashchange', updateActive);
  updateActive();
})();
