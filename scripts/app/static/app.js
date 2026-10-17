// HooperAI test bench – talks to scripts/app/server.py
const $ = (id) => document.getElementById(id);

const state = {
  videos: [],
  video: null,          // selected video name
  jobId: null,
  result: null,         // {fps, total_frames, width, height, shots, ...}
  boxesByFrame: null,   // Map frame → [[cls, x1, y1, x2, y2], ...]
  currentShot: null,    // index into result.shots
};

const BOX_COLORS = { 0: '#e85d26', 1: '#4caf7d', 2: '#6aa9ff' };   // ball, basket, player

// ── API ─────────────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return res.json();
}

function toast(msg, isError = false) {
  const t = $('toast');
  t.textContent = msg;
  t.className = 'toast' + (isError ? ' error' : '');
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, isError ? 6000 : 2500);
}

const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const pct = (x) => (x == null ? '–' : `${Math.round(x * 100)}%`);

// ── startup ─────────────────────────────────────────────────────────────────
async function init() {
  try {
    const st = await api('/api/status');
    const dev = $('device');
    dev.textContent = st.gpu ? `GPU · ${st.device}` : 'CPU (slow detection)';
    dev.className = 'badge ' + (st.gpu ? 'gpu' : 'cpu');
    const sel = $('predictor');
    sel.innerHTML = '';
    for (const p of st.predictors) {
      const o = document.createElement('option');
      o.value = p.config;
      const acc = p.unseen_video_accuracy != null ? ` · ${pct(p.unseen_video_accuracy)} unseen video` : '';
      o.textContent = p.label + acc;
      if (p.config === st.default_predictor) o.selected = true;
      sel.appendChild(o);
    }
    if (!st.predictors.length) toast('No predictor configs in data/model – run lstm_train.py', true);
    if (!st.yolo_found) toast(`YOLO weights missing: ${st.yolo_model}`, true);
  } catch (e) {
    toast(`Server not reachable: ${e.message}`, true);
  }
  await loadVideos();
  await loadStats();
}

async function loadVideos() {
  state.videos = await api('/api/videos');
  renderVideos();
}

function renderVideos() {
  const q = $('search').value.toLowerCase();
  const ul = $('videos');
  ul.innerHTML = '';
  for (const v of state.videos.filter((v) => v.name.toLowerCase().includes(q))) {
    const li = document.createElement('li');
    li.className = v.name === state.video ? 'active' : '';
    li.innerHTML = `<div class="name"></div><div class="meta">${v.size_mb} MB · ${new Date(v.mtime * 1000).toLocaleDateString()}</div>`;
    li.querySelector('.name').textContent = v.name;
    li.onclick = () => selectVideo(v.name);
    ul.appendChild(li);
  }
  if (!ul.children.length) ul.innerHTML = '<li class="meta">No videos in data/raw/videos</li>';
}

async function loadStats() {
  try {
    const s = await api('/api/stats');
    $('stLabeled').textContent = s.labeled_shots;
    $('stVideos').textContent = s.videos;
    $('stFg').textContent = s.labeled_shots ? pct(s.fg_pct) : '–';
    $('stAcc').textContent = s.labeled_shots ? pct(s.model_accuracy) : '–';
  } catch (_) { /* stats are optional */ }
}

// ── video selection / playback ──────────────────────────────────────────────
function selectVideo(name) {
  state.video = name;
  state.jobId = null;
  state.result = null;
  state.boxesByFrame = null;
  state.currentShot = null;
  const v = $('video');
  v.src = `/api/videos/${encodeURIComponent(name)}/stream`;
  v.controls = true;
  $('player').classList.remove('empty');
  $('analyze').disabled = false;
  $('banner').hidden = true;
  renderVideos();
  renderShots();
  renderTimeline();
  renderSummary();
}

// ── analysis ────────────────────────────────────────────────────────────────
async function analyze() {
  if (!state.video) return;
  $('analyze').disabled = true;
  $('progress').hidden = false;
  setProgress(0, 'Queued…');
  try {
    const { job_id } = await api('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video: state.video,
        config: $('predictor').value,
        yolo_conf: parseFloat($('conf').value),
      }),
    });
    state.jobId = job_id;
    await pollJob(job_id);
  } catch (e) {
    toast(`Analysis failed: ${e.message}`, true);
  } finally {
    $('analyze').disabled = false;
  }
}

function setProgress(frac, text) {
  $('progressFill').style.width = `${Math.round(frac * 100)}%`;
  $('progressText').textContent = text;
}

async function pollJob(jobId) {
  const started = Date.now();
  for (;;) {
    const job = await api(`/api/jobs/${jobId}`);
    if (jobId !== state.jobId) return;          // user switched video
    if (job.status === 'done') {
      setProgress(1, `Done in ${((Date.now() - started) / 1000).toFixed(0)} s`);
      state.result = job.result;
      try {
        const det = await api(`/api/detections/${jobId}`);
        state.boxesByFrame = new Map();
        for (const [f, cls, x1, y1, x2, y2] of det.boxes) {
          if (!state.boxesByFrame.has(f)) state.boxesByFrame.set(f, []);
          state.boxesByFrame.get(f).push([cls, x1, y1, x2, y2]);
        }
      } catch (_) { state.boxesByFrame = null; }
      renderShots();
      renderTimeline();
      renderSummary();
      setTimeout(() => { $('progress').hidden = true; }, 2500);
      return;
    }
    if (job.status === 'error') throw new Error(job.error);
    const label = job.status === 'analyzing'
      ? `Analyzing… ${pct(job.progress)}` : `${job.status[0].toUpperCase()}${job.status.slice(1)}…`;
    setProgress(job.progress || 0, label);
    await new Promise((r) => setTimeout(r, 700));
  }
}

// ── shots panel ─────────────────────────────────────────────────────────────
function shotTime(shot) { return (shot.start_frame - 1) / state.result.fps; }

function renderShots() {
  const ol = $('shots');
  ol.innerHTML = '';
  const shots = state.result?.shots || [];
  $('shotCount').textContent = state.result ? `${shots.length} detected` : '';
  if (!state.result) { ol.innerHTML = '<li class="placeholder">No analysis yet.</li>'; return; }
  if (!shots.length) { ol.innerHTML = '<li class="placeholder">No shots detected in this video.</li>'; return; }

  shots.forEach((s, i) => {
    const made = s.prediction === 1;
    const manual = s.prediction == null;
    const notShot = s.label === -1;
    const li = document.createElement('li');
    li.className = 'shot' + (i === state.currentShot ? ' current' : '');
    const verdict = s.label == null || manual ? ''
      : notShot ? '<span class="verdict">not a shot</span>'
      : s.label === s.prediction ? '<span class="verdict ok">✓ model right</span>'
      : '<span class="verdict bad">✗ model wrong</span>';
    const rim = s.rim_offset == null ? '–' : (s.rim_offset >= 2.99 ? 'no rim pass' : s.rim_offset.toFixed(2));
    li.innerHTML = `
      <div class="shot-top">
        <span class="shot-num">#${s.shot_number}</span>
        <span class="shot-time">${fmtTime(shotTime(s))}</span>
        <span class="chip ${manual ? '' : made ? 'made' : 'missed'}">${manual ? 'ADDED' : made ? 'MADE' : 'MISSED'}</span>
      </div>
      <div class="shot-meta">
        <span>score <b>${pct(s.probability)}</b></span>
        <span>conf <b>${s.confidence ? s.confidence.toLowerCase() : 'added by you'}</b></span>
        <span>rim offset <b>${rim}</b></span>
        <span><b>${s.n_frames}</b> frames</span>
      </div>
      <div class="probbar"><div style="width:${(s.probability || 0) * 100}%;background:${made ? 'var(--green)' : 'var(--orange)'}"></div></div>
      <div class="label-row">
        <button class="btn small ${s.label === 1 ? 'on-made' : ''}" data-label="1">Made</button>
        <button class="btn small ${s.label === 0 ? 'on-missed' : ''}" data-label="0">Missed</button>
        ${manual ? '' : `<button class="btn small ${notShot ? 'on-missed' : ''}" data-label="-1">Not a shot</button>`}
        ${verdict}
      </div>`;
    li.onclick = (e) => {
      const btn = e.target.closest('button[data-label]');
      if (btn) { e.stopPropagation(); label(i, parseInt(btn.dataset.label, 10)); return; }
      jumpToShot(i);
    };
    ol.appendChild(li);
  });
}

function jumpToShot(i) {
  const shots = state.result?.shots;
  if (!shots?.length) return;
  state.currentShot = Math.max(0, Math.min(i, shots.length - 1));
  const v = $('video');
  v.currentTime = Math.max(0, shotTime(shots[state.currentShot]) - 1.0);
  v.play().catch(() => {});
  renderShots();
  renderTimeline();
  document.querySelector('.shot.current')?.scrollIntoView({ block: 'nearest' });
}

async function label(i, value) {
  const shot = state.result.shots[i];
  try {
    const r = await api('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ job_id: state.jobId, shot_number: shot.shot_number, label: value }),
    });
    shot.label = value;
    toast(`Shot #${shot.shot_number} labeled ${value === 1 ? 'MADE' : value === 0 ? 'MISSED' : 'NOT A SHOT'} · ${r.labeled_shots} labeled total`);
    renderShots();
    renderTimeline();
    renderSummary();
    loadStats();
  } catch (e) {
    toast(`Could not save label: ${e.message}`, true);
  }
}

// ── timeline + summary ──────────────────────────────────────────────────────
function renderTimeline() {
  const tl = $('timeline');
  tl.innerHTML = '<div class="head" id="head"></div>';
  if (!state.result) return;
  const n = state.result.total_frames || 1;
  state.result.shots.forEach((s, i) => {
    const seg = document.createElement('div');
    seg.className = `seg ${(s.label ?? s.prediction) === 1 ? 'made' : 'missed'}`
      + (s.label != null && s.label !== s.prediction ? ' wrong' : '')
      + (i === state.currentShot ? ' current' : '');
    seg.style.left = `${(s.start_frame / n) * 100}%`;
    seg.style.width = `${((s.end_frame - s.start_frame) / n) * 100}%`;
    seg.title = `#${s.shot_number} ${s.prediction ? 'made' : 'missed'} (${pct(s.probability)})`;
    seg.onclick = (e) => { e.stopPropagation(); jumpToShot(i); };
    tl.appendChild(seg);
  });
}

function renderSummary() {
  const el = $('summary');
  if (!state.result) { el.innerHTML = ''; return; }
  const shots = state.result.shots;
  const made = shots.filter((s) => s.prediction === 1).length;
  const labeled = shots.filter((s) => s.label != null);
  const right = labeled.filter((s) => s.label === s.prediction).length;
  const card = (v, t) => `<div class="card"><b>${v}</b><span>${t}</span></div>`;
  el.innerHTML = card(shots.length, 'shots detected')
    + card(shots.length ? pct(made / shots.length) : '–', `predicted FG% (${made} made)`)
    + card(`${labeled.length}/${shots.length}`, 'labeled in this video')
    + card(labeled.length ? pct(right / labeled.length) : '–', 'model correct on labeled');
}

// ── overlay drawing ─────────────────────────────────────────────────────────
function drawOverlay() {
  const v = $('video');
  const c = $('overlay');
  const r = state.result;
  const ctx = c.getContext('2d');
  const W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * devicePixelRatio) { c.width = W * devicePixelRatio; c.height = H * devicePixelRatio; }
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!r) return;

  const frame = Math.floor(v.currentTime * r.fps) + 1;   // detections are 1-based
  // object-fit: contain → letterbox offsets
  const s = Math.min(W / r.width, H / r.height);
  const ox = (W - r.width * s) / 2, oy = (H - r.height * s) / 2;

  if ($('showBoxes').checked && state.boxesByFrame) {
    for (const [cls, x1, y1, x2, y2] of state.boxesByFrame.get(frame) || []) {
      ctx.strokeStyle = BOX_COLORS[cls] || '#fff';
      ctx.lineWidth = cls === 0 ? 2.5 : 1.5;
      ctx.strokeRect(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s);
    }
  }

  // banner: tracking a shot / last result
  const b = $('banner');
  const live = r.shots.find((sh) => frame >= sh.start_frame && frame <= sh.end_frame);
  const last = [...r.shots].reverse().find((sh) => sh.end_frame < frame && frame - sh.end_frame < r.fps * 3);
  if (live) {
    b.hidden = false; b.className = 'banner tracking'; b.textContent = `TRACKING SHOT #${live.shot_number}`;
  } else if (last) {
    const made = last.prediction === 1;
    b.hidden = false; b.className = `banner ${made ? 'made' : 'missed'}`;
    b.textContent = `#${last.shot_number} ${made ? 'MADE' : 'MISSED'} · ${pct(last.probability)}`;
  } else {
    b.hidden = true;
  }

  const head = $('head');
  if (head) head.style.left = `${(frame / (r.total_frames || 1)) * 100}%`;
}

function loop() { drawOverlay(); requestAnimationFrame(loop); }

// ── events ──────────────────────────────────────────────────────────────────
$('analyze').onclick = analyze;
$('search').oninput = renderVideos;
$('conf').oninput = () => { $('confValue').textContent = parseFloat($('conf').value).toFixed(2); };
$('timeline').onclick = (e) => {
  if (!state.result) return;
  const rect = e.currentTarget.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  $('video').currentTime = (frac * state.result.total_frames) / state.result.fps;
};
$('upload').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append('file', file);
  toast(`Uploading ${file.name}…`);
  try {
    const r = await api('/api/videos', { method: 'POST', body: fd });
    await loadVideos();
    selectVideo(r.name);
  } catch (err) {
    toast(`Upload failed: ${err.message}`, true);
  }
  e.target.value = '';
};
document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  const v = $('video');
  if (e.code === 'Space') { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === 'j' || e.key === 'J') jumpToShot((state.currentShot ?? 1) - 1);
  else if (e.key === 'k' || e.key === 'K') jumpToShot((state.currentShot ?? -1) + 1);
  else if ((e.key === 'm' || e.key === 'M') && state.currentShot != null) label(state.currentShot, 1);
  else if ((e.key === 'x' || e.key === 'X') && state.currentShot != null) label(state.currentShot, 0);
});

init();
loop();
