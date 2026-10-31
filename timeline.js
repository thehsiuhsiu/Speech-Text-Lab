// 錄音裁切軌道：像剪片軟體一樣在波形上切段，勾選要辨識的片段
const SR = 16000;
const RULER = 18;   // 上方時間尺高度
const WAVE = 84;    // 波形高度
const MAX_W = 30000; // 畫布最大寬度（px）

export function createTimeline({ root, player, fmtTime, fmtDur, onChange }) {
  const $ = (sel) => root.querySelector(sel);
  const scroll = $('.tl-scroll');
  const inner = $('.tl-inner');
  const canvas = $('canvas');
  const ctx = canvas.getContext('2d');
  const clipsBox = $('.tl-clips');
  const cutsBox = $('.tl-cuts');
  const playhead = $('.tl-playhead');
  const list = $('.tl-list');
  const summary = $('.tl-summary');
  const zoomInput = $('.tl-zoom');

  let audio = null;
  let duration = 0;
  let clips = [];          // { start, end, sel }
  let zoom = 1;
  let width = 0;           // 內容寬度（px）
  let peaks = null;        // Float32Array [min,max,...]
  let playUntil = null;    // 播單一片段時，播到哪裡停

  const pxPerSec = () => width / Math.max(duration, 0.001);
  const t2x = (t) => t * pxPerSec();
  const x2t = (x) => Math.min(duration, Math.max(0, x / pxPerSec()));

  function setAudio(data) {
    audio = data;
    duration = data.length / SR;
    clips = [{ start: 0, end: duration, sel: true }];
    zoom = 1;
    zoomInput.value = 1;
    root.hidden = false;
    layout();
    changed();
  }

  // ---------- 版面與繪圖 ----------
  function layout(keepTime = null) {
    if (!audio) return;
    const base = scroll.clientWidth || 800;
    width = Math.min(MAX_W, Math.round(base * zoom));
    inner.style.width = width + 'px';
    const dpr = Math.min(window.devicePixelRatio || 1, width * 2 > MAX_W ? 1 : 2);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round((RULER + WAVE) * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = (RULER + WAVE) + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    computePeaks();
    draw();
    renderClips();
    updatePlayhead();
    if (keepTime != null) scroll.scrollLeft = Math.max(0, t2x(keepTime) - scroll.clientWidth / 2);
  }

  function computePeaks() {
    const n = audio.length;
    const W = Math.max(1, Math.floor(width));
    peaks = new Float32Array(W * 2);
    const per = n / W;
    for (let x = 0; x < W; x++) {
      const a = Math.floor(x * per), b = Math.min(n, Math.floor((x + 1) * per));
      let mn = 0, mx = 0;
      const step = Math.max(1, Math.floor((b - a) / 400)); // 每像素最多取 400 點，夠快
      for (let i = a; i < b; i += step) { const v = audio[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      peaks[x * 2] = mn; peaks[x * 2 + 1] = mx;
    }
    // 自動放大音量太小的錄音
    let top = 0;
    for (let i = 0; i < peaks.length; i++) top = Math.max(top, Math.abs(peaks[i]));
    const gain = top > 0 ? Math.min(8, 0.95 / top) : 1;
    for (let i = 0; i < peaks.length; i++) peaks[i] *= gain;
  }

  function draw() {
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue('--accent').trim() || '#5d6b3a';
    const soft = css.getPropertyValue('--accent-soft').trim() || '#eef0e2';
    ctx.clearRect(0, 0, width, RULER + WAVE);
    // 片段底色
    for (const c of clips) {
      ctx.fillStyle = c.sel ? soft : '#ececec';
      ctx.fillRect(t2x(c.start), RULER, t2x(c.end) - t2x(c.start), WAVE);
    }
    // 波形
    const mid = RULER + WAVE / 2;
    const W = peaks.length / 2;
    let ci = 0;
    for (let x = 0; x < W; x++) {
      const t = x2t(x + 0.5);
      while (ci < clips.length - 1 && t >= clips[ci].end) ci++;
      ctx.fillStyle = clips[ci]?.sel ? accent : '#b5b5b5';
      const y1 = mid - peaks[x * 2 + 1] * (WAVE / 2 - 2);
      const y2 = mid - peaks[x * 2] * (WAVE / 2 - 2);
      ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
    }
    // 時間尺
    ctx.fillStyle = '#fafaf7';
    ctx.fillRect(0, 0, width, RULER);
    ctx.fillStyle = '#888';
    ctx.font = '11px ui-monospace, Consolas, monospace';
    const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    const step = steps.find((s) => s * pxPerSec() >= 80) || 3600;
    for (let t = 0; t <= duration; t += step) {
      const x = t2x(t);
      ctx.fillRect(x, RULER - 6, 1, 6);
      ctx.fillText(fmtTime(t).replace(/^00:/, ''), x + 3, 12);
    }
  }

  // 片段方塊、切點把手、清單
  function renderClips() {
    clipsBox.innerHTML = '';
    cutsBox.innerHTML = '';
    clips.forEach((c, i) => {
      const b = document.createElement('div');
      b.className = 'tl-clip' + (c.sel ? ' sel' : '');
      b.style.left = t2x(c.start) + 'px';
      b.style.width = Math.max(2, t2x(c.end) - t2x(c.start)) + 'px';
      b.title = `片段 ${i + 1}：${fmtTime(c.start)}～${fmtTime(c.end)}（點一下切換要不要辨識，按兩下從頭播放）`;
      b.textContent = `${c.sel ? '☑' : '☐'} 片段 ${i + 1}`;
      b.addEventListener('click', () => { c.sel = !c.sel; changed(); });
      b.addEventListener('dblclick', (e) => { e.preventDefault(); playClip(i); });
      clipsBox.appendChild(b);
    });
    for (let i = 0; i < clips.length - 1; i++) {
      const t = clips[i].end;
      const m = document.createElement('div');
      m.className = 'tl-cut';
      m.style.left = t2x(t) + 'px';
      m.title = `切點 ${fmtTime(t)}（拖曳可移動）`;
      const handle = document.createElement('div');
      handle.className = 'tl-cut-handle';
      const del = document.createElement('button');
      del.className = 'tl-cut-del';
      del.textContent = '×';
      del.title = '刪除這個切點';
      del.addEventListener('pointerdown', (e) => e.stopPropagation());
      del.addEventListener('click', (e) => { e.stopPropagation(); removeCut(i); });
      m.append(handle, del);
      dragCut(m, i);
      cutsBox.appendChild(m);
    }
    renderList();
  }

  function renderList() {
    list.innerHTML = '';
    clips.forEach((c, i) => {
      const row = document.createElement('label');
      row.className = 'tl-row' + (c.sel ? ' sel' : '');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = c.sel;
      cb.addEventListener('change', () => { c.sel = cb.checked; changed(); });
      const name = document.createElement('span');
      name.textContent = `片段 ${i + 1}　${fmtTime(c.start)}～${fmtTime(c.end)}（${fmtDur(c.end - c.start)}）`;
      const play = document.createElement('button');
      play.type = 'button';
      play.className = 'secondary small';
      play.textContent = '▶ 播放';
      play.addEventListener('click', (e) => { e.preventDefault(); playClip(i); });
      row.append(cb, name, play);
      list.appendChild(row);
    });
    const sel = clips.filter((c) => c.sel);
    const secs = sel.reduce((a, c) => a + c.end - c.start, 0);
    summary.textContent = sel.length
      ? `已選 ${sel.length}／${clips.length} 個片段，共 ${fmtDur(secs)}，按「開始轉文字」只會辨識這些片段。`
      : '⚠️ 沒有選任何片段。';
  }

  function changed() {
    draw();
    renderClips();
    onChange?.();
  }

  // ---------- 編輯動作 ----------
  function cutAt(t) {
    const i = clips.findIndex((c) => t > c.start + 0.3 && t < c.end - 0.3);
    if (i < 0) return false;
    const c = clips[i];
    clips.splice(i, 1, { start: c.start, end: t, sel: c.sel }, { start: t, end: c.end, sel: c.sel });
    changed();
    return true;
  }
  function removeCut(i) {
    const a = clips[i], b = clips[i + 1];
    clips.splice(i, 2, { start: a.start, end: b.end, sel: a.sel || b.sel });
    changed();
  }
  function clearCuts() {
    const anySel = clips.some((c) => c.sel);
    clips = [{ start: 0, end: duration, sel: anySel || true }];
    changed();
  }
  function setAll(sel) { clips.forEach((c) => { c.sel = sel; }); changed(); }

  // 每 N 秒切一刀，切點挪到前後 15 秒內最安靜的地方
  function autoSplit(sec) {
    const bounds = [0];
    for (let t = sec; t < duration - sec * 0.3; t += sec) bounds.push(quietestNear(t, 15));
    bounds.push(duration);
    clips = [];
    for (let i = 0; i < bounds.length - 1; i++) {
      if (bounds[i + 1] - bounds[i] > 0.5) clips.push({ start: bounds[i], end: bounds[i + 1], sel: true });
    }
    changed();
  }
  function quietestNear(t, range) {
    const win = Math.floor(0.3 * SR);
    let best = t, bv = Infinity;
    for (let x = Math.max(0.5, t - range); x <= Math.min(duration - 0.5, t + range); x += 0.1) {
      const a = Math.floor(x * SR);
      let s = 0;
      for (let i = a; i < a + win && i < audio.length; i += 4) s += audio[i] * audio[i];
      // 越接近原本位置越好（小小加權）
      const v = s * (1 + Math.abs(x - t) / range * 0.3);
      if (v < bv) { bv = v; best = x; }
    }
    return +best.toFixed(2);
  }

  function dragCut(el, i) {
    el.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      el.classList.add('dragging');
      const lo = clips[i].start + 0.3, hi = clips[i + 1].end - 0.3;
      const rect = inner.getBoundingClientRect();
      const move = (ev) => {
        const t = Math.min(hi, Math.max(lo, x2t(ev.clientX - rect.left)));
        clips[i].end = t; clips[i + 1].start = t;
        el.style.left = t2x(t) + 'px';
        el.title = `切點 ${fmtTime(t)}`;
      };
      const up = () => {
        el.removeEventListener('pointermove', move);
        el.removeEventListener('pointerup', up);
        el.classList.remove('dragging');
        changed();
      };
      el.addEventListener('pointermove', move);
      el.addEventListener('pointerup', up);
    });
  }

  // ---------- 播放 ----------
  function playClip(i) {
    const c = clips[i];
    player.currentTime = c.start;
    playUntil = c.end;
    player.play().catch(() => {});
  }
  function updatePlayhead() {
    if (!audio) return;
    const x = t2x(player.currentTime || 0);
    playhead.style.left = x + 'px';
    $('.tl-time').textContent = `播放位置 ${fmtTime(player.currentTime || 0)}／${fmtTime(duration)}`;
    if (!player.paused && (x < scroll.scrollLeft || x > scroll.scrollLeft + scroll.clientWidth - 20)) {
      scroll.scrollLeft = Math.max(0, x - 40);
    }
  }
  let raf = 0;
  const loop = () => {
    updatePlayhead();
    if (playUntil != null && player.currentTime >= playUntil) { player.pause(); playUntil = null; }
    if (!player.paused) raf = requestAnimationFrame(loop);
  };
  player.addEventListener('play', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); });
  player.addEventListener('pause', () => { updatePlayhead(); });
  player.addEventListener('seeked', updatePlayhead);
  player.addEventListener('seeking', () => { playUntil = playUntil != null && player.currentTime < playUntil ? playUntil : null; updatePlayhead(); });

  // 點波形跳到該處
  canvas.addEventListener('click', (e) => {
    const rect = inner.getBoundingClientRect();
    playUntil = null;
    player.currentTime = x2t(e.clientX - rect.left);
    updatePlayhead();
  });

  // ---------- 工具列 ----------
  $('.tl-cut-btn').addEventListener('click', () => {
    if (!cutAt(player.currentTime || 0)) {
      summary.textContent = '⚠️ 這個位置太靠近開頭、結尾或其他切點，請換個位置再切。';
    }
  });
  $('.tl-play-btn').addEventListener('click', () => {
    playUntil = null;
    if (player.paused) player.play().catch(() => {}); else player.pause();
  });
  player.addEventListener('play', () => { $('.tl-play-btn').textContent = '⏸ 暫停'; });
  player.addEventListener('pause', () => { $('.tl-play-btn').textContent = '▶ 播放'; });
  $('.tl-auto').addEventListener('change', (e) => {
    const v = Number(e.target.value);
    if (v) autoSplit(v);
    e.target.value = '';
  });
  $('.tl-clear').addEventListener('click', clearCuts);
  $('.tl-all').addEventListener('click', () => setAll(true));
  $('.tl-none').addEventListener('click', () => setAll(false));
  zoomInput.addEventListener('input', () => {
    const center = x2t(scroll.scrollLeft + scroll.clientWidth / 2);
    zoom = Number(zoomInput.value);
    layout(center);
  });
  // 鍵盤：S 切開、空白鍵播放／暫停（游標不在輸入框時）
  document.addEventListener('keydown', (e) => {
    if (root.hidden || e.target.closest('input, textarea, select, [contenteditable]')) return;
    if (e.key === 's' || e.key === 'S') { e.preventDefault(); $('.tl-cut-btn').click(); }
    else if (e.key === ' ' && document.activeElement?.closest('#timeline-wrap')) { e.preventDefault(); $('.tl-play-btn').click(); }
  });
  let resizeTimer = 0;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => layout(), 200); });

  return {
    setAudio,
    selectedRanges: () => clips.filter((c) => c.sel).map((c) => ({ start: c.start, end: c.end })),
    allSelected: () => clips.every((c) => c.sel),
    hasAudio: () => !!audio,
    autoSplit, cutAt, setAll,
    get clips() { return clips.map((c) => ({ ...c })); },
  };
}
