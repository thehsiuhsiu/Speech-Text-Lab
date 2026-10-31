// 版本號跟著 index.html 的 app.js?v=…，讓背景程式和模型清單也一起更新
const V = new URL(import.meta.url).search;
const { ASR_MODELS, SPEAKER_MODEL } = await import(`./models.js${V}`);
const { createTimeline } = await import(`./timeline.js${V}`);

const $ = (id) => document.getElementById(id);
const SR = 16000;
const COLORS = ['#3b6fb6', '#c0622b', '#3f8f5a', '#8a4fb0', '#b0405f', '#2f8c8c', '#8c7a2f'];
const CACHE_NAME = 'transformers-cache';
const HF = 'https://huggingface.co/';

const state = {
  modelLoaded: false,
  loadedKey: null,
  busy: false,
  audio: null,          // Float32Array 16kHz
  audioName: '',
  duration: 0,
  segments: [],         // {start, end, text, speaker}
  speakers: [],         // {name}
  dirty: false,
  folderFiles: null,    // [{path, file}]
  runStart: 0,
};

let timeline = null; // 裁切軌道（檔尾建立）

// ---------- 背景執行緒 ----------
let worker = null;
function makeWorker() {
  worker = new Worker(new URL(`./worker.js${V}`, import.meta.url), { type: 'module' });
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => {
    console.error(e);
    showError(`背景程式發生錯誤：${e.message || '未知錯誤'}`);
    setBusy(false);
  };
}
makeWorker();

// ---------- 初始化 ----------
function initModelSelect() {
  const sel = $('model-select');
  for (const [key, m] of Object.entries(ASR_MODELS)) {
    const o = document.createElement('option');
    o.value = key; o.textContent = m.label;
    sel.appendChild(o);
  }
  try { sel.value = localStorage.getItem('asr.model') || 'breeze'; } catch { sel.value = 'breeze'; }
  if (!sel.value) sel.value = 'breeze';
  try { $('device-select').value = localStorage.getItem('asr.device') || 'auto'; } catch { /* ignore */ }
  try { $('source-select').value = localStorage.getItem('asr.source') || 'hf'; } catch { /* ignore */ }
  updateModelDesc();
}

async function updateModelDesc() {
  const key = $('model-select').value;
  const m = ASR_MODELS[key];
  let cached = '';
  if ($('source-select').value === 'hf' && 'caches' in self) {
    try {
      const c = await caches.open(CACHE_NAME);
      const hit = await c.match(`${HF}${m.id}/resolve/main/config.json`);
      cached = hit ? '　✅ 這台電腦已下載過，可直接載入。' : '　⬇️ 第一次使用需要下載。';
    } catch { /* ignore */ }
  }
  $('model-desc').textContent = m.desc + cached;
  $('folder-row').hidden = $('source-select').value !== 'folder';
}

async function showEnvInfo() {
  const parts = [];
  let gpu = false;
  if (navigator.gpu) {
    try { gpu = !!(await navigator.gpu.requestAdapter()); } catch { gpu = false; }
  }
  parts.push(gpu ? '顯示卡加速：可用' : '顯示卡加速：不可用（會用 CPU）');
  if (!self.crossOriginIsolated) parts.push('CPU 多執行緒：未啟用');
  if (navigator.storage?.estimate) {
    try {
      const { usage, quota } = await navigator.storage.estimate();
      parts.push(`瀏覽器儲存：已用 ${fmtBytes(usage)}／可用 ${fmtBytes(quota)}`);
    } catch { /* ignore */ }
  }
  $('env-info').textContent = parts.join('・');
}

// ---------- 工具 ----------
function fmtBytes(b) {
  if (!b && b !== 0) return '?';
  if (b > 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b > 1e6) return (b / 1e6).toFixed(0) + ' MB';
  return (b / 1e3).toFixed(0) + ' KB';
}
function fmtTime(t, ms = false) {
  t = Math.max(0, t || 0);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  const base = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  if (!ms) return base;
  return base + ',' + String(Math.round((t % 1) * 1000)).padStart(3, '0');
}
function fmtDur(sec) {
  sec = Math.round(sec);
  if (sec < 60) return `${sec} 秒`;
  const m = Math.floor(sec / 60), s = sec % 60;
  if (m < 60) return `${m} 分 ${s} 秒`;
  return `${Math.floor(m / 60)} 小時 ${m % 60} 分`;
}
function showError(text) {
  const el = $('run-text');
  $('run-progress').hidden = false;
  el.textContent = text;
  el.classList.add('error');
}
function clearError() { $('run-text').classList.remove('error'); $('load-text').classList.remove('error'); }
function setBusy(b) {
  state.busy = b;
  const noClip = timeline && timeline.hasAudio() && timeline.selectedRanges().length === 0;
  $('btn-start').disabled = b || !state.modelLoaded || !state.audio || noClip;
  $('btn-load').disabled = b;
  $('btn-cancel').hidden = !b;
  const rb = $('btn-rediarize');
  if (rb) rb.disabled = b || !state.audio;
}
function updateStartButton() { setBusy(state.busy); }
function speakerName(i) { return state.speakers[i]?.name || `說話者${i + 1}`; }
function ensureSpeakers(n) {
  while (state.speakers.length < n) state.speakers.push({ name: `說話者${state.speakers.length + 1}` });
}

// ---------- 模型載入 ----------
$('model-select').addEventListener('change', () => { try { localStorage.setItem('asr.model', $('model-select').value); } catch { /* */ } updateModelDesc(); });
$('device-select').addEventListener('change', () => { try { localStorage.setItem('asr.device', $('device-select').value); } catch { /* */ } });
$('source-select').addEventListener('change', () => { try { localStorage.setItem('asr.source', $('source-select').value); } catch { /* */ } updateModelDesc(); });
$('btn-pick-folder').addEventListener('click', () => $('folder-input').click());
$('folder-input').addEventListener('change', (e) => {
  const files = [...e.target.files];
  state.folderFiles = files.map((f) => ({ path: f.webkitRelativePath || f.name, file: f }));
  const total = files.reduce((a, f) => a + f.size, 0);
  const key = $('model-select').value;
  const repo = ASR_MODELS[key].id.split('/').pop();
  const has = state.folderFiles.some((f) => f.path.includes(`${repo}/`) && f.path.endsWith('config.json'));
  $('folder-status').textContent = `已選 ${files.length} 個檔案（${fmtBytes(total)}）` +
    (has ? `，找到 ${repo}` : `。⚠️ 沒找到「${repo}」資料夾，請確認選對資料夾或換模型。`) +
    (state.folderFiles.some((f) => f.path.includes(`${SPEAKER_MODEL.id.split('/').pop()}/`)) ? '' : '（沒有分說話者模型，只能用「不分說話者」）');
});

$('btn-load').addEventListener('click', async () => {
  clearError();
  const source = $('source-select').value;
  if (source === 'folder' && !state.folderFiles) {
    $('folder-status').textContent = '請先選擇模型資料夾';
    return;
  }
  if (source === 'hf' && navigator.storage?.persist) {
    try { await navigator.storage.persist(); } catch { /* ignore */ }
  }
  state.modelLoaded = false;
  setBusy(true);
  $('btn-cancel').hidden = true;
  $('load-progress').hidden = false;
  $('load-bar').style.width = '0%';
  $('load-bar').classList.add('indeterminate');
  $('load-text').textContent = '準備中…';
  worker.postMessage({
    type: 'load',
    modelKey: $('model-select').value,
    device: $('device-select').value,
    source,
    files: source === 'folder' ? state.folderFiles : [],
  });
});

// 把瀏覽器快取裡的模型存到資料夾
$('btn-export-models').addEventListener('click', async () => {
  const text = $('load-text');
  $('load-progress').hidden = false;
  if (!window.showDirectoryPicker) {
    text.textContent = '這個瀏覽器不支援存到資料夾，請用 Chrome 或 Edge。';
    return;
  }
  let dir;
  try { dir = await window.showDirectoryPicker({ mode: 'readwrite' }); } catch { return; }
  try {
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();
    const ids = [...Object.values(ASR_MODELS).map((m) => m.id), SPEAKER_MODEL.id];
    let count = 0, bytes = 0;
    const found = new Set();
    for (const req of keys) {
      const url = req.url;
      const id = ids.find((x) => url.startsWith(`${HF}${x}/resolve/`));
      if (!id) continue;
      const filename = decodeURIComponent(url.split('/resolve/')[1].split('/').slice(1).join('/'));
      const repo = id.split('/').pop();
      found.add(repo);
      const parts = [repo, ...filename.split('/')];
      let d = dir;
      for (const p of parts.slice(0, -1)) d = await d.getDirectoryHandle(p, { create: true });
      const fh = await d.getFileHandle(parts[parts.length - 1], { create: true });
      const resp = await cache.match(req);
      const w = await fh.createWritable();
      await resp.body.pipeTo(w);
      count++;
      bytes += Number(resp.headers.get('content-length')) || 0;
      text.textContent = `已存 ${count} 個檔案… ${parts.join('/')}`;
    }
    text.textContent = count
      ? `完成：存了 ${count} 個檔案${bytes ? `（${fmtBytes(bytes)}）` : ''}，模型：${[...found].join('、')}。之後可選「從電腦資料夾載入」並選這個資料夾。`
      : '瀏覽器裡還沒有下載過的模型，請先載入一次。';
  } catch (e) {
    text.textContent = '存檔失敗：' + e.message;
    text.classList.add('error');
  }
});

// ---------- 錄音 ----------
const drop = $('drop');
$('btn-pick-audio').addEventListener('click', () => $('audio-input').click());
$('audio-input').addEventListener('change', (e) => { if (e.target.files[0]) loadAudioFile(e.target.files[0]); });
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => {
  e.preventDefault(); drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) loadAudioFile(f);
});

async function decodeTo16k(file) {
  const buf = await file.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  let ctx;
  try { ctx = new Ctx({ sampleRate: SR }); } catch { ctx = new Ctx(); }
  const decoded = await ctx.decodeAudioData(buf);
  ctx.close?.();
  let data;
  if (decoded.numberOfChannels === 1) {
    data = decoded.getChannelData(0);
  } else {
    const n = decoded.length;
    data = new Float32Array(n);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const ch = decoded.getChannelData(c);
      for (let i = 0; i < n; i++) data[i] += ch[i] / decoded.numberOfChannels;
    }
  }
  if (decoded.sampleRate !== SR) data = resample(data, decoded.sampleRate, SR);
  return data;
}
function resample(data, from, to) {
  const ratio = from / to;
  const n = Math.floor(data.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const x = i * ratio, i0 = Math.floor(x), f = x - i0;
    out[i] = data[i0] * (1 - f) + (data[i0 + 1] ?? data[i0]) * f;
  }
  return out;
}

async function loadAudioFile(file) {
  clearError();
  $('audio-info').textContent = `讀取「${file.name}」中…`;
  try {
    const data = await decodeTo16k(file);
    state.audio = data;
    state.audioName = file.name;
    state.duration = data.length / SR;
    const player = $('player');
    if (player.src) URL.revokeObjectURL(player.src);
    player.src = URL.createObjectURL(file);
    player.hidden = false;
    $('audio-info').textContent = `「${file.name}」長度 ${fmtDur(state.duration)}`;
    timeline.setAudio(data);
    worker.postMessage({ type: 'set-audio', audio: data.slice() });
  } catch (e) {
    console.error(e);
    $('audio-info').textContent = `讀不了這個檔案（${e.message || e}）。請換成 mp3、m4a 或 wav 再試。`;
    $('audio-info').classList.add('error');
    state.audio = null;
  }
  updateStartButton();
}

function selectedSpeakers() {
  const v = $('speakers-select').value;
  if (v === 'auto') return 'auto';
  return Number(v);
}

$('btn-start').addEventListener('click', () => {
  if (!state.audio || !state.modelLoaded) return;
  if (state.segments.length && state.dirty && !confirmInline('重新轉文字會蓋掉目前校對的內容，再按一次「開始轉文字」確認。')) return;
  clearError();
  state.segments = [];
  state.speakers = [];
  state.dirty = false;
  renderAll();
  setBusy(true);
  state.runStart = performance.now();
  $('run-progress').hidden = false;
  $('run-bar').style.width = '0%';
  $('run-text').textContent = '開始…';
  worker.postMessage({
    type: 'transcribe',
    audio: state.audio.slice(),
    opts: { numSpeakers: selectedSpeakers(), toTraditional: $('opt-trad').checked, ranges: timeline.selectedRanges() },
  });
});

// 不用 confirm()：第一次按顯示提示，3 秒內再按一次才執行
let pendingConfirm = 0;
function confirmInline(msg) {
  if (Date.now() - pendingConfirm < 3000) { pendingConfirm = 0; return true; }
  pendingConfirm = Date.now();
  $('run-progress').hidden = false;
  $('run-text').textContent = msg;
  return false;
}

$('btn-cancel').addEventListener('click', () => {
  worker.postMessage({ type: 'cancel' });
  $('run-text').textContent = '停止中（等目前這一段做完）…';
});

// ---------- 背景訊息 ----------
let initTimer = 0;
function stopInitTimer() { if (initTimer) { clearInterval(initTimer); initTimer = 0; } }
function onWorkerMessage(ev) {
  const m = ev.data;
  if (['loaded', 'error', 'status', 'progress', 'segments', 'done', 'cancelled'].includes(m.type)) stopInitTimer();
  switch (m.type) {
    case 'status':
      if (!state.modelLoaded) $('load-text').textContent = m.text;
      else $('run-text').textContent = m.text;
      break;
    case 'load-file':
      break;
    case 'load-init': {
      // 下載完成後，大模型還要花時間載入到顯示卡／記憶體，這段期間沒有進度可以顯示
      const bar = state.modelLoaded ? $('run-bar') : $('load-bar');
      const txt = state.modelLoaded ? $('run-text') : $('load-text');
      bar.classList.add('indeterminate');
      if (!initTimer) {
        const t0 = Date.now();
        const where = $('device-select').value === 'wasm' ? '記憶體' : '顯示卡';
        const tick = () => {
          const sec = Math.round((Date.now() - t0) / 1000);
          let msg = `${m.label}下載完成，正在載入到${where}…已等 ${fmtDur(sec)}（大模型第一次可能要 1～5 分鐘，請不要關閉頁面）`;
          if (sec > 300) msg += '。⚠️ 等太久的話可能是記憶體不足：重新整理後改用較小的模型，或執行方式改「CPU」再試。';
          txt.textContent = msg;
        };
        tick();
        initTimer = setInterval(tick, 1000);
      }
      break;
    }
    case 'load-progress': {
      if (!state.busy) break; // 已結束（成功或失敗）後才到的進度訊息不要蓋掉結果
      const bar = state.modelLoaded ? $('run-bar') : $('load-bar');
      const txt = state.modelLoaded ? $('run-text') : $('load-text');
      if ((m.progress || 0) >= 99.99 && initTimer) break; // 已在「載入中」階段，保留計時訊息
      stopInitTimer();
      bar.classList.remove('indeterminate');
      bar.style.width = `${Math.min(100, m.progress || 0).toFixed(1)}%`;
      txt.textContent = `下載${m.label}：${fmtBytes(m.loaded)}／${fmtBytes(m.total)}（${(m.progress || 0).toFixed(0)}%）`;
      break;
    }
    case 'loaded': {
      state.modelLoaded = true;
      state.loadedKey = m.modelKey;
      $('load-bar').classList.remove('indeterminate');
      $('load-bar').style.width = '100%';
      $('load-text').textContent = `✅ 已載入：${ASR_MODELS[m.modelKey].label}（${m.device === 'webgpu' ? '顯示卡' : 'CPU'}，${typeof m.dtype === 'string' ? m.dtype : Object.values(m.dtype).join('/')}）`;
      setBusy(false);
      updateModelDesc();
      showEnvInfo();
      break;
    }
    case 'progress': {
      const bar = $('run-bar');
      bar.classList.remove('indeterminate');
      const pct = m.total ? (m.done / m.total) * 100 : 0;
      if (m.stage === 'asr') {
        bar.style.width = `${(pct * 0.85).toFixed(1)}%`;
        let eta = '';
        if (m.done > 0 && m.audioDone > 0) {
          const rate = m.elapsed / m.audioDone; // 每秒錄音花幾秒
          eta = `，預估還要 ${fmtDur(rate * (m.audioTotal - m.audioDone))}`;
        }
        $('run-text').textContent = `轉文字中：${fmtTime(m.audioDone)}／${fmtTime(m.audioTotal)}（第 ${Math.min(m.done + 1, m.total)}／${m.total} 段${eta}）`;
      } else {
        bar.style.width = `${(85 + pct * 0.15).toFixed(1)}%`;
        $('run-text').textContent = `分辨說話者中：${m.done}／${m.total} 段`;
      }
      break;
    }
    case 'segments':
      state.segments = m.segments.map((s) => ({ ...s }));
      ensureSpeakers(1);
      renderAll();
      break;
    case 'done': {
      state.segments = m.segments.map((s) => ({ ...s }));
      const n = Math.max(1, ...state.segments.map((s) => s.speaker + 1));
      state.speakers = [];
      ensureSpeakers(n);
      state.dirty = true;
      renderAll();
      setBusy(false);
      $('run-bar').style.width = '100%';
      $('run-text').textContent = `✅ 完成，共 ${state.segments.length} 段，花了 ${fmtDur((performance.now() - state.runStart) / 1000)}。請對照錄音校對。`;
      break;
    }
    case 'rediarized': {
      m.labels.forEach((l, i) => { if (state.segments[i]) state.segments[i].speaker = l; });
      const n = Math.max(1, ...m.labels.map((l) => l + 1));
      const old = state.speakers;
      state.speakers = [];
      ensureSpeakers(n);
      old.forEach((s, i) => { if (state.speakers[i]) state.speakers[i].name = s.name; });
      state.dirty = true;
      renderAll();
      setBusy(false);
      $('run-bar').style.width = '100%';
      $('run-text').textContent = `✅ 已重新分辨說話者（${n} 人）。`;
      break;
    }
    case 'cancelled':
      if (m.segments) { state.segments = m.segments.map((s) => ({ ...s })); ensureSpeakers(1); renderAll(); }
      setBusy(false);
      $('run-text').textContent = '已停止。已完成的部分保留在下方。';
      break;
    case 'error':
      setBusy(false);
      if (m.during === 'load') {
        $('load-bar').classList.remove('indeterminate');
        $('load-text').textContent = /fetch|network|網路/i.test(m.message) && $('source-select').value === 'hf'
          ? '❌ 連不上 Hugging Face 下載模型（可能是網路擋住了）。可以換一個網路，或改用「從電腦資料夾載入」。'
          : /not found|找不到|local/i.test(m.message) && $('source-select').value === 'folder'
            ? `❌ 資料夾裡找不到需要的模型檔（${m.message}）。請確認選的資料夾裡有和模型同名的子資料夾。`
            : `❌ 模型載入失敗：${m.message}。可以換「CPU」模式或較小的模型再試。`;
        $('load-text').classList.add('error');
      } else {
        showError(`❌ 發生錯誤：${m.message}`);
      }
      break;
  }
}

// ---------- 畫面：說話者 ----------
function renderSpeakers() {
  const panel = $('speaker-panel');
  panel.innerHTML = '';
  const counts = state.speakers.map(() => 0);
  state.segments.forEach((s) => { counts[s.speaker] = (counts[s.speaker] || 0) + 1; });
  state.speakers.forEach((sp, i) => {
    const chip = document.createElement('div');
    chip.className = 'speaker-chip';
    chip.innerHTML = `<span class="dot" style="background:${COLORS[i % COLORS.length]}"></span>`;
    const inp = document.createElement('input');
    inp.value = sp.name;
    inp.title = '改名，例如「員警」「被害人」';
    inp.addEventListener('input', () => {
      sp.name = inp.value || `說話者${i + 1}`;
      state.dirty = true;
      document.querySelectorAll(`select.spk option[value="${i}"]`).forEach((o) => { o.textContent = sp.name; });
      renderQaAsker();
      renderExport();
    });
    chip.appendChild(inp);
    const c = document.createElement('span');
    c.className = 'count';
    c.textContent = `${counts[i] || 0} 段`;
    chip.appendChild(c);
    panel.appendChild(chip);
  });
  const add = document.createElement('button');
  add.className = 'secondary';
  add.textContent = '＋ 新增說話者';
  add.addEventListener('click', () => { ensureSpeakers(state.speakers.length + 1); renderAll(); });
  panel.appendChild(add);

  const re = document.createElement('div');
  re.className = 'row';
  re.style.margin = '0';
  re.innerHTML = `<label>重新分辨為 <select id="re-n"><option value="auto">自動</option><option value="2">2 人</option><option value="3">3 人</option><option value="4">4 人</option><option value="5">5 人</option></select></label>`;
  const btn = document.createElement('button');
  btn.id = 'btn-rediarize';
  btn.className = 'secondary';
  btn.textContent = '重新分說話者';
  btn.title = '只重算說話者，不重新轉文字';
  btn.disabled = state.busy || !state.audio;
  btn.addEventListener('click', () => {
    if (!state.audio) return;
    const v = re.querySelector('#re-n').value;
    setBusy(true);
    $('run-progress').hidden = false;
    $('run-text').textContent = '重新分辨說話者…';
    worker.postMessage({
      type: 'rediarize',
      numSpeakers: v === 'auto' ? 'auto' : Number(v),
      segments: state.segments.map((s) => ({ start: s.start, end: s.end })),
    });
  });
  re.appendChild(btn);
  panel.appendChild(re);
  const reSel = re.querySelector('#re-n');
  reSel.value = String(Math.min(5, Math.max(2, state.speakers.length)));
}

// ---------- 畫面：逐字稿 ----------
function renderTranscript() {
  const box = $('transcript');
  const scroll = box.scrollTop;
  box.innerHTML = '';
  const frag = document.createDocumentFragment();
  state.segments.forEach((seg, i) => {
    const row = document.createElement('div');
    row.className = 'seg';
    row.dataset.i = i;
    row.style.borderLeftColor = COLORS[seg.speaker % COLORS.length];

    const t = document.createElement('button');
    t.className = 'time';
    t.textContent = fmtTime(seg.start);
    t.title = '從這裡播放';
    t.addEventListener('click', () => playFrom(seg.start));
    row.appendChild(t);

    const sel = document.createElement('select');
    sel.className = 'spk';
    state.speakers.forEach((sp, si) => {
      const o = document.createElement('option');
      o.value = si; o.textContent = sp.name;
      sel.appendChild(o);
    });
    sel.value = seg.speaker;
    sel.addEventListener('change', () => {
      seg.speaker = Number(sel.value);
      row.style.borderLeftColor = COLORS[seg.speaker % COLORS.length];
      state.dirty = true;
      renderSpeakerCounts();
      renderExport();
    });
    row.appendChild(sel);

    const tx = document.createElement('div');
    tx.className = 'text';
    tx.contentEditable = 'plaintext-only';
    if (tx.contentEditable !== 'plaintext-only') tx.contentEditable = 'true';
    tx.textContent = seg.text;
    tx.addEventListener('input', () => { seg.text = tx.innerText.replace(/\n+/g, ' ').trim(); state.dirty = true; renderExportDebounced(); });
    row.appendChild(tx);

    const tools = document.createElement('div');
    tools.className = 'tools';
    tools.appendChild(toolBtn('分割', '在游標位置把這段拆成兩段', () => splitAtCaret(i, tx)));
    if (i > 0) tools.appendChild(toolBtn('↑併', '和上一段合併', () => mergeUp(i)));
    tools.appendChild(toolBtn('刪', '刪除這段', () => { state.segments.splice(i, 1); state.dirty = true; renderAll(); }));
    row.appendChild(tools);

    frag.appendChild(row);
  });
  box.appendChild(frag);
  box.scrollTop = scroll;
}
function toolBtn(label, title, fn) {
  const b = document.createElement('button');
  b.textContent = label; b.title = title;
  b.addEventListener('mousedown', (e) => e.preventDefault()); // 保留游標位置
  b.addEventListener('click', fn);
  return b;
}
function renderSpeakerCounts() {
  const counts = state.speakers.map(() => 0);
  state.segments.forEach((s) => { counts[s.speaker] = (counts[s.speaker] || 0) + 1; });
  document.querySelectorAll('.speaker-chip .count').forEach((el, i) => { el.textContent = `${counts[i] || 0} 段`; });
}
function caretOffset(el) {
  const sel = window.getSelection();
  if (!sel.rangeCount || !el.contains(sel.anchorNode)) return -1;
  const r = sel.getRangeAt(0).cloneRange();
  r.selectNodeContents(el);
  r.setEnd(sel.anchorNode, sel.anchorOffset);
  return r.toString().length;
}
function splitAtCaret(i, el) {
  const seg = state.segments[i];
  const off = caretOffset(el);
  const text = seg.text;
  if (off <= 0 || off >= text.length) {
    $('run-progress').hidden = false;
    $('run-text').textContent = '請先在文字中間點一下，把游標放在要分割的位置，再按「分割」。';
    return;
  }
  const t = +(seg.start + (seg.end - seg.start) * (off / text.length)).toFixed(2);
  const a = { ...seg, end: t, text: text.slice(0, off).trim() };
  const b = { ...seg, start: t, text: text.slice(off).trim(), speaker: (seg.speaker + 1) % Math.max(2, state.speakers.length) };
  ensureSpeakers(b.speaker + 1);
  state.segments.splice(i, 1, a, b);
  state.dirty = true;
  renderAll();
}
function mergeUp(i) {
  const prev = state.segments[i - 1], cur = state.segments[i];
  prev.end = Math.max(prev.end, cur.end);
  prev.text = (prev.text + cur.text).trim();
  state.segments.splice(i, 1);
  state.dirty = true;
  renderAll();
}

// 播放與跟隨
function playFrom(t) {
  const p = $('player');
  if (!p.src) {
    $('run-progress').hidden = false;
    $('run-text').textContent = '要播放請先在上方選擇對應的錄音檔。';
    return;
  }
  p.currentTime = t;
  p.play().catch(() => {});
}
let lastPlaying = -1;
$('player').addEventListener('timeupdate', () => {
  const t = $('player').currentTime;
  let idx = -1;
  for (let i = 0; i < state.segments.length; i++) {
    if (state.segments[i].start <= t + 0.05) idx = i; else break;
  }
  if (idx === lastPlaying) return;
  document.querySelector('.seg.playing')?.classList.remove('playing');
  const row = document.querySelector(`.seg[data-i="${idx}"]`);
  if (row) {
    row.classList.add('playing');
    if (!$('player').paused) row.scrollIntoView({ block: 'nearest' });
  }
  lastPlaying = idx;
});

// ---------- 匯出 ----------
function renderQaAsker() {
  const sel = $('qa-asker');
  const prev = sel.value;
  sel.innerHTML = '';
  state.speakers.forEach((sp, i) => {
    const o = document.createElement('option');
    o.value = i; o.textContent = sp.name;
    sel.appendChild(o);
  });
  if (prev && Number(prev) < state.speakers.length) sel.value = prev;
}
function mergedSegments() {
  const segs = state.segments.filter((s) => s.text.trim());
  // SRT 字幕每段要短，不合併
  if (!$('opt-merge').checked || $('export-format').value === 'srt') return segs.map((s) => ({ ...s }));
  const out = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    if (last && last.speaker === s.speaker) {
      last.end = s.end;
      last.text += (/[，。？！、,.?!]$/.test(last.text) ? '' : '，') + s.text;
    } else out.push({ ...s });
  }
  return out;
}
function buildExport() {
  const fmt = $('export-format').value;
  const segs = mergedSegments();
  const single = state.speakers.length <= 1;
  if (fmt === 'qa') {
    const asker = Number($('qa-asker').value || 0);
    return segs.map((s) => `${s.speaker === asker ? '問' : '答'}：${s.text}`).join('\n');
  }
  if (fmt === 'txt') return segs.map((s) => `[${fmtTime(s.start)}] ${single ? '' : speakerName(s.speaker) + '：'}${s.text}`).join('\n');
  if (fmt === 'plain') return segs.map((s) => `${single ? '' : speakerName(s.speaker) + '：'}${s.text}`).join('\n');
  // SRT
  return segs.map((s, i) => `${i + 1}\n${fmtTime(s.start, true)} --> ${fmtTime(s.end, true)}\n${single ? '' : speakerName(s.speaker) + '：'}${s.text}\n`).join('\n');
}
function renderExport() {
  $('qa-asker-wrap').hidden = $('export-format').value !== 'qa';
  $('opt-merge').parentElement.hidden = $('export-format').value === 'srt';
  $('export-preview').value = buildExport();
}
let exportTimer = 0;
function renderExportDebounced() { clearTimeout(exportTimer); exportTimer = setTimeout(renderExport, 300); }
$('export-format').addEventListener('change', renderExport);
$('qa-asker').addEventListener('change', renderExport);
$('opt-merge').addEventListener('change', renderExport);

$('btn-copy').addEventListener('click', async () => {
  const text = buildExport();
  try {
    await navigator.clipboard.writeText(text);
    $('copy-status').textContent = '已複製';
  } catch {
    $('export-preview').select();
    document.execCommand('copy');
    $('copy-status').textContent = '已複製';
  }
  setTimeout(() => { $('copy-status').textContent = ''; }, 2000);
});
function baseName() { return (state.audioName || '逐字稿').replace(/\.[^.]+$/, ''); }
function download(name, content, type) {
  const blob = new Blob([content], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}
$('btn-download').addEventListener('click', () => {
  const fmt = $('export-format').value;
  const ext = fmt === 'srt' ? 'srt' : 'txt';
  const suffix = { qa: '_問答', txt: '_逐字稿', plain: '_文字', srt: '' }[fmt];
  download(`${baseName()}${suffix}.${ext}`, '﻿' + buildExport(), 'text/plain;charset=utf-8');
});
$('btn-save-project').addEventListener('click', () => {
  const data = {
    app: 'speech-to-text-lab', version: 1, savedAt: new Date().toISOString(),
    audioName: state.audioName, duration: state.duration,
    speakers: state.speakers, segments: state.segments,
  };
  download(`${baseName()}_專案.json`, JSON.stringify(data, null, 1), 'application/json');
  state.dirty = false;
  $('copy-status').textContent = '已下載專案檔';
  setTimeout(() => { $('copy-status').textContent = ''; }, 2500);
});
$('btn-open-project').addEventListener('click', () => $('project-input').click());
$('project-input').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    const data = JSON.parse(await f.text());
    if (!Array.isArray(data.segments)) throw new Error('格式不對');
    state.segments = data.segments.map((s) => ({ start: +s.start || 0, end: +s.end || 0, text: String(s.text || ''), speaker: +s.speaker || 0 }));
    state.speakers = (data.speakers || []).map((s) => ({ name: String(s.name || '') }));
    ensureSpeakers(Math.max(1, ...state.segments.map((s) => s.speaker + 1)));
    if (!state.audioName) state.audioName = data.audioName || '';
    state.dirty = false;
    renderAll();
    $('run-progress').hidden = false;
    $('run-text').textContent = `已開啟專案（${state.segments.length} 段）。` + (state.audio ? '' : `要邊聽邊校對，請在上方選擇錄音檔「${data.audioName || ''}」。`);
  } catch (err) {
    showError('開不了這個專案檔：' + err.message);
  }
  e.target.value = '';
});

function renderAll() {
  $('sec-result').hidden = state.segments.length === 0;
  renderSpeakers();
  renderTranscript();
  renderQaAsker();
  renderExport();
}

window.addEventListener('beforeunload', (e) => {
  if (state.dirty && state.segments.length) { e.preventDefault(); e.returnValue = ''; }
});

timeline = createTimeline({
  root: $('timeline-wrap'), player: $('player'), fmtTime, fmtDur,
  onChange: () => updateStartButton(),
});

initModelSelect();
showEnvInfo();
