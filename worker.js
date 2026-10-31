// 背景執行緒：載入模型、轉文字、分說話者。錄音資料只在這裡處理，不會送出去。
import {
  pipeline, env, AutoFeatureExtractor, WavLMForXVector,
} from './lib/transformers.min.js';
import * as OpenCC from './lib/opencc-cn2t.js';
// 不用頂層 await，免得背景程式還沒準備好就漏接訊息
const modelsReady = import(`./models.js${new URL(import.meta.url).search}`);

const SR = 16000;

// ---- 執行環境設定 ----
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = {
  mjs: new URL('./lib/ort-wasm-simd-threaded.asyncify.mjs', import.meta.url).href,
  wasm: new URL('./lib/ort-wasm-simd-threaded.asyncify.wasm', import.meta.url).href,
};
if (self.crossOriginIsolated) {
  env.backends.onnx.wasm.numThreads = Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
} else {
  env.backends.onnx.wasm.numThreads = 1;
}

// ---- 從使用者選的資料夾提供模型檔 ----
// 資料夾裡要有和模型同名的子資料夾，例如 Breeze-ASR-25-ONNX/config.json、Breeze-ASR-25-ONNX/onnx/...
let folderFiles = null; // Map<相對路徑, File>
function lookupFolderFile(key) {
  if (!folderFiles) return null;
  const m = String(key).match(/^https?:\/\/[^/]+\/(.+?)\/resolve\/[^/]+\/(.+)$/);
  if (!m) return null;
  const modelId = decodeURIComponent(m[1]);
  const filename = decodeURIComponent(m[2]);
  const repoName = modelId.split('/').pop();
  const want = `${repoName}/${filename}`;
  for (const [path, file] of folderFiles) {
    if (path === want || path.endsWith('/' + want)) return file;
  }
  return null;
}
const folderCache = {
  async match(key) {
    const f = lookupFolderFile(key);
    if (!f) return undefined;
    return new Response(f, { headers: { 'content-length': String(f.size), 'content-type': 'application/octet-stream' } });
  },
  async put() { /* 從資料夾載入時不另外存 */ },
};

function setSource(source, files) {
  if (source === 'folder') {
    folderFiles = new Map(files.map((f) => [f.path, f.file]));
    env.useCustomCache = true;
    env.customCache = folderCache;
    env.useBrowserCache = false;
    // 從資料夾載入時完全不連網（派出所網路擋 Hugging Face 也能用）
    env.allowRemoteModels = false;
    env.allowLocalModels = true; // 兩者不能同時關閉；指到不存在的路徑，找不到就當作沒有
    env.localModelPath = new URL('./__no_local_models__/', import.meta.url).href;
  } else {
    folderFiles = null;
    env.useCustomCache = false;
    env.customCache = null;
    env.useBrowserCache = true;
    env.allowRemoteModels = true;
    env.allowLocalModels = false;
  }
}

// ---- 狀態 ----
let asr = null;
let asrKey = null;
let asrConfig = null;
let spkProcessor = null;
let spkModel = null;
let cancelled = false;
let lastAudio = null;
let lastKey = 0;

const post = (msg) => self.postMessage(msg);
const converter = OpenCC.Converter({ from: 'cn', to: 'twp' });

async function detectDevice(pref) {
  let adapter = null;
  if (navigator.gpu) {
    try { adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }); } catch { adapter = null; }
  }
  const hasGpu = !!adapter;
  const f16 = hasGpu && adapter.features.has('shader-f16');
  let device = pref;
  if (pref === 'auto') device = hasGpu ? 'webgpu' : 'wasm';
  if (device === 'webgpu' && !hasGpu) {
    post({ type: 'status', text: '這台電腦的瀏覽器沒有 WebGPU，改用 CPU。' });
    device = 'wasm';
  }
  return { device, f16, hasGpu };
}

function progressForwarder(label) {
  return (p) => {
    if (p.status === 'progress_total') {
      post({ type: 'load-progress', label, loaded: p.loaded, total: p.total, progress: p.progress });
      if (p.progress >= 99.99) post({ type: 'load-init', label });
    } else if (p.status === 'initiate' || p.status === 'download') {
      post({ type: 'load-file', label, file: p.file });
    }
  };
}

async function loadAsr({ modelKey, device: pref, source, files }) {
  setSource(source, files || []);
  const { ASR_MODELS } = await modelsReady;
  const cfg = ASR_MODELS[modelKey];
  const { device, f16 } = await detectDevice(pref);
  const list = device === 'webgpu' ? (f16 ? cfg.dtypes.webgpu_f16 : cfg.dtypes.webgpu) : cfg.dtypes.wasm;
  if (asr) { try { await asr.dispose(); } catch { /* ignore */ } asr = null; }
  let lastErr = null;
  for (const item of list) {
    const isWrapped = item && typeof item === 'object' && 'dtype' in item;
    const dtype = isWrapped ? item.dtype : item;
    const extra = isWrapped && item.use_external_data_format ? { use_external_data_format: item.use_external_data_format } : {};
    try {
      post({ type: 'status', text: `載入 ${cfg.label}（${device === 'webgpu' ? '顯示卡' : 'CPU'}）…` });
      asr = await pipeline('automatic-speech-recognition', cfg.id, {
        device, dtype, ...extra, progress_callback: progressForwarder('轉文字模型'),
      });
      asrKey = modelKey;
      asrConfig = { device, dtype };
      return { device, dtype };
    } catch (e) {
      lastErr = e;
      console.warn('載入失敗，換下一種格式', dtype, e);
      post({ type: 'status', text: `這種格式載入失敗（${String(e?.message || e).slice(0, 80)}），改試下一種…` });
    }
  }
  throw lastErr || new Error('模型載入失敗');
}

async function loadSpeakerModel() {
  if (spkModel) return;
  const { SPEAKER_MODEL } = await modelsReady;
  let lastErr = null;
  post({ type: 'status', text: '載入分說話者模型（約 100MB，只需下載一次）…' });
  for (const dtype of SPEAKER_MODEL.dtypes) {
    try {
      const cb = progressForwarder('分說話者模型');
      spkProcessor = await AutoFeatureExtractor.from_pretrained(SPEAKER_MODEL.id, { progress_callback: cb });
      spkModel = await WavLMForXVector.from_pretrained(SPEAKER_MODEL.id, { dtype, device: 'wasm', progress_callback: cb });
      return;
    } catch (e) {
      lastErr = e;
      console.warn('分說話者模型載入失敗', dtype, e);
    }
  }
  throw lastErr || new Error('分說話者模型載入失敗');
}

// ---- 依停頓切段（每段 ≤ 28 秒，盡量切在最安靜的地方） ----
function frameRms(audio) {
  const hop = 160; // 10ms
  const n = Math.floor(audio.length / hop);
  const rms = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = i * hop, end = j + hop; j < end; j++) s += audio[j] * audio[j];
    rms[i] = Math.sqrt(s / hop);
  }
  // 300ms 移動平均
  const w = 30;
  const smooth = new Float32Array(n);
  let acc = 0;
  for (let i = 0; i < n; i++) {
    acc += rms[i];
    if (i >= w) acc -= rms[i - w];
    smooth[i] = acc / Math.min(i + 1, w);
  }
  return { rms, smooth };
}

function percentile(arr, p) {
  if (!arr.length) return 0;
  const step = Math.max(1, Math.floor(arr.length / 20000));
  const sample = [];
  for (let i = 0; i < arr.length; i += step) sample.push(arr[i]);
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(p * sample.length))];
}

export function planChunks(audio, maxLen = 28, minLen = 12) {
  const total = audio.length / SR;
  const { rms, smooth } = frameRms(audio);
  const noise = percentile(rms, 0.1);
  const speechThr = Math.max(noise * 3, 0.004);
  const chunks = [];
  let start = 0;
  while (start < total - 0.05) {
    let end;
    if (total - start <= maxLen) {
      end = total;
    } else {
      const a = Math.floor((start + minLen) * 100);
      const b = Math.min(smooth.length - 1, Math.floor((start + maxLen) * 100));
      let best = b, bestV = Infinity;
      for (let f = a; f <= b; f++) {
        if (smooth[f] < bestV) { bestV = smooth[f]; best = f; }
      }
      // smooth 是往回平均，切點取窗中間
      end = Math.max(start + minLen, (best - 15) / 100);
    }
    // 判斷這段有沒有人聲
    const fa = Math.floor(start * 100), fb = Math.min(rms.length, Math.floor(end * 100));
    let voiced = 0;
    for (let f = fa; f < fb; f++) if (rms[f] > speechThr) voiced++;
    const ratio = fb > fa ? voiced / (fb - fa) : 0;
    chunks.push({ start, end, hasSpeech: ratio > 0.03 });
    start = end;
  }
  return chunks;
}

// Whisper 在安靜或雜音段落常見的「幻覺」句子
const HALLUCINATIONS = [
  /字幕(由|製作|提供|志願者|by)/, /請不吝點贊|點贊|訂閱|轉發|打賞/, /Amara\.org/i,
  /^謝謝(大家)?(收看|觀看)[。！!]?$/, /^(優優)?獨播劇場/, /明鏡與點點/, /^\s*[.。,，…]+\s*$/,
];
function isHallucination(text) {
  return HALLUCINATIONS.some((re) => re.test(text));
}
function collapseRepeats(text) {
  // Whisper 偶爾會「跳針」重複同一句，連續重複 3 次以上只留一次
  let t = text.replace(/(.{2,30}?)(?:[，,、。！？!? ]*\1){2,}/g, '$1');
  // 「對對對對對對」這類單字重複縮成 3 個
  t = t.replace(/(.)\1{4,}/g, '$1$1$1');
  return t;
}

async function transcribe({ audio, opts }) {
  cancelled = false;
  lastAudio = audio;
  const myKey = ++lastKey;
  const total = audio.length / SR;
  const needConvert = opts.toTraditional;
  const t0 = performance.now();

  // 先分說話者、切出「每段只有一個人」的片段，再逐段轉文字
  let plan;
  if (opts.numSpeakers !== 0) {
    plan = await speakerTurns(audio, opts.numSpeakers, myKey);
    if (!plan) { post({ type: 'cancelled', segments: [] }); return; }
  } else {
    plan = planChunks(audio).filter((c) => c.hasSpeech).map((c) => ({ start: c.start, end: c.end, speaker: 0 }));
  }

  const segments = [];
  const tAsr = performance.now();
  for (let i = 0; i < plan.length; i++) {
    if (cancelled || myKey !== lastKey) { post({ type: 'cancelled', segments }); return; }
    const c = plan[i];
    post({ type: 'progress', stage: 'asr', done: i, total: plan.length, audioDone: c.start, audioTotal: total, elapsed: (performance.now() - tAsr) / 1000 });
    const slice = audio.subarray(Math.floor(c.start * SR), Math.floor(c.end * SR));
    let out;
    try {
      out = await asr(slice, {
        language: 'zh', task: 'transcribe', return_timestamps: true,
        max_new_tokens: Math.min(440, Math.ceil((c.end - c.start) * 12) + 20),
      });
    } catch (e) {
      console.warn('這一段辨識失敗', c, e);
      continue;
    }
    const dur = c.end - c.start;
    const parts = (out.chunks && out.chunks.length) ? out.chunks : [{ text: out.text, timestamp: [0, dur] }];
    for (const p of parts) {
      let text = (p.text || '').trim();
      if (!text) continue;
      text = collapseRepeats(text);
      if (needConvert) text = converter(text);
      if (isHallucination(text)) continue;
      let [a, b] = p.timestamp || [0, dur];
      if (a == null || Number.isNaN(a)) a = 0;
      if (b == null || Number.isNaN(b) || b > dur + 0.5) b = dur;
      a = Math.min(Math.max(0, a), dur);
      b = Math.min(Math.max(a + 0.2, b), dur);
      segments.push({ start: +(c.start + a).toFixed(2), end: +(c.start + b).toFixed(2), text, speaker: c.speaker });
    }
    post({ type: 'segments', segments });
  }
  post({ type: 'progress', stage: 'asr', done: plan.length, total: plan.length, audioDone: total, audioTotal: total, elapsed: (performance.now() - tAsr) / 1000 });
  // 依出現順序重新編號說話者
  const order = new Map();
  segments.forEach((sg) => { if (!order.has(sg.speaker)) order.set(sg.speaker, order.size); sg.speaker = order.get(sg.speaker); });
  post({ type: 'done', segments, elapsed: (performance.now() - t0) / 1000, asrConfig });
}

// ---- 先分說話者：滑動視窗算聲紋 → 分群 → 時間軸 → 每段一個人 ----
const WIN = 1.5;  // 每個視窗 1.5 秒
const HOP = 0.75; // 每 0.75 秒一個
const MIN_TURN = 1.0; // 少於 1 秒的換人視為誤判，併到前後

async function speakerTurns(audio, k, myKey) {
  await loadSpeakerModel();
  const total = audio.length / SR;
  const { rms } = frameRms(audio);
  const nF = rms.length; // 每格 10ms
  const noise = percentile(rms, 0.1);
  const thr = Math.max(noise * 3, 0.004);
  const voiced = new Uint8Array(nF);
  for (let f = 0; f < nF; f++) voiced[f] = rms[f] > thr ? 1 : 0;
  // 補平 0.3 秒內的短暫停頓
  {
    let f = 0;
    while (f < nF) {
      if (voiced[f]) { f++; continue; }
      let g = f; while (g < nF && !voiced[g]) g++;
      if (f > 0 && g < nF && g - f <= 30) for (let x = f; x < g; x++) voiced[x] = 1;
      f = g;
    }
  }

  // 1. 有人聲的視窗
  const wins = [];
  for (let t = 0; t + WIN <= total + 0.01; t += HOP) {
    const fa = Math.floor(t * 100), fb = Math.min(nF, Math.floor((t + WIN) * 100));
    let v = 0; for (let f = fa; f < fb; f++) v += voiced[f];
    if (fb > fa && v / (fb - fa) >= 0.5) wins.push(t);
  }
  if (wins.length < 2) {
    return planChunks(audio).filter((c) => c.hasSpeech).map((c) => ({ start: c.start, end: c.end, speaker: 0 }));
  }

  // 2. 每個視窗的聲紋
  const embs = [];
  const t0 = performance.now();
  for (let i = 0; i < wins.length; i++) {
    if (cancelled || myKey !== lastKey) return null;
    if (i % 5 === 0) post({ type: 'progress', stage: 'spk', done: i, total: wins.length, elapsed: (performance.now() - t0) / 1000 });
    embs.push(await embedSlice(audio, wins[i], Math.min(total, wins[i] + WIN)));
  }
  post({ type: 'progress', stage: 'spk', done: wins.length, total: wins.length, elapsed: (performance.now() - t0) / 1000 });

  // 3. 分群
  if (k === 'auto') k = guessSpeakerCount(subsample(embs, 300));
  const labels = k <= 1 ? new Array(embs.length).fill(0) : kmeans(embs, k).labels;

  // 4. 每 10ms 一格投票決定是誰
  const votes = Array.from({ length: Math.max(1, k) }, () => new Float32Array(nF));
  wins.forEach((t, i) => {
    const fa = Math.floor(t * 100), fb = Math.min(nF, Math.floor((t + WIN) * 100));
    const mid = (fa + fb) / 2, half = (fb - fa) / 2;
    for (let f = fa; f < fb; f++) votes[labels[i]][f] += 1 - Math.abs(f - mid) / (half + 1) * 0.5; // 視窗中間權重較高
  });
  const lab = new Int16Array(nF).fill(-1);
  for (let f = 0; f < nF; f++) {
    if (!voiced[f]) continue;
    let best = -1, bv = 0;
    for (let c = 0; c < votes.length; c++) if (votes[c][f] > bv) { bv = votes[c][f]; best = c; }
    lab[f] = best;
  }

  // 5. 整理成「連續同一人」的區段（中間的靜音也算進去），太短的換人併掉
  let runs = [];
  for (let f = 0; f < nF; f++) {
    if (lab[f] < 0) continue;
    const last = runs[runs.length - 1];
    if (last && last.speaker === lab[f] && f - last.endF <= 150) last.endF = f + 1; // 1.5 秒內的停頓視為同一段
    else runs.push({ speaker: lab[f], startF: f, endF: f + 1 });
  }
  let changed = true;
  while (changed && runs.length > 1) {
    changed = false;
    for (let i = 0; i < runs.length; i++) {
      const r = runs[i];
      if ((r.endF - r.startF) / 100 >= MIN_TURN) continue;
      const prev = runs[i - 1], next = runs[i + 1];
      const target = prev && next ? ((prev.endF - prev.startF) >= (next.endF - next.startF) ? prev : next) : (prev || next);
      if (!target) continue;
      target.startF = Math.min(target.startF, r.startF);
      target.endF = Math.max(target.endF, r.endF);
      runs.splice(i, 1);
      changed = true;
      break;
    }
    // 合併後相鄰同一人的區段
    const merged = [];
    for (const r of runs) {
      const last = merged[merged.length - 1];
      if (last && last.speaker === r.speaker) last.endF = Math.max(last.endF, r.endF);
      else merged.push({ ...r });
    }
    runs = merged;
  }

  // 6. 轉成秒數，太長的再依停頓切成 ≤ 28 秒
  const plan = [];
  for (const r of runs) {
    const s = Math.max(0, r.startF / 100 - 0.15);
    const e = Math.min(total, r.endF / 100 + 0.15);
    if (e - s < 0.3) continue;
    if (e - s <= 28) { plan.push({ start: s, end: e, speaker: r.speaker }); continue; }
    const sub = audio.subarray(Math.floor(s * SR), Math.floor(e * SR));
    for (const c of planChunks(sub)) {
      if (c.hasSpeech) plan.push({ start: s + c.start, end: s + c.end, speaker: r.speaker });
    }
  }
  // 第一個開口的是說話者 1
  const order = new Map();
  plan.forEach((p) => { if (!order.has(p.speaker)) order.set(p.speaker, order.size); p.speaker = order.get(p.speaker); });
  return plan;
}

function subsample(arr, n) {
  if (arr.length <= n) return arr;
  const step = arr.length / n;
  return Array.from({ length: n }, (_, i) => arr[Math.floor(i * step)]);
}

// 餘弦 k-means（k-means++ 初始化、多次重來取最好），比階層式分群不怕離群值
export function kmeans(X, k, { restarts = 8, iters = 40, weights = null } = {}) {
  const n = X.length, dim = X[0].length;
  const w = weights || new Array(n).fill(1);
  let best = null;
  for (let r = 0; r < restarts; r++) {
    // k-means++
    const cents = [X[Math.floor(Math.random() * n)]];
    const dmin = new Float64Array(n).fill(Infinity);
    while (cents.length < k) {
      const c = cents[cents.length - 1];
      let sum = 0;
      for (let i = 0; i < n; i++) { const d = Math.max(0, 1 - dot(X[i], c)); if (d < dmin[i]) dmin[i] = d; sum += dmin[i] * dmin[i]; }
      let pick = Math.random() * sum, idx = 0;
      for (; idx < n - 1; idx++) { pick -= dmin[idx] * dmin[idx]; if (pick <= 0) break; }
      cents.push(X[idx]);
    }
    let C = cents.map((c) => Float32Array.from(c));
    const lab = new Int32Array(n).fill(-1);
    for (let it = 0; it < iters; it++) {
      let moved = 0;
      for (let i = 0; i < n; i++) {
        let bc = 0, bv = -Infinity;
        for (let c = 0; c < k; c++) { const v = dot(X[i], C[c]); if (v > bv) { bv = v; bc = c; } }
        if (lab[i] !== bc) { lab[i] = bc; moved++; }
      }
      const sums = Array.from({ length: k }, () => new Float32Array(dim));
      const cnt = new Float64Array(k);
      for (let i = 0; i < n; i++) { const S = sums[lab[i]]; for (let d = 0; d < dim; d++) S[d] += X[i][d] * w[i]; cnt[lab[i]] += w[i]; }
      for (let c = 0; c < k; c++) {
        if (cnt[c] > 0) C[c] = normalize(sums[c]);
        else { // 空群：拿離自己群中心最遠的點
          let far = 0, fv = Infinity;
          for (let i = 0; i < n; i++) { const v = dot(X[i], C[lab[i]]); if (v < fv) { fv = v; far = i; } }
          C[c] = Float32Array.from(X[far]);
        }
      }
      if (moved === 0) break;
    }
    let cost = 0;
    for (let i = 0; i < n; i++) cost += w[i] * (1 - dot(X[i], C[lab[i]]));
    if (!best || cost < best.cost) best = { cost, labels: Array.from(lab), centroids: C };
  }
  return best;
}

// ---- 分說話者：每段算聲紋 → 分群 ----
function normalize(v) {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  s = Math.sqrt(s) || 1;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / s;
  return out;
}
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

async function embedSlice(audio, s, e) {
  const slice = audio.slice(Math.floor(s * SR), Math.floor(e * SR));
  const inputs = await spkProcessor(slice);
  const out = await spkModel(inputs);
  return normalize(out.embeddings.data);
}

async function embedSegment(audio, seg, total) {
  let s = seg.start, e = seg.end;
  const dur = e - s;
  if (dur < 1.5) {
    const c = (s + e) / 2;
    s = Math.max(0, c - 0.75);
    e = Math.min(total, c + 0.75);
    return embedSlice(audio, s, e);
  }
  if (dur <= 8) return embedSlice(audio, s, e);
  // 長段落：每 4 秒算一次再平均
  let acc = null, n = 0;
  for (let t = s; t < e - 1; t += 4) {
    const v = await embedSlice(audio, t, Math.min(e, t + 4));
    if (!acc) acc = new Float32Array(v.length);
    for (let i = 0; i < v.length; i++) acc[i] += v[i];
    n++;
    if (n >= 6) break; // 夠了
  }
  return normalize(acc);
}

// 平均連結階層式分群，分成 k 群。
export function agglomerate(X, k, history = null) {
  const n = X.length;
  if (n === 0) return [];
  const D = new Float32Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = 1 - dot(X[i], X[j]);
      D[i * n + j] = d; D[j * n + i] = d;
    }
  }
  const size = new Array(n).fill(1);
  const active = new Array(n).fill(true);
  const parent = Array.from({ length: n }, (_, i) => i);
  let count = n;
  const target = Math.max(1, k);
  while (count > target) {
    let bi = -1, bj = -1, bd = Infinity;
    for (let i = 0; i < n; i++) {
      if (!active[i]) continue;
      for (let j = i + 1; j < n; j++) {
        if (!active[j]) continue;
        const d = D[i * n + j];
        if (d < bd) { bd = d; bi = i; bj = j; }
      }
    }
    if (bi < 0) break;
    if (history) history[count] = bd; // 從 count 群併成 count-1 群的距離
    // 把 bj 併入 bi
    for (let m = 0; m < n; m++) {
      if (!active[m] || m === bi || m === bj) continue;
      const d = (size[bi] * D[bi * n + m] + size[bj] * D[bj * n + m]) / (size[bi] + size[bj]);
      D[bi * n + m] = d; D[m * n + bi] = d;
    }
    size[bi] += size[bj];
    active[bj] = false;
    for (let m = 0; m < n; m++) if (parent[m] === bj) parent[m] = bi;
    count--;
  }
  return parent;
}

// 自動判斷人數：看「再合併下去距離突然變大」的地方（2～6 人）
export function guessSpeakerCount(X) {
  if (X.length < 3) return Math.max(1, X.length);
  const hist = [];
  agglomerate(X, 1, hist);
  if ((hist[2] ?? 0) < 0.12) return 1; // 全部都很像：只有一個人
  let best = 2, bestGap = -Infinity;
  for (let c = 2; c <= Math.min(6, X.length - 1); c++) {
    const gap = (hist[c] ?? 0) - (hist[c + 1] ?? 0);
    if (gap > bestGap) { bestGap = gap; best = c; }
  }
  return best;
}

export function clusterEmbeddings(embs, durations, k) {
  if (k === 'auto') k = guessSpeakerCount(subsample(embs, 300));
  if (k <= 1 || embs.length < 2) return embs.map(() => 0);
  // 長段落的聲紋比較可靠，權重較高
  const weights = durations.map((d) => Math.max(0.3, Math.min(d, 8)));
  const { labels } = kmeans(embs, Math.min(k, embs.length), { weights });
  const order = new Map();
  return labels.map((r) => { if (!order.has(r)) order.set(r, order.size); return order.get(r); });
}

async function diarize(audio, segments, k, myKey) {
  await loadSpeakerModel();
  const total = audio.length / SR;
  const embs = [];
  const t0 = performance.now();
  for (let i = 0; i < segments.length; i++) {
    if (cancelled || myKey !== lastKey) return null;
    if (i % 3 === 0) post({ type: 'progress', stage: 'spk', done: i, total: segments.length, elapsed: (performance.now() - t0) / 1000 });
    embs.push(await embedSegment(audio, segments[i], total));
  }
  post({ type: 'progress', stage: 'spk', done: segments.length, total: segments.length, elapsed: (performance.now() - t0) / 1000 });
  return clusterEmbeddings(embs, segments.map((s) => s.end - s.start), k);
}

// ---- 訊息處理 ----
self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'load') {
      const info = await loadAsr(msg);
      post({ type: 'loaded', modelKey: asrKey, ...info });
    } else if (msg.type === 'transcribe') {
      if (!asr) throw new Error('請先載入模型');
      await transcribe(msg);
    } else if (msg.type === 'rediarize') {
      cancelled = false;
      const myKey = ++lastKey;
      if (!lastAudio) throw new Error('找不到錄音資料，請重新選擇錄音檔');
      const labels = await diarize(lastAudio, msg.segments, msg.numSpeakers, myKey);
      if (labels) post({ type: 'rediarized', labels });
      else post({ type: 'cancelled' });
    } else if (msg.type === 'set-audio') {
      lastAudio = msg.audio;
    } else if (msg.type === 'cancel') {
      cancelled = true;
    }
  } catch (e) {
    console.error(e);
    post({ type: 'error', message: String(e && e.message ? e.message : e), during: msg.type });
  }
};
