// 背景執行緒：載入模型、轉文字、分說話者。錄音資料只在這裡處理，不會送出去。
import {
  pipeline, env, AutoFeatureExtractor, WavLMForXVector,
} from './lib/transformers.min.js';
import * as OpenCC from './lib/opencc-cn2t.js';
import { ASR_MODELS, SPEAKER_MODEL } from './models.js';

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
  const chunks = planChunks(audio);
  const segments = [];
  const needConvert = opts.toTraditional;
  const t0 = performance.now();
  for (let i = 0; i < chunks.length; i++) {
    if (cancelled || myKey !== lastKey) { post({ type: 'cancelled', segments }); return; }
    const c = chunks[i];
    post({ type: 'progress', stage: 'asr', done: i, total: chunks.length, audioDone: c.start, audioTotal: total, elapsed: (performance.now() - t0) / 1000 });
    if (!c.hasSpeech) continue;
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
      segments.push({ start: +(c.start + a).toFixed(2), end: +(c.start + b).toFixed(2), text, speaker: 0 });
    }
    post({ type: 'segments', segments });
  }
  post({ type: 'progress', stage: 'asr', done: chunks.length, total: chunks.length, audioDone: total, audioTotal: total, elapsed: (performance.now() - t0) / 1000 });

  if (opts.numSpeakers !== 0 && segments.length > 1) {
    const labels = await diarize(audio, segments, opts.numSpeakers, myKey);
    if (!labels) { post({ type: 'cancelled', segments }); return; }
    labels.forEach((l, i) => { segments[i].speaker = l; });
  }
  post({ type: 'done', segments, elapsed: (performance.now() - t0) / 1000, asrConfig });
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
  // 段落太多時，先用最長的 300 段分群，其餘段落歸到最近的群
  const n = embs.length;
  const idx = Array.from({ length: n }, (_, i) => i);
  const core = n > 300 ? [...idx].sort((a, b) => durations[b] - durations[a]).slice(0, 300) : idx;
  const X = core.map((i) => embs[i]);
  if (k === 'auto') k = guessSpeakerCount(X);
  const parent = agglomerate(X, k);
  // 建立群中心
  const groups = new Map();
  core.forEach((ci, pos) => {
    const g = parent[pos];
    if (!groups.has(g)) groups.set(g, { sum: new Float32Array(embs[ci].length), w: 0 });
    const G = groups.get(g);
    const w = Math.max(0.5, Math.min(durations[ci], 10));
    for (let d = 0; d < G.sum.length; d++) G.sum[d] += embs[ci][d] * w;
    G.w += w;
  });
  const centroids = [...groups.values()].map((G) => normalize(G.sum));
  // 每段歸到最近的群中心
  const raw = embs.map((e) => {
    let best = 0, bv = -Infinity;
    centroids.forEach((c, ci) => { const v = dot(e, c); if (v > bv) { bv = v; best = ci; } });
    return best;
  });
  // 依出現順序重新編號：第一個開口的是說話者 1
  const order = new Map();
  return raw.map((r) => { if (!order.has(r)) order.set(r, order.size); return order.get(r); });
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
