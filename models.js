// 可選用的轉文字模型。dtype 依序嘗試，前一個失敗（例如檔案不存在、顯示卡不支援）就換下一個。
export const ASR_MODELS = {
  breeze: {
    id: 'markno5/Breeze-ASR-25-ONNX',
    label: 'Breeze ASR 25（台灣華語、中英夾雜最準）',
    desc: '聯發科 Breeze ASR 25 的瀏覽器版本，約 1.5–1.7GB。準確度最好，但需要較好的電腦；建議有獨立顯示卡並使用「顯示卡」模式，用 CPU 會非常慢。',
    traditional: true,
    dtypes: {
      webgpu_f16: ['q4f16', 'q8'],
      webgpu: ['q8'],
      wasm: ['q8'],
    },
  },
  turbo: {
    id: 'onnx-community/whisper-large-v3-turbo',
    label: 'Whisper large-v3 turbo（準確、速度中等）',
    desc: 'OpenAI Whisper 的加速版，約 0.8–1.2GB。中文不錯，速度比 Breeze 快。輸出可能是簡體，會自動轉成繁體。',
    traditional: false,
    dtypes: {
      webgpu_f16: [{ encoder_model: 'fp16', decoder_model_merged: 'q4' }, 'q8'],
      webgpu: [{ encoder_model: 'q4', decoder_model_merged: 'q4' }, 'q8'],
      wasm: ['q8'],
    },
  },
  small: {
    id: 'Xenova/whisper-small',
    label: 'Whisper small（較快、較不準）',
    desc: '約 250MB，一般電腦用 CPU 也跑得動，但錯字較多。',
    traditional: false,
    dtypes: {
      webgpu_f16: [{ encoder_model: 'fp32', decoder_model_merged: 'q4' }, 'q8'],
      webgpu: [{ encoder_model: 'fp32', decoder_model_merged: 'q4' }, 'q8'],
      wasm: ['q8'],
    },
  },
  base: {
    id: 'Xenova/whisper-base',
    label: 'Whisper base（最快，測試用）',
    desc: '約 80MB，速度最快，準確度最低，適合先測試網頁能不能跑。',
    traditional: false,
    dtypes: {
      webgpu_f16: [{ encoder_model: 'fp32', decoder_model_merged: 'q4' }, 'q8'],
      webgpu: [{ encoder_model: 'fp32', decoder_model_merged: 'q4' }, 'q8'],
      wasm: ['q8'],
    },
  },
};

// 分說話者用的「聲紋」模型
export const SPEAKER_MODEL = {
  id: 'Xenova/wavlm-base-plus-sv',
  dtypes: ['q8', 'fp32'],
};
