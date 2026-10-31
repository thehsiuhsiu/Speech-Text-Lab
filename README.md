# 語音轉文字（本機辨識・分說話者）

在瀏覽器裡把錄音轉成逐字稿，並分辨不同說話者（例如員警／被害人）。**錄音只在自己的電腦處理，不會上傳。**

## 功能

- 轉文字模型可選：Breeze ASR 25（台灣華語最準）、Whisper large-v3 turbo、Whisper small、Whisper base
- 分說話者：用 WavLM 聲紋模型替每段話算出聲音特徵，再分群；可指定人數或自動判斷
- 校對：點時間從該處播放、直接改文字、改說話者、分割／合併段落、說話者改名
- 匯出：問答格式（問：／答：）、逐字稿、純文字、SRT 字幕；可儲存專案之後繼續校對
- 模型來源：
  - 從 Hugging Face 下載，存在瀏覽器裡（清除「Cookie 和其他網站資料」會一起刪掉）
  - 從電腦資料夾載入（完全不連網）。可以先用「把已下載的模型存到資料夾」把瀏覽器裡的模型存出來

## 電腦需求

- Chrome 或 Edge（新版）
- 有獨立顯示卡最好（WebGPU）。沒有也能用 CPU 跑，但大模型會很慢
- 第一次使用要下載模型：Breeze 約 1.5–1.7GB、分說話者模型約 100MB

## 模型資料夾的結構

選資料夾時，要選「裝著模型子資料夾」的那一層，例如：

```
我的模型/
├─ Breeze-ASR-25-ONNX/
│  ├─ config.json
│  ├─ ...
│  └─ onnx/encoder_model_quantized.onnx ...
└─ wavlm-base-plus-sv/
   ├─ config.json
   ├─ preprocessor_config.json
   └─ onnx/model_quantized.onnx
```

用網頁上的「把已下載的模型存到資料夾」按鈕存出來的就是這個結構。

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `index.html`、`style.css` | 畫面 |
| `app.js` | 畫面操作、讀取錄音、校對、匯出 |
| `worker.js` | 背景執行：載入模型、轉文字、分說話者 |
| `models.js` | 可選用的模型清單 |
| `lib/` | Transformers.js、ONNX Runtime、opencc-js（簡轉繁）放在本地，不依賴外部 CDN |
| `coi-serviceworker.min.js` | 讓 GitHub Pages 能開啟 CPU 多執行緒 |

## 注意

辨識結果可能有錯字和說話者誤判，正式使用前請對照錄音校對。

## 授權

使用的套件與模型各有授權：Transformers.js（Apache-2.0）、ONNX Runtime（MIT）、opencc-js（MIT）、coi-serviceworker（MIT）、Breeze ASR 25（Apache-2.0）、Whisper（MIT）、WavLM。
