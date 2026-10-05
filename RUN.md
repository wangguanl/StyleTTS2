# 运行命令

- 项目：StyleTTS2
- 生成时间：2026-09-08
- 运行方式：直接运行（uv venv + CLI / Jupyter）
- 硬件评估：**满足**（推理）
  - 官方 Colab Demo 使用 T4 16GB；本机 RTX 4080 16GB，推理时剩余显存约 12GB+，足够 LJSpeech 推理
  - 训练 / 微调（含 SLM adversarial）在单卡 16GB 上可能 OOM，需降 `batch_size` / `max_len` / `batch_percentage`；本交付物以**推理验证**为主，未启动训练

## 环境准备

```powershell
# 1) 全局：PowerShell 7、ffmpeg、eSpeak NG（phonemizer 依赖）
# eSpeak NG：本仓库已带便携版 _tools\espeak-ng（start.ps1 会自动加入 PATH）
# 也可全局安装：winget install --id eSpeak-NG.eSpeak-NG -e（需管理员）

# 2) 虚拟环境与依赖（已用 uv；PyPI 走清华；torch 用 cu124 官方轮子）
cd E:\AI\local-voice\StyleTTS2
uv venv --python 3.10.11 .venv
$env:UV_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'
uv pip install --python .venv\Scripts\python.exe SoundFile munch pydub pyyaml librosa nltk matplotlib accelerate transformers einops einops-exts tqdm typing typing-extensions phonemizer scipy jupyter ipykernel
uv pip install --python .venv\Scripts\python.exe "git+https://github.com/resemble-ai/monotonic_align.git"
# CUDA torch（勿用默认 PyPI 的 +cpu）
uv pip install --python .venv\Scripts\python.exe `
  "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp310-cp310-win_amd64.whl" `
  "https://download.pytorch.org/whl/cu124/torchaudio-2.6.0%2Bcu124-cp310-cp310-win_amd64.whl"

# 3) 下载预训练权重（HF 官方不可达时用镜像）
$env:HF_ENDPOINT = 'https://hf-mirror.com'
hf download yl4579/StyleTTS2-LJSpeech --local-dir .\_hf_LJSpeech
New-Item -ItemType Directory -Force -Path .\Models\LJSpeech | Out-Null
Copy-Item .\_hf_LJSpeech\Models\LJSpeech\* .\Models\LJSpeech\ -Force
```

辅助权重 `Utils/ASR`、`Utils/JDC`、`Utils/PLBERT` 已随仓库自带。

## 启动

- 推荐：`pwsh -NoProfile -File .\start.ps1`
- 说明：多服务时启动后按提示选择；默认 LJSpeech CLI 推理
- 等价手动命令：
  - LJSpeech CLI：`.\.venv\Scripts\python.exe .\infer_ljspeech.py --out outputs\ljspeech_demo.wav`
  - Jupyter Demo：`.\.venv\Scripts\python.exe -m jupyter lab --no-browser --port <port>`，打开 `Demo/Inference_LJSpeech.ipynb`

## 验证

1. `nvidia-smi` 可见推理进程占显存
2. `outputs\ljspeech_demo.wav` 生成，时长 > 1s，用播放器可听清英文
3. 日志打印 `RTF=...` 且无 CUDA / phonemizer 报错

## 备注

- 镜像：PyPI=`https://pypi.tuna.tsinghua.edu.cn/simple`；HF=`https://hf-mirror.com`；torch 轮子直连 `download.pytorch.org`（须钉 `2.6.0+cu124`，勿装到 +cpu）
- ffmpeg：`E:\Programs\ffmpeg-master-latest-win64-gpl\bin`
- eSpeak：便携目录 `_tools\espeak-ng`（管理员 MSI 安装被取消时用此方案）；`start.ps1` / `infer_ljspeech.py` 依赖 `PHONEMIZER_ESPEAK_LIBRARY`
- NLTK `punkt` 若因网络下不去，CLI 会回退空白分词，不影响英文 Demo
- 当前占用：若已有其它 Python 占卡，先腾出显存再推大段文本
- LibriTTS 多说话人需另下 `yl4579/StyleTTS2-LibriTTS` 与 `reference_audio.zip`，未纳入默认启动
- 预训练模型使用约定见 README（合成语音需告知听者）
- 本机产物（默认不提交）：`RUN.md`、`start.ps1`、`infer_ljspeech.py`、`.venv/`、`Models/`、`_tools/`、`outputs/`