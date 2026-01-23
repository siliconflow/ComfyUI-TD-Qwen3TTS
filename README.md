# ComfyUI-TD-Qwen3TTS

[English](#english) | [中文](#chinese)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

ComfyUI nodes for **Qwen3-TTS**, supporting high-quality text-to-speech generation, voice design, and voice cloning.
> **Original Project**: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

---

## <span id="english">English</span>

### Features
- **Qwen3-TTS Integration**: Seamlessly use Qwen3-TTS models within ComfyUI.
- **Voice Design**: Generate speech with specific voice characteristics using natural language prompts.
- **Voice Cloning**: Support for custom voice generation (Voice Cloning) using reference audio or speaker ID.
- **Flexible Configuration**: Support for `bf16`/`fp16`/`fp32` precision and multiple attention implementations (`sdpa`, `flash_attention_2`, `eager`).

### Installation

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone [https://github.com/your-username/ComfyUI-TD-Qwen3TTS.git](https://github.com/AICoderTudou/ComfyUI-TD-Qwen3TTS.git)
   ```

2. Install dependencies:
   ```bash
   cd ComfyUI-TD-Qwen3TTS
   pip install -r requirements.txt
   ```
   *Note: `flash-attn` is optional but recommended for better performance on NVIDIA GPUs. If installation fails on Windows, the plugin will default to `sdpa` (Scaled Dot Product Attention).*

3. Download Models:
   - **Official Download**: [Hugging Face Collection](https://huggingface.co/collections/Qwen/qwen3-tts)
   - **Alternative Download (Quark Drive)**: [https://pan.quark.cn/s/010e3ca25022](https://pan.quark.cn/s/010e3ca25022)
   - Download Qwen3-TTS models (e.g., `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`).
   - Place them in: `ComfyUI/models/Qwen3-TTS-Models/`.
   - The directory structure should look like:
     ```
     ComfyUI/models/Qwen3-TTS-Models/
     └── Qwen3-TTS-12Hz-1.7B-CustomVoice/
         ├── config.json
         ├── model.safetensors
         └── ...
     ```

### Usage

1. **Load Model**: Use the `Qwen3TTSModelLoader` node. Select your model, precision, and acceleration method (`sdpa` is recommended if `flash_attn` is not installed).
2. **Voice Design**: Use `Qwen3TTSVoiceDesign` to generate speech from text with a specific voice description (e.g., "A young female voice, cheerful tone").
3. **Custom Voice**: Use `Qwen3TTSCustomVoice` for specific speakers or voice cloning tasks.

### License
This project is licensed under the [Apache 2.0 License](LICENSE).

---

## <span id="chinese">中文</span>

### 功能特点
- **Qwen3-TTS 集成**: 在 ComfyUI 中无缝使用 Qwen3-TTS 模型。
- **语音设计 (Voice Design)**: 通过自然语言提示词生成具有特定特征的语音。
- **语音克隆 (Voice Cloning)**: 支持使用参考音频或说话人 ID 生成自定义语音。
- **灵活配置**: 支持 `bf16`/`fp16`/`fp32` 精度选择，以及多种注意力加速方式 (`sdpa`, `flash_attention_2`, `eager`)。
> **原项目开源地址**: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)

### 安装说明

1. 将本仓库克隆到您的 ComfyUI `custom_nodes` 目录下：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-TD-Qwen3TTS.git
   ```

2. 安装依赖：
   ```bash
   cd ComfyUI-TD-Qwen3TTS
   pip install -r requirements.txt
   ```
   *注意：`flash-attn` 是可选的，但在 NVIDIA 显卡上推荐使用以获得更好性能。如果在 Windows 上安装失败，插件将默认使用 `sdpa` (Scaled Dot Product Attention) 加速，无需额外操作。*

3. 下载模型：
   - **模型官方下载地址**: [Hugging Face Collection](https://huggingface.co/collections/Qwen/qwen3-tts)
   - **模型网盘下载地址 (夸克)**: [https://pan.quark.cn/s/010e3ca25022](https://pan.quark.cn/s/010e3ca25022)
   - 下载 Qwen3-TTS 模型 (例如 `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)。
   - 将模型文件夹放置在：`ComfyUI/models/Qwen3-TTS-Models/` 目录下。
   - 目录结构应如下所示：
     ```
     ComfyUI/models/Qwen3-TTS-Models/
     └── Qwen3-TTS-12Hz-1.7B-CustomVoice/
         ├── config.json
         ├── model.safetensors
         └── ...
     ```

### 使用方法

1. **加载模型**: 使用 `Qwen3TTSModelLoader` 节点。选择您的模型路径、精度和加速方式（如果未安装 `flash_attn`，请保留默认的 `sdpa`）。
2. **语音设计**: 使用 `Qwen3TTSVoiceDesign` 节点，通过输入提示词（如“一个年轻女性的声音，语气愉快”）来生成语音。
3. **自定义语音**: 使用 `Qwen3TTSCustomVoice` 节点进行特定说话人或语音克隆任务。

### 许可证
本项目采用 [Apache 2.0 许可证](LICENSE) 进行授权。
