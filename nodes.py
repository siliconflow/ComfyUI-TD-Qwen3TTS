import os
import sys
import torch
import numpy as np
import folder_paths

# 将父目录添加到路径以允许导入 qwen_tts
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加本地克隆的 Qwen3-TTS 目录到 sys.path
qwen_repo_path = os.path.join(current_dir, "Qwen3-TTS")
if os.path.exists(qwen_repo_path) and qwen_repo_path not in sys.path:
    sys.path.append(qwen_repo_path)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
except ImportError:
    # 如果找不到 qwen_tts，我们可能是在独立安装中，或者结构不同
    # 但由于我们是在仓库内部创建的，它应该可以工作。
    # 这里可以添加回退或错误处理。
    pass

class Qwen3TTSModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        qwen_models_dir = os.path.join(folder_paths.models_dir, "Qwen3-TTS-Models")
        if not os.path.exists(qwen_models_dir):
            try:
                os.makedirs(qwen_models_dir, exist_ok=True)
            except Exception:
                pass
        
        model_list = []
        if os.path.exists(qwen_models_dir):
            model_list = [d for d in os.listdir(qwen_models_dir) if os.path.isdir(os.path.join(qwen_models_dir, d))]
        
        if not model_list:
            model_list = ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]

        return {
            "required": {
                "model_path": (model_list, {"default": model_list[0] if model_list else "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "attn_implementation": (["sdpa", "flash_attention_2", "eager"], {"default": "sdpa"}),
            }
        }

    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3TTS"

    def load_model(self, model_path, precision, device, attn_implementation):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32
        
        print(f"正在从 {model_path} 加载 Qwen3-TTS 模型，精度: {precision}，设备: {device}，加速方式: {attn_implementation}...")
        
        # 如果 model_path 是相对于此节点的目录，则解析它？
        # 现在假设它是绝对路径或 HF hub id。
        qwen_models_dir = os.path.join(folder_paths.models_dir, "Qwen3-TTS-Models")
        potential_path = os.path.join(qwen_models_dir, model_path)
        if os.path.exists(potential_path):
            model_path_to_load = potential_path
        else:
            model_path_to_load = model_path
        
        model = Qwen3TTSModel.from_pretrained(
            model_path_to_load,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        return (model,)

class Qwen3TTSCustomVoice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，我是Qwen3-TTS。"}),
                "speaker": ("STRING", {"default": "Vivian"}),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {"default": "Auto"}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, speaker, language, instruct=""):
        if language == "Auto":
            language = None
            
        print(f"正在为说话人 {speaker} 生成自定义语音...")
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct if instruct else None
        )
        
        # ComfyUI 音频格式: {'waveform': tensor [batch, channels, samples], 'sample_rate': int}
        # wavs[0] 是 [samples] (单声道)
        audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0) # [1, 1, samples]
        
        return ({"waveform": audio_tensor, "sample_rate": sr},)

class Qwen3TTSVoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "这是一个语音设计测试。"}),
                "instruct": ("STRING", {"multiline": True, "default": "一个年轻女性的声音，语气愉快。"}),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {"default": "Auto"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, instruct, language):
        if language == "Auto":
            language = None
            
        print(f"正在使用指令生成语音设计: {instruct}...")
        wavs, sr = model.generate_voice_design(
            text=text,
            instruct=instruct,
            language=language
        )
        
        audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": sr},)

class Qwen3TTSVoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "这是一个语音克隆测试。"}),
                "ref_audio": ("AUDIO",),
                "language": (["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], {"default": "Auto"}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, ref_audio, language, ref_text=""):
        if language == "Auto":
            language = None
            
        waveform = ref_audio["waveform"] # [batch, channels, samples]
        sr = ref_audio["sample_rate"]
        
        # 取第一个批次，转换为 [samples, channels] 或 [samples]
        # 如果我们传递 numpy，Qwen 通过平均处理立体声
        ref_wav_np = waveform[0].transpose(0, 1).numpy()
        
        print(f"正在生成语音克隆...")
        wavs, out_sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=(ref_wav_np, sr),
            ref_text=ref_text if ref_text else None
        )
        
        audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": out_sr},)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSModelLoader": Qwen3TTSModelLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSModelLoader": "Qwen3 TTS Model Loader",
    "Qwen3TTSCustomVoice": "Qwen3 TTS Custom Voice",
    "Qwen3TTSVoiceDesign": "Qwen3 TTS Voice Design",
    "Qwen3TTSVoiceClone": "Qwen3 TTS Voice Clone",
}
