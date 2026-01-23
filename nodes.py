import os
import sys
import json
import torch
import numpy as np
import folder_paths

# 将父目录添加到路径以允许导入 qwen_tts
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加本地克隆的 cc 目录到 sys.path
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

def get_all_speakers():
    speakers = set()
    qwen_models_dir = os.path.join(folder_paths.models_dir, "Qwen3-TTS-Models")
    if os.path.exists(qwen_models_dir):
        for d in os.listdir(qwen_models_dir):
            path = os.path.join(qwen_models_dir, d)
            if os.path.isdir(path):
                config_path = os.path.join(path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config = json.load(f)
                            if "talker_config" in config and "spk_id" in config["talker_config"]:
                                speakers.update(config["talker_config"]["spk_id"].keys())
                    except Exception as e:
                        print(f"Failed to read config for {d}: {e}")
    
    if not speakers:
        return ["Vivian", "Serena", "Ryan"] # Fallback defaults
    
    return sorted(list(speakers))

KNOWN_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
]

class TDQwen3TTSModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        qwen_models_dir = os.path.join(folder_paths.models_dir, "Qwen3-TTS-Models")
        if not os.path.exists(qwen_models_dir):
            try:
                os.makedirs(qwen_models_dir, exist_ok=True)
            except Exception:
                pass
        
        # Get existing directories
        existing_models = []
        if os.path.exists(qwen_models_dir):
            existing_models = [d for d in os.listdir(qwen_models_dir) if os.path.isdir(os.path.join(qwen_models_dir, d))]
        
        # Merge known models and existing models (deduplicate by checking if known model's basename exists)
        model_list = list(KNOWN_MODELS)
        for existing in existing_models:
            # Check if this existing folder is already covered by a known model (by basename)
            is_known = False
            for known in KNOWN_MODELS:
                if known.split("/")[-1] == existing:
                    is_known = True
                    break
            if not is_known:
                model_list.append(existing)
        
        if not model_list:
            model_list = ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]

        return {
            "required": {
                "model_path": (model_list, {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "attn_implementation": (["sdpa", "flash_attention_2", "eager"], {"default": "sdpa"}),
                "auto_download": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
                "download_source": (["ModelScope", "HuggingFace"], {"default": "ModelScope"}),
            }
        }

    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3TTS"

    def load_model(self, model_path, precision, device, attn_implementation, auto_download, download_source):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32
        
        print(f"Loading Qwen3-TTS Model from {model_path} (Precision: {precision}, Device: {device}, Attn: {attn_implementation})...")
        
        qwen_models_dir = os.path.join(folder_paths.models_dir, "Qwen3-TTS-Models")
        
        # Determine local path
        if "/" in model_path:
            local_name = model_path.split("/")[-1]
        else:
            local_name = model_path
            
        local_path = os.path.join(qwen_models_dir, local_name)
        
        # Check if model exists
        if not os.path.exists(local_path):
            if auto_download:
                print(f"Model not found at {local_path}. Attempting to download from {download_source}...")
                try:
                    self.download_model(model_path, local_path, download_source)
                except Exception as e:
                    raise RuntimeError(f"Failed to download model {model_path}: {e}")
            else:
                 raise FileNotFoundError(f"Model not found at {local_path} and auto_download is disabled. Please enable auto_download or manually place the model in {qwen_models_dir}.")
        
        print(f"Loading model from {local_path}...")
        model = Qwen3TTSModel.from_pretrained(
            local_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        return (model,)

    def download_model(self, model_id, local_dir, source):
        # Handle cases where user selected a custom folder name that isn't a repo ID
        if "/" not in model_id:
            # If it's not a repo ID, we can't download it unless we know the mapping.
            # But our INPUT_TYPES puts Repo IDs in the list. 
            # If user manually typed a folder name, we assume they know what they are doing or it exists.
            # If it doesn't exist and isn't a repo ID, we can't download.
            raise ValueError(f"Cannot download '{model_id}' because it does not look like a valid Repo ID (e.g. User/Model).")

        if source == "ModelScope":
            try:
                from modelscope import snapshot_download
            except ImportError:
                raise ImportError("ModelScope is not installed. Please run: pip install -U modelscope")
            
            print(f"Downloading {model_id} from ModelScope to {local_dir}...")
            snapshot_download(model_id, local_dir=local_dir)
            
        elif source == "HuggingFace":
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError("HuggingFace Hub is not installed. Please run: pip install -U huggingface_hub")
            
            print(f"Downloading {model_id} from HuggingFace to {local_dir}...")
            snapshot_download(repo_id=model_id, local_dir=local_dir)

class TDQwen3TTSCustomVoice:
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_all_speakers()
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "你好，我是Qwen3-TTS。"}),
                "speaker": (speakers, {"default": speakers[0] if speakers else "Vivian"}),
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

class TDQwen3TTSVoiceDesign:
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

class TDQwen3TTSVoiceClone:
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
                "x_vector_only_mode": ("BOOLEAN", {"default": False, "label_on": "Enable", "label_off": "Disable"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, ref_audio, language, ref_text="", x_vector_only_mode=False):
        if language == "Auto":
            language = None
            
        waveform = ref_audio["waveform"] # [batch, channels, samples]
        sr = ref_audio["sample_rate"]
        
        # 处理参考音频：取第一个批次，如果多声道则混合为单声道，最后转为 numpy
        ref_wav_tensor = waveform[0] # [channels, samples]
        if ref_wav_tensor.shape[0] > 1:
            print(f"检测到多声道音频，正在混合为单声道...")
            ref_wav_tensor = torch.mean(ref_wav_tensor, dim=0, keepdim=True)
            
        ref_wav_np = ref_wav_tensor.squeeze(0).numpy() # [samples]
        
        # 参考 test_model_12hz_base.py 的实现
        # 如果未启用 x_vector_only_mode，则必须提供 ref_text
        if not x_vector_only_mode and not ref_text.strip():
             raise ValueError("必须填写参考文本(ref_text)，内容为参考音频的实际说话内容。如果无法提供文本，请开启 'x_vector_only_mode'。")
            
        print(f"正在生成语音克隆 (x_vector_only_mode={x_vector_only_mode})...")
        wavs, out_sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=(ref_wav_np, sr),
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=x_vector_only_mode
        )
        
        audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": out_sr},)

NODE_CLASS_MAPPINGS = {
    "TDQwen3TTSModelLoader": TDQwen3TTSModelLoader,
    "TDQwen3TTSCustomVoice": TDQwen3TTSCustomVoice,
    "TDQwen3TTSVoiceDesign": TDQwen3TTSVoiceDesign,
    "TDQwen3TTSVoiceClone": TDQwen3TTSVoiceClone,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TDQwen3TTSModelLoader": "TD Qwen3 TTS Model Loader",
    "TDQwen3TTSCustomVoice": "TD Qwen3 TTS Custom Voice",
    "TDQwen3TTSVoiceDesign": "TD Qwen3 TTS Voice Design",
    "TDQwen3TTSVoiceClone": "TD Qwen3 TTS Voice Clone",
}
