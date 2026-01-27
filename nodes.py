import os
import sys
import json
import torch
import torchaudio
import random
import string
import numpy as np
import folder_paths
import zlib
import ast

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
except ImportError as e:
    print("\n" + "!" * 50)
    print("Warning / 警告:")
    print("Failed to import Qwen3TTS dependencies. / 无法导入 Qwen3TTS 依赖项。")
    print(f"Error details / 错误详情: {e}")
    print("\nPlease try the following steps to fix this issue / 请尝试以下步骤解决此问题:")
    print("1. Open terminal in the node directory / 在节点目录下打开终端:")
    print(f"   cd {os.path.dirname(os.path.abspath(__file__))}")
    print("2. Install dependencies / 安装依赖:")
    print("   pip install -r requirements.txt")
    print("3. If using portable ComfyUI / 如果使用便携版 ComfyUI:")
    print("   path/to/python_embeded/python.exe -m pip install -r requirements.txt")
    print("!" * 50 + "\n")
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
                                # Capitalize keys for better UI display (e.g. "vivian" -> "Vivian")
                                # This helps match default values like "Vivian" in workflows
                                keys = [k.capitalize() for k in config["talker_config"]["spk_id"].keys()]
                                speakers.update(keys)
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
        audio_tensor = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": sr},)

class TDQwen3TTSDefineSpeaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "name": ("STRING", {"default": "MySpeaker"}),
            }
        }

    RETURN_TYPES = ("SPEAKER",)
    RETURN_NAMES = ("speaker",)
    FUNCTION = "define_speaker"
    CATEGORY = "Qwen3TTS"

    def define_speaker(self, audio, name):
        # Remove existing with same name to avoid duplicates/allow override
        name_clean = name.strip()
        print(f"Defined speaker '{name_clean}'")
        return ({"name": name_clean, "audio": audio},)

class TDQwen3TTSBatchGenerateSpeaker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "speakers_config": ("STRING", {"default": "[]", "multiline": True}), # JSON string (Widget or Input)
            }
        }

    RETURN_TYPES = ("SPEAKER_LIST",)
    RETURN_NAMES = ("speakers",)
    FUNCTION = "batch_generate"
    CATEGORY = "Qwen3TTS"

    def batch_generate(self, model, speakers_config):
        config_list = []
        if isinstance(speakers_config, list):
             config_list = speakers_config
        elif isinstance(speakers_config, str):
             try:
                config_list = json.loads(speakers_config)
             except json.JSONDecodeError:
                # Try parsing as Python literal (single quotes)
                try:
                    config_list = ast.literal_eval(speakers_config)
                    if not isinstance(config_list, list):
                        print("Error: speakers_config evaluated to non-list.")
                        return ([],)
                except (ValueError, SyntaxError):
                    print("Error parsing speakers config. Ensure it is valid JSON or Python List.")
                    return ([],)
        else:
             print(f"Error: speakers_config must be a JSON string or a list, got {type(speakers_config)}")
             return ([],)
        
        generated_speakers = []
        batch_outputs = [] # For UI preview
        
        print(f"Batch generating {len(config_list)} speakers...")
        
        output_dir = folder_paths.get_temp_directory()
        
        for item in config_list:
            name = item.get("name", "").strip()
            instruct = item.get("instruct", "").strip()
            text = item.get("text", "").strip()
            
            if not name or not instruct or not text:
                print(f"Skipping invalid speaker config: {item}")
                continue
                
            print(f"Generating speaker '{name}' with instruct: '{instruct}'...")
            try:
                # Use Voice Design logic
                wavs, sr = model.generate_voice_design(
                    text=text,
                    instruct=instruct,
                    language=None # Auto
                )
                
                # Check if wavs is empty or None
                if not wavs or len(wavs) == 0:
                    print(f"Warning: No audio generated for speaker '{name}'")
                    continue
                    
                # Convert to ComfyUI AUDIO format:
                wav_data = wavs[0]
                
                # Ensure it's a tensor
                if isinstance(wav_data, np.ndarray):
                    wav_tensor = torch.from_numpy(wav_data)
                else:
                    wav_tensor = wav_data
                    
                # Ensure dimensions: [batch, channels, samples]
                if wav_tensor.ndim == 1:
                     wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, samples]
                elif wav_tensor.ndim == 2:
                     wav_tensor = wav_tensor.unsqueeze(0) # [1, channels, samples]
                
                audio_dict = {"waveform": wav_tensor, "sample_rate": sr}
                
                generated_speakers.append({"name": name, "audio": audio_dict, "instruct": instruct})
                
                # Save preview audio
                filename = f"Qwen3TTS_Batch_{name}_{''.join(random.choices(string.ascii_letters + string.digits, k=8))}.wav"
                filepath = os.path.join(output_dir, filename)
                
                # Save using torchaudio
                # torchaudio.save expects [channels, samples]
                # wav_tensor is [1, 1, samples], so squeeze(0) -> [1, samples]
                torchaudio.save(filepath, wav_tensor.squeeze(0), sr)
                
                batch_outputs.append({
                    "name": name,
                    "filename": filename,
                    "subfolder": "",
                    "type": "temp"
                })
                
                print(f"Successfully generated speaker '{name}'")
                
            except Exception as e:
                print(f"Failed to generate speaker '{name}': {e}")
                import traceback
                traceback.print_exc()
                
        return {"ui": {"batch_outputs": batch_outputs}, "result": (generated_speakers,)}

class TDQwen3TTSMultiDialog:
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_all_speakers()
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Vivian: 你好！\nRyan: 你好，今天天气真不错。\nRoleA: 我是自定义角色A。\nVivian: 欢迎你！"}),
                "interval": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "speaker_list": ("SPEAKER_LIST",), # New input for batch speakers
                "speaker_1": ("SPEAKER",), # Add explicit initial slot to ensure visibility before JS loads
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, interval=0.5, seed=0, speaker_list=None, **kwargs):
        
        language = None # Auto identify from audio
        
        # 构建自定义角色字典
        custom_roles = {}
        
        # Add from speaker_list (batch input)
        if speaker_list:
            # Handle case where speaker_list might be a single dictionary (if connected incorrectly but passed)
            if isinstance(speaker_list, dict):
                 if "name" in speaker_list:
                      custom_roles[speaker_list["name"]] = speaker_list
            elif isinstance(speaker_list, list):
                for spk in speaker_list:
                    if isinstance(spk, dict) and "name" in spk:
                        # Store full speaker object (name, audio, instruct)
                        custom_roles[spk["name"]] = spk
        
        # Add from individual inputs (kwargs)
        for key, value in kwargs.items():
            if key.startswith("speaker_") and value is not None:
                custom_roles[value["name"]] = value
        
        if custom_roles:
            print(f"Loaded {len(custom_roles)} custom speakers: {list(custom_roles.keys())}")
        else:
            print("No custom speakers loaded.")

        # Create lowercase mapping for custom roles to support case-insensitive matching
        custom_roles_lower = {k.lower(): k for k in custom_roles.keys()}
            
        # 1. 解析文本
        dialog_items = []
        lines = text.strip().split('\n')
        
        # Default to the first custom speaker if available, otherwise "Vivian"
        if custom_roles:
            current_speaker = list(custom_roles.keys())[0]
        else:
            current_speaker = "Vivian"
            
        available_speakers = get_all_speakers()
        # Create a lowercase mapping for case-insensitive lookup
        available_speakers_lower = {s.lower(): s for s in available_speakers}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 尝试解析 "Speaker: Text" 或 "[Speaker]: Text"
            if ":" in line or "：" in line:
                # 统一冒号
                line_fixed = line.replace("：", ":")
                parts = line_fixed.split(":", 1)
                possible_speaker = parts[0].strip().replace("[", "").replace("]", "")
                content = parts[1].strip()
                
                # 优先级：自定义角色 > 预设角色
                
                # 1. Exact match custom
                if possible_speaker in custom_roles:
                    current_speaker = possible_speaker
                    dialog_items.append((current_speaker, content))
                # 2. Case-insensitive match custom
                elif possible_speaker.lower() in custom_roles_lower:
                    current_speaker = custom_roles_lower[possible_speaker.lower()]
                    dialog_items.append((current_speaker, content))
                # 3. Exact match preset
                elif possible_speaker in available_speakers:
                    current_speaker = possible_speaker
                    dialog_items.append((current_speaker, content))
                # 4. Case-insensitive match preset
                elif possible_speaker.lower() in available_speakers_lower:
                    current_speaker = available_speakers_lower[possible_speaker.lower()]
                    dialog_items.append((current_speaker, content))
                else:
                    # 如果不是已知角色，归为当前 speaker 的延续
                    print(f"Warning: Speaker '{possible_speaker}' not found in custom or preset speakers. Treating as text for '{current_speaker}'.")
                    dialog_items.append((current_speaker, line))
            else:
                # 没有冒号，认为是当前角色的延续
                dialog_items.append((current_speaker, line))
        
        # 2. 生成音频
        all_wavs = []
        sample_rate = None
        
        print(f"开始生成多人对话，共 {len(dialog_items)} 个片段...")
        
        # Detect model type
        model_type = getattr(model.model, "tts_model_type", "unknown")
        print(f"Detected model type: {model_type}")

        for i, (speaker, content) in enumerate(dialog_items):
            if not content:
                continue
            
            # Set deterministic seed based on global seed + speaker name hash
            # This ensures that the same speaker always gets the same random seed for generation
            # Use & 0xffffffff to ensure adler32 result is treated consistently
            speaker_seed = (seed + (zlib.adler32(speaker.encode('utf-8')) & 0xffffffff)) % 0xffffffffffffffff
            torch.manual_seed(speaker_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(speaker_seed)
                
            print(f"[{i+1}/{len(dialog_items)}] 正在为 {speaker} 生成: {content[:20]}...")
            try:
                wav = None
                sr = None
                
                # Check if it is a custom role
                if speaker in custom_roles:
                    role_data = custom_roles[speaker]
                    
                    if model_type == "voice_design":
                        # For Voice Design model, we MUST use instruct
                        instruct = role_data.get("instruct")
                        if instruct:
                            print(f"  - 使用 Voice Design (Instruct: {instruct[:20]}...)...")
                            wavs, sr = model.generate_voice_design(
                                text=content,
                                instruct=instruct,
                                language=language
                            )
                            wav = wavs[0]
                        else:
                            print(f"Error: Speaker '{speaker}' has no 'instruct' and model is Voice Design. Skipping.")
                            continue
                            
                    elif model_type == "base":
                         # For Base model, use Voice Clone (requires audio)
                         if "audio" in role_data and role_data["audio"]:
                            ref_audio = role_data["audio"]
                            waveform = ref_audio["waveform"]
                            ref_sr = ref_audio["sample_rate"]
                            
                            # Prepare ref audio
                            ref_wav_tensor = waveform[0]
                            if ref_wav_tensor.shape[0] > 1:
                                ref_wav_tensor = torch.mean(ref_wav_tensor, dim=0, keepdim=True)
                            ref_wav_np = ref_wav_tensor.squeeze(0).numpy()
                            
                            print(f"  - 使用 Voice Clone (Base Model)...")
                            wavs, sr = model.generate_voice_clone(
                                text=content,
                                language=language,
                                ref_audio=(ref_wav_np, ref_sr),
                                ref_text=None,
                                x_vector_only_mode=True
                            )
                            wav = wavs[0]
                         else:
                            print(f"Error: Speaker '{speaker}' has no 'audio' and model is Base. Skipping.")
                            continue
                    
                    elif model_type == "custom_voice":
                         # Try fallback to preset if name matches
                         supported = model.get_supported_speakers()
                         if supported and speaker.lower() in [s.lower() for s in supported]:
                              print(f"  - Fallback to Preset Speaker '{speaker}'...")
                              wavs, sr = model.generate_custom_voice(
                                    text=content,
                                    speaker=speaker,
                                    language=language
                                )
                              wav = wavs[0]
                         else:
                              print(f"Error: CustomVoice model cannot generate custom speaker '{speaker}'.")
                              continue
                    else:
                         print(f"Error: Unknown model type '{model_type}'. Cannot determine generation method.")
                         continue

                else:
                    # Preset Speaker (not in custom_roles)
                    if model_type == "custom_voice":
                        print(f"  - 使用 Custom Voice (Preset: {speaker})...")
                        wavs, sr = model.generate_custom_voice(
                            text=content,
                            speaker=speaker,
                            language=language,
                            instruct=None 
                        )
                        wav = wavs[0]
                    else:
                        print(f"Error: Model type '{model_type}' does not support preset speaker '{speaker}'. Please define '{speaker}' as a custom role with an 'instruct' (for Voice Design) or 'audio' (for Base).")
                        continue
                
                if sample_rate is None:
                    sample_rate = sr
                elif sample_rate != sr:
                    print(f"Warning: Sample rate mismatch {sr} vs {sample_rate}, skipping resampling for now (assuming compatible).")
                
                all_wavs.append(wav)
                
                # 添加间隔
                if interval > 0 and i < len(dialog_items) - 1:
                    silence_samples = int(interval * sr)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    all_wavs.append(silence)
                    
            except Exception as e:
                print(f"Error generating segment for {speaker}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_wavs:
            # 返回静音或报错
            print("Warning: No audio generated.")
            # 返回 1秒静音
            sr = sample_rate if sample_rate else 24000
            silence = np.zeros(sr, dtype=np.float32)
            audio_tensor = torch.from_numpy(silence).unsqueeze(0).unsqueeze(0)
            return ({"waveform": audio_tensor, "sample_rate": sr},)
            
        # 3. 拼接
        final_wav = np.concatenate(all_wavs)
        
        # 转换为 tensor [1, 1, samples]
        audio_tensor = torch.from_numpy(final_wav).unsqueeze(0).unsqueeze(0)
        
        return ({"waveform": audio_tensor, "sample_rate": sample_rate},)

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

class TDParseJson:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_string": ("STRING", {"multiline": True, "default": "{}"}),
                "key": ("STRING", {"default": "key"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value_json",)
    FUNCTION = "parse_json"
    CATEGORY = "Qwen3TTS"

    def parse_json(self, json_string, key):
        try:
            data = json.loads(json_string)
        except Exception:
            try:
                data = ast.literal_eval(json_string)
            except Exception:
                print("TDParseJson: Failed to parse input string")
                raise ValueError("TDParseJson: 传入的数据格式不正确，请仔细检查JSON数据格式。")

        val = None
        if isinstance(data, dict):
            val = data.get(key, None)
            if val is None and key not in data:
                 print(f"TDParseJson: Key '{key}' not found in dict")
                 return ("{}",)
        elif isinstance(data, list):
            try:
                idx = int(key)
                if 0 <= idx < len(data):
                    val = data[idx]
                else:
                    print(f"TDParseJson: Index {idx} out of bounds")
                    return ("{}",)
            except ValueError:
                print(f"TDParseJson: Key '{key}' is not a valid integer index for list")
                return ("{}",)
        else:
             print("TDParseJson: Input is not a dict or list")
             return ("{}",)

        # Convert back to JSON string
        try:
            res = json.dumps(val, ensure_ascii=False)
            return (res,)
        except Exception as e:
            print(f"TDParseJson: Failed to serialize result: {e}")
            return ("{}",)

NODE_CLASS_MAPPINGS = {
    "TDQwen3TTSModelLoader": TDQwen3TTSModelLoader,
    "TDQwen3TTSCustomVoice": TDQwen3TTSCustomVoice,
    "TDQwen3TTSMultiDialog": TDQwen3TTSMultiDialog,
    "TDQwen3TTSDefineSpeaker": TDQwen3TTSDefineSpeaker,
    "TDQwen3TTSBatchGenerateSpeaker": TDQwen3TTSBatchGenerateSpeaker,
    "TDQwen3TTSVoiceDesign": TDQwen3TTSVoiceDesign,
    "TDQwen3TTSVoiceClone": TDQwen3TTSVoiceClone,
    "TDParseJson": TDParseJson,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TDQwen3TTSModelLoader": "TD Qwen3 TTS Model Loader",
    "TDQwen3TTSCustomVoice": "TD Qwen3 TTS Custom Voice",
    "TDQwen3TTSMultiDialog": "TD Qwen3 TTS Multi Dialog",
    "TDQwen3TTSDefineSpeaker": "TD Qwen3 TTS Define Speaker",
    "TDQwen3TTSBatchGenerateSpeaker": "TD Qwen3 TTS Batch Generate Speaker",
    "TDQwen3TTSVoiceDesign": "TD Qwen3 TTS Voice Design",
    "TDQwen3TTSVoiceClone": "TD Qwen3 TTS Voice Clone",
    "TDParseJson": "TD Parse Json",
}
