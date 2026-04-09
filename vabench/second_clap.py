import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
import os
from transformers import AutoProcessor, ClapModel, ClapProcessor, ClapFeatureExtractor

from vabench.utils import load_dimension_info

CACHE_DIR = os.environ.get('VABENCH_CACHE_DIR')
cache_dir = os.path.join(CACHE_DIR, 'clap')


def _get_clap_model(device, submodules_list):
    model_name = submodules_list.get('second_clap_model_name')
    os.makedirs(cache_dir, exist_ok=True)
    is_local_dir = os.path.isdir(model_name)
    try:
        model = ClapModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=is_local_dir
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=is_local_dir
        )
    except Exception:
        raise Exception(f"Error loading CLAP model from submodules_list: {submodules_list}")
    model.to(device=device)
    return model, processor


def _calc_clap_score_for_audio(audio_path, prompt, device, model, processor):
    import subprocess
    import os
    import soundfile as sf
    from vabench.memory_utils import managed_temp_file

    try:
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        with managed_temp_file(prefix='clap', suffix='.wav') as safe_path:
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', audio_path,
                    '-ac', '1', '-ar', '48000', '-c:a', 'pcm_s16le', safe_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"FFmpeg failed for {audio_path}: {e}")
                return 0.0

            # Step 2: 读取音频
            try:
                audio_data, sample_rate = sf.read(safe_path, dtype='float32')
                if sample_rate != 48000:
                    return 0.0
                if audio_data.ndim == 2:
                    audio_data = audio_data[:, 0]  # 取第一声道
                elif audio_data.ndim > 2:
                    return 0.0
            except Exception as e:
                print(f"SoundFile read failed for {safe_path}: {e}")
                return 0.0
            
            # 假设 audio_data 是 (N,) 的 numpy array，采样率 48kHz
            # MAX_LEN = 48000 * 10
            # if len(audio_data) > MAX_LEN:
            #     audio_data = audio_data[:MAX_LEN]  # 确定性截断
            inputs = processor(
                text=[prompt],
                audio=audio_data,
                return_tensors="pt",
                padding=True,
                sampling_rate=48000
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}  

            with torch.no_grad():
                outputs = model(**inputs)

            score = cos(outputs.text_embeds, outputs.audio_embeds)
            return float(score.item())

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return 0.0


def compute_second_clap(json_dir, device, submodules_list, **kwargs):

    try:
        video_list, audio_list, prompt_dict_ls = load_dimension_info(json_dir, dimension='second_clap', lang='en')

        # 构建路径映射，避免分发后索引错配
        video_to_prompt = {}
        video_to_audio = {}
        video_to_prompt_audio = {}
        
        for pd in (prompt_dict_ls or []):
            ptxt = pd.get('prompt', '')
            prompt_audio = pd.get('prompt_audio', '')
            vls = (pd.get('video_list') or [])
            als = (pd.get('audio_list') or [])
            
            # 为每个视频映射 prompt
            for vp in vls:
                video_to_prompt[vp] = ptxt
                video_to_prompt_audio[vp] = prompt_audio
            for vp, ap in zip(vls, als):
                video_to_audio[vp] = ap


        model, processor = _get_clap_model(device, submodules_list)
        
        video_results = []
        for i, video_path in enumerate(tqdm(video_list)):
            try:
                # 获取配对的音频路径
                audio_path = video_to_audio.get(video_path)
                prompt = video_to_prompt.get(video_path, '')
                prompt_audio = video_to_prompt_audio.get(video_path, '')
                clap_score = _calc_clap_score_for_audio(audio_path, prompt_audio, device, model, processor)
                
                video_results.append({
                    'video_path': video_path,
                    'video_results': round(float(clap_score), 5),
                    'prompt': prompt
                })
                
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                video_results.append({
                    'video_path': video_path,
                    'video_results': 0.0,
                    'prompt': ""
                })

        return None, video_results
        
    except Exception as e:
        print(f"Error in second_clap evaluation: {e}")
        return None, []
