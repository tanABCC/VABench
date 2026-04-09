import torch
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
import importlib.util

from vabench.utils import load_dimension_info
import subprocess
import os
import soundfile as sf
from vabench.memory_utils import managed_temp_file

# 全局模型缓存
_DNSMOS_MODEL_CACHE = {}

# VABench 根目录和 third_party 路径
_VABENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DNSMOS_LOCAL_PATH = os.path.join(_VABENCH_ROOT, 'third_party', 'DNS-Challenge', 'DNSMOS', 'dnsmos_local.py')

# Cache 目录用于模型文件
CACHE_DIR = os.environ.get('VABENCH_CACHE_DIR')
ComputeScore = None

def _try_load_dnsmos_from_path(file_path):
    """尝试从给定的绝对路径动态加载 dnsmos_local.py 并返回 ComputeScore。失败返回 None。"""
    try:
        if not file_path:
            return None
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return None
        spec = importlib.util.spec_from_file_location("dnsmos_local_dynamic", str(p))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "ComputeScore", None)
    except Exception as e:
        print(f"Warning: 动态加载 dnsmos_local 失败: {e}")
        return None



def _load_dns_mos_model(sample_rate=16000, primary_model_path=None, p808_model_path=None, device=None):
    global _DNSMOS_MODEL_CACHE
    cache_key = (sample_rate, primary_model_path, p808_model_path, device)
    
    if cache_key not in _DNSMOS_MODEL_CACHE:
        if ComputeScore is None:
            raise RuntimeError("DNSMOS库未正确加载，请检查DNS-Challenge/DNSMOS路径")
        if not primary_model_path or not Path(primary_model_path).exists():
            raise FileNotFoundError(f"DNSMOS主模型文件不存在: {primary_model_path}")
        if not p808_model_path or not Path(p808_model_path).exists():
            raise FileNotFoundError(f"DNSMOS P.808模型文件不存在: {p808_model_path}")
        
        try:
            model = ComputeScore(primary_model_path, p808_model_path)
            _DNSMOS_MODEL_CACHE[cache_key] = model
        except Exception as e:
            raise RuntimeError(f"DNSMOS模型加载失败: {e}")
    
    return _DNSMOS_MODEL_CACHE[cache_key]


def compute_audio_mos_single(audio_data, sr, model, device=None):
    """
    对单段音频计算 DNSMOS 分数
    返回: {'sig': float, 'bak': float, 'ovr': float, 'p808': float}
    """
    
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=-1)
    
    if isinstance(audio_data, torch.Tensor):
        if device is not None and device != torch.device('cpu'):
            try:
                audio_data = audio_data.to(device)
            except Exception as e:
                print(f"Warning: cannot move audio data to {device}, use CPU: {e}")
        audio_data = audio_data.cpu().numpy()
    
    try:
        # DNSMOS require float32 [-1, 1]
        if audio_data.max() > 1.0:
            audio_data = audio_data / 32768.0  # assume int16
        
        import tempfile
        import soundfile as sf
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_data, sr)
                tmp_path = tmp_file.name
            
            result = model(tmp_path, sr, is_personalized_MOS=False)
            return {
                'sig': round(float(result.get('SIG', 0)), 5),
                'bak': round(float(result.get('BAK', 0)), 5),
                'ovrl': round(float(result.get('OVRL', 0)), 5),
                'p808': round(float(result.get('P808_MOS', 0)), 5)
            }
        finally:
            # 安全清理临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    print(f"Warning: cannot delete temporary file {tmp_path}: {e}")
            
    except Exception as e:
        print(f"DNSMOS processing failed: {e}")
        return {'sig': 0.0, 'bak': 0.0, 'ovrl': 0.0, 'p808': 0.0}


def _process_dnsmos_audio(model, audio_path, device):
    
    with managed_temp_file(prefix='dnsmos', suffix='.wav') as safe_path:
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', audio_path,
                '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', safe_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            audio, sr = sf.read(safe_path)
            # 确保是单声道（虽然ffmpeg已经处理，但作为安全检查）
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            
            scores = compute_audio_mos_single(audio, sr, model, device)
            result = {
                'audio_path': audio_path,
                'video_results': scores['ovrl'],
                'mos_sig': scores['sig'],
                'mos_bak': scores['bak'],
                'mos_ovrl': scores['ovrl'],
                'mos_p808': scores['p808']
            }
            return result, scores
            
        except Exception as e:
            raise RuntimeError(f"处理音频文件失败 ({audio_path}): {e}")


def compute_first_dnsmos(json_dir, device, submodules_list, **kwargs):
    # 1. 从 third_party 目录加载 DNSMOS 库
    global ComputeScore
    
    _cs = _try_load_dnsmos_from_path(_DNSMOS_LOCAL_PATH)
    
    if _cs is not None:
        ComputeScore = _cs
        print(f"DNSMOS library loaded successfully from {_DNSMOS_LOCAL_PATH}")
    else:
        raise RuntimeError(
            f"DNSMOS library not found. Please ensure the file exists:\n"
            f"  {_DNSMOS_LOCAL_PATH}"
        )
        
    primary_path = submodules_list['dns_mos_primary']
    p808_path = submodules_list['dns_mos_p808']

    
    if not Path(primary_path).exists():
        raise FileNotFoundError(f"DNSMOS primary model file not found: {primary_path}")
    if not Path(p808_path).exists():
        raise FileNotFoundError(f"DNSMOS P.808 model file not found: {p808_path}")
    
    model = _load_dns_mos_model(sample_rate=16000, 
                              primary_model_path=primary_path, 
                              p808_model_path=p808_path,
                              device=device)

    video_list, audio_list, prompt_dict_ls = load_dimension_info(json_dir, dimension='first_dnsmos', lang='en')

    video_results = []

    # 逐个处理音频文件
    for audio_path in tqdm(audio_list, desc="DNSMOS"):
        if not os.path.isfile(audio_path):
            print(f"skip missing audio: {audio_path}")
            continue
        result, scores = _process_dnsmos_audio(model, audio_path, device)
        video_results.append(result)

    return None, video_results

