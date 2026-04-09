import os
import numpy as np
from tqdm import tqdm
import torch

from vabench.utils import load_dimension_info
import subprocess



_HAS_NISQA = True
try:
    # note: in NISQA 2.x, the class name is nisqaModel, and the parameter is pretrained_model
    import nisqa.NISQA_model as nisqa_model_mod
except Exception:
    _HAS_NISQA = False


_NISQA_MODEL_CACHE = {}


def _load_nisqa_model(model_path: str, device: torch.device):
    """加载 NISQA 模型（全局单例，只加载一次）"""
    global _NISQA_MODEL_CACHE
    if not _HAS_NISQA:
        raise ImportError("nisqa package not installed, please install the dependency or provide an alternative scoring method.")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"NISQA model not found: {model_path}")

    # 缓存键：基于模型路径和设备（模型类型从 checkpoint 自动识别）
    cache_key = (model_path, str(device))
    
    if cache_key not in _NISQA_MODEL_CACHE:
        tmp_out_dir = os.path.join(os.path.dirname(model_path), 'nisqa_tmp_outputs')
        try:
            os.makedirs(tmp_out_dir, exist_ok=True)
        except Exception:
            pass
        args = {
            'mode': 'predict_file',
            'pretrained_model': model_path,
            'deg': '',  # 空字符串，稍后动态设置
            'output_dir': tmp_out_dir,
            'ms_channel': None,  # 必需：在 _loadDatasetsFile 中使用
            'device': str(device),  # 可能被 _getDevice 使用
        }
        model = nisqa_model_mod.nisqaModel(args)
        _NISQA_MODEL_CACHE[cache_key] = model
    
    return _NISQA_MODEL_CACHE[cache_key]


def compute_first_nisqa(json_dir, device, submodules_list, **kwargs):

    model_path = submodules_list.get('nisqa_model_path')
    if not model_path:
        raise ValueError("缺少 NISQA 模型路径: submodules_list['nisqa_model_path']")

    video_list, audio_list, prompt_dict_ls = load_dimension_info(json_dir, dimension='first_nisqa', lang='en')

    # 只加载一次模型（全局复用）
    model = _load_nisqa_model(model_path, device)

    video_results = []

    def _predict_one(path: str) -> float:
        """预测单个音频文件的 MOS 分数"""
        from .memory_utils import managed_temp_file
        
        with managed_temp_file(prefix='nisqa', suffix='.wav') as safe_path:
            # 转换为 NISQA 要求的格式（48kHz 单声道）
            try:
                subprocess.run([
                    'ffmpeg','-y','-i', path,
                    '-ac','1','-ar','48000','-c:a','pcm_s16le', safe_path
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                raise RuntimeError(f'FFmpeg 重采样转码失败: {path}') from e

            model.args['deg'] = safe_path
            model._loadDatasetsFile()

            # NISQA predict() 返回 pandas DataFrame
            df = model.predict()
            return float(df['mos_pred'].iloc[0])
    
    # 逐个处理模式
    pbar = tqdm(audio_list, desc="NISQA")
    for audio_path in pbar:
            mos = _predict_one(audio_path)
            video_results.append({
                'audio_path': audio_path,
                'video_results': round(float(mos), 5),
                'nisqa_mos': round(float(mos), 5)
            })

    return None, video_results

