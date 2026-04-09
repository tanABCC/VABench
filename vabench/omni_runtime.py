import os
from typing import Tuple

import torch
import random
import numpy as np


_GLOBAL_OMNI_MODEL = None
_GLOBAL_OMNI_PROCESSOR = None
_GLOBAL_OMNI_CKPT = None


def set_deterministic_mode(seed=42):
    """设置确定性模式，确保结果可重复
    
    Args:
        seed: 随机种子，默认42
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 设置CUDA确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_ckpt_path(explicit_ckpt: str | None) -> str:
    """解析权重路径：优先使用传入值；否则从环境 VABENCH_CACHE_DIR 拼接本地默认目录；
    若均不存在，回退到 Hub 名称（但会强制 local_files_only，避免联网）。"""
    if explicit_ckpt and isinstance(explicit_ckpt, str) and len(explicit_ckpt.strip()) > 0:
        return os.path.realpath(os.path.abspath(explicit_ckpt))

    cache_dir = os.environ.get("VABENCH_CACHE_DIR") or os.environ.get("VABENCH_CACHE_DIR".upper())
    if cache_dir:
        local_default = os.path.join(cache_dir, "Qwen2.5_omni_7B")
        return os.path.realpath(os.path.abspath(local_default))

    # 最后回退（不建议）：Hub 名称。注意我们将使用 local_files_only=True，缺文件会报错而不是下载
    return "Qwen/Qwen2.5-Omni-7B"


def get_omni_model_and_processor(checkpoint: str = None, **kwargs) -> Tuple[object, object]:
    """
    获取（并缓存）全局唯一的 Qwen2.5-Omni 模型与处理器。
    不同维度共享同一实例，避免重复 from_pretrained。
    """
    global _GLOBAL_OMNI_MODEL, _GLOBAL_OMNI_PROCESSOR, _GLOBAL_OMNI_CKPT

    # 设置确定性模式，确保结果可重复
    set_deterministic_mode(seed=42)

    if _GLOBAL_OMNI_MODEL is not None and _GLOBAL_OMNI_PROCESSOR is not None:
        # 若请求的权重不同，则认为需要重新加载（极少场景）；否则复用
        requested_raw = (checkpoint or os.environ.get("QWEN_OMNI_CKPT", None))
        requested = _resolve_ckpt_path(requested_raw)
        if _GLOBAL_OMNI_CKPT == requested:
            return _GLOBAL_OMNI_MODEL, _GLOBAL_OMNI_PROCESSOR
        # 权重不同，先释放，再重新加载
        release_omni()

    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    ckpt = _resolve_ckpt_path(checkpoint or os.environ.get("QWEN_OMNI_CKPT"))

    # 初始化 device_map 和 max_memory（在 if 之前，确保变量总是被定义）
    device_map = kwargs.get('device_map', None) or 'auto'
    max_memory = kwargs.get('max_memory', None)
    
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        # raise RuntimeError("No available CUDA device detected. Qwen2.5-Omni requires at least one GPU.")

        # 单进程环境：可以使用模型切片（auto + max_memory）
        if device_map == 'auto' and max_memory:
            print(f"[Model Sharding] Using device_map='auto' with max_memory={max_memory}")
        else:
            print(f"[Single Process] Using device_map='{device_map}'")

    torch_dtype = kwargs.get('torch_dtype', 'auto')
    attn_implementation = kwargs.get('attn_implementation', None)

    # 可选的视频像素全局限制
    video_max_pixels = kwargs.get('video_max_pixels', None)
    if video_max_pixels:
        os.environ['VIDEO_MAX_PIXELS'] = str(video_max_pixels)

    model_kwargs = {
        'torch_dtype': torch_dtype,
        'device_map': device_map,
    }

    if max_memory is not None:
        model_kwargs['max_memory'] = max_memory
    if attn_implementation:
        model_kwargs['attn_implementation'] = attn_implementation

    # 强制只用本地文件，避免误触发下载
    local_files_only = bool(kwargs.get('local_files_only', True))
    model_kwargs['local_files_only'] = local_files_only
    processor_kwargs = {'local_files_only': local_files_only}

    _GLOBAL_OMNI_MODEL = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        ckpt,
        **model_kwargs
    )
    _GLOBAL_OMNI_MODEL.eval()

    _GLOBAL_OMNI_PROCESSOR = Qwen2_5OmniProcessor.from_pretrained(ckpt, **processor_kwargs)
    _GLOBAL_OMNI_CKPT = ckpt
    return _GLOBAL_OMNI_MODEL, _GLOBAL_OMNI_PROCESSOR


def release_omni():
    global _GLOBAL_OMNI_MODEL, _GLOBAL_OMNI_PROCESSOR, _GLOBAL_OMNI_CKPT
    try:
        # 先删除引用，确保对象能被垃圾回收
        if _GLOBAL_OMNI_MODEL is not None:
            del _GLOBAL_OMNI_MODEL
        if _GLOBAL_OMNI_PROCESSOR is not None:
            del _GLOBAL_OMNI_PROCESSOR
        _GLOBAL_OMNI_MODEL = None
        _GLOBAL_OMNI_PROCESSOR = None
        _GLOBAL_OMNI_CKPT = None
        import gc as _gc
        # 多次垃圾回收确保彻底清理
        _gc.collect()
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


