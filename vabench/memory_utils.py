"""
内存管理和临时文件工具函数
用于防止长循环中的显存累积和临时文件管理
"""
import gc
import os
import uuid
import tempfile
import torch
from functools import wraps
from contextlib import contextmanager


def cleanup_every_n(n=10):
    """
    装饰器：每处理 n 个样本后自动清理显存
    
    Usage:
        @cleanup_every_n(10)
        def process_samples(samples):
            for i, sample in enumerate(samples):
                # 处理样本
                yield result
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            gen = func(*args, **kwargs)
            for i, result in enumerate(gen):
                yield result
                if (i + 1) % n == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return wrapper
    return decorator


def cleanup_tensors(*tensors):

    for tensor in tensors:
        if isinstance(tensor, dict):
            for v in tensor.values():
                if isinstance(v, torch.Tensor):
                    del v
        elif isinstance(tensor, torch.Tensor):
            del tensor
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def periodic_cleanup(iteration, interval=10):

    if (iteration + 1) % interval == 0:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    return False



def get_unique_temp_path(prefix='vabench', suffix='.wav', base_dir=None, include_rank=True):
    """
    生成唯一的临时文件路径（支持分布式环境）
    
    Args:
        prefix: 文件名前缀
        suffix: 文件后缀
        base_dir: 基础目录（默认为系统临时目录）
        include_rank: 是否在文件名中包含 rank（分布式环境推荐）
    
    Returns:
        str: 唯一的临时文件路径
    
    Example:
        >>> path = get_unique_temp_path('nisqa', '.wav')
        >>> # /tmp/vabench_nisqa_tmp/nisqa_rank0_abc12345.wav
    """
    if base_dir is None:
        base_dir = os.path.join(tempfile.gettempdir(), f'{prefix}_tmp')
    
    os.makedirs(base_dir, exist_ok=True)
    
    # 生成唯一ID
    unique_id = uuid.uuid4().hex[:8]
    
    # 生成文件名（不使用 rank）
    filename = f"{prefix}_{unique_id}{suffix}"
    
    return os.path.join(base_dir, filename)


@contextmanager
def managed_temp_file(prefix='vabench', suffix='.wav', base_dir=None, include_rank=True):
    """
    上下文管理器：自动清理的临时文件
    
    Args:
        prefix: 文件名前缀
        suffix: 文件后缀
        base_dir: 基础目录
        include_rank: 是否包含 rank
    
    Yields:
        str: 临时文件路径
    
    Example:
        >>> with managed_temp_file('nisqa', '.wav') as temp_path:
        ...     # 使用 temp_path
        ...     subprocess.run(['ffmpeg', '-i', input, temp_path])
        ...     process(temp_path)
        ... # 文件自动删除
    """
    temp_path = get_unique_temp_path(prefix, suffix, base_dir, include_rank)
    try:
        yield temp_path
    finally:
        # 自动清理
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file {temp_path}: {e}")


def cleanup_temp_directory(prefix='vabench', max_age_seconds=3600):
    """
    清理旧的临时文件
    
    Args:
        prefix: 要清理的临时目录前缀
        max_age_seconds: 超过此时间的文件将被删除（默认1小时）
    
    Returns:
        int: 删除的文件数量
    """
    import time
    
    base_dir = os.path.join(tempfile.gettempdir(), f'{prefix}_tmp')
    if not os.path.exists(base_dir):
        return 0
    
    deleted_count = 0
    current_time = time.time()
    
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        try:
            # 检查文件年龄
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.unlink(file_path)
                    deleted_count += 1
        except Exception as e:
            print(f"Warning: Failed to delete {file_path}: {e}")
    
    return deleted_count

