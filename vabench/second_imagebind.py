import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import sys
import os

# 添加 ImageBind 路径
VABENCH_CACHE_DIR = os.environ.get('VABENCH_CACHE_DIR')
IMAGEBIND_PATH = os.path.join(VABENCH_CACHE_DIR, 'imagebind')

# ========== Monkey Patch for pytorchvideo decord bug ==========
# 重要：必须在导入 imagebind 之前应用 patch！
# 修复 pytorchvideo 在多次调用后返回 NDArray 或 numpy.ndarray 而不是 Tensor 的问题
def _patch_pytorchvideo_decord():
    """
    Monkey patch pytorchvideo 的 encoded_video_decord 模块
    彻底修复 decord 在多次调用后返回 NDArray 的问题
    
    基于源代码分析，问题出在 get_clip 方法第 193 行：
    video = video.to(torch.float32)  # 当 video 是 NDArray 时会报错
    """
    try:
        import pytorchvideo.data.encoded_video_decord as decord_module
        from pytorchvideo.data.utils import thwc_to_cthw
        import torch
        import numpy as np
        import math
        import logging
        
        logger = logging.getLogger(__name__)
        
        # 完全替换 get_clip 方法，确保 NDArray 转 Tensor
        def patched_get_clip(self, start_sec: float, end_sec: float):
            """
            修复版本的 get_clip，在调用 .to() 之前确保 video 是 Tensor
            """
            if start_sec > end_sec or start_sec > self._duration:
                raise RuntimeError(
                    f"Incorrect time window for Decord decoding for video: {self._video_name}."
                )

            start_idx = math.ceil(self._fps * start_sec)
            end_idx = math.ceil(self._fps * end_sec)
            end_idx = min(end_idx, len(self._av_reader))
            frame_idxs = list(range(start_idx, end_idx))
            audio = None

            try:
                outputs = self._av_reader.get_batch(frame_idxs)
            except Exception as e:
                logger.debug(f"Failed to decode video with Decord: {self._video_name}. {e}")
                raise e

            if self._decode_audio:
                audio, video = outputs
                if audio is not None:
                    audio = list(audio)
                    audio = torch.cat(audio, dim=1)
                    audio = torch.flatten(audio)
                    audio = audio.to(torch.float32)
            else:
                video = outputs

            # 关键修复：在调用 .to() 之前，确保 video 是 Tensor
            if video is not None:
                # 检查是否是 NDArray 或 numpy.ndarray
                if hasattr(video, '__class__'):
                    class_name = video.__class__.__name__
                    if 'NDArray' in class_name:
                        # decord.NDArray -> torch.Tensor
                        video = torch.from_numpy(video.asnumpy())
                    elif isinstance(video, np.ndarray) and not isinstance(video, torch.Tensor):
                        # numpy.ndarray -> torch.Tensor
                        video = torch.from_numpy(video)
                
                # 现在可以安全地调用 .to() 了
                video = video.to(torch.float32)
                video = thwc_to_cthw(video)

            return {
                "video": video,
                "audio": audio,
            }
        
        # 应用 patch
        decord_module.EncodedVideoDecord.get_clip = patched_get_clip
        
        # 确保 sys.modules 中的引用也被更新
        import sys
        module_name = 'pytorchvideo.data.encoded_video_decord'
        if module_name in sys.modules:
            sys.modules[module_name].EncodedVideoDecord.get_clip = patched_get_clip
        
        return True
    except Exception as e:
        print(f"Warning: Failed to patch pytorchvideo: {e}")
        import traceback
        traceback.print_exc()
        return False

# 应用 patch - 只 patch pytorchvideo 内部，不影响其他模块
_patch_applied = _patch_pytorchvideo_decord()
if _patch_applied:
    # 验证 patch 是否真的生效
    try:
        import pytorchvideo.data.encoded_video_decord as test_module
        print(f"✓ Pytorchvideo decord patch applied successfully")
        print(f"  - get_clip method: {test_module.EncodedVideoDecord.get_clip.__name__}")
        print(f"  - Module location: {test_module.__file__}")
    except Exception as e:
        print(f"✗ Failed to verify patch: {e}")
else:
    print(f"✗ Pytorchvideo decord patch failed to apply")
    

# ========== End Monkey Patch ==========

from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from vabench.utils import load_dimension_info


def _get_imagebind_model(device, submodules_list):

    ckpt_path = submodules_list.get('second_imagebind_model_name')
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ImageBind model file not found: {ckpt_path}")

    model = imagebind_model.imagebind_huge(pretrained=False)
    state = torch.load(ckpt_path, map_location='cpu')
    # 兼容可能包含 'state_dict' 的保存格式
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"ImageBind local ckpt loaded with missing={len(missing)}, unexpected={len(unexpected)}")

    model.eval()
    model.to(device)
    return model


def _calc_imagebind_scores_for_single(
    video_path: str,
    audio_path: str,
    prompt_vision: str,
    prompt_audio: str,
    device,
    model,
):

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    # 合并文本提示（视觉 + 音频），顺序很重要：前者用于与视频对齐，后者用于与音频对齐
    all_prompts = [prompt_vision, prompt_audio]
    try:
        # 强制清理 decord 缓存，避免 NDArray 类型错误
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(all_prompts, device),
            ModalityType.VISION: data.load_and_transform_video_data([video_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([audio_path], device),
        }
        with torch.no_grad():
            embeddings = model(inputs)
        text_embed, audio_text_embed = embeddings[ModalityType.TEXT].chunk(2, dim=0)
        video_embed = embeddings[ModalityType.VISION]
        audio_embed = embeddings[ModalityType.AUDIO]
        sim_tv = cos(text_embed, video_embed).item()
        sim_ta = cos(audio_text_embed, audio_embed).item()
        sim_av = cos(video_embed, audio_embed).item()
        
        # 清理输入数据
        del inputs, embeddings, text_embed, audio_text_embed, video_embed, audio_embed
        gc.collect()
        
        return float(sim_tv), float(sim_ta), float(sim_av)
    except Exception as e:
        print(f"Error processing sample {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, 0.0, 0.0


def compute_second_imagebind(json_dir, device, submodules_list, **kwargs):
    
    try:
        video_list, audio_list, prompt_dict_ls = load_dimension_info(
            json_dir, 
            dimension='second_imagebind', 
            lang='en'
        )
        
        video_to_prompt = {}
        video_to_audio = {}
        video_to_prompt_vision = {}
        video_to_prompt_audio = {}
        
        for pd in (prompt_dict_ls or []):
            prompt = pd.get('prompt', '')
            prompt_vision = pd.get('prompt_vision', prompt)  # 默认使用 prompt
            prompt_audio = pd.get('prompt_audio', prompt)   # 默认使用 prompt
            
            vls = (pd.get('video_list') or [])
            als = (pd.get('audio_list') or [])
            
            # 为每个视频建立 prompt 映射
            for vp in vls:
                video_to_prompt[vp] = prompt
                video_to_prompt_vision[vp] = prompt_vision
                video_to_prompt_audio[vp] = prompt_audio
            for vp, ap in zip(vls, als):
                video_to_audio[vp] = ap
        
        # 加载模型
        model = _get_imagebind_model(device, submodules_list)
        
        video_results = []
        
        # 逐样本计算分数
        for video_path in tqdm(video_list, desc='ImageBind (per-sample)'):
            # 获取配对的音频路径
            audio_path = video_to_audio.get(video_path)
            prompt = video_to_prompt.get(video_path, '')
            prompt_vision = video_to_prompt_vision.get(video_path, prompt)
            prompt_audio = video_to_prompt_audio.get(video_path, prompt)
            sim_tv, sim_ta, sim_av = _calc_imagebind_scores_for_single(
                video_path, audio_path, prompt_vision, prompt_audio, device, model
            )
            sim_tv = round(float(sim_tv), 5)
            sim_ta = round(float(sim_ta), 5)
            sim_av = round(float(sim_av), 5)
            
            video_results.append({
                'video_path': video_path,
                'video_results': sim_av,
                'sim_tv': sim_tv,
                'sim_ta': sim_ta,
                'sim_av': sim_av,
                'prompt': prompt,
            })
        
        try:
            del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup ImageBind resources: {cleanup_error}")
        
        return None, video_results
        
    except Exception as e:
        print(f"Error in second_imagebind evaluation: {e}")
        import traceback
        traceback.print_exc()
        # 确保即使出错也清理资源
        try:
            if 'model' in locals():
                del model
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass
        return None, []
