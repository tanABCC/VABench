import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import os

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
except ImportError:
    pass

from vabench.utils import load_dimension_info
from third_party.ViCLIP.viclip import ViCLIP


def _get_clip_model(device, submodules_list):
    # 使用 ViCLIP 模型
    model_path = submodules_list.get('second_viclip_model_name')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ViCLIP model file not found: {model_path}")

    import sys
    import types
    import easydict
    from easydict import EasyDict  # noqa: F401
    if 'utils.easydict' not in sys.modules:
        # 确保存在顶级 utils 包占位
        sys.modules.setdefault('utils', types.ModuleType('utils'))
        # 将 easydict 作为 utils.easydict 暴露
        sys.modules['utils.easydict'] = easydict
    import torch as _torch
    # PyTorch>=2.6 提供 add_safe_globals，用于安全加载允许的类型
    if hasattr(_torch, 'serialization') and hasattr(_torch.serialization, 'add_safe_globals'):
        _torch.serialization.add_safe_globals([EasyDict])

    model = ViCLIP(pretrain=model_path, freeze_text=True)
    model = model.to(device)
    model.eval()
    
    def preprocess(image):
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),  
            transforms.CenterCrop(224),  
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                               std=(0.26862954, 0.26130258, 0.27577711))
        ])
        return transform(image)
    
    return model, preprocess


def _calc_clip_score_for_video(video_path, prompt, device, model, preprocess, num_frames=8):

    def _frame_transform(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = preprocess(frame)
        return frame

    import decord
    from decord import VideoReader
    decord.bridge.set_bridge('native')
    vr = VideoReader(video_path, num_threads=1)
    vlen = len(vr)
    if vlen == 0:
        return 0.0
    num = min(num_frames, vlen)
    
    # 将视频分成 num 个区间，从每个区间的中点采样
    intervals = np.linspace(start=0, stop=vlen, num=num + 1).astype(int)
    ranges = [(intervals[idx], intervals[idx + 1] - 1) for idx in range(len(intervals) - 1)]
    indices = [(x[0] + x[1]) // 2 for x in ranges]  # 取每个区间的中点
    
    frames_np = vr.get_batch(indices).asnumpy()  # (T, H, W, C) uint8

    # 预处理帧
    frames_tensor = []
    for f in frames_np:
        frames_tensor.append(_frame_transform(f))
    frames_tensor = torch.stack(frames_tensor, dim=0).to(device)  # (T, 3, H, W)
    
    # ViCLIP 的 encode_vision 会在内部将 (B, T, C, H, W) 变换为 (B, C, T, H, W)
    # 因此这里保持 (B, T, C, H, W) 即可，避免维度被二次交换导致通道错误
    frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, 3, H, W)

    with torch.no_grad():
            text_features = model.encode_text(prompt)  # (1, D)
            image_features = model.encode_vision(frames_tensor, test=True)  # (1, D)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            cos_sim = (image_features @ text_features.T).squeeze(-1)  # [1]
            clip_score = cos_sim.item()
            
            return clip_score


def compute_second_viclip(json_dir, device, submodules_list, **kwargs):

    try:
        video_list, _, prompt_dict_ls = load_dimension_info(json_dir, dimension='second_viclip', lang='en')

        video_to_prompt = {}
        video_to_prompt_vision = {}
        for pd in (prompt_dict_ls or []):
            ptxt = pd.get('prompt', '')
            prompt_vision = pd.get('prompt_vision', '')
            for vp in (pd.get('video_list') or []):
                video_to_prompt[vp] = ptxt
                video_to_prompt_vision[vp] = prompt_vision

        num_frames = kwargs.get('num_frames', 8)
        model, preprocess = _get_clip_model(device, submodules_list)
        
        video_results = []

        for i, video_path in enumerate(tqdm(video_list)):
            prompt = video_to_prompt.get(video_path, '')
            prompt_vision = video_to_prompt_vision.get(video_path, '')
            
            clip_score = _calc_clip_score_for_video(
                video_path, prompt_vision, device, model, preprocess, num_frames
            )
            
            video_results.append({
                'video_path': video_path,
                'video_results': round(float(clip_score), 5),
                'prompt': prompt
            })
                

        return None, video_results
        
    except Exception as e:
        print(f"Error in second_viclip evaluation: {e}")
        return None, []
