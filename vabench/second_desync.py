import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

# 为 PyTorch 2.6+ 注册 OmegaConf 的 DictConfig 为安全全局变量
# 这样 torch.load 在 weights_only=True 时也能加载包含 DictConfig 的 checkpoint
try:
    torch.serialization.add_safe_globals([DictConfig])
except Exception:
    pass  # 如果 PyTorch 版本较旧，没有 add_safe_globals，忽略即可

# 从 third_party 目录加载 Synchformer 代码
_VABENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # VABench 根目录
_SYNCH_ROOT = os.path.join(_VABENCH_ROOT, 'third_party', 'Synchformer')
_NEEDED_SUBDIRS = [
    _SYNCH_ROOT,
    os.path.join(_SYNCH_ROOT, 'model', 'modules', 'feat_extractors', 'visual', 'motionformer_src'),
    os.path.join(_SYNCH_ROOT, 'model', 'modules', 'feat_extractors', 'train_clip_src'),
    os.path.join(_SYNCH_ROOT, 'utils'),
]
for _p in _NEEDED_SUBDIRS:
    if os.path.isdir(_p):
        if _p in sys.path:
            sys.path.remove(_p)
        sys.path.insert(0, _p)

# Cache 目录仍然用于存储配置和权重文件
_CACHE_DIR = os.environ.get('VABENCH_CACHE_DIR', os.path.join(os.path.expanduser('~'), '.cache', 'vabench'))

from vabench.utils import load_dimension_info

# Synchformer 相关模块延迟导入（避免与 vabench.utils 冲突）
def _lazy_import_synchformer():
    """延迟导入 Synchformer 模块，使用绝对路径避免包名冲突"""
    global get_model, get_transforms, prepare_inputs, get_video_and_audio
    global make_class_grid, quantize_offset, instantiate_from_config
    
    try:
        import importlib.util
        
        # 使用 third_party 目录中的 Synchformer
        synch_root = _SYNCH_ROOT
        
        # 临时调整sys.path，确保Synchformer的utils包被优先找到
        original_sys_path = sys.path.copy()
        
        # 将Synchformer根目录和所有子目录放到sys.path最前面
        needed_paths = [
            synch_root,
            os.path.join(synch_root, 'model', 'modules', 'feat_extractors', 'visual'),  # motionformer_src的父目录
            os.path.join(synch_root, 'model', 'modules', 'feat_extractors', 'train_clip_src'),
            os.path.join(synch_root, 'utils'),
        ]
        
        for path in needed_paths:
            if os.path.isdir(path):
                if path in sys.path:
                    sys.path.remove(path)
                sys.path.insert(0, path)
            
        try:
            # 首先导入 utils.utils 并将其注册到 sys.modules 中，这样其他模块就能找到它了
            utils_utils_path = os.path.join(synch_root, 'utils', 'utils.py')
            spec = importlib.util.spec_from_file_location("synch_utils_utils", utils_utils_path)
            utils_utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils_utils_module)
            instantiate_from_config = utils_utils_module.instantiate_from_config
            
            # 将 utils.utils 模块注册到 sys.modules 中，使其他模块能够通过 import utils.utils 找到它
            sys.modules['utils.utils'] = utils_utils_module
            utils_init_spec = importlib.util.spec_from_file_location("utils", os.path.join(synch_root, 'utils', '__init__.py'))
            utils_init_module = importlib.util.module_from_spec(utils_init_spec)
            utils_init_spec.loader.exec_module(utils_init_module)
            sys.modules['utils'] = utils_init_module
            sys.modules['utils'].utils = utils_utils_module
            
            # 导入并注册 motionformer_src 包
            motionformer_src_path = os.path.join(synch_root, 'model', 'modules', 'feat_extractors', 'visual', 'motionformer_src')
            motionformer_src_init_spec = importlib.util.spec_from_file_location(
                "motionformer_src", 
                os.path.join(motionformer_src_path, '__init__.py')
            )
            motionformer_src_module = importlib.util.module_from_spec(motionformer_src_init_spec)
            motionformer_src_init_spec.loader.exec_module(motionformer_src_module)
            sys.modules['motionformer_src'] = motionformer_src_module
            
            # 导入 scripts.train_utils
            train_utils_path = os.path.join(synch_root, 'scripts', 'train_utils.py')
            spec = importlib.util.spec_from_file_location("synch_train_utils", train_utils_path)
            train_utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_utils_module)
            get_model = train_utils_module.get_model
            get_transforms = train_utils_module.get_transforms
            prepare_inputs = train_utils_module.prepare_inputs
            
            # 导入 dataset.dataset_utils
            dataset_utils_path = os.path.join(synch_root, 'dataset', 'dataset_utils.py')
            spec = importlib.util.spec_from_file_location("synch_dataset_utils", dataset_utils_path)
            dataset_utils_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dataset_utils_module)
            get_video_and_audio = dataset_utils_module.get_video_and_audio
            
            # 导入 dataset.transforms
            transforms_path = os.path.join(synch_root, 'dataset', 'transforms.py')
            spec = importlib.util.spec_from_file_location("synch_transforms", transforms_path)
            transforms_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(transforms_module)
            make_class_grid = transforms_module.make_class_grid
            quantize_offset = transforms_module.quantize_offset
            
            return True
        
        finally:
            # 恢复原始的sys.path
            sys.path[:] = original_sys_path
        
    except Exception as e:
        print(f"Failed to import Synchformer modules: {e}")
        return False


def _load_synchformer_model(cfg_path: str, ckpt_path: str, device: torch.device):

    # 延迟导入 Synchformer 模块
    if not _lazy_import_synchformer():
        raise ImportError("Failed to import required Synchformer modules")
    
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Synchformer config not found: {cfg_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Synchformer checkpoint not found: {ckpt_path}")
    
    # 加载配置
    cfg = OmegaConf.load(cfg_path)
    
    # 修复配置（根据README，完整的同步模型checkpoint已包含特征提取器权重）
    # 将特征提取器路径设为null，避免重复加载
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target.replace(
        '.modules.feature_selector.', '.sync_model.'
    )
    
    # 设置随机种子（确保结果可复现）
    import random
    import numpy as np
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    np.random.seed(42)
    
    # 加载模型
    _, model = get_model(cfg, device)
    
    # 加载checkpoint（DictConfig已在模块顶部注册为安全全局变量）
    try:
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=True)
    except Exception as e:
        # 如果仍然失败，回退到 weights_only=False（对于可信任的checkpoint）
        print(f"Warning: Loading checkpoint with weights_only=False due to: {e}")
        ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 自动适配两种checkpoint格式
    # 1. 训练checkpoint (dict with 'model' key): ckpt['model']
    # 2. 纯权重文件 (OrderedDict): ckpt 本身
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)

    model.eval()
    
    return model, cfg



def _get_synchformer_model(device, submodules_list):

    cfg_path = submodules_list.get('cfg_path')
    ckpt_path = submodules_list.get('ckpt_path')

    if not cfg_path or not ckpt_path:
        raise ValueError(
            "Synchformer model configuration not found in submodules_list. "
            "Expected 'synchformer' or 'second_desync' with 'cfg_path' and 'ckpt_path'"
        )
    model, cfg = _load_synchformer_model(cfg_path, ckpt_path, device)
    return model, cfg



def _evaluate_desync_direct(video_path: str, audio_path: str, model, cfg, device: torch.device) -> float:

    # 导入 make_class_grid
    if not _lazy_import_synchformer():
        raise ImportError("Failed to import required Synchformer modules")
    
    # 提取特征
    afeats = _extract_audio_features_avbench_style(audio_path, model, device)
    vfeats = _extract_video_features_avbench_style(video_path, model, cfg, device)
    
    # 访问模型组件
    sync_mod = getattr(model, 'sync_model', None) or getattr(model, 'model', None) or model
    vproj = getattr(sync_mod, 'vproj', None)
    aproj = getattr(sync_mod, 'aproj', None)
    transformer = getattr(sync_mod, 'transformer', None)
    
    if vproj is None or aproj is None or transformer is None:
        raise RuntimeError('Required model components (vproj/aproj/transformer) not found')
    
    # 使用 make_class_grid
    sync_grid = make_class_grid(-2, 2, 21)
    
    # 检查 segment 数量
    S_v = vfeats.shape[0]
    S_a = afeats.shape[0]
    
    if S_v < 14 or S_a < 14:
        # 如果 segment 不足14，使用全部数据
        vfeats_front = vfeats
        afeats_front = afeats
        vfeats_back = vfeats
        afeats_back = afeats
    else:
        # 正常情况：前14段和后14段
        vfeats_front = vfeats[:14]
        afeats_front = afeats[:14]
        vfeats_back = vfeats[-14:]
        afeats_back = afeats[-14:]
    
    with torch.no_grad():
        # === 前14段 ===
        vfeats_front_batch = vfeats_front.unsqueeze(0).to(device)
        afeats_front_batch = afeats_front.unsqueeze(0).to(device)
        
        vis_proj_front = vproj(vfeats_front_batch)
        aud_proj_front = aproj(afeats_front_batch)
        
        B, S, tv, D = vis_proj_front.shape
        _, _, ta, _ = aud_proj_front.shape
        vis_flat_front = vis_proj_front.view(B, S * tv, D)
        aud_flat_front = aud_proj_front.view(B, S * ta, D)
        
        logits_front = transformer(vis_flat_front, aud_flat_front)
        pred_front = torch.argmax(logits_front, dim=-1).item()
        score_front = abs(sync_grid[pred_front].item())
        
        # === 后14段 ===
        vfeats_back_batch = vfeats_back.unsqueeze(0).to(device)
        afeats_back_batch = afeats_back.unsqueeze(0).to(device)
        
        vis_proj_back = vproj(vfeats_back_batch)
        aud_proj_back = aproj(afeats_back_batch)
        
        vis_flat_back = vis_proj_back.view(B, S * tv, D)
        aud_flat_back = aud_proj_back.view(B, S * ta, D)
        
        logits_back = transformer(vis_flat_back, aud_flat_back)
        pred_back = torch.argmax(logits_back, dim=-1).item()
        score_back = abs(sync_grid[pred_back].item())
    
    # 计算平均分数
    avg_score = (score_front + score_back) / 2.0
    return avg_score


def compute_second_desync(json_dir, device, submodules_list, **kwargs):

    try:
        # 读取 JSON（由 YAML 构建）并兼容返回四元组
        video_list,audio_list,prompt_dict_ls = load_dimension_info(json_dir, dimension='second_desync', lang='en')

        # 构建视频到音频的映射
        video_to_audio = {}
        video_to_prompt = {}
        
        for prompt_dict in prompt_dict_ls:
            prompt = prompt_dict.get('prompt', '')
            cur_video_list = prompt_dict.get('video_list', [])
            cur_audio_list = prompt_dict.get('audio_list', [])
            
            # 假设每个视频对应一个音频，一一配对
            for v, a in zip(cur_video_list, cur_audio_list):
                video_to_audio[v] = a
                video_to_prompt[v] = prompt

        # 加载 Synchformer 模型
        model, cfg = _get_synchformer_model(device, submodules_list)

        video_results = []
        for idx, video_path in enumerate(tqdm(list(video_to_audio.keys()))):
            try:
                audio_path = video_to_audio.get(video_path)

                prompt = video_to_prompt.get(video_path, "")

                # from pathlib import Path
                # video_stem = Path(video_path).stem
                # audio_stem = Path(audio_path).stem
                # match_status = "✓" if video_stem == audio_stem else "✗ MISMATCH"
                # print(f"\n[DEBUG] 样本 #{idx+1}/{len(video_list)} {match_status}")
                # print(f"  视频: {video_stem}")
                # print(f"  音频: {audio_stem}")

                desync_score = _evaluate_desync_direct(video_path, audio_path, model, cfg, device)
                
                # print(f"  分数: {desync_score:.6f}")

                video_results.append({
                    'video_path': video_path,
                    'video_results': round(float(desync_score), 5),
                    'prompt': prompt
                })

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                import traceback
                traceback.print_exc()
                video_results.append({
                    'video_path': video_path,
                    'video_results': 1.0,
                    'prompt': ""
                })

        return None, video_results

    except Exception as e:
        print(f"Error in second_desync evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, []


def _pad_or_truncate(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """将最后一维填充或截断到目标长度（用于梅尔帧）。"""
    cur_len = x.shape[-1]
    if cur_len < target_len:
        pad_len = target_len - cur_len
        return torch.nn.functional.pad(x, (0, pad_len))
    else:
        return x[..., :target_len]


def _extract_audio_features_avbench_style(audio_path: str, model, device: torch.device) -> torch.Tensor:
    """严格按 av-benchmark 音频特征提取流程：
    读取wav/flac → 分段(10240, step=5120) → 梅尔 → log → 归一化 → pad/trunc(66) → AST → (S, ta, D)
    """
    # 读取音频（16kHz单声道，与av-benchmark一致）
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # (T,)

    # 音频处理：保持与视频长度一致
    total_len = waveform.shape[0]

    # 分段（与 av-benchmark 一致：segment_size=10240, step=5120）
    segment_size = 10240
    step_size = 5120
    if total_len < segment_size:
        # 若音频太短，填充到一个段
        print(f"     音频太短，填充到 {segment_size} samples")
        waveform = torch.nn.functional.pad(waveform, (0, segment_size - total_len))
        segments = [waveform]
    else:
        num_segments = (total_len - segment_size) // step_size + 1
        segments = []
        for i in range(num_segments):
            start = i * step_size
            end = start + segment_size
            if end > total_len:
                seg = waveform[start:]
                pad_len = segment_size - seg.shape[0]
                print(f"    最后一段需要填充 {pad_len} samples")
                seg = torch.nn.functional.pad(seg, (0, pad_len))
            else:
                seg = waveform[start:end]
            segments.append(seg)
        # print(f"    音频segments数量: {len(segments)}")
    x = torch.stack(segments, dim=0).unsqueeze(0)  # (1, S, T)

    # 梅尔频谱（与 av-benchmark 一致：n_fft=512, hop=160, n_mels=128）
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, win_length=400, hop_length=160, n_fft=1024, n_mels=128
    ).to(device)
    x = x.to(device)
    x = mel_transform(x)
    x = torch.log(x + 1e-6)
    x = _pad_or_truncate(x, 66)
    # x = mel_transform(x)  # (1, S, 128, T_mel)
    # x = _pad_or_truncate(x, 66)
    # x = torch.log(x + 1e-6)

    # 归一化（av-benchmark 固定均值/方差）
    mean = -4.2677393
    std = 4.5689974
    x = (x - mean) / (2 * std)

    # AST 提取特征
    sync_mod = getattr(model, 'sync_model', None) or getattr(model, 'model', None) or model
    a_extractor = getattr(sync_mod, 'afeat_extractor', None)
    if a_extractor is None:
        raise RuntimeError('afeat_extractor not found')

    # AST 期望输入 (B, S, T_mel, F)，当前为 (B, S, F, T_mel)
    x = x.permute(0, 1, 3, 2)  # -> (1, S, 66, 128)
    with torch.no_grad():
        afeats, *_ = a_extractor(x)
    return afeats.squeeze(0).cpu()  # (S, ta, D)


def _extract_video_features_avbench_style(video_path: str, model, cfg, device: torch.device) -> torch.Tensor:
    """严格按 av-benchmark 视频特征提取流程：
    使用与 av-benchmark 完全一致的预处理方式，避免 Synchformer 复杂 transforms 导致的差异
    
    关键改进：
    1. 使用 Resize(224) 而非 Resize(256)+CenterCrop(224)
    2. 使用 float32 而非 float16
    3. 简化的预处理流程，与 av-benchmark/extract_video.py 完全一致
    """
    import decord
    import numpy as np
    from torchvision.transforms import v2
    from einops import rearrange
    
    # 使用 decord 加载并采样视频（与 av-benchmark 一致）
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    video_fps = vr.get_avg_fps()
    
    # 使用实际视频时长（支持5秒或8秒视频）
    duration_sec = total_frames / video_fps  # 动态计算实际时长
    sync_fps = 25.0
    sync_expected_length = int(duration_sec * sync_fps)  # 根据实际时长计算采样帧数
    sync_timestamps = np.arange(sync_expected_length) / sync_fps  # 时间戳（秒）
    sync_frame_indices = (sync_timestamps * video_fps).astype(int)  # 转换为帧索引
    sync_frame_indices = np.clip(sync_frame_indices, 0, total_frames - 1)  # 确保不超出范围
    sync_frames = vr.get_batch(sync_frame_indices).asnumpy()  # (T, H, W, C)
    rgb = torch.from_numpy(sync_frames).permute(0, 3, 1, 2)  # (T, C, H, W)
    
    # 检查采样结果
    if rgb.shape[0] < sync_expected_length:
        raise RuntimeError(f'视频采样失败: {video_path}, 期望 {sync_expected_length}, 实际 {rgb.shape[0]}')
    
    rgb = rgb[:sync_expected_length]
    
    # 【关键修复】使用与 av-benchmark 完全一致的 transforms
    # 参考: av-benchmark/av_bench/data/video_dataset.py (52-58行)
    sync_transform = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC),  # 直接resize到224，不先resize到256
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # 使用float32而非float16
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # 应用transforms
    rgb = sync_transform(rgb)  # (T=125, C=3, H=224, W=224)
    
    # 【关键修复】使用与 av-benchmark 完全一致的分段方式
    # 参考: av-benchmark/extract_video.py (52-70行)
    x = rgb.unsqueeze(0)  # (B=1, T=125, C=3, H=224, W=224)
    b, t, c, h, w = x.shape
    
    # 分段: segment_size=16, step_size=8
    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1  # (125-16)//8+1 = 14
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size:i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B=1, S=14, T=16, C=3, H=224, W=224)
    
    # 准备输入给 MotionFormer（与 av-benchmark 一致）
    x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')  # (14, 1, 16, 3, 224, 224)
    
    # 转换为模型期望的维度顺序
    x = x.permute(0, 1, 3, 2, 4, 5)  # (14, 1, 3, 16, 224, 224) = (B*S, 1, C, T, H, W)
    
    # 提取特征
    sync_mod = getattr(model, 'sync_model', None) or getattr(model, 'model', None) or model
    v_extractor = getattr(sync_mod, 'vfeat_extractor', None)
    if v_extractor is None:
        raise RuntimeError('vfeat_extractor not found')
    
    x = x.to(device)
    with torch.no_grad():
        vfeats_output = v_extractor(x)
        # MotionFormer 返回元组 (segment_features, global_features)，只取第一个
        if isinstance(vfeats_output, tuple):
            vfeats = vfeats_output[0]
        else:
            vfeats = vfeats_output
    
    # vfeats shape: (B*S, 1, tv, D) 例如 (14, 1, 8, 768)
    # 去掉维度1（segment内的batch维度）
    vfeats = vfeats.squeeze(1)  # (B*S, tv, D) 例如 (14, 8, 768)
    
    # 恢复batch维度
    vfeats = rearrange(vfeats, '(b s) tv d -> b s tv d', b=b)  # (1, S, tv, D) 例如 (1, 14, 8, 768)
    
    return vfeats.squeeze(0).cpu()  # (S, tv, D) 例如 (14, 8, 768)

