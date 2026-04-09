import os
import re
import json
from typing import List, Tuple

import torch
from tqdm import tqdm
import gc

from vabench.utils import load_dimension_info
from vabench.omni_runtime import get_omni_model_and_processor


def _build_system_prompt() -> str:

    return """
    You are an expert AV Synchronization and Coherence Analyst. Evaluate whether audio and video are not only temporally aligned but also semantically and expressively consistent in a way that feels natural to a typical viewer.

    Assess key cues:
    - **Lip movements** vs. spoken words (e.g., does "p" sound match lip closure?)
    - **On-screen actions** (e.g., a clap, door slam) vs. their sound
    - **Visual cuts or flashes** vs. musical hits or stingers
    - **Speaker prosody** vs. facial/body expression (e.g., does a rising intonation match eyebrow lift or head tilt?)
    - **Audio context** vs. visual scene (e.g., is background sound appropriate? Does music mood match the emotional tone of the shot?)

    Use this practical scoring scale (based on 24fps video; 1 frame ≈ 42ms):
    5: Excellent sync and coherence — audio within ±0.1s (≈ ±2 frames) of video, and all modalities (speech, emotion, sound effects, music) align naturally with visuals. Feels seamless.
    4: Good sync — offset ≤0.15s (≈ 3-4 frames). Minor expressive mismatch (e.g., flat tone with excited face), but not distracting.
    3: Acceptable but flawed — consistent offset of ~0.15-0.3s (4-7 frames) or noticeable prosody-expression mismatch (e.g., sad voice with smiling face). Slightly unprofessional.
    2: Poor sync or coherence — clear temporal mismatch (0.3-0.5s) OR strong semantic conflict (e.g., laughter audio over a crying face). Distracting and immersion-breaking.
    1: Unusable — audio/video offset >0.5s (>12 frames) OR severe multimodal contradiction (e.g., explosion sound with silent, static scene). Feels disoriented or hallucinated.

    Output Requirements:
    - Return ONLY a single JSON object.
    - Must contain exactly two keys: "score" (integer 1-5) and "reason" (string, ≥15 characters).
    - In "reason", mention at least one specific issue with approximate timestamp (e.g., "At 0:12, footstep sound arrives ~4 frames after heel hits ground" or "At 0:08, speaker says 'I'm devastated' with a wide smile") and explain its impact.
    - Do NOT include markdown, extra text, or explanations.

    Example valid output:
    {"score": 3, "reason": "At 0:14, the character says 'Wow!' with flat intonation while jumping excitedly, creating emotional dissonance."}
    """


def _infer_batch_with_omni(paths: List[str], path_to_prompt: dict = None, use_audio_in_video: bool = True, **kwargs) -> List[Tuple[int, str]]:
    """Run a batch of paths (video/audio/image) through local Qwen2.5-Omni and parse (score, reasoning).
    
    Args:
        paths: List of file paths to evaluate
        path_to_prompt: Optional dict mapping video paths to their original generation prompts
        use_audio_in_video: Whether to use audio from video
        **kwargs: Additional arguments for model
    """
    from qwen_omni_utils import process_mm_info

    # 将全局/维度 kwargs 透传给模型加载，确保使用 YAML 中的 GPU 配置
    model, processor = get_omni_model_and_processor(**kwargs)

    def path_to_content(p: str):
        low = p.lower()
        if low.endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
            return [{"type": "video", "video": p}]
        if low.endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
            return [{"type": "audio", "audio": p}]
        if low.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            return [{"type": "image", "image": p}]
        return [{"type": "video", "video": p}]


    system_text = _build_system_prompt()
    conversations = []
    for p in paths:
        # 构建用户消息，包含文件名、原始prompt（如果有）和媒体内容
        user_text_parts = [f"Path: {os.path.basename(p)}"]
        
        # 添加原始prompt（如果提供）
        if path_to_prompt and p in path_to_prompt:
            original_prompt = path_to_prompt[p]
            user_text_parts.append(f"\nOriginal Generation Prompt: {original_prompt}")
        
        user_content = [{"type": "text", "text": "\n".join(user_text_parts)}] + path_to_content(p)
        
        conversations.append([
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ])

    text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=use_audio_in_video)
    
    # 设置视频处理参数
    processor_kwargs = {
        'text': text,
        'audio': audios,
        'images': images,
        'videos': videos,
        'return_tensors': "pt",
        'padding': True,
        'use_audio_in_video': use_audio_in_video,
    }
    
    
    inputs = processor(**processor_kwargs)
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        text_ids = model.generate(
            **inputs, 
            use_audio_in_video=use_audio_in_video, 
            return_audio=False,
            do_sample=False,      
            temperature=1.0,     
            num_beams=1,         
        )
    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    results: List[Tuple[int, str]] = []
    for out in decoded:
        try:
            # 优先：抓取最后一个 JSON 对象
            json_matches = list(re.finditer(r"\{[\s\S]*?\}", out))
            if json_matches:
                obj = json.loads(json_matches[-1].group(0))
                score = int(obj.get("score", 0.0))
                reason = str(obj.get("reason", obj.get("reasoning", "")))
            else:
                # 兜底：从 Markdown 提取，取最后一次出现
                score_all = re.findall(r"\*\*score:?\*\*\s*(\d+)", out, flags=re.IGNORECASE)
                score = int(score_all[-1]) if score_all else 0.0
                reason_all = re.findall(r"\*\*reason(?:ing)?:?\*\*\s*([\s\S]+?)(?=\n\*\*|$)", out, flags=re.IGNORECASE)
                reason = reason_all[-1].strip() if reason_all else ""
            score = max(1, min(5, score))
            results.append((score, reason))
        except Exception:
            results.append((0.0, out.strip() if isinstance(out, str) else ""))
    # 严格释放中间变量
    try:
        del inputs, text_ids, decoded, audios, images, videos, text, conversations
    except Exception:
        pass
    torch.cuda.empty_cache()
    return results


def compute_third_alignment(json_dir, device, submodules_list, **kwargs):
    """
    读取 JSON 中的待评估路径（视频/音频/图像），使用本地 Qwen2.5-Omni 批量打分。
    返回 (all_results, video_results)，其中：
      - all_results: 平均分（1-5）
      - video_results: [{video_path, video_results(分数), reasoning, original_prompt }]
    """
    try:
        use_audio_in_video = bool(kwargs.get('use_audio_in_video', True))
        checkpoint = kwargs.get('qwen_omni_ckpt', None)

        get_omni_model_and_processor(checkpoint, **kwargs)

        video_list, _, prompt_dict_ls = load_dimension_info(json_dir, dimension='third_alignment', lang='en')
        
        path_to_prompt = {}
        for prompt_info in prompt_dict_ls:
            prompt = prompt_info.get('prompt', '')
            for video_path in prompt_info.get('video_list', []):
                path_to_prompt[video_path] = prompt

        video_results = []
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'use_audio_in_video'}
        
        for p in tqdm(video_list):
            try:
                scores = _infer_batch_with_omni([p], path_to_prompt=path_to_prompt, use_audio_in_video=use_audio_in_video, **filtered_kwargs)
                score, reasoning = scores[0]
            except torch.cuda.OutOfMemoryError:
                print(f"Warning: OOM processing {p}, skipping")
                score, reasoning = 0.0, "OOM_ERROR"
                torch.cuda.empty_cache()
            
            _reasoning = reasoning if isinstance(reasoning, str) and len(reasoning.strip()) > 0 else "No reasoning provided (empty or OOM/parse fallback)."
            result_dict = {
                'video_path': p,
                'video_results': float(f"{float(score):.5f}"),
                'reasoning': _reasoning,
            }
            
            if p in path_to_prompt:
                result_dict['original_prompt'] = path_to_prompt[p]
            
            video_results.append(result_dict)

        try:
            del  video_list
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        return None, video_results
    except Exception as e:
        import traceback
        print(f"Error evaluating artistry: {e}")
        traceback.print_exc()
        return None, []