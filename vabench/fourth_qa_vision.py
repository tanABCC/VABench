import os
import re
import json
import subprocess
from typing import List, Tuple

import torch
from tqdm import tqdm
import gc

from vabench.utils import load_dimension_info
from vabench.omni_runtime import get_omni_model_and_processor
# Distributed imports removed


def _has_audio_track(video_path: str) -> bool:

    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-select_streams', 'a', 
            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path
        ], capture_output=True, text=True, timeout=10)
        
        # 如果返回非空且包含audio，说明有音频轨道
        return result.returncode == 0 and 'audio' in result.stdout.strip()
    except Exception as e:
        print(f"Warning: Failed to check audio track for {video_path}: {e}")
        return False



def _build_system_prompt() -> str:
    return """You are a video generation quality evaluation expert. You will be given a target reference text that describes the desired visual characteristics, along with a specific question about the generated video's visual content. Your task is to evaluate whether the video's visuals align with the expectations set by the provided reference text.

    Strictly follow these instructions:
    1. Do not answer with only "yes" or "no".
    2. Your response must begin with either "Yes:" or "No:", followed by a space and a concise, factual explanation.
    3. Base your judgment solely on whether the video's visual content matches the visual quality, attributes, or conditions described in the provided reference text.
    4. The reference text represents the ideal or expected state. Use it as the ground truth for determining desirability.
    5. Determine polarity as follows:
    - "Polarity: positive" means the video's visuals meet or align with the desired state described in the reference text (i.e., the outcome is beneficial for visual fidelity as defined by your input).
    - "Polarity: negative" means the video's visuals deviate from or violate the desired state in the reference text (i.e., the outcome is detrimental relative to your expectation).
    - Always interpret the question in the context of the reference text you provided. For example:
        • If the reference text says "a bright, sunlit park with green grass", and the question is "Is the scene well-lit with daylight?", then "Yes" → positive, "No" → negative.
        • If the reference text says "the person is wearing a red jacket", and the question is "Is the person wearing a red jacket?", then "Yes" → positive, "No" → negative.
        • If the reference text implies stillness (e.g., "a sleeping baby"), and the question is "Is the baby moving actively?", then "Yes" → negative, "No" → positive.
    6. At the very end of your response, output exactly one line—and only one line—in the following format:
    Polarity: positive
    or
    Polarity: negative

    Your entire response must conform to this structure strictly, especially end with "Polarity: positive" or "Polarity: negative". Any deviation is unacceptable."""


def _infer_batch_with_omni(paths: List[str], questions: List[str], use_audio_in_video: bool = True, **kwargs) -> List[Tuple[str, str]]:
    """Run a batch of paths (video/audio/image) through local Qwen2.5-Omni and parse (answer, reason)."""
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
        # fallback: treat as video path
        return [{"type": "video", "video": p}]

    system_text = _build_system_prompt()
    conversations = []
    contexts = kwargs.get('contexts')  # 可选：与 questions 对齐的上下文文本
    for idx, (p, question) in enumerate(zip(paths, questions)):
        # 将 reference 与 question 以固定模板嵌入，便于 7B 模型严格遵循
        reference_text = ""
        if isinstance(contexts, list) and idx < len(contexts) and isinstance(contexts[idx], str) and contexts[idx].strip():
            reference_text = contexts[idx].strip()
        instruction_text = (
        "You are a video generation quality evaluator. "
        "The 'Reference' below describes the desired visual state. "
        "Your job is to check if the video's visuals match that desired state.\n\n"
        
        "Reference: \"{reference_text}\"\n"
        "Question: \"{question}\"\n\n"
        
        "Important rules for Polarity:\n"
        "- Polarity is POSITIVE if the video's visuals MATCH the desired state in the Reference.\n"
        "- Polarity is NEGATIVE if the video's visuals VIOLATE or LACK what the Reference expects.\n"
        "- Do NOT assume 'Yes = positive' or 'No = negative'.\n"
        "  Example: If Reference says 'bright sunlit park', and the question is 'Is the scene well-lit?', then:\n"
        "    - Answer 'Yes' → Polarity: positive (well-lit is good)\n"
        "    - Answer 'No' → Polarity: negative (dark is bad)\n\n"
        
        "Output format (EXACTLY two lines, no extra text):\n"
        "Yes: [One short sentence].\nPolarity: positive\n"
        "OR\n"
        "No: [One short sentence].\nPolarity: negative\n\n"
        
        "Now generate your response:"
        ).format(reference_text=reference_text, question=question)
        conversations.append([
            {
                "role": "system",
                "content": [{"type": "text", "text": system_text}],
            },
            {
                "role": "user",
                "content": ([{"type": "text", "text": instruction_text}] + path_to_content(p)),
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
            do_sample=False,      # 禁用采样，使用贪婪解码确保结果确定性
            temperature=1.0,      # 温度参数（do_sample=False时不起作用）
            num_beams=1,          # 使用贪婪搜索
        )
    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    results: List[Tuple[str, str]] = []
    for out in decoded:
        clean_reason = out.strip()
        if "assistant" in clean_reason:
            # 提取assistant后面的内容
            assistant_part = clean_reason.split("assistant")[-1].strip()
            if assistant_part:
                clean_reason = assistant_part
        answer = clean_reason.lower()[0:3]
        if answer == "yes":
            answer = "yes"
        elif answer:
            answer = "no"

        # 规范化极性标记到理由文本开头，便于后续解析
        pol = None
        for line in clean_reason.splitlines()[::-1]:  # 从结尾往前找
            low = line.strip().lower()
            if low.startswith("polarity:"):
                if "positive" in low:
                    pol = "positive"
                elif "negative" in low:
                    pol = "negative"
                break
        if pol in ("positive", "negative"):
            clean_reason = f"[polarity: {pol}] " + clean_reason

        results.append((answer, clean_reason))
    # 严格释放中间变量
    try:
        del inputs, text_ids, decoded, audios, images, videos, text, conversations, processor_kwargs
    except Exception:
        pass
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    return results


def compute_fourth_qa_vision(json_dir, device, submodules_list, **kwargs):
    """
    使用Qwen2.5-Omni进行视频QA问答评估
    从JSON文件的auxiliary_info中读取所有问题，进行yes/no问答
    视频通过路径输入模型进行分析
    返回 (all_results, video_results)，其中：
      - all_results: 平均分（0-1）
      - video_results: [{video_path, video_results(分数), answers}]
    """
    try:
        use_audio_in_video = bool(kwargs.get('use_audio_in_video', True))
        checkpoint = kwargs.get('qwen_omni_ckpt', None)

        # 预热模型（可选指定权重）
        get_omni_model_and_processor(checkpoint, **kwargs)

        # 读取列表并按 rank 分发 - 使用 fourth_qa_vision 维度（与 full_info.json 对齐）
        video_list, audio_list, prompt_dict_ls = load_dimension_info(json_dir, dimension='fourth_qa_vision', lang='en')
        
        video_results = []

        for prompt_dict in tqdm(prompt_dict_ls):
            video_paths = prompt_dict.get('video_list', [])
            print(prompt_dict)
            auxiliary_info = prompt_dict.get('auxiliary_info_vision', [])
            
            # 确保auxiliary_info是列表
            if isinstance(auxiliary_info, str):
                auxiliary_info = [auxiliary_info]
            elif not isinstance(auxiliary_info, list):
                auxiliary_info = []
            
            # 使用所有auxiliary_info中的问题
            questions = auxiliary_info
            
            if len(questions) == 0:
                print(f"Warning: auxiliary_info is empty")
                continue
                
            # 逐个视频处理，但将该视频的所有问题一次性批处理
            for video_path in video_paths:
                answers = []

                if len(questions) > 0:
                    # 检查视频是否包含音频轨道，动态设置use_audio_in_video
                    has_audio = _has_audio_track(video_path)
                    current_use_audio = use_audio_in_video and has_audio
                    
                    if use_audio_in_video and not has_audio:
                        print(f"Warning: Video {video_path} has no audio track, setting use_audio_in_video=False for vision QA")
                    
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'use_audio_in_video'}
                    prompt_context = prompt_dict.get('prompt').strip()
                    try:
                        # 一次性处理所有问题
                        batched_paths = [video_path] * len(questions)
                        batched_contexts = [prompt_context] * len(questions)
                        batched_results = _infer_batch_with_omni(
                            batched_paths,
                            list(questions),
                            use_audio_in_video=current_use_audio,
                            contexts=batched_contexts,
                            **filtered_kwargs,
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(f"Warning: OOM processing {video_path} batched questions, skipping")
                        batched_results = [("no", "OOM_ERROR")] * len(questions)
                        torch.cuda.empty_cache()

                    # 汇总批次结果
                    for i, (answer, reason) in enumerate(batched_results):
                        answers.append({"question": questions[i], "answer": answer, "reason": reason})

                # 计算最终分数：基于极性（positive=1, negative=0）/ 问题总数
                if len(questions) > 0:
                    def infer_polarity_from_reason(text: str) -> str:
                        low = (text or "").lower()
                        # 优先解析标准化前缀
                        if "[polarity: positive]" in low:
                            return "positive"
                        if "[polarity: negative]" in low:
                            return "negative"
                        # 退化解析：末尾行 "Polarity: positive/negative"
                        for line in (text or "").splitlines()[::-1]:
                            l = line.strip().lower()
                            if l.startswith("polarity:"):
                                if "positive" in l:
                                    return "positive"
                                if "negative" in l:
                                    return "negative"
                                break
                        return ""

                    def heuristic_polarity(question: str, answer_yes_no: str, reason_text: str) -> str:
                        ql = (question or "").lower()
                        al = (answer_yes_no or "").lower()
                        rl = (reason_text or "").lower()
                        issue_cues = [
                            "blurry", "blur", "dark", "overexposed", "underexposed", "flickering",
                            "artifact", "distortion", "pixelated", "glitch", "jitter", "shaky",
                            "unnatural", "incorrect color", "wrong color", "color shift", "bad lighting",
                            "模糊", "过暗", "过曝", "曝光不足", "闪烁", "失真", "像素化", "抖动", "不自然", "色彩错误",
                        ]
                        good_cues = [
                            "clear", "sharp", "well-lit", "good quality", "high quality", "stable", "smooth",
                            "natural", "correct color", "proper lighting", "balanced exposure",
                            "清晰", "清楚", "锐利", "光线好", "高质量", "质量好", "稳定", "流畅", "自然", "色彩正确",
                        ]
                        # 若问题包含问题类关键词，则 yes => negative, no => positive
                        if any(k in ql for k in issue_cues):
                            return "negative" if al.startswith("yes") else "positive"
                        # 若问题包含正向条件关键词，则 yes => positive, no => negative
                        if any(k in ql for k in good_cues):
                            return "positive" if al.startswith("yes") else "negative"
                        # 从理由中再尝试一次
                        if any(k in rl for k in issue_cues):
                            return "negative"
                        if any(k in rl for k in good_cues):
                            return "positive"
                        # 无法判断时，回退为 yes 视为 positive
                        return "positive" if al.startswith("yes") else "negative"

                    positive_count = 0
                    for i, (ans, reason) in enumerate(batched_results):
                        pol = infer_polarity_from_reason(reason)
                        if pol == "":
                            pol = heuristic_polarity(questions[i], str(ans), reason)
                        if pol == "positive":
                            positive_count += 1
                    final_score = round(positive_count / len(questions), 5)
                else:
                    final_score = 0

                video_results.append({
                    'video_path': video_path,
                    'video_results': final_score,
                    'answers': answers,
                })

                # 每个视频处理完后清理显存和变量
                try:
                    # 注意：这些变量只在 len(questions) > 0 时存在
                    if len(questions) > 0:
                        del batched_paths, batched_contexts, batched_results
                    del answers
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

        # 共享单例：不在此处释放，由外层统一释放
        try:
            del video_list, audio_list, prompt_dict_ls
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        
        return None, video_results
        
    except Exception as e:
        print(f"Error evaluating fourth_qa_vision: {e}")
        return None, []
