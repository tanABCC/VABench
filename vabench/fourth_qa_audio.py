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


def _build_system_prompt() -> str:
    return """You are a video generation quality evaluation expert. You will be given a target reference text that describes the desired audio characteristics, along with a specific question about the generated video's audio. Your task is to evaluate whether the video's audio aligns with the expectations set by the provided reference text.

    Strictly follow these instructions:
    1. Do not answer with only "yes" or "no".
    2. Your response must begin with either "Yes:" or "No:", followed by a space and a concise, factual explanation.
    3. Base your judgment solely on whether the video's audio matches the audio quality or condition described in the provided reference text.
    4. The reference text represents the ideal or expected state. Use it as the ground truth for determining desirability.
    5. Determine polarity as follows:
    - "Polarity: positive" means the video's audio meets or aligns with the desired state described in the reference text (i.e., the outcome is beneficial for audio quality as defined by your input).
    - "Polarity: negative" means the video's audio deviates from or violates the desired state in the reference text (i.e., the outcome is detrimental relative to your expectation).
    - Always interpret the question in the context of the reference text you provided. For example:
        • If the reference text says "clean and echo-free audio", and the question is "Is there echo?", then "Yes" → negative, "No" → positive.
        • If the reference text says "audio should contain background music", and the question is "Is background music present?", then "Yes" → positive, "No" → negative.
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
        "The 'Reference' below describes the desired audio state. "
        "Your job is to check if the video's audio matches that desired state.\n\n"
        
        "Reference: \"{reference_text}\"\n"
        "Question: \"{question}\"\n\n"
        
        "Important rules for Polarity:\n"
        "- Polarity is POSITIVE if the video's audio MATCHES the desired state in the Reference.\n"
        "- Polarity is NEGATIVE if the video's audio VIOLATES or LACKS what the Reference expects.\n"
        "- Do NOT assume 'Yes = positive' or 'No = negative'.\n"
        "  Example: If Reference says 'no echo', and the question is 'Is there echo?', then:\n"
        "    - Answer 'Yes' → Polarity: negative (echo is bad)\n"
        "    - Answer 'No' → Polarity: positive (no echo is good)\n\n"
        
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
            do_sample=False,      
            temperature=1.0,      
            num_beams=1,          
        )
    decoded = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # print('--------------------------decoded',decoded)
    results: List[Tuple[str, str]] = []
    for out in decoded:
        # print('-------------------out', out)
        clean_reason = out.strip()
        if "assistant" in clean_reason:
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
    try:
        del inputs, text_ids, decoded, audios, images, videos, text, conversations, processor_kwargs
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return results


def compute_fourth_qa_audio(json_dir, device, submodules_list, **kwargs):
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

        get_omni_model_and_processor(checkpoint, **kwargs)
        video_list, audio_list, prompt_dict_ls = load_dimension_info(json_dir, dimension='fourth_qa_audio', lang='en')
        
        video_results = []
        for prompt_dict in tqdm(prompt_dict_ls):
            video_paths = prompt_dict.get('video_list', [])
            auxiliary_info = prompt_dict.get('auxiliary_info_audio', [])
            questions = auxiliary_info

            for video_path in video_paths:
                answers = []

                if len(questions) > 0:
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'use_audio_in_video'}
                    prompt_context = prompt_dict.get('prompt').strip()
                    
                    try:
                        batched_paths = [video_path] * len(questions)
                        batched_contexts = [prompt_context] * len(questions)
                        batched_results = _infer_batch_with_omni(
                            batched_paths,
                            list(questions),
                            use_audio_in_video=use_audio_in_video,
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
                        if "[polarity: positive]" in low:
                            return "positive"
                        if "[polarity: negative]" in low:
                            return "negative"
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
                            "noise", "noisy", "distortion", "echo", "reverb", "clipping",
                            "artifact", "buzz", "hum", "hiss", "pop", "crackle",
                            "fluctuation", "unstable volume", "loudness variation",
                            "噪音", "杂音", "失真", "回声", "混响", "削波", "爆音", "嗡嗡", "嘶声", "噼啪",
                        ]
                        good_cues = [
                            "clean", "clear", "good quality", "high quality", "stable volume", "balanced",
                            "清晰", "干净", "高质量", "质量好", "音量稳定", "平衡",
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

        try:
            del video_list, audio_list, prompt_dict_ls
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
        
        return None, video_results
        
    except Exception as e:
        print(f"Error evaluating fourth_qa_audio: {e}")
        return None, []
