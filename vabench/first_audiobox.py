import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from audiobox_aesthetics.infer import initialize_predictor
from vabench.utils import load_dimension_info


def _get_audiobox_predictor(submodules_list):
    try:
        ckpt_path = submodules_list.get('first_audiobox_model_path')
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Audiobox model file not found: {ckpt_path}")
        
        model = initialize_predictor(ckpt_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize audiobox_aesthetics predictor: {e}")
    return model


def _process_audiobox_audio(audio_path, predictor):

    items = [{"path": audio_path}]
    outputs = predictor.forward(items) or []
    out = outputs[0] if len(outputs) > 0 else {}
    ce = float(out.get('CE', 0.0)) if out is not None else 0.0
    cu = float(out.get('CU', 0.0)) if out is not None else 0.0
    pc = float(out.get('PC', 0.0)) if out is not None else 0.0
    pq = float(out.get('PQ', 0.0)) if out is not None else 0.0
    video_avg_score = (ce + cu - pc + pq) / 4
    result = {
        'audio_path': audio_path,
        'video_results': round(video_avg_score, 5),
        'aesthetic_ce': round(ce, 5),
        'aesthetic_cu': round(cu, 5),
        'aesthetic_pc': round(pc, 5),
        'aesthetic_pq': round(pq, 5),
    }
    scores = {
        'CE': ce,
        'CU': cu,
        'PC': pc,
        'PQ': pq,
    }
    return result, scores


def compute_first_audiobox(json_dir, device, submodules_list, **kwargs):

    predictor = _get_audiobox_predictor(submodules_list)

    video_list, audio_list, prompt_dict_ls = load_dimension_info(json_dir, dimension='first_audiobox', lang='en')

    video_results = []

    for audio_path in tqdm(audio_list, desc="AudioBox"):
        result, scores = _process_audiobox_audio(audio_path, predictor)
        video_results.append(result)

    return None, video_results
