import os
import csv
import gc
import torch
from .utils import init_submodules, save_json, load_json
import importlib
from itertools import chain
from pathlib import Path
from .omni_runtime import release_omni
import json

def _cleanup_models_and_memory(device):
    """
    通用的模型和显存清理函数
    """
    try:
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # 打印显存回收情况
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            print(f'GPU Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB')

    except Exception as e:
        print(f'Warning: Failed to cleanup memory: {e}')

class VABench(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device
        self.full_info_dir = full_info_dir
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.prompt_mapping = self.load_prompt_mapping()
        
    def load_prompt_mapping(self):
        mapping_file = "./mapping_text/final_idx_to_prompt.csv"
        mapping = {}
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    prompt = row['prompt']
                    prompt = '"' + prompt + '"'
                    prompt = json.loads(prompt)
                    # 直接使用原始字符串，不需要JSON解析
                    mapping[row['idx']] = prompt
        except Exception as e:
            print(f"加载映射表失败: {e}")
        return mapping
        
    def find_video_by_prompt(self, prompt, video_names):
        """根据prompt查找对应的视频文件"""
        for idx, mapped_prompt in self.prompt_mapping.items():
            idx = idx.split('.')[0]
            if mapped_prompt == prompt:
                for video_name in video_names:
                    video_name_nosuffix = video_name.split('.')[0]
                    video_name_prefix= video_name_nosuffix
                    if idx == video_name_prefix:
                        return video_name

    def build_full_info_json(self, videos_path, audios_path, name, dimension_list, prompt_list=[], special_str='', verbose=False, **kwargs):
        cur_full_info_list = []
        # 构建音频文件名到完整路径的映射（用 stem 匹配，忽略扩展名）

        audio_names = os.listdir(audios_path)
        audio_map = {Path(x).stem: os.path.join(audios_path, x) for x in audio_names}


        full_info_list = load_json(self.full_info_dir)
        video_names = os.listdir(videos_path)
        postfix = Path(video_names[0]).suffix
        for prompt_dict in full_info_list:
            if set(dimension_list) & set(prompt_dict["dimension"]): 
                prompt = prompt_dict['prompt_en']
                prompt_dict['video_list'] = []
                prompt_dict['audio_list'] = []
                for i in range(1):
                    # intended_video_name = f'{prompt[:180]}{special_str}-{str(i)}{postfix}'
                    intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'

                    found_video = self.find_video_by_prompt(prompt, video_names)
                    
                    if found_video:
                        intended_video_path = os.path.join(videos_path, found_video)
                        prompt_dict['video_list'].append(intended_video_path)
                        # print(f'成功找到视频: {found_video} (原始期望: {intended_video_name})')
                        
                        # 匹配音频
                        audio_stem = Path(found_video).stem
                        if audio_stem in audio_map:
                            prompt_dict['audio_list'].append(audio_map[audio_stem])
                        else:
                            print(f'WARNING!!! This required audio is not found! Missing audio can lead to unfair evaluation result. The missing audio is: {audio_stem}')
                            # 不抛出异常，继续处理
                        
                        
                        if verbose:
                            print(f'Successfully found video: {found_video}')
                    else:
                        print(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
                        # 不抛出异常，继续处理
                cur_full_info_list.append(prompt_dict)

        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path
        

    def evaluate(self, videos_path, audios_path, name, prompt_list=[], dimension_list=None, local=False, **kwargs):
        results_dict = {}
        submodules_dict = init_submodules(dimension_list, local=local) # TODO me: to fix
        cur_full_info_path = self.build_full_info_json(videos_path, audios_path, name, dimension_list, prompt_list, **kwargs)
        
        for dimension in dimension_list:
            try:
                dimension_change = dimension
                dimension_module = importlib.import_module(f'vabench.{dimension_change.lower()}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension_change.lower()}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            # 为不同维度构造专属参数（支持 YAML 的 dim_params，以及 namespaced 参数映射）
            dim_key = str(dimension_change).lower()
            # 基于全局 kwargs，但去掉 dim_params，防止透传到子模块
            dim_kwargs = {k: v for k, v in (kwargs.items() if kwargs else []) if k != 'dim_params'}
            # 从 YAML 注入维度专属参数（不会影响其他维度）
            if isinstance(kwargs.get('dim_params'), dict):
                yaml_dim_cfg = kwargs['dim_params'].get(dim_key, {})
                if isinstance(yaml_dim_cfg, dict):
                    dim_kwargs.update(yaml_dim_cfg)

            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **dim_kwargs)

            # 所有子模块统一返回 (None, video_results)
            unified_video_results = results[1] if isinstance(results, tuple) and len(results) == 2 else []
            results_dict[dimension] = {'video_results': unified_video_results}
            
            try:
                if submodules_list:
                    del submodules_list
            except Exception as e:
                print(f'Warning: Failed to clean submodules: {e}')
            
            # 清理结果中的大对象
            try:
                del results
            except Exception:
                pass
            
            # 检查是否需要释放 Omni 模型
            # 如果当前维度使用 Omni，但下一个维度不使用，则释放
            current_dim_lower = str(dimension).lower()
            uses_omni_current = current_dim_lower.startswith('third_') or current_dim_lower in ['fourth_qa_audio', 'fourth_qa_vision']
            
            if uses_omni_current:
                # 查找下一个维度
                current_idx = dimension_list.index(dimension)
                next_uses_omni = False
                if current_idx + 1 < len(dimension_list):
                    next_dim = str(dimension_list[current_idx + 1]).lower()
                    next_uses_omni = next_dim.startswith('third_') or next_dim in ['fourth_qa_audio', 'fourth_qa_vision']
                
                # 如果下一个维度不使用 Omni，则释放
                if not next_uses_omni:
                    print(f'Next dimension does not use Omni model, releasing Omni...')
                    release_omni()
            
            # 使用通用清理函数
            _cleanup_models_and_memory(self.device)
            print(f'Memory cleanup completed for {dimension}\n')
        
        # 保存结果
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')

        # 读取 full_info，建立路径到 prompt 的映射
        full_info_list = load_json(cur_full_info_path)
        prompt_list_all = []
        video_to_prompt = {}
        audio_to_prompt = {}
        for item in full_info_list:
            prompt_en = item.get('prompt_en')
            if prompt_en not in prompt_list_all:
                prompt_list_all.append(prompt_en)
            for v in item.get('video_list', []) or []:
                video_to_prompt[v] = prompt_en
            for a in item.get('audio_list', []) or []:
                audio_to_prompt[a] = prompt_en

        # 聚合每个维度的 per-prompt 分数
        from collections import defaultdict
        dim_to_prompt_scores = {dim: defaultdict(list) for dim in (dimension_list or [])}

        def _append_score_for_path(dim, path, score):
            if not isinstance(score, (int, float)):
                return
            p = video_to_prompt.get(path) or audio_to_prompt.get(path)
            if p is not None:
                dim_to_prompt_scores[dim][p].append(float(score))

        for dim, dim_results in results_dict.items():

            video_results = dim_results.get('video_results')
            if not isinstance(video_results, list):
                continue
            for rec in video_results:
                if not isinstance(rec, dict):
                    continue
                path = rec.get('video_path') or rec.get('audio_path')
                score = rec.get('video_results')
                if path is not None and isinstance(score, (int, float)):
                    _append_score_for_path(dim, path, score)

          # 写出按 prompt 聚合后的 CSV（每个维度一列）
            csv_path = os.path.join(self.output_path, name + '_eval_results.csv')
            header = ['prompt_name'] + list(dimension_list or [])
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for prompt in prompt_list_all:
                    row = [prompt]
                    for dim in (dimension_list or []):
                        scores = dim_to_prompt_scores.get(dim, {}).get(prompt, [])
                        if len(scores) == 0:
                            row.append(-1)
                        else:
                            row.append(sum(scores) / len(scores))
                    writer.writerow(row)
        print(f'CSV results saved to {csv_path}')

        # 入口级：若有使用 Omni，统一释放
        dims_lower = [str(d).lower() for d in (dimension_list or [])]
        needs_omni = any(d.startswith('third_') or d == 'fourth_qa_audio' or d == 'fourth_qa_vision' for d in dims_lower)
        if needs_omni:
            release_omni()


