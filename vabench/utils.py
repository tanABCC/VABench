import os
import re
import json
import logging

from PIL import Image
from pathlib import Path
import gdown
import subprocess
from huggingface_hub import snapshot_download


CACHE_DIR = os.environ.get('VABENCH_CACHE_DIR')
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'vabench')

# logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
 
def load_dimension_info(json_dir, dimension, lang):
    video_list = []
    audio_list = []
    prompt_dict_ls = []
    full_prompt_list = load_json(json_dir)


    def ensure_list(x):
        if x is None:
            return []
        return x if isinstance(x, list) else [x]

    for prompt_dict in full_prompt_list:
        dims = prompt_dict.get('dimension', [])
        if isinstance(dims, list) and dimension in dims and 'video_list' in prompt_dict:
            prompt = (
                prompt_dict.get(f'prompt_{lang}')
                or prompt_dict.get('prompt_en')
                or prompt_dict.get('prompt')
                or ''
            )
            cur_video_list = ensure_list(prompt_dict.get('video_list'))
            cur_audio_list = ensure_list(prompt_dict.get('audio_list'))

            video_list += cur_video_list
            audio_list += cur_audio_list

            prompt_vision = prompt_dict.get('prompt_vision', '')
            prompt_audio = prompt_dict.get('prompt_audio', '')

            prompt_dict_item = {
                'prompt': prompt,
                'video_list': cur_video_list,
                'audio_list': cur_audio_list,
                'prompt_vision': prompt_vision,
                'prompt_audio': prompt_audio,
            }
            if 'auxiliary_info_vision' in prompt_dict:
                prompt_dict_item['auxiliary_info_vision'] = prompt_dict['auxiliary_info_vision']
            if 'auxiliary_info_audio' in prompt_dict:
                prompt_dict_item['auxiliary_info_audio'] = prompt_dict['auxiliary_info_audio']
            prompt_dict_ls.append(prompt_dict_item)
    return video_list, audio_list, prompt_dict_ls


def google_drive(model, file_id, output_path):
    file = f"{CACHE_DIR}/{model}"
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(file, exist_ok=True)
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Model downloaded successfully to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        

def init_submodules(dimension_list, local=False):
    submodules_dict = {}
    for dimension in dimension_list:
        os.makedirs(CACHE_DIR, exist_ok=True)


        if dimension == 'first_dnsmos':
            submodules_dict[dimension] = {
                "dns_mos_primary": f'{CACHE_DIR}/dns_mos/sig_bak_ovr.onnx',
                "dns_mos_p808": f'{CACHE_DIR}/dns_mos/model_v8.onnx'
            }

            if not os.path.exists(submodules_dict[dimension]['dns_mos_primary']):
                wget_command = ['wget', 'https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx', '-P', os.path.dirname(submodules_dict[dimension]["dns_mos_primary"])]
                subprocess.run(wget_command, check=True)

            if not os.path.exists(submodules_dict[dimension]['dns_mos_p808']):
                wget_command = ['wget', 'https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx', '-P', os.path.dirname(submodules_dict[dimension]["dns_mos_p808"])]
                subprocess.run(wget_command, check=True)


        elif dimension == 'first_nisqa':
            submodules_dict[dimension] = {
                'nisqa_model_path': f'{CACHE_DIR}/nisqa/nisqa_mos_only.tar',
                'nisqa_model_type': None
            }

            if not os.path.exists(submodules_dict[dimension]['nisqa_model_path']):
                wget_command = ['wget', 'https://raw.githubusercontent.com/gabrielmittag/NISQA/master/weights/nisqa_mos_only.tar', '-P', os.path.dirname(submodules_dict[dimension]["nisqa_model_path"])]
                subprocess.run(wget_command, check=True)



        elif dimension == 'first_audiobox':
            submodules_dict[dimension] = {
                "first_audiobox_model_path": f'{CACHE_DIR}/audiobox_aesthetics/checkpoint.pt'
            }

            if not os.path.exists(submodules_dict[dimension]['first_audiobox_model_path']):
                wget_command = ['wget', 'https://huggingface.co/facebook/audiobox-aesthetics/blob/main/checkpoint.pt', '-P', os.path.dirname(submodules_dict[dimension]["first_audiobox_model_path"])]
                subprocess.run(wget_command, check=True)



        elif dimension == 'second_clap':
            submodules_dict[dimension] = {
                "second_clap_model_name": f"{CACHE_DIR}/clap/laion__clap-htsat-unfused"
            }
            if not os.path.exists(submodules_dict[dimension]['second_clap_model_name']):
                model_dir = os.path.dirname(submodules_dict[dimension]["second_clap_model_name"])
                os.makedirs(model_dir, exist_ok=True)

                snapshot_download(
                    repo_id="laion/clap-htsat-unfused",
                    local_dir=model_dir,
                    local_dir_use_symlinks=False 
                )


        elif dimension  == 'second_viclip':
            submodules_dict[dimension] = {
                "second_viclip_model_name": f"{CACHE_DIR}/viclip/ViCLIP-L_InternVid-FLT-10M.pth"
            }

            if not os.path.exists(submodules_dict[dimension]['second_viclip_model_name']):
                wget_command = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/blob/main/ViCLIP-L_InternVid-FLT-10M.pth', '-P', os.path.dirname(submodules_dict[dimension]["second_viclip_model_name"])]
                subprocess.run(wget_command, check=True)


        elif dimension == 'second_lsa':
            # SyncNet2 模型配置
            submodules_dict[dimension] = {
                "syncnet2_model_path": f'{CACHE_DIR}/syncnet2/syncnet_v2.model'
            }

            if not os.path.exists(submodules_dict[dimension]['syncnet2_model_path']):
                wget_command = ['wget', 'https://huggingface.co/ByteDance/LatentSync-1.6/blob/main/auxiliary/syncnet_v2.model', '-P', os.path.dirname(submodules_dict[dimension]["syncnet2_model_path"])]
                subprocess.run(wget_command, check=True)



        elif dimension == 'second_desync':
            # Synchformer（去同步检测）所需配置
            cfg_dir = f"{CACHE_DIR}/synchformer/configs"
            ckpt_dir = f"{CACHE_DIR}/synchformer/checkpoints"
            
            model_id = "24-01-04T16-39-21"  # 21类偏移量检测

            submodules_dict[dimension] = {
                'cfg_path':  f"{cfg_dir}/cfg-{model_id}.yaml",
                'ckpt_path': f"{ckpt_dir}/{model_id}.pt",
            }

            if not os.path.exists(submodules_dict[dimension]['cfg_path']):
                ckpt_dir = os.path.dirname(submodules_dict[dimension]["cfg_path"])
                os.makedirs(ckpt_dir, exist_ok=True)  # 确保目录存在

                url = "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/cfg-24-01-04T16-39-21.yaml"
                
                wget_command = ['wget', '-O', submodules_dict[dimension]['cfg_path'], url]
                subprocess.run(wget_command, check=True)

            if not os.path.exists(submodules_dict[dimension]['ckpt_path']):
                ckpt_dir = os.path.dirname(submodules_dict[dimension]["ckpt_path"])
                os.makedirs(ckpt_dir, exist_ok=True)  # 确保目录存在

                url = "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/sync/sync_models/24-01-04T16-39-21/24-01-04T16-39-21.pt"
                
                wget_command = ['wget', '-O', submodules_dict[dimension]['ckpt_path'], url]
                subprocess.run(wget_command, check=True)

        elif dimension == 'second_imagebind':
            submodules_dict[dimension] = {
                'second_imagebind_model_name': f'{CACHE_DIR}/imagebind/imagebind_huge.pth'
            }
 

        else:
            submodules_dict[dimension]={}

    return submodules_dict



def save_json(data, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def load_json(path):
    """
    Load a JSON file from the given file path.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    
    Returns:
    - data (dict or list): The data loaded from the JSON file, which could be a dictionary or a list.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_prompt_from_filename(path: str):
    """
    1. prompt-0.suffix -> prompt
    2. prompt.suffix -> prompt
    """
    prompt = Path(path).stem
    number_ending = r'-\d+$' # checks ending with -<number>
    if re.search(number_ending, prompt):
        return re.sub(number_ending, '', prompt)
    return prompt