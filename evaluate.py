import torch
import os
from vabench import VABench
from datetime import datetime
import argparse
import yaml


def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VABench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./evaluation_results/',
        help="output directory to save the evaluation results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="main data directory containing video, audio, json subdirectories",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default=f'{CUR_DIR}/vabench/config.yaml',
        help="Path to a YAML config that defines per-dimension params and global params.")
    args = parser.parse_args()
    return args


def get_subfolders(data_dir):
    videos_path = os.path.join(data_dir, "video")
    audios_path = os.path.join(data_dir, "audio")
    json_path = os.path.join(data_dir, "json")
    
    
    video_subfolders = set(os.listdir(videos_path))
    audio_subfolders = set(os.listdir(audios_path))
    
    if os.path.exists(json_path):
        json_files = [f.replace('.json', '') for f in os.listdir(json_path) if f.endswith('.json')]
        json_subfolders = set(json_files)
        common_subfolders = video_subfolders.intersection(audio_subfolders).intersection(json_subfolders)
    
    return sorted(list(common_subfolders)), videos_path, audios_path, json_path


def evaluate_single_folder(videos_path, audios_path, subfolder, output_path, device,
                         dimensions, local,
                         json_path=None, **kwargs):
    current_videos_path = os.path.join(videos_path, subfolder)
    current_audios_path = os.path.join(audios_path, subfolder)
    
    if json_path:
        json_file_path = os.path.join(json_path, f"{subfolder}.json")
    else:
        print("Error: No JSON path provided")
        return
    
    my_VABench = VABench(device, json_file_path, output_path)
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    
    
    prompt = []
    my_VABench.evaluate(
        videos_path=current_videos_path,
        audios_path=current_audios_path,
        name=f'{subfolder}_{current_time}',
        prompt_list=prompt,
        dimension_list=dimensions,
        local=local,
        **kwargs
    )
    
    print(f"完成子文件夹 {subfolder} 的评估")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    kwargs = {}
    dimensions = None
    local = False
    # category = ''
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        global_params = cfg.get('params', {})
        if isinstance(global_params, dict):
            kwargs.update(global_params)
        if cfg.get('dimensions'):
            dimensions = cfg.get('dimensions')
    
    global_audios_path = []
    global_videos_path = []
        
    # 获取匹配的子文件夹和路径
    subfolders, videos_path, audios_path, json_path = get_subfolders(args.data_dir)

    for subfolder in subfolders:
        subfolder_output_path = os.path.join(args.output_dir, subfolder)
        os.makedirs(subfolder_output_path, exist_ok=True)

        generated_audios_path = os.path.join(audios_path, subfolder)
        generated_videos_path = os.path.join(videos_path, subfolder)
        generated_wavs = sorted([f for f in os.listdir(generated_audios_path)])
        generated_videos = sorted([f for f in os.listdir(generated_videos_path)])
        
        for wav in generated_wavs:
            global_audios_path.append(os.path.join(generated_audios_path, wav))
        for video in generated_videos:
            global_videos_path.append(os.path.join(generated_videos_path, video))
        
        # 添加同步：如果某个进程遇到异常，所有进程一起跳过
        evaluate_single_folder(
            videos_path, 
            audios_path, 
            subfolder, 
            subfolder_output_path,
            device, 
            dimensions, 
            local, 
            json_path,  # 传递JSON路径
            **kwargs
        )

        # 每个子文件夹评估后清理显存（但保留模型）
        import gc
        print(f"Cleaning memory after {subfolder}...")
        gc.collect()
        torch.cuda.empty_cache()

    print("Batch evaluation completed")


if __name__ == "__main__":

    exit_code = 0
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
        exit_code = 130  # Standard exit code for Ctrl+C
    except Exception as e:
        print(f"\n[Main] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
