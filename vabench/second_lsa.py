import os
import shutil
import gc
from tqdm import tqdm
import torch
from statistics import fmean

try:
    import onnxruntime as ort
    ort.set_default_logger_severity(3)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
except ImportError:
    pass

# 禁用其他日志
import logging
logging.getLogger('insightface').setLevel(logging.ERROR)
logging.getLogger('onnxruntime').setLevel(logging.ERROR)
os.environ['GLOG_minloglevel'] = '2'

CACHE_DIR = os.environ.get('VABENCH_CACHE_DIR')

from vabench.utils import load_dimension_info
from vabench.memory_utils import periodic_cleanup

try:
    from third_party.syncnet_eval.syncnet_eval import SyncNetEval
    from third_party.syncnet_eval.syncnet_detect import SyncNetDetector
    SYNCNET_AVAILABLE = True
except ImportError as e:
    print(f"SyncNet not available: {e}")
    SYNCNET_AVAILABLE = False


class IntegratedVideoProcessor:
    
    def __init__(self, device='cuda', submodules_list=None):
        if isinstance(device, torch.device):
            self.device = str(device)
        elif isinstance(device, str):
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if not SYNCNET_AVAILABLE:
            raise ImportError("SyncNet 不可用，无法进行 LSA 评估")
        
        syncnet_model_path = submodules_list.get('syncnet2_model_path')
        if not os.path.isfile(syncnet_model_path):
            raise FileNotFoundError(f"SyncNet model file not found: {syncnet_model_path}")
        
        self.syncnet = SyncNetEval(device=self.device)
        self.syncnet.loadParameters(syncnet_model_path)
        
        self.syncnet_detector = SyncNetDetector(
            device=self.device,
            detect_results_dir="detect_results"
        )
        
        print(f"SyncNet loaded from: {syncnet_model_path}")

    def compute_lsa_score(self, video_path: str, temp_dir: str = "temp") -> float:

        try:
            detect_results_dir = "detect_results"
            
            self.syncnet_detector(video_path=video_path, min_track=50)
            
            crop_dir = os.path.join(detect_results_dir, "crop")
            crop_videos = os.listdir(crop_dir) if os.path.exists(crop_dir) else []
            
            if not crop_videos:
                print(f"No face detected in {video_path}")
                return 0.0
            
            av_offset_list = []
            conf_list = []
            
            for video in crop_videos:
                crop_video_path = os.path.join(crop_dir, video)
                av_offset, _, conf = self.syncnet.evaluate(
                    video_path=crop_video_path,
                    temp_dir=temp_dir
                )
                av_offset_list.append(av_offset)
                conf_list.append(conf)
            
            avg_conf = float(fmean(conf_list))

            
            return avg_conf
            
        except Exception as e:
            print(f"Error computing LSA score: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def process_and_evaluate_video(self, input_path: str, output_dir: str, temp_dir: str, prompt: str = ""):
        video_name = os.path.basename(input_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            lsa_score = self.compute_lsa_score(input_path, temp_dir=temp_dir)
            
            if lsa_score == 0.0:
                print(f"人脸检测失败或评估失败: {video_name}")
                return None
            
        except Exception as e:
            import datetime
            import traceback
            
            error_str = str(e).lower()
            error_traceback = traceback.format_exc()
            
            is_face_detection_error = any(keyword in error_str for keyword in [
                'no face detected',
                'face not detected',
                'no face'
            ])
            
            if not is_face_detection_error:
                # 只记录非人脸检测错误
                print(f"视频评估失败: {video_name} - {e}")
                
                log_file = "video_evaluation_errors.txt"
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"时间: {timestamp}\n")
                    f.write(f"步骤: 整视频LSA评估\n")
                    f.write(f"视频: {input_path}\n")
                    f.write(f"错误类型: {type(e).__name__}\n")
                    f.write(f"错误信息: {str(e)}\n")
                    f.write(f"详细堆栈:\n{error_traceback}\n")
                    f.write(f"{'='*80}\n")
            else:
                print(f"人脸检测失败: {video_name}")
            
            return None
        
        # 返回结果
        result = {
            'video_path': input_path,
            'video_results': round(float(lsa_score), 5),
            'prompt': prompt,
            'sync_score': round(float(lsa_score), 5)
        }
        
        del lsa_score
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def cleanup(self, temp_integrated_dir: str = None):
        # 清理通用临时目录
        temp_dirs = ["temp", "detect_results"]
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"清理 {temp_dir} 失败: {e}")
        
        # 清理主临时目录（temp_integrated）
        if temp_integrated_dir and os.path.exists(temp_integrated_dir):
            try:
                shutil.rmtree(temp_integrated_dir)
            except Exception as e:
                print(f"清理 {temp_integrated_dir} 失败: {e}")


def compute_second_lsa(json_dir, device, submodules_list, **kwargs):
    try:
        video_list, _, prompt_dict_ls = load_dimension_info(
            json_dir, 
            dimension='second_lsa', 
            lang='en'
        )
        
        video_to_prompt = {}
        for prompt_dict in prompt_dict_ls:
            prompt = prompt_dict.get('prompt', prompt_dict.get('prompt_en', ''))
            for video_path in prompt_dict['video_list']:
                video_to_prompt[video_path] = prompt
        
        output_dir = kwargs.get('output_dir', 'processed_videos')
        temp_dir = kwargs.get('temp_dir', 'temp_integrated')
        
        processor = IntegratedVideoProcessor(
            device=device,
            submodules_list=submodules_list
        )
        
        video_results = []
        for idx, video_path in enumerate(tqdm(video_list, desc="LSA Evaluation")):
            prompt = video_to_prompt.get(video_path, "")
            
            result = processor.process_and_evaluate_video(
                input_path=video_path,
                output_dir=output_dir,
                temp_dir=temp_dir,
                prompt=prompt
            )
            
            if result is not None:
                video_results.append(result)
            
            del result, video_path, prompt
            
            # 周期性内存清理：每处理 2 个视频清理一次（LSA 处理较重）
            periodic_cleanup(idx, interval=2)
        
        processor.cleanup(temp_integrated_dir=temp_dir)
        del processor, video_to_prompt
        gc.collect()        
        return None, video_results
        
    except Exception as e:
        print(f"Error in LSA evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, []

