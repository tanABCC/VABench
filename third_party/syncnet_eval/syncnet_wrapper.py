import os
import sys
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

# 动态添加 LatentSync 路径以支持其内部导入
LATENTSYNC_ROOT = "/root/autodl-tmp/LatentSync"   #TODO me 换地方
if LATENTSYNC_ROOT not in sys.path:
    sys.path.insert(0, LATENTSYNC_ROOT)

from .syncnet_eval import SyncNetEval
from .syncnet_detect import SyncNetDetector


class VABenchSyncNet:
    """VABench 专用 SyncNet 包装器"""
    
    def __init__(self, device: str = "cuda", syncnet_model_path: Optional[str] = None):
        """
        初始化 SyncNet 评估器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
            syncnet_model_path: SyncNet 模型路径，默认使用 LatentSync 的模型
        """
        self.device = device
        
        if syncnet_model_path is None:
            syncnet_model_path = os.path.join(LATENTSYNC_ROOT, "checkpoints/auxiliary/syncnet_v2.model")
        
        if not os.path.exists(syncnet_model_path):
            raise FileNotFoundError(
                f"SyncNet model not found at: {syncnet_model_path}\n"
            )
        
        # 初始化 SyncNet 评估器
        self.syncnet = SyncNetEval(device=device)
        self.syncnet.loadParameters(syncnet_model_path)
        
        # 初始化人脸检测器
        self.syncnet_detector = SyncNetDetector(
            device=device, 
            detect_results_dir="vabench_syncnet_temp"
        )
        
        print(f"✅ SyncNet loaded from: {syncnet_model_path}")
    
    def compute_sync_score(
        self, 
        video_path: str, 
        temp_dir: str = "vabench_syncnet_temp",
        min_track: int = 30,
        normalize: bool = True
    ) -> Tuple[float, int, float]:
        """
        计算视频的音唇同步分数
        
        Args:
            video_path: 视频文件路径
            temp_dir: 临时文件目录
            min_track: 最小人脸跟踪帧数
            normalize: 是否归一化到 0-1 范围（VABench 标准）
            
        Returns:
            tuple: (归一化分数, AV偏移, 原始置信度分数)
                - 归一化分数: 0-1 范围，用于 VABench
                - AV偏移: 音视频偏移帧数（负值表示音频滞后）
                - 原始置信度: SyncNet 原始分数 0-10
        """
        try:
            # 为此次评估创建独立的检测结果目录
            detect_results_dir = os.path.join(temp_dir, "detect_results")
            os.makedirs(detect_results_dir, exist_ok=True)
            
            # Step 1: 人脸检测与裁剪（使用独立的临时目录）
            # 临时更新 syncnet_detector 的检测结果目录
            original_detect_dir = self.syncnet_detector.detect_results_dir
            self.syncnet_detector.detect_results_dir = detect_results_dir
            
            self.syncnet_detector(video_path=video_path, min_track=min_track)
            
            # 恢复原始设置
            self.syncnet_detector.detect_results_dir = original_detect_dir
            
            crop_dir = os.path.join(detect_results_dir, "crop")
            crop_videos = os.listdir(crop_dir) if os.path.exists(crop_dir) else []
            
            if not crop_videos:
                print(f"❌ No face detected in {video_path}, returning default score")
                return 0.0, 0, 0.0
            
            # Step 2: 计算同步分数
            av_offset_list = []
            conf_list = []
            
            # 评估阶段使用与裁剪目录不同的子临时目录，避免被清理掉裁剪结果
            eval_temp_root = os.path.join(temp_dir, "eval")

            for video in crop_videos:
                crop_path = os.path.join(crop_dir, video)
                # 为每个裁剪视频单独的评估临时目录，防止并发/多样本相互影响
                eval_temp_dir = os.path.join(eval_temp_root, os.path.splitext(video)[0])
                av_offset, _, conf = self.syncnet.evaluate(
                    video_path=crop_path,
                    temp_dir=eval_temp_dir
                )
                av_offset_list.append(av_offset)
                conf_list.append(conf)
            
            # 取平均值
            avg_av_offset = int(np.mean(av_offset_list))
            avg_conf = float(np.mean(conf_list))
            
            # 归一化到 0-1（SyncNet 输出范围 0-10）
            normalized_score = min(1.0, max(0.0, avg_conf / 10.0)) if normalize else avg_conf
            
            return normalized_score, avg_av_offset, avg_conf
            
        except Exception as e:
            import traceback
            import datetime
            
            error_str = str(e).lower()
            error_traceback = traceback.format_exc()
            
            # 判断是否是人脸检测失败错误（不记录日志）
            is_face_detection_error = any(keyword in error_str for keyword in [
                'no face detected',
                'face not detected',
                'no face',
                '未检测到人脸',
                '人脸检测失败',
                '未找到人脸'
            ])
            
            if not is_face_detection_error:
                # 只记录非人脸检测错误
                print(f"❌ Error computing sync score for {video_path}: {e}")
                
                log_file = "syncnet_evaluation_errors.txt"
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"时间: {timestamp}\n")
                    f.write(f"步骤: SyncNet评估\n")
                    f.write(f"视频: {video_path}\n")
                    f.write(f"临时目录: {temp_dir}\n")
                    f.write(f"错误类型: {type(e).__name__}\n")
                    f.write(f"错误信息: {str(e)}\n")
                    f.write(f"详细堆栈:\n{error_traceback}\n")
                    f.write(f"{'='*80}\n")
                
                print(f"   详细错误信息已记录到: {log_file}")
                traceback.print_exc()
            
            return 0.0, 0, 0.0
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        import glob
        
        # 清理所有 vabench_syncnet_temp_* 目录（每个片段的独立临时目录）
        temp_patterns = ["vabench_syncnet_temp_*", "detect_results", "temp"]
        for pattern in temp_patterns:
            for temp_dir in glob.glob(pattern):
                if os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception as e:
                        print(f"Warning: Failed to remove {temp_dir}: {e}")


