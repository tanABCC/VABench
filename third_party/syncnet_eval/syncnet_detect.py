# Adapted from https://github.com/joonson/syncnet_python/blob/master/run_pipeline.py

import os, pdb, subprocess, glob, cv2
import numpy as np
from shutil import rmtree
import torch

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from .detectors import S3FD
from .latentsync.utils.face_detector import FaceDetector


class SyncNetDetector:
    def __init__(self, device, detect_results_dir="detect_results"):
        self.s3f_detector = S3FD(device=device)
        self.latentsync_face_detector = FaceDetector(device=device)
        self.detect_results_dir = detect_results_dir

    def __call__(self, video_path: str, min_track=50, scale=False):
        crop_dir = os.path.join(self.detect_results_dir, "crop")
        video_dir = os.path.join(self.detect_results_dir, "video")
        frames_dir = os.path.join(self.detect_results_dir, "frames")
        temp_dir = os.path.join(self.detect_results_dir, "temp")

        # ========== DELETE EXISTING DIRECTORIES ==========
        if os.path.exists(crop_dir):
            rmtree(crop_dir)

        if os.path.exists(video_dir):
            rmtree(video_dir)

        if os.path.exists(frames_dir):
            rmtree(frames_dir)

        if os.path.exists(temp_dir):
            subdirs = os.listdir(temp_dir)
            # print(f"DEBUG: temp_dir 子目录: {subdirs}")
        
        if os.path.exists(crop_dir):
            # print(f"DEBUG: 删除 crop_dir: {crop_dir}")
            rmtree(crop_dir)
        if os.path.exists(video_dir):
            # print(f"DEBUG: 删除 video_dir: {video_dir}")
            rmtree(video_dir)
        if os.path.exists(frames_dir):
            # print(f"DEBUG: 删除 frames_dir: {frames_dir}")
            rmtree(frames_dir)
        # 不再删除整个 temp_dir，避免删除 affined 目录
        
        # print(f"DEBUG: 目录删除后状态:")
        # print(f"DEBUG: temp_dir 存在: {os.path.exists(temp_dir)}")
        if os.path.exists(temp_dir):
            subdirs = os.listdir(temp_dir)
                # print(f"DEBUG: temp_dir 子目录: {subdirs}")

        # ========== MAKE NEW DIRECTORIES ==========

        os.makedirs(crop_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========
        
        # 日志记录函数
        def log_ffmpeg_error(step_name, command, returncode, stderr, video_path):
            """记录 ffmpeg 错误到文件"""
            import datetime
            log_file = "video_conversion_errors.txt"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"时间: {timestamp}\n")
                f.write(f"步骤: {step_name}\n")
                f.write(f"视频: {video_path}\n")
                f.write(f"命令: {' '.join(command)}\n")
                f.write(f"返回码: {returncode}\n")
                f.write(f"错误信息:\n{stderr}\n")
                f.write(f"{'='*80}\n")
            print(f"    ❌ {step_name} 失败 (返回码: {returncode})，详情已记录到 {log_file}")

        # 检查输入视频是否存在
        if not os.path.exists(video_path):
            error_msg = f"输入视频不存在: {video_path}"
            print(f"❌ {error_msg}")
            log_ffmpeg_error("输入检查", ["N/A"], -1, error_msg, video_path)
            raise FileNotFoundError(error_msg)
        
        # 检查输入视频大小
        video_size = os.path.getsize(video_path)
        if video_size == 0:
            error_msg = f"输入视频为空文件: {video_path} (大小: 0 bytes)"
            print(f"❌ {error_msg}")
            log_ffmpeg_error("输入检查", ["N/A"], -1, error_msg, video_path)
            raise ValueError(error_msg)

        # 步骤1: 缩放视频（如果需要）
        if scale:
            scaled_video_path = os.path.join(video_dir, "scaled.mp4")
            command = ["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", video_path, "-vf", "scale=224:224", scaled_video_path]
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0 or not os.path.exists(scaled_video_path):
                print(f"❌ 视频缩放失败: {video_path}")
                log_ffmpeg_error("缩放视频", command, result.returncode, result.stderr, video_path)
                raise RuntimeError(f"视频缩放失败: {result.stderr}")
            
            video_path = scaled_video_path

        # 步骤2: 转换视频到25fps
        output_video = os.path.join(video_dir, 'video.mp4')
        command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", video_path, "-qscale:v", "2", "-async", "1", "-r", "25", output_video]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0 or not os.path.exists(output_video):
            print(f"❌ 视频转换到25fps失败: {video_path}")
            log_ffmpeg_error("转换视频到25fps", command, result.returncode, result.stderr, video_path)
            raise RuntimeError(f"视频转换失败: {result.stderr}")

        # 步骤3: 提取视频帧
        command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", output_video, "-qscale:v", "2", "-f", "image2", os.path.join(frames_dir, '%06d.jpg')]
        result = subprocess.run(command, capture_output=True, text=True)
        
        # 检查是否有帧被提取
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')] if os.path.exists(frames_dir) else []
        
        if result.returncode != 0 or len(frame_files) == 0:
            print(f"❌ 提取视频帧失败: {video_path} (提取到 {len(frame_files)} 帧)")
            log_ffmpeg_error("提取视频帧", command, result.returncode, result.stderr, video_path)
            raise RuntimeError(f"提取视频帧失败: {result.stderr}")

        # 步骤4: 提取音频
        audio_path = os.path.join(video_dir, 'audio.wav')
        command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", output_video, "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", audio_path]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0 or not os.path.exists(audio_path):
            # 有些视频可能没有音频轨道，是正常情况，不记录日志
            pass

        # 步骤5: 人脸检测
        try:
            faces = self.detect_face(frames_dir)
            num_frames_with_faces = sum(1 for frame_faces in faces if len(frame_faces) > 0)
            
            if num_frames_with_faces == 0:
                # 人脸检测失败是正常情况，不记录日志
                pass
        except Exception as e:
            # 只有真正的异常（非人脸检测失败）才记录
            error_msg = f"人脸检测失败: {str(e)}"
            print(f"❌ {error_msg}: {video_path}")
            log_ffmpeg_error("人脸检测", ["detect_face"], -1, error_msg, video_path)
            raise

        # 步骤6: 场景检测
        try:
            scene = self.scene_detect(video_dir)
        except Exception as e:
            error_msg = f"场景检测失败: {str(e)}"
            print(f"❌ {error_msg}: {video_path}")
            log_ffmpeg_error("场景检测", ["scene_detect"], -1, error_msg, video_path)
            raise

        # 步骤7: 人脸跟踪
        alltracks = []
        for shot in scene:
            if shot[1].frame_num - shot[0].frame_num >= min_track:
                tracks = self.track_face(faces[shot[0].frame_num : shot[1].frame_num], min_track=min_track)
                alltracks.extend(tracks)
        
        if len(alltracks) == 0:
            # 未找到人脸轨迹是正常情况，不记录日志
            pass

        # 步骤8: 裁剪人脸视频
        for ii, track in enumerate(alltracks):
            self.crop_video(track, os.path.join(crop_dir, "%05d" % ii), frames_dir, 25, temp_dir, video_dir)

        # 清理临时目录（添加错误处理）
        try:
            if os.path.exists(temp_dir):
                rmtree(temp_dir)
        except Exception as e:
            pass  # 忽略清理错误，不影响主流程

    def scene_detect(self, video_dir):
        video_manager = VideoManager([os.path.join(video_dir, "video.mp4")])
        stats_manager = StatsManager()
        scene_manager = SceneManager(stats_manager)
        # Add ContentDetector algorithm with higher threshold to reduce sensitivity
        scene_manager.add_detector(ContentDetector())  
        base_timecode = video_manager.get_base_timecode()
        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list(base_timecode)

        if scene_list == []:
            scene_list = [(video_manager.get_base_timecode(), video_manager.get_current_timecode())]

        return scene_list

    def track_face(self, scenefaces, num_failed_det=25, min_track=50, min_face_size=100):

        iouThres = 0.5  # Minimum IOU between consecutive face detections
        tracks = []

        while True:
            track = []
            for framefaces in scenefaces:
                for face in framefaces:
                    if track == []:
                        track.append(face)
                        framefaces.remove(face)
                    elif face["frame"] - track[-1]["frame"] <= num_failed_det:
                        iou = bounding_box_iou(face["bbox"], track[-1]["bbox"])
                        if iou > iouThres:
                            track.append(face)
                            framefaces.remove(face)
                            continue
                    else:
                        break

            if track == []:
                break
            elif len(track) > min_track:

                framenum = np.array([f["frame"] for f in track])
                bboxes = np.array([np.array(f["bbox"]) for f in track])

                frame_i = np.arange(framenum[0], framenum[-1] + 1)

                bboxes_i = []
                for ij in range(0, 4):
                    interpfn = interp1d(framenum, bboxes[:, ij])
                    bboxes_i.append(interpfn(frame_i))
                bboxes_i = np.stack(bboxes_i, axis=1)

                if (
                    max(np.mean(bboxes_i[:, 2] - bboxes_i[:, 0]), np.mean(bboxes_i[:, 3] - bboxes_i[:, 1]))
                    > min_face_size
                ):
                    tracks.append({"frame": frame_i, "bbox": bboxes_i})

        return tracks

    def detect_face(self, frames_dir, facedet_scale=0.25):
        flist = glob.glob(os.path.join(frames_dir, "*.jpg"))
        flist.sort()

        dets = []
        total_faces = 0

        for fidx, fname in enumerate(flist):
            image = cv2.imread(fname)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 使用 S3FD 检测器（与 LatentSync 保持一致）
            bboxes = self.s3f_detector.detect_faces(image_np, conf_th=0.9, scales=[facedet_scale])
            dets.append([])
            for bbox in bboxes:
                dets[-1].append({"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]})
        return dets

    def crop_video(self, track, cropfile, frames_dir, frame_rate, temp_dir, video_dir, crop_scale=0.4):
        
        flist = glob.glob(os.path.join(frames_dir, "*.jpg"))
        flist.sort()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_video = cropfile + "t.mp4"
        vOut = cv2.VideoWriter(temp_video, fourcc, frame_rate, (224, 224))
        
        if not vOut.isOpened():
            print(f"    警告: VideoWriter 无法打开 {temp_video}")
            return None

        dets = {"x": [], "y": [], "s": []}

        for det in track["bbox"]:

            dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
            dets["y"].append((det[1] + det[3]) / 2)  # crop center x
            dets["x"].append((det[0] + det[2]) / 2)  # crop center y

        # Smooth detections
        dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
        dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
        dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

        for fidx, frame in enumerate(track["frame"]):

            cs = crop_scale

            bs = dets["s"][fidx]  # Detection box size
            bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount

            image = cv2.imread(flist[frame])

            frame = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
            my = dets["y"][fidx] + bsi  # BBox center Y
            mx = dets["x"][fidx] + bsi  # BBox center X

            face = frame[int(my - bs) : int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs))]

            vOut.write(cv2.resize(face, (224, 224)))

        audiotmp = os.path.join(temp_dir, "audio.wav")
        audiostart = (track["frame"][0]) / frame_rate
        audioend = (track["frame"][-1] + 1) / frame_rate

        vOut.release()

        # ========== CROP AUDIO FILE ==========

        command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", os.path.join(video_dir, "audio.wav"), "-ss", f"{audiostart:.3f}", "-to", f"{audioend:.3f}", audiotmp]
        output = subprocess.run(command, stdout=None, stderr=subprocess.PIPE)
        
        if output.returncode != 0:
            print(f"    警告: 音频裁剪失败: {output.stderr.decode()}")

        sample_rate, audio = wavfile.read(audiotmp)

        # ========== COMBINE AUDIO AND VIDEO FILES ==========

        final_video = cropfile + ".mp4"
        command = ["ffmpeg", "-y", "-nostdin", "-loglevel", "error", "-i", temp_video, "-i", audiotmp, "-c:v", "copy", "-c:a", "aac", final_video]
        output = subprocess.run(command, stdout=None, stderr=subprocess.PIPE)
        
        if output.returncode != 0:
            print(f"    警告: ffmpeg 合并失败: {output.stderr.decode()}")
            return None
        
        # 清理临时文件
        if os.path.exists(temp_video):
            os.remove(temp_video)

        return {"track": track, "proc_track": dets}


def bounding_box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou
