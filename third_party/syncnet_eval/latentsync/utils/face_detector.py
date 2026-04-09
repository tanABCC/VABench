from insightface.app import FaceAnalysis
import numpy as np
import torch
import os
import logging

INSIGHTFACE_DETECT_SIZE = 512

# 优先从环境变量 VABENCH_CACHE_DIR 中读取 checkpoints 路径
_cache_root = os.environ.get("VABENCH_CACHE_DIR")
if _cache_root:
    _INSIGHTFACE_ROOT = os.path.join(_cache_root, "checkpoints", "auxiliary")
else:
    # 回退到原来的相对路径，兼容旧用法
    _INSIGHTFACE_ROOT = "checkpoints/auxiliary"


class FaceDetector:
    def __init__(self, device="cuda"):
        # 禁用 insightface 和 onnxruntime 的详细日志
        os.environ['GLOG_minloglevel'] = '2'  # 禁用 glog
        logging.getLogger('insightface').setLevel(logging.ERROR)
        logging.getLogger('onnxruntime').setLevel(logging.ERROR)
        
        device_id = cuda_to_int(device)
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        
        print(f"    🔧 FaceDetector: device={device}, device_id={device_id}, current_device={torch.cuda.current_device()}")
        
        # 动态检测可用的providers，并明确指定 device_id
        try:
            import onnxruntime as ort
            # 禁用 onnxruntime 的详细输出
            ort.set_default_logger_severity(3)  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
            
            available_providers = ort.get_available_providers()
            
            if "CUDAExecutionProvider" in available_providers:
                cuda_provider_options = {
                    'device_id': device_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB 限制
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                providers = [
                    ('CUDAExecutionProvider', cuda_provider_options),
                    'CPUExecutionProvider'
                ]
                # print(f"  使用 GPU {device_id}")
            else:
                providers = ["CPUExecutionProvider"]
                # print(f" CUDA 不可用，使用 CPU")
        except Exception as e:
            # print(f" 无法检测 providers，使用 CPU: {e}")
            providers = ["CPUExecutionProvider"]
        
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root=_INSIGHTFACE_ROOT,
            providers=providers,
        )
        
        self.app.prepare(ctx_id=device_id, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))
        
        # 验证是否使用GPU
        # try:
        #     providers = getattr(self.app, 'providers', ['CPUExecutionProvider'])
        #     if "CUDAExecutionProvider" in providers:
        #         print("🚀 FaceDetector 已启用 GPU 加速")
        #     else:
        #         print("⚠️  FaceDetector 使用 CPU，可能较慢")
        # except AttributeError:
        #     print("⚠️  FaceDetector 使用 CPU（无法检测providers）")

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape

        faces = self.app.get(frame)

        get_face_store = None
        max_size = 0

        if len(faces) == 0:
            return None, None
        else:
            for face in faces:
                bbox = face.bbox.astype(np.int_).tolist()
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w < 50 or h < 80:
                    continue
                if w / h > 1.5 or w / h < 0.2:
                    continue
                if face.det_score < threshold:
                    continue
                size_now = w * h

                if size_now > max_size:
                    max_size = size_now
                    get_face_store = face

        if get_face_store is None:
            return None, None
        else:
            face = get_face_store
            lmk = np.round(face.landmark_2d_106).astype(np.int_)

            halk_face_coord = np.mean([lmk[74], lmk[73]], axis=0)  # lmk[73]

            sub_lmk = lmk[LMK_ADAPT_ORIGIN_ORDER]
            halk_face_dist = np.max(sub_lmk[:, 1]) - halk_face_coord[1]
            upper_bond = halk_face_coord[1] - halk_face_dist  # *0.94

            x1, y1, x2, y2 = (np.min(sub_lmk[:, 0]), int(upper_bond), np.max(sub_lmk[:, 0]), np.max(sub_lmk[:, 1]))

            if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
                x1, y1, x2, y2 = face.bbox.astype(np.int_).tolist()

            y2 += int((x2 - x1) * 0.1)
            x1 -= int((x2 - x1) * 0.05)
            x2 += int((x2 - x1) * 0.05)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(f_w, x2)
            y2 = min(f_h, y2)

            return (x1, y1, x2, y2), lmk


def cuda_to_int(cuda_like) -> int:
    """
    将输入（如 "cuda", "cuda:0", torch.device('cuda') 等）转换为 CUDA 设备整型 ID。
    """
    # 已是整型（防御）
    if isinstance(cuda_like, int):
        return int(cuda_like)

    # torch.device 输入
    if isinstance(cuda_like, torch.device):
        if cuda_like.type != "cuda":
            raise ValueError(f"Device type must be 'cuda', got: {cuda_like.type}")
        return torch.cuda.current_device() if cuda_like.index is None else int(cuda_like.index)

    # 字符串输入
    if isinstance(cuda_like, str):
        if cuda_like == "cuda":
            return torch.cuda.current_device()
        device = torch.device(cuda_like)
        if device.type != "cuda":
            raise ValueError(f"Device type must be 'cuda', got: {device.type}")
        return torch.cuda.current_device() if device.index is None else int(device.index)

    # 其他类型，尝试字符串化再解析
    device = torch.device(str(cuda_like))
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


LMK_ADAPT_ORIGIN_ORDER = [
    1,
    10,
    12,
    14,
    16,
    3,
    5,
    7,
    0,
    23,
    21,
    19,
    32,
    30,
    28,
    26,
    17,
    43,
    48,
    49,
    51,
    50,
    102,
    103,
    104,
    105,
    101,
    73,
    74,
    86,
]
