# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
import os
from typing import Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path=None) -> torch.Tensor:
    if mask_image_path is None:
        mask_image_path = os.path.join(os.path.dirname(__file__), "mask.png")
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        
        # 将 tensor 转换为 numpy 数组供 face_detector 使用
        if isinstance(image, torch.Tensor):
            # 确保是 (H, W, C) 格式
            if len(image.shape) == 3 and image.shape[0] == 3:  # (C, H, W)
                image_for_detection = image.permute(1, 2, 0).numpy()
            else:  # (H, W, C)
                image_for_detection = image.numpy()
        else:
            image_for_detection = image
        
        # 确保数据类型正确（uint8）
        if image_for_detection.dtype != np.uint8:
            image_for_detection = (image_for_detection * 255).astype(np.uint8) if image_for_detection.max() <= 1.0 else image_for_detection.astype(np.uint8)
        
        bbox, landmark_2d_106 = self.face_detector(image_for_detection)
        if bbox is None:
            raise RuntimeError("Face not detected")

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        landmarks3 = np.round([pt_left_eye, pt_right_eye, pt_nose])

        # 使用已经转换好的 numpy 数组
        face, affine_matrix = self.restorer.align_warp_face(image_for_detection.copy(), landmarks3=landmarks3, smooth=True)
        
        # 验证返回的face格式
        if face is None:
            raise RuntimeError("align_warp_face returned None")
        
        if not isinstance(face, np.ndarray):
            face = np.array(face)
        
        if len(face.shape) != 3 or face.shape[2] != 3:
            raise ValueError(f"Invalid face shape: {face.shape}, expected (H, W, 3)")
        
        if face.dtype != np.uint8:
            face = face.astype(np.uint8)
        
        # 确保数组是连续的
        if not face.flags['C_CONTIGUOUS']:
            face = np.ascontiguousarray(face)
        
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        
        # 调整大小到目标分辨率
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        success_count = 0
        
        for i, frame in enumerate(video_frames):
            try:
                # 确保帧格式正确
                if isinstance(frame, torch.Tensor):
                    frame_tensor = frame
                else:
                    frame_tensor = torch.from_numpy(frame)
                
                # 确保是 (C, H, W) 格式
                if len(frame_tensor.shape) == 3 and frame_tensor.shape[2] == 3:
                    frame_tensor = frame_tensor.permute(2, 0, 1)
                
                frame, _, _ = self.image_processor.affine_transform(frame_tensor)
                results.append(frame)
                success_count += 1
            except Exception as e:
                continue
        
        if success_count == 0:
            raise RuntimeError("所有帧都处理失败，未检测到人脸")
        
        # print(f"    成功处理 {success_count}/{len(video_frames)} 帧")
        results = torch.stack(results)
        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
