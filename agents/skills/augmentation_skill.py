"""
图像与标注同步增强技能 (AugmentationSkill)
"""
from typing import Any, Dict, List, Tuple
import os
from agents.base_agent import BaseSkill


class AugmentationSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="augmentation_skill",
            description="对图像及其 YOLO 格式标注边界框执行同步几何与颜色空间数据增强"
        )

    def execute(self, image_path: str, boxes: List[Dict[str, Any]],
                transforms: List[str] = None) -> List[Tuple[Any, List[Dict[str, Any]], str]]:
        """
        对单张图片和对应标注执行增强，返回增强后的样本列表
        """
        if not os.path.exists(image_path):
            return []

        try:
            import cv2
            import numpy as np
        except ImportError:
            raise RuntimeError("执行图像增强需要安装 opencv-python 与 numpy 依赖库")

        image = cv2.imread(image_path)
        if image is None:
            return []

        if transforms is None:
            transforms = ["flip_h", "hsv_bright"]

        h, w, _ = image.shape
        augmented_results = []

        # 1. 水平翻转 (Horizontal Flip)
        if "flip_h" in transforms:
            flipped_img = cv2.flip(image, 1)
            flipped_boxes = []
            for b in boxes:
                flipped_boxes.append({
                    "class_id": b["class_id"],
                    "x_center": max(0.0, min(1.0, 1.0 - b["x_center"])),
                    "y_center": b["y_center"],
                    "width": b["width"],
                    "height": b["height"],
                    "confidence": b.get("confidence", 1.0)
                })
            augmented_results.append((flipped_img, flipped_boxes, "aug_fliph"))

        # 2. 垂直翻转 (Vertical Flip)
        if "flip_v" in transforms:
            flipped_img = cv2.flip(image, 0)
            flipped_boxes = []
            for b in boxes:
                flipped_boxes.append({
                    "class_id": b["class_id"],
                    "x_center": b["x_center"],
                    "y_center": max(0.0, min(1.0, 1.0 - b["y_center"])),
                    "width": b["width"],
                    "height": b["height"],
                    "confidence": b.get("confidence", 1.0)
                })
            augmented_results.append((flipped_img, flipped_boxes, "aug_flipv"))

        # 3. 亮度调亮 (Brightness Increase)
        if "hsv_bright" in transforms:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * 1.25
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            hsv = np.array(hsv, dtype=np.uint8)
            bright_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            augmented_results.append((bright_img, [b.copy() for b in boxes], "aug_bright"))

        # 4. 亮度调暗 (Brightness Decrease)
        if "hsv_dark" in transforms:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * 0.75
            hsv = np.array(hsv, dtype=np.uint8)
            dark_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            augmented_results.append((dark_img, [b.copy() for b in boxes], "aug_dark"))

        # 5. 高斯模糊 (Gaussian Blur)
        if "blur" in transforms:
            blurred_img = cv2.GaussianBlur(image, (5, 5), 0)
            augmented_results.append((blurred_img, [b.copy() for b in boxes], "aug_blur"))

        return augmented_results
