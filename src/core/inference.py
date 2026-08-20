"""
实时推理与模型测试引擎 (Inference Engine)
"""
from typing import List, Tuple, Dict, Any, Optional, Union
import os
import time

from src.core.annotation import BoundingBox, get_class_color
from src.utils.logger import logger


class InferenceEngine:
    """支持 PyTorch (.pt) 与 ONNX (.onnx) 双后端的统一推理引擎"""

    def __init__(self):
        self.model = None
        self.model_path: Optional[str] = None
        self.class_names: List[str] = []
        self.is_onnx: bool = False
        self.onnx_session = None

    def load_model(self, model_path: str, device: str = "auto") -> bool:
        """加载模型权重文件"""
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return False

        self.model_path = model_path
        ext = os.path.splitext(model_path)[1].lower()

        try:
            if ext == ".onnx":
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
                self.onnx_session = ort.InferenceSession(model_path, providers=providers)
                self.is_onnx = True
                self.model = None
                self.class_names = ["target"]
                logger.info(f"成功加载 ONNX 模型: {model_path}")
                return True
            else:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.is_onnx = False
                self.onnx_session = None
                if hasattr(self.model, "names") and self.model.names:
                    if isinstance(self.model.names, dict):
                        self.class_names = [self.model.names[k] for k in sorted(self.model.names.keys())]
                    elif isinstance(self.model.names, list):
                        self.class_names = self.model.names
                logger.info(f"成功加载 YOLO 模型: {model_path}，包含类别: {self.class_names}")
                return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False

    def predict_image(
        self,
        image_input: Any,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Tuple[List[BoundingBox], float, Any]:
        """
        对单张图片执行推理
        :param image_input: 图像文件路径或 numpy BGR 图像数组
        :param conf_threshold: 置信度阈值
        :param iou_threshold: NMS IoU 阈值
        :return: (检测框列表, 前向耗时毫秒, 绘制了检测结果的 BGR 图像)
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            return [], 0.0, None

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                return [], 0.0, np.zeros((100, 100, 3), dtype=np.uint8)
            img = cv2.imread(image_input)
        else:
            img = image_input.copy()

        if img is None:
            return [], 0.0, np.zeros((100, 100, 3), dtype=np.uint8)

        h, w, _ = img.shape
        start_time = time.perf_counter()
        boxes: List[BoundingBox] = []

        if self.model is not None:
            results = self.model.predict(
                source=img,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            if results and len(results) > 0:
                res = results[0]
                pred_boxes = res.boxes
                if pred_boxes is not None and len(pred_boxes) > 0:
                    for i in range(len(pred_boxes)):
                        xywhn = pred_boxes.xywhn[i].cpu().numpy().tolist()
                        cls_id = int(pred_boxes.cls[i].item())
                        conf = float(pred_boxes.conf[i].item())
                        boxes.append(BoundingBox(
                            class_id=cls_id,
                            x_center=float(xywhn[0]),
                            y_center=float(xywhn[1]),
                            width=float(xywhn[2]),
                            height=float(xywhn[3]),
                            confidence=conf
                        ))

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        annotated_img = self.render_boxes(img, boxes)
        return boxes, latency_ms, annotated_img

    def render_boxes(self, image: Any, boxes: List[BoundingBox]) -> Any:
        """在图像上绘制边界框、标签与置信度"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            return image

        canvas = image.copy()
        h, w, _ = canvas.shape

        for b in boxes:
            x1, y1, x2, y2 = b.to_xyxy(w, h)
            cls_name = self.class_names[b.class_id] if b.class_id < len(self.class_names) else f"class_{b.class_id}"
            label_text = f"{cls_name} {b.confidence:.2f}"

            hex_color = get_class_color(b.class_id).lstrip("#")
            bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), bgr_color, 2)

            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = max(0, y1 - text_h - 6)
            label_y2 = y1
            cv2.rectangle(canvas, (x1, label_y1), (x1 + text_w + 6, label_y2), bgr_color, -1)
            cv2.putText(
                canvas,
                label_text,
                (x1 + 3, label_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return canvas
