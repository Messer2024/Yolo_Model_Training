"""
AI 辅助智能预标注技能 (AutoLabelSkill)
"""
from typing import Any, Dict, List, Optional
import os
from agents.base_agent import BaseSkill


class AutoLabelSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="auto_label_skill",
            description="利用预训练 YOLO 模型自动对海量图像进行目标检测与边界框预打标"
        )
        self._model = None
        self._current_model_path = None

    def execute(self, image_paths: List[str], model_name: str = "yolov8n.pt",
                conf_threshold: float = 0.35, iou_threshold: float = 0.45,
                target_classes: Optional[List[int]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        执行批量预标注
        :param image_paths: 图像路径列表
        :param model_name: 模型权重名称或路径 (如 yolov8n.pt)
        :param conf_threshold: 置信度阈值
        :param iou_threshold: NMS IoU 阈值
        :param target_classes: 目标过滤类别 ID 列表
        :return: 映射字典 { image_path: [ {class_id, x_center, y_center, width, height, confidence} ] }
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError("未检测到 ultralytics 库，请先安装 requirements.txt")

        # 延迟加载或复用模型
        if self._model is None or self._current_model_path != model_name:
            self._model = YOLO(model_name)
            self._current_model_path = model_name

        results_map = {}
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue

            preds = self._model.predict(
                source=img_path,
                conf=conf_threshold,
                iou=iou_threshold,
                classes=target_classes,
                verbose=False
            )

            box_list = []
            if preds and len(preds) > 0:
                result = preds[0]
                img_h, img_w = result.orig_shape
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        xywhn = boxes.xywhn[i].cpu().numpy().tolist()
                        cls_id = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        box_list.append({
                            "class_id": cls_id,
                            "x_center": float(xywhn[0]),
                            "y_center": float(xywhn[1]),
                            "width": float(xywhn[2]),
                            "height": float(xywhn[3]),
                            "confidence": conf
                        })
            results_map[img_path] = box_list

        return results_map
