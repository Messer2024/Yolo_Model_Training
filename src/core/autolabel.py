"""
AI 辅助智能预标注服务 (Auto-label Service)
"""
from typing import List, Dict, Any, Optional
from agents.skills.auto_label_skill import AutoLabelSkill
from src.core.annotation import BoundingBox
from src.utils.logger import logger


class AutoLabelEngine:
    """包装 AutoLabelSkill，提供给 UI 和工作流使用的预标注引擎"""

    def __init__(self):
        self.skill = AutoLabelSkill()

    def label_images(
        self,
        image_paths: List[str],
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None
    ) -> Dict[str, List[BoundingBox]]:
        """
        批量预打标并返回 BoundingBox 实体字典
        """
        logger.info(f"启动 AI 预打标: 共 {len(image_paths)} 张图片，使用模型 {model_name}")
        raw_results = self.skill.execute(
            image_paths=image_paths,
            model_name=model_name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            target_classes=target_classes
        )

        boxes_map: Dict[str, List[BoundingBox]] = {}
        for img_path, box_dicts in raw_results.items():
            box_list = [BoundingBox.from_dict(d) for d in box_dicts]
            boxes_map[img_path] = box_list

        logger.info(f"AI 预打标完成，累计生成 {sum(len(b) for b in boxes_map.values())} 个检测框")
        return boxes_map
