"""
数据集健康度审查与质量诊断智能体 (DatasetAuditAgent)
"""
from typing import Any, Dict, List
import os
from agents.base_agent import BaseAgent
from agents.skills.auto_label_skill import AutoLabelSkill
from agents.skills.augmentation_skill import AugmentationSkill


class DatasetAuditAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DatasetAuditAgent",
            role="数据集质量审查、异常检测与健康度诊断专家"
        )
        self.register_skill(AutoLabelSkill())
        self.register_skill(AugmentationSkill())

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行数据集审计并生成报告与改进建议
        context:
          - image_files: List[str]
          - labels_map: Dict[str, List[Dict[str, Any]]]  # {image_path: [box, ...]}
          - class_names: List[str]
        """
        image_files = context.get("image_files", [])
        labels_map = context.get("labels_map", {})
        class_names = context.get("class_names", [])

        total_images = len(image_files)
        total_boxes = 0
        empty_images = []
        out_of_bounds_boxes = 0
        zero_area_boxes = 0
        class_counts = {i: 0 for i in range(len(class_names))}
        unregistered_class_boxes = 0

        for img_path in image_files:
            boxes = labels_map.get(img_path, [])
            if not boxes:
                empty_images.append(img_path)
                continue

            for b in boxes:
                total_boxes += 1
                cls_id = b.get("class_id", -1)
                xc = b.get("x_center", 0.0)
                yc = b.get("y_center", 0.0)
                w = b.get("width", 0.0)
                h = b.get("height", 0.0)

                # 检查边界越界
                if xc < 0.0 or xc > 1.0 or yc < 0.0 or yc > 1.0 or w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
                    out_of_bounds_boxes += 1
                if w * h <= 0.00001:
                    zero_area_boxes += 1

                if 0 <= cls_id < len(class_names):
                    class_counts[cls_id] += 1
                else:
                    unregistered_class_boxes += 1

        # 计算健康度评分 (满分 100)
        score = 100
        issues = []
        suggestions = []

        if total_images == 0:
            score = 0
            issues.append("当前未导入任何图像。")
            suggestions.append("请先点击【打开目录】导入需要标注或训练的图片。")
        else:
            if len(empty_images) > 0:
                empty_ratio = len(empty_images) / total_images
                if empty_ratio > 0.5:
                    score -= 30
                    issues.append(f"有 {len(empty_images)} 张图片（占比 {empty_ratio:.1%}）未做标注。")
                    suggestions.append("可点击【AI 辅助预标注】技能对未标注图片进行批量快速打标。")
                elif empty_ratio > 0.1:
                    score -= 10
                    issues.append(f"发现 {len(empty_images)} 张未标注图片。")

            if out_of_bounds_boxes > 0:
                score -= 15
                issues.append(f"发现 {out_of_bounds_boxes} 个越界标注框。")
                suggestions.append("系统可自动裁剪越界坐标至合法图像边界内。")

            if zero_area_boxes > 0:
                score -= 15
                issues.append(f"发现 {zero_area_boxes} 个面积极小或退化的无效框。")
                suggestions.append("建议在标注界面批量清理无效噪点框。")

            if unregistered_class_boxes > 0:
                score -= 20
                issues.append(f"发现 {unregistered_class_boxes} 个类别 ID 超出定义的框。")

            # 类别分布均匀度
            if total_boxes > 0 and len(class_names) > 1:
                counts = list(class_counts.values())
                max_c, min_c = max(counts), min(counts)
                if min_c == 0:
                    score -= 15
                    issues.append("存在未被任何标注框使用的类别。")
                elif max_c / (min_c + 1e-5) > 5.0:
                    score -= 10
                    issues.append("不同类别间样本数量差异较大（不平衡）。")
                    suggestions.append("可对稀缺类别样本使用【数据增强】技能扩充样本数量。")

        score = max(0, min(100, score))

        return {
            "health_score": score,
            "total_images": total_images,
            "total_boxes": total_boxes,
            "empty_images_count": len(empty_images),
            "class_distribution": {class_names[i] if i < len(class_names) else f"class_{i}": cnt
                                  for i, cnt in class_counts.items()},
            "issues": issues,
            "suggestions": suggestions
        }
