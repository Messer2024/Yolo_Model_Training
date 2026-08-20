"""
自适应训练策略与超参推荐智能体 (AutoTrainAgent)
"""
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from agents.skills.export_skill import ExportSkill


class AutoTrainAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="AutoTrainAgent",
            role="训练配置自适应决策与超参调优专家"
        )
        self.register_skill(ExportSkill())

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据硬件环境与数据集规模，智能推荐训练方案与超参数
        context:
          - total_images: int
          - total_classes: int
          - is_cuda_available: bool
          - gpu_vram_gb: float
          - gpu_name: str
        """
        total_images = context.get("total_images", 0)
        total_classes = context.get("total_classes", 1)
        is_cuda = context.get("is_cuda_available", False)
        vram_gb = context.get("gpu_vram_gb", 0.0)
        gpu_name = context.get("gpu_name", "CPU")

        # 推荐模型
        if not is_cuda or vram_gb < 4.0:
            recommended_model = "yolov8n.pt"
            recommended_batch = 8 if is_cuda else 4
            recommended_workers = 2
            reason = "检测到使用 CPU 或轻量级显卡，推荐使用轻量级 YOLOv8n 模型保证训练速度与稳定性。"
        elif vram_gb < 8.0:
            recommended_model = "yolov8s.pt"
            recommended_batch = 16
            recommended_workers = 4
            reason = f"检测到 {gpu_name} (显存 {vram_gb:.1f} GB)，推荐使用平衡型 YOLOv8s 模型。"
        elif vram_gb < 16.0:
            recommended_model = "yolov8m.pt"
            recommended_batch = 16
            recommended_workers = 8
            reason = f"检测到高性能显卡 {gpu_name} (显存 {vram_gb:.1f} GB)，推荐使用高精度 YOLOv8m/YOLO11m 模型。"
        else:
            recommended_model = "yolov8x.pt"
            recommended_batch = 32
            recommended_workers = 8
            reason = f"检测到专业级大显存显卡 {gpu_name} (显存 {vram_gb:.1f} GB)，推荐使用顶级精度 YOLOv8x 模型。"

        # 推荐训练轮数 (Epochs)
        if total_images < 200:
            recommended_epochs = 100
        elif total_images < 1000:
            recommended_epochs = 80
        else:
            recommended_epochs = 50

        # 学习率与优化器
        recommended_optimizer = "auto"
        recommended_imgsz = 640

        return {
            "recommended_model": recommended_model,
            "recommended_epochs": recommended_epochs,
            "recommended_batch": recommended_batch,
            "recommended_imgsz": recommended_imgsz,
            "recommended_workers": recommended_workers,
            "recommended_optimizer": recommended_optimizer,
            "device": "0" if is_cuda else "cpu",
            "reason": reason
        }
