"""
多端模型导出与量化技能 (ExportSkill)
"""
from typing import Any, Dict, List, Optional
import os
from agents.base_agent import BaseSkill


class ExportSkill(BaseSkill):
    def __init__(self):
        super().__init__(
            name="export_skill",
            description="将训练好的 YOLO PyTorch 权重一键转换为 ONNX, TensorRT, OpenVINO, CoreML, TFLite 等多端格式"
        )

    def execute(self, weights_path: str, format: str = "onnx",
                imgsz: int = 640, half: bool = False,
                dynamic: bool = False, simplify: bool = True) -> Dict[str, Any]:
        """
        执行模型导出
        :param weights_path: .pt 权重文件路径
        :param format: 目标格式 (onnx, engine, openvino, coreml, tflite, torchscript)
        :param imgsz: 输入分辨率
        :param half: 是否开启 FP16 半精度
        :param dynamic: 是否启用动态 Batch/Input 分辨率 (针对 ONNX)
        :param simplify: 是否简化 ONNX 计算图
        :return: { "success": bool, "exported_path": str, "message": str }
        """
        if not os.path.exists(weights_path):
            return {"success": False, "exported_path": "", "message": f"权重文件不存在: {weights_path}"}

        try:
            from ultralytics import YOLO
            model = YOLO(weights_path)
            export_result = model.export(
                format=format,
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                simplify=simplify
            )
            return {
                "success": True,
                "exported_path": str(export_result),
                "message": f"成功导出为 {format.upper()} 格式: {export_result}"
            }
        except Exception as e:
            return {
                "success": False,
                "exported_path": "",
                "message": f"导出失败: {str(e)}"
            }
