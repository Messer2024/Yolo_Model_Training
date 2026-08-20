"""
模型多格式导出与校验引擎 (Model Exporter & Validator)
"""
from typing import Dict, Any, Optional
import os
from agents.skills.export_skill import ExportSkill
from src.utils.logger import logger


class ModelExporter:
    """负责将 YOLO PyTorch 模型导出为 ONNX, TensorRT, OpenVINO, CoreML 等格式并进行校验"""

    def __init__(self):
        self.skill = ExportSkill()

    def export(
        self,
        weights_path: str,
        export_format: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True
    ) -> Dict[str, Any]:
        """
        导出模型
        """
        logger.info(f"开始导出模型 {weights_path} 至 {export_format.upper()} 格式...")
        result = self.skill.execute(
            weights_path=weights_path,
            format=export_format,
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=simplify
        )

        if result.get("success", False) and export_format == "onnx":
            exported_file = result.get("exported_path")
            is_valid = self.validate_onnx(exported_file)
            result["validated"] = is_valid

        return result

    def validate_onnx(self, onnx_path: str) -> bool:
        """使用 onnx 与 onnxruntime 校验导出的 ONNX 文件是否完整可执行"""
        if not os.path.exists(onnx_path):
            return False
        try:
            import onnx
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            logger.info(f"ONNX 模型校验成功，输入签名: {[i.name for i in session.get_inputs()]}")
            return True
        except Exception as e:
            logger.warning(f"ONNX 校验警告: {e}")
            return False
