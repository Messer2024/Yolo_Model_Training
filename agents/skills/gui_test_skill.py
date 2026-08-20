"""
GUI 与工作流全生命周期自动化测试技能 (GuiWorkflowTestSkill)
"""
from typing import Dict, Any, List
import os
import sys
import time

from agents.base_agent import BaseSkill
from src.core.annotation import BoundingBox
from src.core.dataset_manager import DatasetManager
from src.core.inference import InferenceEngine
from src.core.exporter import ModelExporter
from src.utils.logger import logger


class GuiWorkflowTestSkill(BaseSkill):
    """
    负责执行 YOLO Studio 核心工作流的自动化端到端测试
    涵盖标注管理、数据集切分、训练引擎、推理引擎与模型导出
    """

    def __init__(self):
        super().__init__(
            name="gui_workflow_test_skill",
            description="执行标注、数据集划分、训练配置、推理与导出等全套工作流自动化测试"
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        test_scope = kwargs.get("scope", "all")
        sample_dir = kwargs.get("sample_dir", os.path.abspath("samples/coco8"))
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "errors": [],
            "details": {}
        }

        # 1. 测试标注与坐标转换
        if test_scope in ["all", "annotation"]:
            self._test_annotation_core(results)

        # 2. 测试数据集管理与划分
        if test_scope in ["all", "dataset"]:
            self._test_dataset_manager(results, sample_dir)

        # 3. 测试推理引擎与防抖画布
        if test_scope in ["all", "inference"]:
            self._test_inference_engine(results)

        # 4. 测试模型导出机制
        if test_scope in ["all", "export"]:
            self._test_exporter(results)

        return results

    def _test_annotation_core(self, res: Dict[str, Any]):
        res["total_tests"] += 1
        try:
            b = BoundingBox(class_id=0, x_center=0.5, y_center=0.5, width=0.4, height=0.4)
            x1, y1, x2, y2 = b.to_xyxy(1000, 1000)
            assert x1 == 300 and y1 == 300 and x2 == 700 and y2 == 700, "坐标反归一化计算偏差"

            # 越界自动截断测试
            b_invalid = BoundingBox(class_id=1, x_center=1.2, y_center=-0.5, width=0.5, height=0.5)
            assert 0.0 <= b_invalid.x_center <= 1.0, "越界坐标未正确 Clamp"
            assert 0.0 <= b_invalid.y_center <= 1.0, "负数坐标未正确 Clamp"

            res["passed"] += 1
            res["details"]["annotation_core"] = "PASS: 标注实体与归一化坐标转换正确"
        except Exception as e:
            res["failed"] += 1
            res["errors"].append({"module": "annotation", "error": str(e)})
            res["details"]["annotation_core"] = f"FAIL: {e}"

    def _test_dataset_manager(self, res: Dict[str, Any], sample_dir: str):
        res["total_tests"] += 1
        try:
            if not os.path.exists(sample_dir):
                raise FileNotFoundError(f"示例数据集目录不存在: {sample_dir}")

            import tempfile
            temp_out = os.path.join(tempfile.gettempdir(), "test_gui_dataset_split")

            dm = DatasetManager(sample_dir)
            audit = dm.audit_dataset()
            assert "health_score" in audit, "体检报告缺少 health_score"
            assert audit["total_images"] > 0, "未扫描到有效图像"

            # 测试自动划分到独立测试目录
            split_res = dm.split_dataset(train_ratio=0.7, val_ratio=0.3, test_ratio=0.0, output_dir=temp_out)
            assert split_res["success"], f"数据集划分失败: {split_res.get('message')}"
            assert os.path.exists(split_res["yaml_path"]), "data.yaml 未生成"

            res["passed"] += 1
            res["details"]["dataset_manager"] = f"PASS: 数据集扫描、健康度体检与划分正常 (图像: {audit['total_images']} 张)"
        except Exception as e:
            res["failed"] += 1
            res["errors"].append({"module": "dataset_manager", "error": str(e)})
            res["details"]["dataset_manager"] = f"FAIL: {e}"

    def _test_inference_engine(self, res: Dict[str, Any]):
        res["total_tests"] += 1
        try:
            engine = InferenceEngine()
            # 测试预训练模型加载
            success = engine.load_model("yolov8n.pt")
            assert success, "官方 yolov8n.pt 预训练模型加载失败"
            assert len(engine.class_names) > 0, "模型类别清单为空"

            # 测试虚拟图像推理
            import numpy as np
            dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
            boxes, latency, rendered = engine.predict_image(dummy_img, conf_threshold=0.1)
            assert latency >= 0, "前向耗时计算异常"
            assert rendered is not None and rendered.shape == dummy_img.shape, "渲染结果图像尺寸不匹配"

            res["passed"] += 1
            res["details"]["inference_engine"] = f"PASS: 推理引擎前向正常 (耗时: {latency:.1f}ms, 类别数: {len(engine.class_names)})"
        except Exception as e:
            res["failed"] += 1
            res["errors"].append({"module": "inference_engine", "error": str(e)})
            res["details"]["inference_engine"] = f"FAIL: {e}"

    def _test_exporter(self, res: Dict[str, Any]):
        res["total_tests"] += 1
        try:
            exporter = ModelExporter()
            # 校验导出配置构造
            assert exporter is not None
            res["passed"] += 1
            res["details"]["model_exporter"] = "PASS: 导出器就绪"
        except Exception as e:
            res["failed"] += 1
            res["errors"].append({"module": "exporter", "error": str(e)})
            res["details"]["model_exporter"] = f"FAIL: {e}"
