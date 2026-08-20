"""
硬件检测与 Agents/Skills 单元测试 (test_agents.py)
"""
import unittest
from src.utils.hardware import detect_hardware
from agents.dataset_audit_agent import DatasetAuditAgent
from agents.autotrain_agent import AutoTrainAgent


class TestAgentsAndHardware(unittest.TestCase):
    def test_hardware_detection(self):
        info = detect_hardware()
        self.assertIn("os", info)
        self.assertIn("cpu", info)
        self.assertIn("is_cuda_available", info)
        self.assertIn("default_device", info)

    def test_autotrain_agent_recommendations(self):
        agent = AutoTrainAgent()
        # 模拟 CPU 环境
        ctx_cpu = {
            "total_images": 50,
            "total_classes": 1,
            "is_cuda_available": False,
            "gpu_vram_gb": 0.0,
            "gpu_name": "CPU"
        }
        rec_cpu = agent.run(ctx_cpu)
        self.assertEqual(rec_cpu["recommended_model"], "yolov8n.pt")
        self.assertEqual(rec_cpu["device"], "cpu")

        # 模拟 16GB 显卡环境
        ctx_gpu = {
            "total_images": 1500,
            "total_classes": 5,
            "is_cuda_available": True,
            "gpu_vram_gb": 16.0,
            "gpu_name": "NVIDIA RTX 4090"
        }
        rec_gpu = agent.run(ctx_gpu)
        self.assertIn("yolov8x.pt", rec_gpu["recommended_model"])
        self.assertEqual(rec_gpu["device"], "0")
        self.assertEqual(rec_gpu["recommended_epochs"], 50)

    def test_dataset_audit_agent(self):
        agent = DatasetAuditAgent()
        ctx = {
            "image_files": ["img1.jpg", "img2.jpg"],
            "labels_map": {
                "img1.jpg": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.2, "height": 0.2}],
                "img2.jpg": []
            },
            "class_names": ["person"]
        }
        res = agent.run(ctx)
        self.assertEqual(res["total_images"], 2)
        self.assertEqual(res["total_boxes"], 1)
        self.assertEqual(res["empty_images_count"], 1)
        self.assertLess(res["health_score"], 100)


if __name__ == "__main__":
    unittest.main()
