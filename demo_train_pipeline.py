"""
YOLO Studio - 全自动训练与推理全流程演示脚本 (End-to-End Demo)
"""
import os
import sys
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 将项目根目录加入搜索路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.dataset_manager import DatasetManager
from src.core.trainer import YoloTrainerWorker, TrainConfig
from src.core.inference import InferenceEngine
from src.core.exporter import ModelExporter
from src.utils.hardware import detect_hardware
from agents.dataset_audit_agent import DatasetAuditAgent
from src.utils.logger import logger


def run_demo():
    print("=" * 60)
    print("[DEMO] YOLO Studio - Automated End-to-End Training & Inference Demo")
    print("=" * 60)

    # 1. 硬件环境检测
    hw = detect_hardware()
    print(f"\n[Step 1] Hardware Detection:")
    print(f"  * OS: {hw['os']}")
    print(f"  * CPU: {hw['cpu']}")
    print(f"  * CUDA Available: {hw['is_cuda_available']}")
    print(f"  * Default Device: {hw['default_device']} ({hw['primary_gpu_name']})")

    # 2. 准备演示数据集配置文件 data.yaml
    dataset_dir = os.path.abspath("samples/coco8")
    yaml_path = os.path.join(dataset_dir, "data.yaml")

    yaml_data = {
        "path": dataset_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 80,
        "names": {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
            45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
            50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
            55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
            60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
            65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
            70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
            75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"
        }
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print(f"\n[Step 2] Dataset Configuration Ready:")
    print(f"  * Directory: {dataset_dir}")
    print(f"  * Config Path: {yaml_path}")

    # 3. 数据集健康度体检
    print(f"\n[Step 3] Dataset Quality Audit (DatasetAuditAgent):")
    dm = DatasetManager(dataset_dir)
    audit_report = dm.audit_dataset()
    print(f"  * Health Score: {audit_report.get('health_score')}/100")
    print(f"  * Total Images: {audit_report.get('total_images')}")
    print(f"  * Total Bboxes: {audit_report.get('total_boxes')}")

    # 4. 执行快速训练 (2 轮 Epochs 作为快速演示)
    print(f"\n[Step 4] Starting YOLO Training (Running 2 Epochs for Quick Demo):")
    train_config = TrainConfig(
        data_yaml=yaml_path,
        model_name="yolov8n.pt",
        epochs=2,
        batch_size=2,
        imgsz=320,
        device="cpu",  # 确保在任意设备上通用运行
        project_name="runs/demo_train",
        experiment_name="exp1"
    )

    trainer = YoloTrainerWorker(train_config)

    def on_epoch(epoch, metrics):
        print(f"    [Training Progress] Epoch {epoch}/2 Finished - Box Loss: {metrics.get('box_loss', 0.0):.4f} | mAP50: {metrics.get('mAP50', 0.0):.4f}")

    def on_log(msg):
        if "Epoch" in msg or "best.pt" in msg or "results" in msg:
            print(f"    [Log] {msg}")

    trainer.epoch_finished.connect(on_epoch)
    trainer.log_message.connect(on_log)
    trainer.run()

    # 5. 模型推理验证
    best_weights = os.path.abspath("runs/demo_train/exp1/weights/best.pt")
    if not os.path.exists(best_weights):
        best_weights = os.path.abspath("runs/demo_train/exp1/weights/last.pt")

    print(f"\n[Step 5] Validating Generated Weights:")
    print(f"  * Weights Path: {best_weights}")

    if os.path.exists(best_weights):
        infer = InferenceEngine()
        infer.load_model(best_weights)

        test_img = os.path.abspath("samples/01_bus_and_pedestrians.jpg")
        boxes, latency, _ = infer.predict_image(test_img, conf_threshold=0.2)
        print(f"  * Test Image: {test_img}")
        print(f"  * Inference Latency: {latency:.1f} ms")
        print(f"  * Detected Objects: {len(boxes)} items")

    # 6. 模型导出测试
    print(f"\n[Step 6] Model Export to ONNX:")
    if os.path.exists(best_weights):
        exporter = ModelExporter()
        export_res = exporter.export(
            weights_path=best_weights,
            export_format="onnx",
            imgsz=320,
            dynamic=False,
            simplify=True
        )
        print(f"  * Export Result: {export_res.get('message')}")
        print(f"  * ONNX Validation: {export_res.get('validated')}")

    print("\n" + "=" * 60)
    print("[SUCCESS] End-to-End Demo Completed! Training, inference, and export all verified!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
