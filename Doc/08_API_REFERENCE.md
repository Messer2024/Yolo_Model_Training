# YOLO Studio - 核心模块 API 接口参考手册 (API Reference)

**文档版本**：v1.0.0  
**编制日期**：2026-08-20  
**项目名称**：YOLO Studio

---

## 1. 标注与几何模型 API (`src/core/annotation.py`)

### 1.1 `class BoundingBox`
目标检测矩形框的数据封装实体类。

```python
class BoundingBox:
    def __init__(self, class_id: int, x_center: float, y_center: float, 
                 width: float, height: float, confidence: float = 1.0, 
                 box_id: str = None)
```
- **属性**：
  - `class_id (int)`: 类别索引编号；
  - `x_center (float)`: 归一化中心 X 坐标 $[0.0, 1.0]$；
  - `y_center (float)`: 归一化中心 Y 坐标 $[0.0, 1.0]$；
  - `width (float)`: 归一化宽度 $[0.0, 1.0]$；
  - `height (float)`: 归一化高度 $[0.0, 1.0]$；
  - `confidence (float)`: 预测置信度（预标注或推理时有效，默认 1.0）；
  - `box_id (str)`: 唯一标识 UUID 字符串。
- **方法**：
  - `to_yolo_str() -> str`: 转换为标准 YOLO 格式字符串 `class_id x y w h`；
  - `to_xyxy(img_w: int, img_h: int) -> tuple[int, int, int, int]`: 转换为绝对像素坐标 $(x_1, y_1, x_2, y_2)$；
  - `from_xyxy(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, class_id: int) -> BoundingBox`: 静态工厂方法，从绝对坐标构建实例。

---

## 2. 数据集管理 API (`src/core/dataset_manager.py`)

### 2.1 `class DatasetManager`
负责数据集文件扫描、格式解析、健康度体检与子集切分。

- `load_project(project_dir: str) -> bool`: 加载并解析指定目录下的图片、标注与 `classes.txt`。
- `add_class(class_name: str, color_hex: str = None) -> int`: 添加新类别，返回类别 ID。
- `split_dataset(train_ratio: float = 0.8, val_ratio: float = 0.15, test_ratio: float = 0.05, output_dir: str = None) -> dict`: 按比例执行分层随机抽样，生成标准目录结构与 `data.yaml`。
- `audit_dataset() -> dict`: 执行健康度体检，返回包含 `corrupted_images`, `out_of_bounds_boxes`, `empty_images`, `class_distribution` 的字典。

---

## 3. YOLO 训练引擎 API (`src/core/trainer.py`)

### 3.1 `class YoloTrainerWorker(QThread)`
在独立子线程中承载 Ultralytics YOLO 训练任务。

- **Qt 信号 (Signals)**：
  - `epoch_end_signal = Signal(int, dict)`: 每轮 Epoch 结束时触发，回传当前轮数与指标字典（包含 `loss`, `mAP50`, `mAP50-95`, `precision`, `recall` 等）；
  - `progress_signal = Signal(int, int, float, str)`: 当前批次进度、总批次、实时损失与阶段描述；
  - `finished_signal = Signal(str)`: 训练完成时触发，回传生成的最优模型权重路径 `best.pt`；
  - `error_signal = Signal(str)`: 异常或 OOM 时触发，回传错误堆栈与排查建议。
- **核心方法**：
  - `start_training(config: TrainConfig) -> None`: 启动训练工作线程；
  - `stop_training() -> None`: 安全请求中断训练并保存当前断点；
  - `pause_training() / resume_training()`: 暂停与恢复控制。

---

## 4. 推理与评测引擎 API (`src/core/inference.py`)

### 4.1 `class InferenceEngine`
提供统一的 PyTorch `.pt` 与 ONNX `.onnx` 目标检测推理。

- `load_model(model_path: str, device: str = "auto") -> bool`: 加载并预热模型权重。
- `predict_image(image: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> tuple[list[BoundingBox], float]`: 对单张图像进行前向预测，返回检测框列表及前向耗时（ms）。
- `predict_batch(image_paths: list[str], ...) -> list[tuple[str, list[BoundingBox]]]`: 批量预测。

---

## 5. 模型多端导出 API (`src/core/exporter.py`)

### 5.1 `class ModelExporter`
负责模型格式转换与量化。

- `export_onnx(model_path: str, output_path: str = None, imgsz: int = 640, half: bool = False, dynamic: bool = True) -> str`: 导出为 ONNX 格式；
- `export_tensorrt(model_path: str, imgsz: int = 640, half: bool = True) -> str`: 导出为 TensorRT Engine；
- `validate_exported_model(exported_model_path: str, test_image: np.ndarray = None) -> bool`: 运行 ONNX Runtime 验证导出的模型是否可用。
