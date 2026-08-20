# YOLO Studio - 开发者指南与二次开发维护手册 (Developer Guide)

**文档版本**：v1.0.0  
**适用对象**：系统维护人员、二次开发工程师、功能扩展开发者

---

## 1. 开发环境搭建与依赖配置

### 1.1 Python 运行时要求
- **Python 版本**：推荐 Python 3.10 ~ 3.12（兼容 3.9+）
- **包管理工具**：`pip` / `conda` / `uv`

### 1.2 安装依赖
在项目根目录下执行：
```bash
# 1. 创建并激活虚拟环境 (推荐)
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# 2. 安装 PyTorch (建议根据本地 GPU CUDA 版本安装匹配版本)
# 例如 CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖清单
pip install -r requirements.txt
```

---

## 2. 核心架构分层与目录导览

```text
yolo_studio/
├── Doc/                        # 技术规范与使用文档
├── agents/                     # 自动化与辅助管理模块 (数据集体检、参数推荐、自动化测试与自愈)
│   ├── base_agent.py           # 模块基类与生命周期协议
│   ├── dataset_audit_agent.py  # 数据集审查模块
│   ├── autotrain_agent.py      # 自适应训练策略模块
│   ├── app_test_agent.py       # 应用自动化测试与巡检模块
│   ├── bug_fix_agent.py        # 缺陷分析与自愈辅助模块
│   └── skills/                 # 独立工具库 (预打标/数据增强/导出/测试)
├── src/
│   ├── app.py                  # 应用启动主入口 (初始化 QApplication 与 MainWindow)
│   ├── core/                   # 领域逻辑与模型服务层 (无 UI 依赖)
│   │   ├── annotation.py       # 标注框几何模型与命令模式实现 (Undo/Redo)
│   │   ├── dataset_manager.py  # 数据集操作、目录组织、格式转换与划分
│   │   ├── trainer.py          # Ultralytics 训练线程包装与指标信号捕获
│   │   ├── inference.py        # 推理会话 (PyTorch & ONNX 双后端)
│   │   ├── autolabel.py        # 批量自动预打标流水线
│   │   └── exporter.py         # 多端模型导出与验证器
│   ├── ui/                     # 表现层 (PySide6 / Qt6)
│   │   ├── main_window.py      # 主界面框架与 Tab 路由
│   │   ├── canvas.py           # QGraphicsView 标注画布实现
│   │   ├── views/              # 各功能子工作区 (View)
│   │   ├── widgets/            # 复用控件 (图表、终端、类别列表、防抖画布)
│   │   └── styles/             # QSS 样式表与调色板
│   └── utils/                  # 基础设施工具层 (配置、日志、硬件探测)
└── tests/                      # 自动化测试用例
```

---

## 3. 核心机制实现指南

### 3.1 标注画布与命令模式 (Undo/Redo)
在 `src/core/annotation.py` 中，所有的画布编辑行为通过命令模式进行管理：
```python
from PySide6.QtGui import QUndoCommand

class AddBoxCommand(QUndoCommand):
    def __init__(self, annotation_model, box_data):
        super().__init__("Add Bounding Box")
        self.model = annotation_model
        self.box = box_data

    def redo(self):
        self.model.add_box(self.box)

    def undo(self):
        self.model.remove_box(self.box.id)
```
**规范**：数据变更通过 `undo_stack.push(Command)` 触发，保证用户可随时进行历史回退。

### 3.2 训练器工作线程与信号槽通信
在 `src/core/trainer.py` 中，训练任务运行在 `QThread` 中，通过 Qt Signal 向 UI 主线程分发事件：
```python
from PySide6.QtCore import QThread, Signal

class YoloTrainerWorker(QThread):
    # 定义强类型信号
    epoch_finished = Signal(int, dict)          # (epoch, metrics_dict)
    batch_progress = Signal(int, int, float, str)  # (current_batch, total_batches, loss, status)
    training_done = Signal(str)                  # (best_model_path)
    training_failed = Signal(str)                # (error_message)
    log_message = Signal(str)                    # (log_text)
```

### 3.3 自适应防抖画布与视口约束
在 `src/ui/widgets/video_canvas.py` 中，采用 `ImageDisplayCanvas` 重写 `paintEvent`，设置 `setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)`，防止连续视频帧渲染改变控件 `sizeHint` 从而导致窗口尺寸异常变化。

---

## 4. 扩展开发常见场景

### 4.1 新增自定义图像增强方法
在 `agents/skills/augmentation_skill.py` 或 `src/core/dataset_manager.py` 中添加对应的图像转换函数，确保在图像变换的同时按照对应仿射公式同步更新边界框坐标 $(x_{center}, y_{center}, width, height)$。

### 4.2 接入新型目标检测模型
1. 在 `src/ui/views/train_view.py` 的模型下拉列表中添加新模型代号（如 `yolo12n.pt`）；
2. 底层 `YoloTrainerWorker` 自动调用 Ultralytics 接口进行下载与微调；
3. 在 `src/core/exporter.py` 中确认新模型的导出参数兼容性。

---

## 5. 打包与分发

在项目根目录下使用 PyInstaller 进行单文件或目录打包：
```bash
pip install pyinstaller
pyinstaller --noconsole --name "YOLO_Studio" --icon="src/ui/assets/icon.ico" src/app.py
```
打包完成后，可执行文件将生成在 `dist/YOLO_Studio/` 目录下。
