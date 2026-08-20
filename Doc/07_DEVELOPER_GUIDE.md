# YOLO Studio - 开发者指南与二次开发维护手册 (Developer Guide)

**文档版本**：v1.0.0  
**适用对象**：系统维护人员、二次开发工程师、算法插件开发者

---

## 1. 开发环境搭建与依赖配置

### 1.1 Python 运行时要求
- **Python 版本**：推荐 Python 3.10 ~ 3.12（或兼容 3.9+）
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

# 2. 安装 PyTorch (建议根据本地 GPU CUDA 版本安装官方匹配版本)
# 例如 CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖清单
pip install -r requirements.txt
```

---

## 2. 核心架构分层与目录导览

```
yolo_studio/
├── Doc/                        # 10 大标准软件工程文档体系
├── agents/                     # 智能体层 (Agents & Skills 插件体系)
│   ├── base_agent.py           # 智能体基类与生命周期协议
│   ├── dataset_audit_agent.py  # 数据集审查智能体
│   ├── autotrain_agent.py      # 自适应训练策略智能体
│   └── skills/                 # 技能插件 (预打标/数据增强/导出)
├── src/
│   ├── app.py                  # 应用启动主入口 (初始化 QApplication 与 MainWindow)
│   ├── core/                   # 领域逻辑与模型服务层 (无 UI 依赖)
│   │   ├── annotation.py       # 标注框几何模型与命令模式实现 (Undo/Redo)
│   │   ├── dataset_manager.py  # 数据集操作、目录组织、格式转换与划分
│   │   ├── trainer.py          # Ultralytics 训练线程包装与指标信号捕获
│   │   ├── inference.py        # 推理会话 (PyTorch & ONNX 双后端)
│   │   ├── autolabel.py        # AI 辅助预打标流水线
│   │   └── exporter.py         # 多端模型导出与验证器
│   ├── ui/                     # 表现层 (PySide6 / Qt6)
│   │   ├── main_window.py      # 主界面框架与 Tab 路由
│   │   ├── canvas.py           # QGraphicsView 标注画布实现
│   │   ├── views/              # 各功能子工作区 (View)
│   │   ├── widgets/            # 复用控件 (图表、终端、类别列表)
│   │   └── styles/             # QSS 样式表与调色板
│   └── utils/                  # 基础设施工具层 (配置、日志、硬件探测)
└── tests/                      # 自动化测试用例
```

---

## 3. 核心机制实现指南

### 3.1 标注画布与命令模式 (Undo/Redo)
在 `src/core/annotation.py` 中，所有的画布编辑行为必须封装为 `QUndoCommand`：
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
**规范**：严禁在 UI 控件中直接硬编码删除或修改数据，必须通过 `undo_stack.push(Command)` 触发，以保证用户可随时撤销。

### 3.2 训练器子线程与信号槽通信机制
在 `src/core/trainer.py` 中，训练任务运行在 `QThread` 中，通过 Qt Signal 向 UI 主线程分发事件：
```python
from PySide6.QtCore import QThread, Signal

class YoloTrainWorker(QThread):
    # 定义强类型信号
    epoch_finished = Signal(int, dict)       # (epoch, metrics_dict)
    batch_progress = Signal(int, int, float)  # (current_batch, total_batches, loss)
    training_done = Signal(str)               # (best_model_path)
    training_failed = Signal(str)             # (error_message)

    def run(self):
        try:
            # 挂载 Ultralytics 回调并启动训练
            ...
        except Exception as e:
            self.training_failed.emit(str(e))
```

---

## 4. 如何扩展新功能 (二次开发教程)

### 4.1 新增一种模型架构支持 (如 YOLO-World 或 RT-DETR)
1. 打开 `src/core/trainer.py`，在 `SUPPORTED_MODELS` 字典中注册新增的模型权重或 YAML 配置名称；
2. 在 `src/ui/views/train_view.py` 中的模型下拉列表增加对应选项；
3. 更新 `tests/test_trainer.py` 添加对应的模型初始化单元测试。

### 4.2 编写并挂载一个新的 Skill 插件
1. 在 `agents/skills/` 目录下创建 `my_custom_skill.py`；
2. 继承 `BaseSkill` 并实现 `execute(**kwargs)` 方法；
3. 在 `agents/skills/__init__.py` 中暴露该技能，系统启动时将自动注册。

---

## 5. 打包为独立可执行程序 (.exe / .app)

项目使用 `PyInstaller` 实现单文件或文件夹形式的一键分发打包：

```bash
# Windows 平台打包
pyinstaller --noconfirm --onedir --windowed \
    --name "YOLO-Studio" \
    --add-data "src/ui/styles;src/ui/styles" \
    --hidden-import "ultralytics" \
    --hidden-import "torch" \
    --icon "assets/icon.ico" \
    src/app.py
```
打包生成的可执行文件位于 `dist/YOLO-Studio/YOLO-Studio.exe`。
