# YOLO Studio - 图形化目标检测模型训练工作站

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v9%20%7C%20v10%20%7C%2011-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**YOLO Studio** 是一个基于 Python 和 PySide6 开发的桌面端目标检测工作站。它将日常模型开发中的**数据标注、数据集划分与增强、模型配置与训练、效果验证以及格式导出**整合为一个清晰直观的图形化工具，帮助开发者更高效地完成模型迭代与交付。

---

## 主要功能

- **图形化交互标注**：基于 Qt 2D 图元渲染框架，支持以鼠标为中心的平滑缩放、画布拖拽平移、十字交叉辅助线、类别与颜色管理，并提供完整的撤销/重做（`Ctrl+Z` / `Ctrl+Y`）与快捷键操作。
- **批量自动预标注**：可载入现有的模型权重对批量图片进行初筛打标，只需在此基础上人工微调或核对，减少重复框选工作。
- **数据集管理与体检**：支持可视化调整 Train / Val / Test 切分比例，自动生成符合 YOLO 规范的目录结构与 `data.yaml` 文件；内置数据检查功能，排查坐标越界、空标注与破损图片。
- **离线数据增强**：支持水平翻转、垂直翻转、色彩调整、模糊与旋转等常见图像增强方式，并在生成新图片时自动完成边界框坐标的同步映射。
- **异步训练与实时大屏**：训练任务运行在独立工作线程中，主界面在计算过程中保持流畅响应；内置 Matplotlib 实时图表，动态刷新 Loss 损失曲线、mAP50、mAP50-95、Precision 和 Recall 等关键指标。
- **交互式推理测试台**：支持导入单张图片、批量图片或本地视频进行检测测试，可通过滑块实时调节置信度（Confidence）与 NMS IoU 阈值，直观对比检测效果。
- **多端模型导出**：支持一键将训练好的 PyTorch 权重（`.pt`）导出为 **ONNX、TensorRT Engine、OpenVINO、CoreML、TFLite** 等格式，并自动完成 ONNX 模型加载校验。
- **纯本地运行支持**：全套流程支持在完全离线、无外网环境下独立运行，确保数据隐私与资产安全。

---

## 目录结构

```text
Yolo_Model_Training/
├── Doc/                  # 详细的技术文档与使用手册
├── agents/               # 自动化辅助模块 (数据集体检、参数推荐、自动化测试与修复)
│   └── skills/           # 独立工具函数 (预标注、数据增强、模型转换、测试巡检)
├── samples/              # 演示图片与示例微缩数据集
├── src/
│   ├── app.py            # 程序主入口
│   ├── core/             # 标注数据模型、数据集管理、训练调度、推理与导出引擎
│   ├── ui/               # 基于 PySide6 的界面布局、画布组件与样式
│   │   ├── views/        # 标注、数据管理、训练、推理与导出等功能页面
│   │   ├── widgets/      # 绘图看板、终端日志、类别列表等通用控件
│   │   └── styles/       # 深色与浅色 QSS 主题
│   └── utils/            # 配置文件读写、日志工具与硬件环境检测
└── tests/                # 自动化单元测试与集成测试用例
```

---

## 快速开始

### 1. 环境准备

推荐使用 Python 3.10 ~ 3.12 环境。首先创建并激活虚拟环境：

```bash
# 创建虚拟环境
python -m venv venv

# Windows 激活:
.\venv\Scripts\activate

# Linux / macOS 激活:
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **说明**：如需使用 NVIDIA 显卡进行 GPU 加速训练，请根据本机显卡驱动与 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应的 PyTorch GPU 版本。

### 3. 启动软件

```bash
python src/app.py
```

---

## 基础使用流程

1. **导入与标注**：打开包含待标注图片的文件夹，在右侧添加需要的类别名称，在画布中框选目标对象。如果图片较多，可点击「批量自动预标注」快速生成初步候选框。
2. **数据集准备**：切换到「数据集管理」页面，检查数据质量，调整训练集与验证集比例，点击「划分数据集」生成标准 `data.yaml`。
3. **配置与训练**：切换到「模型训练」页面，选择模型尺寸（如 YOLOv8s / YOLO11s）和训练轮数，点击「开始训练」，右侧图表将实时绘制损失与精度变化曲线。
4. **推理与验证**：训练完成后，在「推理测试」页面可直接载入刚生成的 `best.pt`，选择测试图片或视频即可查看检测结果。
5. **模型导出**：在「模型导出」页面选择目标格式（如 ONNX 或 TensorRT），点击导出即可用于后续生产部署。

---

## 运行测试

项目中包含核心数据转换、几何计算、GUI 联动以及自动化巡检的测试用例，可通过以下命令执行验证：

```bash
python -m unittest discover tests
```

---

## 详细文档索引

更多详细设计说明与操作指引已整理在 `Doc/` 目录下：

- [01. 项目可行性与技术选型报告](Doc/01_FEASIBILITY_REPORT.md)
- [02. 软件需求规格说明书 (SRS)](Doc/02_REQUIREMENTS_SPECIFICATION.md)
- [03. 系统架构与详细设计说明书](Doc/03_SYSTEM_ARCHITECTURE.md)
- [04. 数据集结构与标注格式规范](Doc/04_DATASET_SPECIFICATION.md)
- [05. 自动化辅助模块与扩展规范](Doc/05_AGENTS_AND_SKILLS_SPECIFICATION.md)
- [06. 用户使用与操作指南](Doc/06_USER_MANUAL.md)
- [07. 开发者指南与二次开发手册](Doc/07_DEVELOPER_GUIDE.md)
- [08. 核心模块 API 接口参考手册](Doc/08_API_REFERENCE.md)
- [09. 测试计划与多平台打包部署规范](Doc/09_TEST_AND_DEPLOYMENT_PLAN.md)
- [10. Git 版本管理与 GitHub 备份指南](Doc/10_GITHUB_BACKUP_GUIDE.md)

---

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
