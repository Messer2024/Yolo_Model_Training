# YOLO Studio - 图形化目标检测模型训练工作站

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v9%20%7C%20v10%20%7C%2011-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**YOLO Studio** 是一个基于 Python 和 PySide6 构建的 YOLO 模型图形化训练与部署工具。它将目标检测模型开发中的**数据标注、数据集划分与增强、模型配置与训练、效果验证以及格式导出**整合到了一个统一的桌面界面中，方便算法工程师、科研人员以及视觉应用开发者快速完成模型迭代。

---

## 功能特点

- **交互式图像标注**：基于 Qt Graphics View 框架开发，支持以鼠标为中心的平滑缩放、画布平移、十字线辅助对齐、类别管理以及完整的快捷键与撤销/重做（Undo/Redo）操作。
- **AI 辅助预标注**：可载入现有的预训练权重对批量图片进行初筛打标，用户只需在此基础上进行微调或纠错，有效减少重复标注工作量。
- **数据集划分与质量体检**：提供可视化的 Train / Val / Test 比例切分，自动生成符合 YOLO 规范的目录结构与 `data.yaml` 文件；内置数据健康度检查，排查坐标越界、空标注与损坏图像。
- **离线数据增强**：支持水平/垂直翻转、亮度与对比度调整、模糊等常见增强方式，并在生成新图片时自动完成边界框坐标的同步转换。
- **异步训练与实时监控**：训练任务运行在独立工作线程中，避免主界面在计算过程中出现未响应现象；界面内嵌实时折线图，动态展示 Loss、mAP50、mAP50-95、Precision 和 Recall 等关键指标。
- **交互式推理测试台**：支持导入单图、批量图片或本地视频进行推理测试，并可通过滑块实时调节置信度（Confidence）与 NMS IoU 阈值，直观对比检测效果。
- **多端模型导出**：支持一键将训练好的 PyTorch 权重（`.pt`）导出为 **ONNX、TensorRT Engine、OpenVINO、CoreML、TFLite** 等部署格式，并支持 ONNX 模型的自动加载校验。
- **模块化设计**：采用清晰的分层架构（UI 表现层、业务控制层、核心引擎层与数据层解耦），方便后续扩展新的网络结构、自定义增强算子或自动化调参策略。

---

## 模块结构

```text
Yolo_Model_Training/
├── Doc/                  # 详细的技术文档与使用手册
├── agents/               # 智能体与自动化决策逻辑 (数据集审计、参数推荐、辅助技能)
│   └── skills/           # 独立工具函数 (预标注、数据增强、模型转换)
├── src/
│   ├── app.py            # 程序主入口
│   ├── core/             # 标注模型、数据集管理、训练调度、推理与导出引擎
│   ├── ui/               # 基于 PySide6 的界面布局、画布组件与样式
│   │   ├── views/        # 标注、数据管理、训练、推理与导出等功能页面
│   │   ├── widgets/      # 绘图看板、终端日志、类别列表等通用控件
│   │   └── styles/       # 深色与浅色 QSS 主题
│   └── utils/            # 配置文件读写、日志工具与硬件环境检测
└── tests/                # 单元测试用例
```

---

## 快速开始

### 1. 环境准备

建议使用 Python 3.10 ~ 3.12 环境。首先创建并激活虚拟环境：

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

> **提示**：如果需要使用 NVIDIA 显卡进行 GPU 加速训练，请先根据本机 CUDA 版本从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应的 PyTorch GPU 版本。

### 3. 启动软件

```bash
python src/app.py
```

---

## 核心使用流程

1. **导入与标注**：打开包含待标注图片的目录，在右侧添加需要的类别名称，在画布中框选目标对象。如果图片较多，可点击「AI 辅助预标注」批量生成初步检测框。
2. **数据集准备**：切换到「数据集管理」页面，点击「深度体检」检查数据质量，调整训练集与验证集比例，点击「划分数据集」生成标准 `data.yaml`。
3. **配置与训练**：切换到「模型训练」页面，选择模型尺寸（如 YOLOv8s / YOLO11s）和训练轮数，点击「开始训练」，右侧图表将实时绘制损失与精度变化曲线。
4. **推理与验证**：训练完成后，在「推理测试」页面载入生成的 `best.pt`，拖入测试图片或视频即可查看检测结果。
5. **模型导出**：在「模型导出」页面选择需要的格式（如 ONNX 或 TensorRT），点击导出即可用于后续生产部署。

---

## 运行测试

项目中包含核心数据转换、几何计算与数据集管理的单元测试，可通过以下命令执行验证：

```bash
python -m unittest discover tests
```

---

## 详细文档索引

更深入的设计说明与操作指南已整理在 `Doc/` 目录下：

- [01. 项目可行性与选型报告](Doc/01_FEASIBILITY_REPORT.md)
- [02. 软件需求规格说明书 (SRS)](Doc/02_REQUIREMENTS_SPECIFICATION.md)
- [03. 系统架构与详细设计说明书](Doc/03_SYSTEM_ARCHITECTURE.md)
- [04. 数据集结构与标注格式规范](Doc/04_DATASET_SPECIFICATION.md)
- [05. 智能体与技能扩展系统规范](Doc/05_AGENTS_AND_SKILLS_SPECIFICATION.md)
- [06. 用户使用与操作指南](Doc/06_USER_MANUAL.md)
- [07. 开发者指南与二次开发手册](Doc/07_DEVELOPER_GUIDE.md)
- [08. 核心模块 API 接口参考手册](Doc/08_API_REFERENCE.md)
- [09. 测试计划与多平台打包部署规范](Doc/09_TEST_AND_DEPLOYMENT_PLAN.md)
- [10. Git 版本管理与 GitHub 备份指南](Doc/10_GITHUB_BACKUP_GUIDE.md)

---

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
