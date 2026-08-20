# 🚀 YOLO Studio - 一站式图形化智能标注与模型训练工作站

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://wiki.qt.io/Qt_for_Python)
[![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v9%20%7C%20v10%20%7C%2011-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**YOLO Studio** 是一款专为计算机视觉开发者、算法工程师和零基础行业用户设计的**工业级一站式 YOLO 模型训练与部署图形化软件**。用户无需手动编写代码或配置复杂的命令行参数，即可在现代化图形界面中完成从**数据标注、AI 辅助打标、数据清洗、一键训练、实时监控到模型推理与多端导出**的全流程闭环。

---

## 🌟 核心特性 (Key Features)

- 🎨 **极速交互标注画布**：基于 PySide6 `QGraphicsView` 硬件加速，支持超大分辨率 60FPS 缩放、平移、十字线辅助、类别色彩管理与完整的 `Ctrl+Z/Y` 撤销重做机制。
- 🤖 **AI 辅助智能预标注**：集成预训练大模型，一键对海量未标注图片进行自动识别框选，仅需微调即可完成标注，效率提升 10 倍。
- 📊 **智能数据集管线**：支持 Train/Val/Test 自动比例划分、数据健康度一键体检（破损图/越界框/空图排查）、离线数据增强扩充与 `data.yaml` 自动生成。
- ⚡ **一键训练与进程隔离**：无缝支持 YOLOv8、YOLOv9、YOLOv10、YOLO11 全系列；训练在独立子线程/进程中运行，UI 界面永不卡死。
- 📈 **实时指标动态大屏**：实时绘制 Loss、mAP50、mAP50-95、Precision、Recall 动态折线图与 ETA 倒计时。
- 🔍 **交互式推理测试台 (Playground)**：支持图片拖拽测试、批量预测、本地视频与实时 Webcam 摄像头检测，置信度阈值动态拖拽过滤。
- 📦 **多端模型一键导出**：一键将 `.pt` 模型转换为 **ONNX, TensorRT, OpenVINO, CoreML, TFLite**，并自动执行 ONNX Runtime 精度校验。
- 🧩 **Agents & Skills 插件体系**：模块化架构设计，支持自定义智能体与技能扩展，维护与升级极其轻松。

---

## 📁 快速文档索引 (`Doc/`)

项目严格遵循专业软件工程标准，所有设计与开发文档均存放在 `Doc/` 目录下：

1. 📄 [01_项目可行性与技术选型评估报告](Doc/01_FEASIBILITY_REPORT.md)
2. 📄 [02_软件需求规格说明书 (SRS)](Doc/02_REQUIREMENTS_SPECIFICATION.md)
3. 📄 [03_系统架构与详细设计说明书](Doc/03_SYSTEM_ARCHITECTURE.md)
4. 📄 [04_数据集结构与标注格式规范](Doc/04_DATASET_SPECIFICATION.md)
5. 📄 [05_智能体 (Agents) 与技能 (Skills) 扩展系统规范](Doc/05_AGENTS_AND_SKILLS_SPECIFICATION.md)
6. 📄 [06_用户使用与操作指南 (零基础图文手册)](Doc/06_USER_MANUAL.md)
7. 📄 [07_开发者指南与二次开发手册](Doc/07_DEVELOPER_GUIDE.md)
8. 📄 [08_核心模块 API 接口参考手册](Doc/08_API_REFERENCE.md)
9. 📄 [09_测试计划与多平台打包部署规范](Doc/09_TEST_AND_DEPLOYMENT_PLAN.md)
10. 📄 [10_Git 版本管理与 GitHub 仓库备份指南](Doc/10_GITHUB_BACKUP_GUIDE.md)

---

## 🛠️ 快速安装与启动

### 1. 克隆/打开项目并配置环境
```bash
# 建议使用 Python 3.10 ~ 3.12
python -m venv venv

# Windows 激活虚拟环境:
.\venv\Scripts\activate

# Linux / macOS 激活虚拟环境:
source venv/bin/activate
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动图形化工作站
```bash
python src/app.py
```

---

## 🧪 运行单元测试
```bash
python -m unittest discover tests
```

---

## 📄 开源许可证
本项目遵循 [MIT License](LICENSE) 开源协议。
