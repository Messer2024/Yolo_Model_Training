# YOLO Studio - 测试计划与多平台打包部署规范 (Test & Deployment Plan)

**文档版本**：v1.0.0  
**编制日期**：2026-08-20  
**项目名称**：YOLO Studio

---

## 1. 测试策略与分层设计

为了确保 YOLO Studio 的工业级稳定性与可靠性，项目执行 **三层测试金字塔架构**：

```
       ▲
      / \     1. 端到端系统验收测试 (GUI E2E & Workflow Test)
     /   \    2. 集成测试 (Dataset Pipeline & Engine Integration)
    /_____\   3. 核心单元测试 (Coordinate Math, Commands, Config, Hardware)
```

---

## 2. 单元测试用例设计 (Unit Tests)

### 2.1 标注坐标计算与边界约束测试 (`test_annotation.py`)
- **UT-01-01**：测试 `(x1, y1, x2, y2)` 与归一化 `(x_center, y_center, w, h)` 的双向无损互转；
- **UT-01-02**：测试当用户框选超出图片尺寸边界时（如 $x < 0$ 或 $x > W$），边界框是否被自动限制在合法区间；
- **UT-01-03**：测试 `QUndoStack` 在执行 10 次连续增加/删除/修改标注框后的撤销与重做一致性。

### 2.2 数据集管理与划分测试 (`test_dataset_manager.py`)
- **UT-02-01**：测试数据集加载时缺失 `classes.txt` 时的自动推断与修复能力；
- **UT-02-02**：测试 `split_dataset` 按 8:1:1 划分后，各子集图片与标注文件的对应关系无一遗漏；
- **UT-02-03**：测试 `data.yaml` 输出路径与格式是否完全符合 Ultralytics 解析规范。

### 2.3 硬件探测与自适应配置测试 (`test_hardware.py`)
- **UT-03-01**：测试无 CUDA 环境下是否优雅降级为 `cpu` 且不引发崩溃；
- **UT-03-02**：测试 CUDA 可用时能否准确读取显卡型号与可用显存。

---

## 3. 集成与端到端测试流程 (Integration & E2E)

1. **导入与自动标注闭环**：
   - 自动生成包含 10 张合成测试图的临时项目；
   - 触发 `AutoLabelSkill` 执行预标注，验证生成的文件与边界框数量一致；
2. **轻量训练闭环**：
   - 触发 `YoloTrainerWorker` 运行 1 个 Epoch（使用超轻量测试权重），验证信号发送完整且无内存泄漏；
3. **模型导出与验证闭环**：
   - 导出为 ONNX 格式，使用 ONNX Runtime 加载并完成单次前向测试。

---

## 4. 自动化测试执行命令

在项目根目录下使用 `pytest` 执行测试：
```bash
# 运行全部单元测试并输出覆盖率报告
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 5. 多平台打包与分发方案 (Deployment & Packaging)

### 5.1 Windows 独立可执行程序打包 (PyInstaller)
在 Windows 环境下执行打包脚本：
```bash
pip install pyinstaller

pyinstaller --noconfirm --windowed --onedir \
    --name "YOLO-Studio" \
    --add-data "src/ui/styles;src/ui/styles" \
    --hidden-import "ultralytics" \
    --hidden-import "torch" \
    --hidden-import "torchvision" \
    --hidden-import "PySide6" \
    src/app.py
```

### 5.2 Linux & macOS 一键分发脚本
提供 `run.sh` 脚本，自动创建虚拟环境、安装必要依赖并启动主程序：
```bash
chmod +x run.sh
./run.sh
```
