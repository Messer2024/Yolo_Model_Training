# YOLO Studio - 数据集结构与标注格式规范

**文档版本**：v1.0.0  
**编制日期**：2026-08-20  
**项目名称**：YOLO Studio

---

## 1. 数据集目录结构规范

YOLO Studio 严格遵循 Ultralytics YOLO 标准数据集组织规范。整个项目在被导出或训练时，目录结构组织如下：

```
my_yolo_dataset/
├── data.yaml                   # 数据集元数据与类别配置文件
├── classes.txt                 # 类别名称列表 (每行一个类别)
├── images/
│   ├── train/                  # 训练集图像 (如 0001.jpg, 0002.png)
│   ├── val/                    # 验证集图像 (如 0003.jpg, 0004.png)
│   └── test/                   # 测试集图像 (可选，用于最终盲测)
└── labels/
    ├── train/                  # 训练集标注文件 (与 images/train 同名 .txt)
    ├── val/                    # 验证集标注文件 (与 images/val 同名 .txt)
    └── test/                   # 测试集标注文件 (可选)
```

---

## 2. 标注格式规范 (YOLO TXT Format)

### 2.1 目标检测 (Object Detection) 标注格式
每个图像对应的标注文件均为同名的 `.txt` 文本文件。每一行代表图像中的一个检测目标对象，格式如下：

$$\text{<class\_id>} \quad \text{<x\_center>} \quad \text{<y\_center>} \quad \text{<width>} \quad \text{<height>}$$

所有坐标均为**相对于图像宽度和高度的归一化浮点数**，取值范围在 $[0.0, 1.0]$ 之间：

- `class_id`：整数，从 `0` 开始的类别索引号（例如 `0` 代表 `person`, `1` 代表 `car`）；
- `x_center`：目标边界框中心点的 X 坐标 / 图像总宽度；
- `y_center`：目标边界框中心点的 Y 坐标 / 图像总高度；
- `width`：目标边界框的宽度 / 图像总宽度；
- `height`：目标边界框的高度 / 图像总高度。

#### 示例：
假设图像分辨率为 $1920 \times 1080$，图像中有一个属于类别 0 的目标，边界框左上角为 $(100, 200)$，右下角为 $(500, 800)$：
- 宽度 $W = 500 - 100 = 400$
- 高度 $H = 800 - 200 = 600$
- 中心点 $X_c = 100 + 400/2 = 300$
- 中心点 $Y_c = 200 + 600/2 = 500$
- 归一化：
  - $x\_center = 300 / 1920 \approx 0.156250$
  - $y\_center = 500 / 1080 \approx 0.462963$
  - $width = 400 / 1920 \approx 0.208333$
  - $height = 600 / 1080 \approx 0.555556$

对应的 `label.txt` 文件内容为：
```text
0 0.156250 0.462963 0.208333 0.555556
1 0.654120 0.321450 0.120500 0.240000
```

---

## 3. 配置文件规范 (`data.yaml`)

`data.yaml` 是驱动 YOLO 训练引擎的核心元数据文件，包含数据集路径、子集相对路径与类别映射信息：

```yaml
# YOLO Studio 自动生成的数据集配置文件
path: C:/Users/username/Datasets/my_project   # 数据集根目录 (绝对路径或相对路径)
train: images/train                          # 训练集图片路径 (相对于 path)
val: images/val                              # 验证集图片路径 (相对于 path)
test: images/test                            # 测试集图片路径 (可选)

# 类别总数与类别名称映射
nc: 3                                        # 类别数量 (Number of Classes)
names:
  0: person
  1: bicycle
  2: car
```

---

## 4. 数据格式互转支持

YOLO Studio 内置自动化格式转换器（`DatasetConverter`），支持在以下主流格式之间无损互转：

| 格式名称 | 标注文件形态 | 坐标表示方式 | 适用场景 |
| :--- | :--- | :--- | :--- |
| **YOLO TXT** | 每张图对应同名 `.txt` | 归一化中心点及宽高 $[x_c, y_c, w, h]$ | YOLO 训练原生格式 |
| **Pascal VOC** | 每张图对应同名 `.xml` | 绝对像素坐标 $[x_{min}, y_{min}, x_{max}, y_{max}]$ | 传统 CV 标注工具兼容 |
| **MS COCO** | 单个全量 `instances_train.json` | 绝对像素坐标 $[x_{min}, y_{min}, w, h]$ | 学术基准评测与跨平台迁移 |

---

## 5. 数据质量与健康度校验规则

在开始训练前，系统的 `DatasetAuditor` 自动执行以下 6 项健康检查：
1. **空标注检测 (Missing Labels)**：检查是否存在无对应 `.txt` 标注文件的图像，并提示用户是否作为负样本（Background Image）加入训练；
2. **坐标越界检测 (Out-of-Bounds Check)**：检查 $x_c, y_c, w, h$ 是否位于 $(0.0, 1.0]$ 区间内，若超出则自动裁剪限制在边缘边界内；
3. **退化框检测 (Degenerate Box Check)**：检查是否存在 $width \le 0$ 或 $height \le 0$ 的无效或极小噪点框；
4. **未定义类别检测 (Undefined Class ID)**：检查标注文件中的 `class_id` 是否在 `data.yaml` 的类别字典范围 $[0, nc-1]$ 内；
5. **类别失衡评估 (Class Imbalance Analysis)**：统计各类别在训练集中的样本数量比例，若出现极端不平衡（如 100:1），弹出警告并推荐数据增强；
6. **图像格式与通道校验 (Corrupted Image Check)**：验证图片能否通过 OpenCV / PIL 正常解码，剔除损坏或 0 字节的图像。
