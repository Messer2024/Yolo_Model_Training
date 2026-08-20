"""
标注数据模型与几何计算 (Annotation Model & Geometry)
"""
from typing import List, Tuple, Dict, Any, Optional
import uuid
import math


class BoundingBox:
    """目标检测矩形边界框实体"""

    def __init__(
        self,
        class_id: int,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        confidence: float = 1.0,
        box_id: Optional[str] = None
    ):
        self.class_id = int(class_id)
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)
        self.confidence = float(confidence)
        self.box_id = box_id or str(uuid.uuid4())
        self.clip()

    def clip(self) -> None:
        """限制坐标在 [0.0, 1.0] 合法区间内，并保证宽高为正数"""
        self.width = max(0.0001, min(1.0, self.width))
        self.height = max(0.0001, min(1.0, self.height))
        self.x_center = max(self.width / 2.0, min(1.0 - self.width / 2.0, self.x_center))
        self.y_center = max(self.height / 2.0, min(1.0 - self.height / 2.0, self.y_center))

    def to_yolo_str(self) -> str:
        """转换为 YOLO 格式字符串: class_id x_center y_center width height"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"

    @classmethod
    def from_yolo_str(cls, line: str, box_id: Optional[str] = None) -> Optional["BoundingBox"]:
        """从 YOLO 格式的一行文本解析构建 BoundingBox"""
        parts = line.strip().split()
        if len(parts) < 5:
            return None
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            conf = float(parts[5]) if len(parts) >= 6 else 1.0
            return cls(class_id, x_center, y_center, width, height, confidence=conf, box_id=box_id)
        except Exception:
            return None

    def to_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """将归一化坐标转换为绝对像素坐标 (x_min, y_min, x_max, y_max)"""
        w_px = self.width * img_w
        h_px = self.height * img_h
        x_min = int(round((self.x_center * img_w) - (w_px / 2.0)))
        y_min = int(round((self.y_center * img_h) - (h_px / 2.0)))
        x_max = int(round(x_min + w_px))
        y_max = int(round(y_min + h_px))
        return max(0, x_min), max(0, y_min), min(img_w, x_max), min(img_h, y_max)

    @classmethod
    def from_xyxy(
        cls,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        img_w: int,
        img_h: int,
        class_id: int,
        confidence: float = 1.0,
        box_id: Optional[str] = None
    ) -> "BoundingBox":
        """从绝对像素坐标构建归一化 BoundingBox"""
        if img_w <= 0 or img_h <= 0:
            return cls(class_id, 0.5, 0.5, 0.1, 0.1, confidence, box_id)

        # 确保 min < max
        x1, x2 = min(x_min, x_max), max(x_min, x_max)
        y1, y2 = min(y_min, y_max), max(y_min, y_max)

        # 限制在图像边界内
        x1 = max(0.0, min(float(img_w), x1))
        x2 = max(0.0, min(float(img_w), x2))
        y1 = max(0.0, min(float(img_h), y1))
        y2 = max(0.0, min(float(img_h), y2))

        w_px = max(1.0, x2 - x1)
        h_px = max(1.0, y2 - y1)
        xc_px = x1 + w_px / 2.0
        yc_px = y1 + h_px / 2.0

        return cls(
            class_id=class_id,
            x_center=xc_px / img_w,
            y_center=yc_px / img_h,
            width=w_px / img_w,
            height=h_px / img_h,
            confidence=confidence,
            box_id=box_id
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box_id": self.box_id,
            "class_id": self.class_id,
            "x_center": self.x_center,
            "y_center": self.y_center,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        return cls(
            class_id=data.get("class_id", 0),
            x_center=data.get("x_center", 0.5),
            y_center=data.get("y_center", 0.5),
            width=data.get("width", 0.1),
            height=data.get("height", 0.1),
            confidence=data.get("confidence", 1.0),
            box_id=data.get("box_id")
        )

    def copy(self) -> "BoundingBox":
        return BoundingBox(
            class_id=self.class_id,
            x_center=self.x_center,
            y_center=self.y_center,
            width=self.width,
            height=self.height,
            confidence=self.confidence,
            box_id=self.box_id
        )


# 预定义高对比度现代调色板
CLASS_PALETTE = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7"
]


def get_class_color(class_id: int) -> str:
    """根据类别 ID 获取固定美观的十六进制色彩代码"""
    return CLASS_PALETTE[abs(int(class_id)) % len(CLASS_PALETTE)]
