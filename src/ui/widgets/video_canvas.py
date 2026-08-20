"""
自适应高性能图像与视频渲染画布 (Image & Video Display Canvas Widget)
解决普通 QLabel.setPixmap 导致的尺寸无限放大与布局递归抖动问题
"""
from typing import Optional
import numpy as np
import cv2
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QPainter, QImage, QPixmap, QColor, QFont


class ImageDisplayCanvas(QWidget):
    """
    自适应比例渲染画布
    在保持几何约束的同时，以最高性能平滑绘制图像与实时视频帧
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_pixmap: Optional[QPixmap] = None
        self._placeholder_text: str = "请载入模型并选择测试图片/视频以查看检测结果"

        # 设置尺寸策略为 Ignored，防止 pixmap 改变父级布局尺寸
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumSize(240, 180)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)

    def set_placeholder(self, text: str):
        """设置未加载图像时的提示文字"""
        self._placeholder_text = text
        self.update()

    def set_bgr_image(self, bgr_img: Optional[np.ndarray]):
        """接收 OpenCV BGR 格式的 numpy 图像并渲染"""
        if bgr_img is None or bgr_img.size == 0:
            self._current_pixmap = None
            self.update()
            return

        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._current_pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def set_pixmap(self, pixmap: Optional[QPixmap]):
        """直接接收 QPixmap 进行渲染"""
        self._current_pixmap = pixmap
        self.update()

    def clear(self):
        """清空当前画面"""
        self._current_pixmap = None
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(640, 480)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. 绘制暗色背景与边框
        w = self.width()
        h = self.height()
        bg_rect = QRectF(0, 0, w, h)

        painter.fillRect(bg_rect, QColor("#141419"))
        painter.setPen(QColor("#2d2d38"))
        painter.drawRoundedRect(bg_rect.adjusted(1, 1, -1, -1), 8, 8)

        # 2. 绘制图像或提示文本
        if self._current_pixmap and not self._current_pixmap.isNull():
            pm_w = self._current_pixmap.width()
            pm_h = self._current_pixmap.height()

            if pm_w > 0 and pm_h > 0:
                # 计算等比例居中适应矩形
                scale = min((w - 10) / pm_w, (h - 10) / pm_h)
                target_w = pm_w * scale
                target_h = pm_h * scale
                target_x = (w - target_w) / 2.0
                target_y = (h - target_h) / 2.0

                target_rect = QRectF(target_x, target_y, target_w, target_h)
                painter.drawPixmap(target_rect.toRect(), self._current_pixmap)
        else:
            # 绘制占位提示文本
            painter.setPen(QColor("#707080"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(bg_rect, Qt.AlignCenter | Qt.TextWordWrap, self._placeholder_text)
