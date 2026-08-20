"""
高性能交互式标注画布 (Annotation Canvas - QGraphicsView & QGraphicsScene)
"""
from typing import List, Dict, Optional, Tuple
import math
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsItem, QGraphicsSimpleTextItem
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, Slot
from PySide6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor,
    QFont, QUndoStack, QUndoCommand
)

from src.core.annotation import BoundingBox, get_class_color


class BoxItem(QGraphicsRectItem):
    """画布上的可交互矩形标注图元 (包含角点缩放手柄与类别标签)"""

    HANDLE_SIZE = 8

    def __init__(self, bbox: BoundingBox, img_w: int, img_h: int, class_name: str = "", parent=None):
        super().__init__(parent)
        self.bbox = bbox
        self.img_w = img_w
        self.img_h = img_h
        self.class_name = class_name
        self.is_resizing = False
        self.resize_handle = None

        self.setFlags(
            QGraphicsItem.ItemIsSelectable |
            QGraphicsItem.ItemIsMovable |
            QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.update_geometry_from_bbox()

    def update_geometry_from_bbox(self):
        """根据归一化 bbox 刷新图元像素坐标"""
        x1, y1, x2, y2 = self.bbox.to_xyxy(self.img_w, self.img_h)
        self.setRect(0, 0, max(2, x2 - x1), max(2, y2 - y1))
        self.setPos(x1, y1)

    def sync_bbox_from_geometry(self):
        """将当前像素位置与大小同步回归一化 bbox"""
        pos = self.pos()
        rect = self.rect()
        x1 = pos.x()
        y1 = pos.y()
        x2 = x1 + rect.width()
        y2 = y1 + rect.height()

        updated = BoundingBox.from_xyxy(
            x1, y1, x2, y2, self.img_w, self.img_h,
            class_id=self.bbox.class_id,
            confidence=self.bbox.confidence,
            box_id=self.bbox.box_id
        )
        self.bbox.x_center = updated.x_center
        self.bbox.y_center = updated.y_center
        self.bbox.width = updated.width
        self.bbox.height = updated.height

    def paint(self, painter: QPainter, option, widget=None):
        rect = self.rect()
        color_hex = get_class_color(self.bbox.class_id)
        base_color = QColor(color_hex)

        # 选中态高亮
        is_selected = self.isSelected()
        pen_width = 3 if is_selected else 2
        pen = QPen(base_color, pen_width)
        if is_selected:
            pen.setStyle(Qt.SolidLine)
        painter.setPen(pen)

        fill_color = QColor(base_color)
        fill_color.setAlpha(60 if is_selected else 30)
        painter.setBrush(QBrush(fill_color))
        painter.drawRect(rect)

        # 绘制顶部类别标签
        label_text = f"{self.class_name or f'cls_{self.bbox.class_id}'}"
        if self.bbox.confidence < 1.0:
            label_text += f" {self.bbox.confidence:.2f}"

        font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(label_text)
        text_h = fm.height()

        tag_rect = QRectF(rect.left(), rect.top() - text_h - 4, text_w + 8, text_h + 4)
        if tag_rect.top() < 0:
            tag_rect = QRectF(rect.left(), rect.top(), text_w + 8, text_h + 4)

        painter.setPen(Qt.NoPen)
        tag_color = QColor(base_color)
        tag_color.setAlpha(220)
        painter.setBrush(QBrush(tag_color))
        painter.drawRoundedRect(tag_rect, 3, 3)

        painter.setPen(QPen(Qt.white))
        painter.drawText(tag_rect.adjusted(4, 2, 0, 0), label_text)

        # 选中时绘制角点手柄
        if is_selected:
            painter.setBrush(QBrush(Qt.white))
            painter.setPen(QPen(base_color, 1.5))
            hs = self.HANDLE_SIZE
            handles = [
                QRectF(rect.left() - hs/2, rect.top() - hs/2, hs, hs),
                QRectF(rect.right() - hs/2, rect.top() - hs/2, hs, hs),
                QRectF(rect.left() - hs/2, rect.bottom() - hs/2, hs, hs),
                QRectF(rect.right() - hs/2, rect.bottom() - hs/2, hs, hs)
            ]
            for h in handles:
                painter.drawRect(h)


class AnnotationCanvas(QGraphicsView):
    """高性能 60FPS 标注画布"""

    box_created = Signal(BoundingBox)
    box_modified = Signal(BoundingBox)
    box_selected = Signal(Optional[BoundingBox])
    box_deleted = Signal(BoundingBox)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self.box_items: List[BoxItem] = []
        self.current_class_id = 0
        self.class_names: List[str] = ["target"]

        # 画布状态模式: 'draw', 'select', 'pan'
        self.mode = "draw"
        self.is_drawing = False
        self.draw_start_pos = QPointF()
        self.temp_rect_item: Optional[QGraphicsRectItem] = None

        # 平移支持
        self.is_panning = False
        self.pan_start_pos = QPointF()

        # 十字线支持
        self.show_crosshair = True
        self.cursor_pos = QPointF()

        self.img_w = 1
        self.img_h = 1

        self.init_view_settings()

    def init_view_settings(self):
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#141419")))

    def load_image(self, img_path: str, boxes: List[BoundingBox], class_names: List[str]):
        """加载新图片与标注信息"""
        self.scene.clear()
        self.box_items.clear()
        self.class_names = class_names

        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            return

        self.img_w = pixmap.width()
        self.img_h = pixmap.height()
        self.scene.setSceneRect(0, 0, self.img_w, self.img_h)

        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.pixmap_item.setZValue(0)

        for b in boxes:
            cls_name = class_names[b.class_id] if b.class_id < len(class_names) else f"cls_{b.class_id}"
            item = BoxItem(b, self.img_w, self.img_h, class_name=cls_name)
            item.setZValue(10)
            self.scene.addItem(item)
            self.box_items.append(item)

        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def set_current_class(self, class_id: int):
        self.current_class_id = class_id

    def set_mode(self, mode: str):
        self.mode = mode
        if mode == "pan":
            self.setCursor(Qt.OpenHandCursor)
        elif mode == "draw":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    # --- 鼠标与交互事件 ---
    def wheelEvent(self, event):
        """以鼠标指针为中心平滑缩放"""
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:
            zoom_factor = 1.0 / zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())

        # 中键或空格+左键: 平移
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.AltModifier):
            self.is_panning = True
            self.pan_start_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        if self.mode == "draw" and event.button() == Qt.LeftButton:
            # 限制在图片范围内
            if 0 <= scene_pos.x() <= self.img_w and 0 <= scene_pos.y() <= self.img_h:
                self.is_drawing = True
                self.draw_start_pos = scene_pos
                self.temp_rect_item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
                color = QColor(get_class_color(self.current_class_id))
                pen = QPen(color, 2, Qt.DashLine)
                self.temp_rect_item.setPen(pen)
                fill_color = QColor(color)
                fill_color.setAlpha(40)
                self.temp_rect_item.setBrush(QBrush(fill_color))
                self.temp_rect_item.setZValue(99)
                self.scene.addItem(self.temp_rect_item)
                return

        super().mousePressEvent(event)

        # 检查选中状态
        selected = self.scene.selectedItems()
        if selected and isinstance(selected[0], BoxItem):
            self.box_selected.emit(selected[0].bbox)
        else:
            self.box_selected.emit(None)

    def mouseMoveEvent(self, event):
        self.cursor_pos = self.mapToScene(event.pos())

        if self.is_panning:
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            return

        if self.is_drawing and self.temp_rect_item:
            cur_pos = self.cursor_pos
            rect = QRectF(
                min(self.draw_start_pos.x(), cur_pos.x()),
                min(self.draw_start_pos.y(), cur_pos.y()),
                abs(cur_pos.x() - self.draw_start_pos.x()),
                abs(cur_pos.y() - self.draw_start_pos.y())
            )
            self.temp_rect_item.setRect(rect)
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.CrossCursor if self.mode == "draw" else Qt.ArrowCursor)
            return

        if self.is_drawing and self.temp_rect_item:
            self.is_drawing = False
            rect = self.temp_rect_item.rect()
            self.scene.removeItem(self.temp_rect_item)
            self.temp_rect_item = None

            # 过滤过小点击误触框 (宽或高小于 5 像素)
            if rect.width() >= 5 and rect.height() >= 5:
                new_bbox = BoundingBox.from_xyxy(
                    rect.left(), rect.top(), rect.right(), rect.bottom(),
                    self.img_w, self.img_h,
                    class_id=self.current_class_id
                )
                cls_name = self.class_names[self.current_class_id] if self.current_class_id < len(self.class_names) else ""
                item = BoxItem(new_bbox, self.img_w, self.img_h, class_name=cls_name)
                item.setZValue(10)
                self.scene.addItem(item)
                self.box_items.append(item)
                item.setSelected(True)
                self.box_created.emit(new_bbox)
            return

        super().mouseReleaseEvent(event)

        # 检查是否移动或调整了已有的框
        for item in self.box_items:
            if item.isSelected():
                item.sync_bbox_from_geometry()
                self.box_modified.emit(item.bbox)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected_box()
        else:
            super().keyPressEvent(event)

    def delete_selected_box(self):
        selected = self.scene.selectedItems()
        for item in selected:
            if isinstance(item, BoxItem):
                self.scene.removeItem(item)
                if item in self.box_items:
                    self.box_items.remove(item)
                self.box_deleted.emit(item.bbox)

    def get_all_boxes(self) -> List[BoundingBox]:
        return [item.bbox for item in self.box_items]
