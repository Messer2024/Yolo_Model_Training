"""
类别管理与列表控件 (Class List Widget)
"""
from typing import List, Dict, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QInputDialog, QMessageBox, QLabel, QMenu
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPixmap, QPainter

from src.core.annotation import get_class_color


class ClassListWidget(QWidget):
    """类别管理列表控件"""

    class_selected = Signal(int)  # 当前选中的 class_id
    class_added = Signal(str)     # 新增类别
    class_renamed = Signal(int, str)  # 类别重命名 (class_id, new_name)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.class_names: List[str] = []
        self.class_counts: Dict[int, int] = {}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 标题栏
        header = QHBoxLayout()
        title = QLabel("🏷️ 目标类别管理")
        title.setStyleSheet("font-weight: bold; color: #00d4bb;")
        header.addWidget(title)

        header.addStretch()

        self.btn_add = QPushButton("+ 添加")
        self.btn_add.setFixedWidth(65)
        self.btn_add.clicked.connect(self._on_add_class)
        header.addWidget(self.btn_add)

        layout.addLayout(header)

        # 列表
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.list_widget)

    def set_classes(self, class_names: List[str], counts: Optional[Dict[int, int]] = None):
        """刷新类别列表数据"""
        self.class_names = class_names
        self.class_counts = counts or {}
        self.list_widget.clear()

        for i, name in enumerate(class_names):
            color_hex = get_class_color(i)
            count = self.class_counts.get(i, 0)
            item_text = f"[{i}] {name}  ({count})"

            item = QListWidgetItem(item_text)

            # 生成色块图标
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(color_hex))
            item.setIcon(QIcon(pixmap))
            item.setData(Qt.UserRole, i)

            self.list_widget.addItem(item)

        if len(class_names) > 0 and self.list_widget.currentRow() < 0:
            self.list_widget.setCurrentRow(0)

    def get_selected_class_id(self) -> int:
        row = self.list_widget.currentRow()
        return row if row >= 0 else 0

    def _on_row_changed(self, row: int):
        if row >= 0:
            self.class_selected.emit(row)

    def _on_add_class(self):
        text, ok = QInputDialog.getText(self, "添加类别", "请输入新类别英文/中文名称:")
        if ok and text.strip():
            self.class_added.emit(text.strip())

    def _show_context_menu(self, pos):
        item = self.list_widget.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        rename_action = menu.addAction("✏️ 重命名类别")
        action = menu.exec(self.list_widget.mapToGlobal(pos))

        if action == rename_action:
            cls_id = item.data(Qt.UserRole)
            old_name = self.class_names[cls_id] if cls_id < len(self.class_names) else ""
            new_name, ok = QInputDialog.getText(self, "重命名类别", "请输入新名称:", text=old_name)
            if ok and new_name.strip() and new_name.strip() != old_name:
                self.class_renamed.emit(cls_id, new_name.strip())
