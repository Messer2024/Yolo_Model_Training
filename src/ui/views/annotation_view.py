"""
交互式标注工作台视图 (Annotation View)
"""
from typing import Optional, List
import os
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QListWidget,
    QListWidgetItem, QPushButton, QLabel, QFileDialog, QMessageBox,
    QToolBar, QComboBox, QSpinBox, QProgressBar
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QIcon, QAction, QColor

from src.core.dataset_manager import DatasetManager
from src.core.annotation import BoundingBox
from src.core.autolabel import AutoLabelEngine
from src.ui.canvas import AnnotationCanvas
from src.ui.widgets.class_list_widget import ClassListWidget
from src.utils.logger import logger


class AnnotationView(QWidget):
    """标注工作区：图片列表 + 高性能交互画布 + 类别管理与属性栏"""

    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self.dm = dataset_manager
        self.autolabel_engine = AutoLabelEngine()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # 顶部快捷操作栏
        toolbar = QHBoxLayout()

        self.btn_open = QPushButton("📁 打开项目目录")
        self.btn_open.setObjectName("primaryButton")
        self.btn_open.clicked.connect(self._on_open_folder)
        toolbar.addWidget(self.btn_open)

        self.btn_prev = QPushButton("◀ 上一张 (A)")
        self.btn_prev.clicked.connect(self._on_prev_image)
        toolbar.addWidget(self.btn_prev)

        self.btn_next = QPushButton("下一张 (D) ▶")
        self.btn_next.clicked.connect(self._on_next_image)
        toolbar.addWidget(self.btn_next)

        self.btn_save = QPushButton("💾 保存标注 (Ctrl+S)")
        self.btn_save.clicked.connect(self._on_save_current)
        toolbar.addWidget(self.btn_save)

        self.btn_delete_box = QPushButton("🗑️ 删除选中框 (Del)")
        self.btn_delete_box.clicked.connect(self._on_delete_box)
        toolbar.addWidget(self.btn_delete_box)

        toolbar.addSpacing(15)

        # 绘图 / 平移模式切换
        self.btn_mode_draw = QPushButton("✏️ 框选模式")
        self.btn_mode_draw.setCheckable(True)
        self.btn_mode_draw.setChecked(True)
        self.btn_mode_draw.clicked.connect(lambda: self._set_mode("draw"))
        toolbar.addWidget(self.btn_mode_draw)

        self.btn_mode_pan = QPushButton("✋ 平移视野")
        self.btn_mode_pan.setCheckable(True)
        self.btn_mode_pan.clicked.connect(lambda: self._set_mode("pan"))
        toolbar.addWidget(self.btn_mode_pan)

        toolbar.addStretch()

        # AI 智能预打标按钮
        self.btn_autolabel = QPushButton("🤖 AI 辅助预标注")
        self.btn_autolabel.setStyleSheet("background-color: #7b2cbf; color: white; font-weight: bold;")
        self.btn_autolabel.clicked.connect(self._on_auto_label)
        toolbar.addWidget(self.btn_autolabel)

        main_layout.addLayout(toolbar)

        # 主分割区域 (左侧图片列表 - 中间画布 - 右侧类别管理器)
        splitter = QSplitter(Qt.Horizontal)

        # 1. 左侧图像列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_img_count = QLabel("图片列表 (0)")
        self.lbl_img_count.setStyleSheet("font-weight: bold; color: #00d4bb;")
        left_layout.addWidget(self.lbl_img_count)

        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self._on_image_selected)
        left_layout.addWidget(self.image_list_widget)
        splitter.addWidget(left_panel)

        # 2. 中间交互式标注画布
        self.canvas = AnnotationCanvas()
        self.canvas.box_created.connect(self._on_box_changed)
        self.canvas.box_modified.connect(self._on_box_changed)
        self.canvas.box_deleted.connect(self._on_box_changed)
        splitter.addWidget(self.canvas)

        # 3. 右侧类别与属性管理面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.class_list_widget = ClassListWidget()
        self.class_list_widget.class_selected.connect(self._on_class_selected)
        self.class_list_widget.class_added.connect(self._on_class_added)
        self.class_list_widget.class_renamed.connect(self._on_class_renamed)
        right_layout.addWidget(self.class_list_widget)

        # 当前框属性详情
        self.lbl_box_info = QLabel("未选中任何标注框")
        self.lbl_box_info.setStyleSheet("color: #888899; padding: 6px;")
        right_layout.addWidget(self.lbl_box_info)

        splitter.addWidget(right_panel)

        # 设置分割栏初始宽度比例 (18% : 64% : 18%)
        splitter.setSizes([200, 700, 200])
        main_layout.addWidget(splitter)

    def reload_data(self):
        """当项目重新加载时刷新界面"""
        self.image_list_widget.clear()
        for img_path in self.dm.image_files:
            name = os.path.basename(img_path)
            boxes = self.dm.labels_map.get(img_path, [])
            tag = f"✓ ({len(boxes)})" if boxes else "·"
            item = QListWidgetItem(f"{tag} {name}")
            if boxes:
                item.setForeground(QColor("#00d4bb"))
            self.image_list_widget.addItem(item)

        self.lbl_img_count.setText(f"图片列表 ({len(self.dm.image_files)})")
        self._update_class_widget()

        if len(self.dm.image_files) > 0:
            self.image_list_widget.setCurrentRow(0)

    def _update_class_widget(self):
        # 统计每个类别的框总数
        counts = {}
        for boxes in self.dm.labels_map.values():
            for b in boxes:
                counts[b.class_id] = counts.get(b.class_id, 0) + 1
        self.class_list_widget.set_classes(self.dm.class_names, counts)

    def _set_mode(self, mode: str):
        self.btn_mode_draw.setChecked(mode == "draw")
        self.btn_mode_pan.setChecked(mode == "pan")
        self.canvas.set_mode(mode)

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片或项目目录")
        if folder:
            if self.dm.load_project(folder):
                self.reload_data()

    def _on_image_selected(self, row: int):
        if 0 <= row < len(self.dm.image_files):
            # 先保存上一张
            self._on_save_current()

            self.dm.current_image_index = row
            img_path = self.dm.image_files[row]
            boxes = self.dm.labels_map.get(img_path, [])
            self.canvas.load_image(img_path, boxes, self.dm.class_names)
            self.canvas.set_current_class(self.class_list_widget.get_selected_class_id())

    def _on_prev_image(self):
        cur = self.image_list_widget.currentRow()
        if cur > 0:
            self.image_list_widget.setCurrentRow(cur - 1)

    def _on_next_image(self):
        cur = self.image_list_widget.currentRow()
        if cur < len(self.dm.image_files) - 1:
            self.image_list_widget.setCurrentRow(cur + 1)

    def _on_save_current(self):
        if 0 <= self.dm.current_image_index < len(self.dm.image_files):
            img_path = self.dm.image_files[self.dm.current_image_index]
            boxes = self.canvas.get_all_boxes()
            self.dm.save_annotation(img_path, boxes)

            # 刷新列表项标记
            item = self.image_list_widget.item(self.dm.current_image_index)
            if item:
                name = os.path.basename(img_path)
                tag = f"✓ ({len(boxes)})" if boxes else "·"
                item.setText(f"{tag} {name}")
                item.setForeground(QColor("#00d4bb") if boxes else QColor("#e0e0e0"))

            self._update_class_widget()

    def _on_delete_box(self):
        self.canvas.delete_selected_box()

    def _on_box_changed(self, bbox: BoundingBox):
        self._on_save_current()

    def _on_class_selected(self, class_id: int):
        self.canvas.set_current_class(class_id)

    def _on_class_added(self, class_name: str):
        self.dm.add_class(class_name)
        self._update_class_widget()

    def _on_class_renamed(self, class_id: int, new_name: str):
        self.dm.update_class_name(class_id, new_name)
        self._update_class_widget()
        # 刷新画布标签
        if 0 <= self.dm.current_image_index < len(self.dm.image_files):
            img_path = self.dm.image_files[self.dm.current_image_index]
            self.canvas.load_image(img_path, self.canvas.get_all_boxes(), self.dm.class_names)

    def _on_auto_label(self):
        if not self.dm.image_files:
            QMessageBox.warning(self, "提示", "请先打开包含图像的项目目录！")
            return

        reply = QMessageBox.question(
            self,
            "AI 辅助预标注",
            f"将使用预训练 YOLOv8n 模型对当前项目的 {len(self.dm.image_files)} 张图片进行自动预打标。\n是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                results_map = self.autolabel_engine.label_images(self.dm.image_files)
                for img_path, boxes in results_map.items():
                    if boxes:
                        self.dm.save_annotation(img_path, boxes)
                self.reload_data()
                QMessageBox.information(self, "完成", "AI 预标注完成！已为所有图片生成候选目标框，请核对微调。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"AI 预标注失败: {e}")
