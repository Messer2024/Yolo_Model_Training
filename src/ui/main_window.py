"""
主工作台窗口 (Main Window)
"""
import os
import sys
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QMenuBar,
    QMenu, QStatusBar, QFileDialog, QMessageBox, QLabel
)
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtCore import Qt

from src.core.dataset_manager import DatasetManager
from src.ui.views.annotation_view import AnnotationView
from src.ui.views.dataset_view import DatasetView
from src.ui.views.train_view import TrainView
from src.ui.views.inference_view import InferenceView
from src.ui.views.export_view import ExportView
from src.utils.config import ConfigManager
from src.utils.hardware import detect_hardware
from src.utils.logger import logger


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Studio - 一站式图形化智能标注与模型训练工作站 v1.0.0")
        self.resize(1366, 850)

        self.config_manager = ConfigManager()
        self.dataset_manager = DatasetManager()
        self.hardware_info = detect_hardware()

        self.init_ui()
        self.apply_theme("dark")

        # 尝试加载上次打开的项目
        last_proj = self.config_manager.get("last_project_path", "")
        if last_proj and os.path.exists(last_proj):
            if self.dataset_manager.load_project(last_proj):
                self.annotation_view.reload_data()
                self._update_status_bar()

    def init_ui(self):
        # 1. 顶部菜单栏
        self._create_menu_bar()

        # 2. 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status_proj = QLabel("当前未加载项目")
        self.lbl_status_device = QLabel(f"硬件设备: {self.hardware_info['default_device'].upper()} ({self.hardware_info['primary_gpu_name']})")
        self.status_bar.addWidget(self.lbl_status_proj, stretch=1)
        self.status_bar.addPermanentWidget(self.lbl_status_device)

        # 3. 核心 5 大功能标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)

        self.annotation_view = AnnotationView(self.dataset_manager, self)
        self.dataset_view = DatasetView(self.dataset_manager, self)
        self.train_view = TrainView(self.dataset_manager, self)
        self.inference_view = InferenceView(self.dataset_manager, self)
        self.export_view = ExportView(self)

        self.tab_widget.addTab(self.annotation_view, "✏️ 1. 交互式标注工作台")
        self.tab_widget.addTab(self.dataset_view, "📊 2. 数据集管理与增强")
        self.tab_widget.addTab(self.train_view, "🚀 3. 模型训练与实时大屏")
        self.tab_widget.addTab(self.inference_view, "🔍 4. 推理测试与验证")
        self.tab_widget.addTab(self.export_view, "📦 5. 模型多端导出与部署")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # 训练完成信号联动
        self.train_view.trainer_worker = None  # 占位

        self.setCentralWidget(self.tab_widget)

    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # 【文件】菜单
        file_menu = menu_bar.addMenu("📁 文件 (File)")
        open_action = QAction("打开项目/图片目录...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_action)

        file_menu.addSeparator()
        exit_action = QAction("退出 (Exit)", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 【视图】菜单
        view_menu = menu_bar.addMenu("🎨 视图 (View)")
        dark_action = QAction("🌙 暗黑主题 (Dark)", self)
        dark_action.triggered.connect(lambda: self.apply_theme("dark"))
        view_menu.addAction(dark_action)

        light_action = QAction("☀️ 明亮主题 (Light)", self)
        light_action.triggered.connect(lambda: self.apply_theme("light"))
        view_menu.addAction(light_action)

        # 【帮助】菜单
        help_menu = menu_bar.addMenu("❓ 帮助 (Help)")
        doc_action = QAction("📖 用户使用手册 (User Manual)", self)
        doc_action.triggered.connect(self._show_user_manual)
        help_menu.addAction(doc_action)

        about_action = QAction("ℹ️ 关于 YOLO Studio", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def apply_theme(self, theme_name: str = "dark"):
        """动态加载 QSS 样式表"""
        base_dir = os.path.dirname(__file__)
        qss_file = os.path.join(base_dir, "styles", f"{theme_name}_theme.qss")
        if os.path.exists(qss_file):
            try:
                with open(qss_file, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                logger.warning(f"加载样式表失败: {e}")

    def _on_open_project(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片或项目目录")
        if folder:
            if self.dataset_manager.load_project(folder):
                self.config_manager.add_recent_project(folder)
                self.annotation_view.reload_data()
                self._update_status_bar()
                self.train_view.auto_detect_yaml()

    def _update_status_bar(self):
        if self.dataset_manager.project_dir:
            n_imgs = len(self.dataset_manager.image_files)
            n_classes = len(self.dataset_manager.class_names)
            self.lbl_status_proj.setText(f"当前项目: {self.dataset_manager.project_dir} ({n_imgs} 张图像, {n_classes} 个类别)")
        else:
            self.lbl_status_proj.setText("当前未加载项目")

    def _on_tab_changed(self, index: int):
        if index == 1:
            # 切换到数据集管理页面
            self.dataset_view.refresh_view()
        elif index == 2:
            # 切换到训练大屏页面
            self.train_view.auto_detect_yaml()
        elif index == 3:
            # 切换到推理测试页面
            if self.train_view.best_model_path:
                self.inference_view.load_model(self.train_view.best_model_path)
        elif index == 4:
            # 切换到模型导出页面
            if self.train_view.best_model_path:
                self.export_view.set_weights_path(self.train_view.best_model_path)

    def _show_user_manual(self):
        doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Doc", "06_USER_MANUAL.md"))
        QMessageBox.information(
            self,
            "用户使用手册",
            f"详细用户指南已存放在工程文档中：\n{doc_path}\n\n请在 Doc/06_USER_MANUAL.md 中查看完整图文教程！"
        )

    def _show_about(self):
        QMessageBox.about(
            self,
            "关于 YOLO Studio",
            "<h3>🚀 YOLO Studio v1.0.0</h3>"
            "<p>工业级一站式 YOLO 模型智能标注、训练与部署工作站。</p>"
            "<p>支持 YOLOv8, YOLOv9, YOLOv10, YOLO11 全系列模型。</p>"
            "<p>遵循 Clean Architecture 架构设计与 Agents & Skills 插件体系。</p>"
        )
