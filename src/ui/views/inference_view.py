"""
模型推理与效果验证工作台视图 (Inference View)
"""
from typing import Optional
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QSlider, QFileDialog, QMessageBox,
    QSplitter, QListWidget, QListWidgetItem, QCheckBox, QComboBox
)
from PySide6.QtCore import Qt, QTimer, Slot

from src.core.inference import InferenceEngine
from src.core.dataset_manager import DatasetManager
from src.ui.widgets.video_canvas import ImageDisplayCanvas
from src.utils.logger import logger


class InferenceView(QWidget):
    """交互式模型测试与推理验证台 (Playground)"""

    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self.dm = dataset_manager
        self.engine = InferenceEngine()
        self.current_img: Optional[np.ndarray] = None
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self._on_video_frame)
        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_trained_weights: str = ""

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 顶部操作与参数控制区
        top_group = QGroupBox("🔍 模型加载与推理控制")
        top_layout = QGridLayout(top_group)

        # 1. 权重文件选择快捷下拉与浏览
        top_layout.addWidget(QLabel("模型权重来源:"), 0, 0)

        model_select_layout = QHBoxLayout()
        self.combo_quick_model = QComboBox()
        self.combo_quick_model.addItem("⚡ 官方轻量预训练模型 (yolov8n.pt)", "yolov8n.pt")
        self.combo_quick_model.addItem("📂 自定义外部模型文件...", "custom")
        self.combo_quick_model.currentIndexChanged.connect(self._on_quick_model_changed)
        model_select_layout.addWidget(self.combo_quick_model, stretch=1)

        self.btn_load_model = QPushButton("📂 浏览权重...")
        self.btn_load_model.setObjectName("primaryButton")
        self.btn_load_model.clicked.connect(self._on_browse_model)
        model_select_layout.addWidget(self.btn_load_model)

        top_layout.addLayout(model_select_layout, 0, 1, 1, 3)

        # 当前加载状态
        top_layout.addWidget(QLabel("当前加载状态:"), 1, 0)
        self.lbl_model_path = QLabel("🟢 默认已就绪: yolov8n.pt (官方预训练 80 类别)")
        self.lbl_model_path.setStyleSheet("color: #00d4bb; font-weight: bold;")
        top_layout.addWidget(self.lbl_model_path, 1, 1, 1, 3)

        # 2. 置信度滑块
        top_layout.addWidget(QLabel("置信度阈值 (Confidence):"), 2, 0)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(1, 100)
        self.slider_conf.setValue(20)  # 默认 0.20 方便查看新训练模型效果
        self.slider_conf.valueChanged.connect(self._on_params_changed)
        self.lbl_conf_val = QLabel("0.20")
        self.lbl_conf_val.setFixedWidth(40)
        conf_box = QHBoxLayout()
        conf_box.addWidget(self.slider_conf)
        conf_box.addWidget(self.lbl_conf_val)
        top_layout.addLayout(conf_box, 2, 1)

        # 3. IoU 滑块
        top_layout.addWidget(QLabel("NMS IoU 阈值:"), 2, 2)
        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setRange(1, 100)
        self.slider_iou.setValue(45)
        self.slider_iou.valueChanged.connect(self._on_params_changed)
        self.lbl_iou_val = QLabel("0.45")
        self.lbl_iou_val.setFixedWidth(40)
        iou_box = QHBoxLayout()
        iou_box.addWidget(self.slider_iou)
        iou_box.addWidget(self.lbl_iou_val)
        top_layout.addLayout(iou_box, 2, 3)

        # 4. 测试源按钮
        src_box = QHBoxLayout()
        self.btn_test_image = QPushButton("🖼️ 测试单张图片...")
        self.btn_test_image.clicked.connect(self._on_test_image)
        src_box.addWidget(self.btn_test_image)

        self.btn_test_video = QPushButton("🎬 测试视频文件...")
        self.btn_test_video.clicked.connect(self._on_test_video)
        src_box.addWidget(self.btn_test_video)

        self.cb_webcam = QCheckBox("📹 开启摄像头 (Webcam)")
        self.cb_webcam.toggled.connect(self._on_toggle_webcam)
        src_box.addWidget(self.cb_webcam)

        self.btn_stop_media = QPushButton("⏹️ 停止视频/摄像头")
        self.btn_stop_media.setEnabled(False)
        self.btn_stop_media.clicked.connect(self._stop_media)
        src_box.addWidget(self.btn_stop_media)

        src_box.addStretch()
        self.lbl_perf = QLabel("前向耗时: -- ms | 检出目标: 0 个")
        self.lbl_perf.setStyleSheet("font-weight: bold; color: #ffb86c; font-size: 13px;")
        src_box.addWidget(self.lbl_perf)

        top_layout.addLayout(src_box, 3, 0, 1, 4)
        main_layout.addWidget(top_group)

        # 中部图像展示与结果大屏
        splitter = QSplitter(Qt.Horizontal)

        # 使用自适应防抖画布 (ImageDisplayCanvas)
        self.canvas_widget = ImageDisplayCanvas()
        splitter.addWidget(self.canvas_widget)

        # 右侧目标清单与状态面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("🎯 检出目标明细列表:"))
        self.list_detections = QListWidget()
        right_layout.addWidget(self.list_detections)

        self.lbl_hint = QLabel("💡 提示：若使用刚刚训练的自定义模型未出结果，可向左滑动适当调低置信度（如 0.05-0.15）。")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("color: #a0a0b0; font-size: 11px; padding: 4px;")
        right_layout.addWidget(self.lbl_hint)

        splitter.addWidget(right_panel)

        splitter.setSizes([750, 250])
        main_layout.addWidget(splitter, stretch=1)

        # 初始化时默认载入官方模型或准备就绪
        self.load_model("yolov8n.pt")

    def set_newly_trained_model(self, weights_path: str):
        """当训练完成时自动注册并优先选中刚刚训练的模型"""
        if not weights_path or not os.path.exists(weights_path):
            return

        abs_path = os.path.abspath(weights_path)
        self.latest_trained_weights = abs_path

        # 检查是否已在下拉列表
        tag = f"🎯 刚刚训练的最优模型 ({os.path.basename(abs_path)})"
        idx = -1
        for i in range(self.combo_quick_model.count()):
            if self.combo_quick_model.itemData(i) == abs_path:
                idx = i
                break

        if idx == -1:
            self.combo_quick_model.insertItem(0, tag, abs_path)
            self.combo_quick_model.setCurrentIndex(0)
        else:
            self.combo_quick_model.setCurrentIndex(idx)

        self.load_model(abs_path)

    def load_model(self, model_path: str) -> bool:
        """外部调用载入模型"""
        if not model_path:
            return False

        # 如果是官方预训练模型且不存在本地，YOLO 会自动从线上下载，无需检查本地文件存在
        if not os.path.exists(model_path) and not model_path.endswith(".pt"):
            logger.error(f"模型文件不存在: {model_path}")
            return False

        success = self.engine.load_model(model_path)
        if success:
            n_classes = len(self.engine.class_names)
            # 如果当前项目有类别定义且模型自带类别不足，则同步类别信息
            if self.dm.class_names and (n_classes == 0 or "class_0" in self.engine.class_names):
                self.engine.class_names = self.dm.class_names.copy()
                n_classes = len(self.engine.class_names)

            model_name = os.path.basename(model_path)
            self.lbl_model_path.setText(f"🟢 已加载: {model_name} (包含 {n_classes} 个识别类别)")
            self.lbl_model_path.setStyleSheet("color: #00d4bb; font-weight: bold;")
            logger.info(f"InferenceView 成功加载模型: {model_path}")

            # 如果当前已有图片，立即重新推理
            if self.current_img is not None and not self.webcam_timer.isActive():
                self._run_inference_on_current()
            return True
        else:
            self.lbl_model_path.setText(f"❌ 加载失败: {model_path}")
            self.lbl_model_path.setStyleSheet("color: #ff5555; font-weight: bold;")
            return False

    def _on_quick_model_changed(self, index: int):
        data = self.combo_quick_model.itemData(index)
        if data == "custom":
            self._on_browse_model()
        elif data:
            self.load_model(data)

    def _on_browse_model(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "选择模型权重", filter="Model Files (*.pt *.onnx)")
        if fpath:
            abs_path = os.path.abspath(fpath)
            # 添加到下拉框并选中
            item_text = f"📦 自定义模型 ({os.path.basename(abs_path)})"
            self.combo_quick_model.insertItem(0, item_text, abs_path)
            self.combo_quick_model.setCurrentIndex(0)
            self.load_model(abs_path)

    def _on_params_changed(self):
        conf = self.slider_conf.value() / 100.0
        iou = self.slider_iou.value() / 100.0
        self.lbl_conf_val.setText(f"{conf:.2f}")
        self.lbl_iou_val.setText(f"{iou:.2f}")

        # 如果已有静态图片，重新推理
        if self.current_img is not None and not self.webcam_timer.isActive():
            self._run_inference_on_current()

    def _on_test_image(self):
        if self.engine.model is None and not self.engine.is_onnx:
            QMessageBox.warning(self, "提示", "请先载入模型权重 (.pt / .onnx)！")
            return

        self._stop_media()
        fpath, _ = QFileDialog.getOpenFileName(self, "选择测试图片", filter="Images (*.jpg *.png *.jpeg *.bmp *.webp)")
        if fpath:
            self.current_img = cv2.imread(fpath)
            self._run_inference_on_current()

    def _run_inference_on_current(self):
        if self.current_img is None:
            return

        conf = self.slider_conf.value() / 100.0
        iou = self.slider_iou.value() / 100.0
        boxes, latency, rendered_bgr = self.engine.predict_image(self.current_img, conf, iou)

        self.lbl_perf.setText(f"前向耗时: {latency:.1f} ms | 检出目标: {len(boxes)} 个")

        # 刷新目标列表
        self.list_detections.clear()
        if len(boxes) == 0:
            item = QListWidgetItem("（当前阈值下未检出目标）")
            item.setForeground(Qt.gray)
            self.list_detections.addItem(item)
        else:
            for b in boxes:
                cls_name = self.engine.class_names[b.class_id] if b.class_id < len(self.engine.class_names) else f"class_{b.class_id}"
                item_text = f"🎯 {cls_name} (置信度: {b.confidence:.1%})"
                self.list_detections.addItem(item_text)

        # 在无抖动画布上渲染
        self.canvas_widget.set_bgr_image(rendered_bgr)

    def _on_test_video(self):
        if self.engine.model is None and not self.engine.is_onnx:
            QMessageBox.warning(self, "提示", "请先载入模型权重！")
            return

        fpath, _ = QFileDialog.getOpenFileName(self, "选择测试视频", filter="Videos (*.mp4 *.avi *.mov *.mkv)")
        if fpath:
            self._stop_media()
            self.cap = cv2.VideoCapture(fpath)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", f"无法打开视频文件: {fpath}")
                return
            self.btn_stop_media.setEnabled(True)
            self.webcam_timer.start(30)

    def _on_toggle_webcam(self, checked: bool):
        if checked:
            if self.engine.model is None and not self.engine.is_onnx:
                QMessageBox.warning(self, "提示", "请先载入模型权重！")
                self.cb_webcam.setChecked(False)
                return
            self._stop_media()
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头设备！")
                self.cb_webcam.setChecked(False)
                return
            self.btn_stop_media.setEnabled(True)
            self.webcam_timer.start(30)
        else:
            self._stop_media()

    def _stop_media(self):
        """停止视频或摄像头播放"""
        self.webcam_timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_stop_media.setEnabled(False)
        self.cb_webcam.blockSignals(True)
        self.cb_webcam.setChecked(False)
        self.cb_webcam.blockSignals(False)

    def _on_video_frame(self):
        """视频与摄像头逐帧推理回调"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_img = frame
                self._run_inference_on_current()
            else:
                self._stop_media()
