"""
模型推理与效果验证工作台视图 (Inference View)
"""
from typing import Optional
import os
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QSlider, QFileDialog, QMessageBox,
    QSplitter, QListWidget, QListWidgetItem, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap

from src.core.inference import InferenceEngine
from src.core.dataset_manager import DatasetManager
from src.utils.logger import logger


class InferenceView(QWidget):
    """交互式模型测试与推理验证台 (Playground)"""

    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self.dm = dataset_manager
        self.engine = InferenceEngine()
        self.current_img = None
        self.webcam_timer = QTimer(self)
        self.webcam_timer.timeout.connect(self._on_webcam_frame)
        self.cap = None

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 顶部操作与参数控制区
        top_group = QGroupBox("🔍 模型加载与推理控制")
        top_layout = QGridLayout(top_group)

        # 1. 权重文件选择
        top_layout.addWidget(QLabel("模型权重 (.pt / .onnx):"), 0, 0)
        self.lbl_model_path = QLabel("未加载模型 (可载入训练好的 best.pt 或官方预训练权重)")
        self.lbl_model_path.setStyleSheet("color: #00d4bb; font-weight: bold;")
        top_layout.addWidget(self.lbl_model_path, 0, 1)

        self.btn_load_model = QPushButton("📂 载入权重...")
        self.btn_load_model.setObjectName("primaryButton")
        self.btn_load_model.clicked.connect(self._on_browse_model)
        top_layout.addWidget(self.btn_load_model, 0, 2)

        # 2. 置信度滑块
        top_layout.addWidget(QLabel("置信度阈值 (Confidence):"), 1, 0)
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(1, 100)
        self.slider_conf.setValue(25)
        self.slider_conf.valueChanged.connect(self._on_params_changed)
        self.lbl_conf_val = QLabel("0.25")
        self.lbl_conf_val.setFixedWidth(40)
        conf_box = QHBoxLayout()
        conf_box.addWidget(self.slider_conf)
        conf_box.addWidget(self.lbl_conf_val)
        top_layout.addLayout(conf_box, 1, 1)

        # 3. IoU 滑块
        top_layout.addWidget(QLabel("NMS IoU 阈值:"), 1, 2)
        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setRange(1, 100)
        self.slider_iou.setValue(45)
        self.slider_iou.valueChanged.connect(self._on_params_changed)
        self.lbl_iou_val = QLabel("0.45")
        self.lbl_iou_val.setFixedWidth(40)
        iou_box = QHBoxLayout()
        iou_box.addWidget(self.slider_iou)
        iou_box.addWidget(self.lbl_iou_val)
        top_layout.addLayout(iou_box, 1, 3)

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

        src_box.addStretch()
        self.lbl_perf = QLabel("耗时: -- ms | 目标数: 0")
        self.lbl_perf.setStyleSheet("font-weight: bold; color: #ffb86c;")
        src_box.addWidget(self.lbl_perf)

        top_layout.addLayout(src_box, 2, 0, 1, 4)
        main_layout.addWidget(top_group)

        # 中部图像展示与结果大屏
        splitter = QSplitter(Qt.Horizontal)

        # 图像显示 QLabel
        self.lbl_canvas = QLabel("请载入模型并选择测试图片/视频以查看检测结果")
        self.lbl_canvas.setAlignment(Qt.AlignCenter)
        self.lbl_canvas.setStyleSheet("""
            background-color: #141419;
            border: 1px solid #2d2d38;
            border-radius: 8px;
            color: #707080;
            font-size: 15px;
        """)
        splitter.addWidget(self.lbl_canvas)

        # 右侧目标清单
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("🎯 检出目标列表:"))
        self.list_detections = QListWidget()
        right_layout.addWidget(self.list_detections)
        splitter.addWidget(right_panel)

        splitter.setSizes([750, 250])
        main_layout.addWidget(splitter, stretch=1)

    def load_model(self, model_path: str):
        """外部调用载入模型"""
        if self.engine.load_model(model_path):
            self.lbl_model_path.setText(model_path)

    def _on_browse_model(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "选择模型权重", filter="Model Files (*.pt *.onnx)")
        if fpath:
            self.load_model(fpath)

    def _on_params_changed(self):
        conf = self.slider_conf.value() / 100.0
        iou = self.slider_iou.value() / 100.0
        self.lbl_conf_val.setText(f"{conf:.2f}")
        self.lbl_iou_val.setText(f"{iou:.2f}")

        # 如果已有图片，重新推理
        if self.current_img is not None and not self.cb_webcam.isChecked():
            self._run_inference_on_current()

    def _on_test_image(self):
        if self.engine.model is None and not self.engine.is_onnx:
            QMessageBox.warning(self, "提示", "请先载入模型权重 (.pt / .onnx)！")
            return

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
        for b in boxes:
            cls_name = self.engine.class_names[b.class_id] if b.class_id < len(self.engine.class_names) else f"class_{b.class_id}"
            item_text = f"🎯 {cls_name} (置信度: {b.confidence:.1%})"
            self.list_detections.addItem(item_text)

        # 转换为 QPixmap 显示
        rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.lbl_canvas.setPixmap(pix.scaled(self.lbl_canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_test_video(self):
        if self.engine.model is None and not self.engine.is_onnx:
            QMessageBox.warning(self, "提示", "请先载入模型权重！")
            return
        fpath, _ = QFileDialog.getOpenFileName(self, "选择测试视频", filter="Videos (*.mp4 *.avi *.mov *.mkv)")
        if fpath:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(fpath)
            self.webcam_timer.start(30)

    def _on_toggle_webcam(self, checked: bool):
        if checked:
            if self.engine.model is None and not self.engine.is_onnx:
                QMessageBox.warning(self, "提示", "请先载入模型权重！")
                self.cb_webcam.setChecked(False)
                return
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头！")
                self.cb_webcam.setChecked(False)
                return
            self.webcam_timer.start(30)
        else:
            self.webcam_timer.stop()
            if self.cap:
                self.cap.release()
                self.cap = None

    def _on_webcam_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_img = frame
                self._run_inference_on_current()
            else:
                self.webcam_timer.stop()
