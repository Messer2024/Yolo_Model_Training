"""
模型导出与多端量化工作台视图 (Export View)
"""
from typing import Optional
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QCheckBox, QSpinBox,
    QTextEdit, QFileDialog, QMessageBox, QProgressBar
)
from PySide6.QtCore import Qt

from src.core.exporter import ModelExporter
from src.utils.logger import logger


class ExportView(QWidget):
    """模型多端格式转换、半精度量化与自动校验工作台"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.exporter = ModelExporter()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 1. 导出设置面板
        opt_group = QGroupBox("📦 模型多端格式转换与量化选项")
        opt_layout = QGridLayout(opt_group)

        # 模型权重路径
        opt_layout.addWidget(QLabel("PyTorch 权重 (.pt):"), 0, 0)
        self.lbl_weights_path = QLabel("未选择模型权重")
        self.lbl_weights_path.setStyleSheet("color: #00d4bb; font-weight: bold;")
        opt_layout.addWidget(self.lbl_weights_path, 0, 1)

        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._on_browse_weights)
        opt_layout.addWidget(self.btn_browse, 0, 2)

        # 导出格式
        opt_layout.addWidget(QLabel("目标导出格式:"), 1, 0)
        self.combo_format = QComboBox()
        self.combo_format.addItems([
            "onnx (Open Neural Network Exchange - 工业通用标准)",
            "engine (NVIDIA TensorRT - 极速 GPU 加速引擎)",
            "openvino (Intel OpenVINO - CPU/iGPU 加速)",
            "coreml (Apple CoreML - iOS/macOS 部署)",
            "tflite (TensorFlow Lite - Android/移动端轻量格式)",
            "torchscript (C++ LibTorch 生产环境格式)"
        ])
        opt_layout.addWidget(self.combo_format, 1, 1, 1, 2)

        # 分辨率
        opt_layout.addWidget(QLabel("输入分辨率 (imgsz):"), 2, 0)
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems(["640", "512", "416", "320", "1024"])
        opt_layout.addWidget(self.combo_imgsz, 2, 1)

        # 高级复选框
        adv_box = QHBoxLayout()
        self.cb_half = QCheckBox("启用 FP16 半精度加速")
        self.cb_half.setChecked(False)
        adv_box.addWidget(self.cb_half)

        self.cb_dynamic = QCheckBox("启用 Dynamic 动态输入维度 (ONNX)")
        self.cb_dynamic.setChecked(False)
        adv_box.addWidget(self.cb_dynamic)

        self.cb_simplify = QCheckBox("简化计算图 (ONNX-Simplifier)")
        self.cb_simplify.setChecked(True)
        adv_box.addWidget(self.cb_simplify)

        opt_layout.addLayout(adv_box, 3, 0, 1, 3)

        # 导出按钮
        self.btn_export = QPushButton("🚀 一键开始导出模型")
        self.btn_export.setObjectName("primaryButton")
        self.btn_export.setMinimumHeight(38)
        self.btn_export.clicked.connect(self._on_start_export)
        opt_layout.addWidget(self.btn_export, 4, 0, 1, 3)

        main_layout.addWidget(opt_group)

        # 2. 导出日志与校验结果
        log_group = QGroupBox("📋 导出控制台与精度校验报告")
        log_layout = QVBoxLayout(log_group)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("""
            background-color: #16161c;
            color: #dcdcdc;
            font-family: Consolas;
            border-radius: 6px;
        """)
        log_layout.addWidget(self.txt_log)
        main_layout.addWidget(log_group, stretch=1)

    def set_weights_path(self, path: str):
        if os.path.exists(path):
            self.lbl_weights_path.setText(path)

    def _on_browse_weights(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "选择 PyTorch 权重", filter="PyTorch Weights (*.pt)")
        if fpath:
            self.set_weights_path(fpath)

    def _on_start_export(self):
        weights_path = self.lbl_weights_path.text()
        if not os.path.exists(weights_path):
            QMessageBox.warning(self, "警告", "请先选择有效的 .pt 权重文件！")
            return

        fmt_key = self.combo_format.currentText().split()[0].lower()
        imgsz = int(self.combo_imgsz.currentText())
        half = self.cb_half.isChecked()
        dynamic = self.cb_dynamic.isChecked()
        simplify = self.cb_simplify.isChecked()

        self.txt_log.append(f"📦 正在启动模型导出...")
        self.txt_log.append(f"• 原始权重: {weights_path}")
        self.txt_log.append(f"• 导出格式: {fmt_key.upper()} | imgsz: {imgsz} | FP16: {half} | Dynamic: {dynamic}")

        try:
            res = self.exporter.export(
                weights_path=weights_path,
                export_format=fmt_key,
                imgsz=imgsz,
                half=half,
                dynamic=dynamic,
                simplify=simplify
            )

            if res.get("success", False):
                out_path = res.get("exported_path", "")
                self.txt_log.append(f"🎉 导出成功！产物路径: {out_path}")
                if res.get("validated", False):
                    self.txt_log.append("✅ ONNX Runtime 仿真推理校验通过！模型结构完整合规。")
                QMessageBox.information(self, "导出成功", f"🎉 模型已成功导出为 {fmt_key.upper()} 格式！\n路径:\n{out_path}")
            else:
                self.txt_log.append(f"❌ 导出失败: {res.get('message')}")
                QMessageBox.critical(self, "导出失败", res.get("message", "未知错误"))
        except Exception as e:
            self.txt_log.append(f"❌ 发生异常: {str(e)}")
            QMessageBox.critical(self, "导出异常", str(e))
