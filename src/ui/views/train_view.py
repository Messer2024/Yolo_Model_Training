"""
模型训练与实时监控大屏视图 (Train View)
"""
from typing import Optional
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QFileDialog, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, Slot

from src.core.dataset_manager import DatasetManager
from src.core.trainer import YoloTrainerWorker, TrainConfig
from src.ui.widgets.metric_plots import MetricPlotsWidget
from src.ui.widgets.log_console import LogConsoleWidget
from src.utils.hardware import detect_hardware
from agents.autotrain_agent import AutoTrainAgent
from src.utils.logger import logger


class TrainView(QWidget):
    """模型训练配置、智能调优决策、训练状态机控制与实时大屏"""

    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self.dm = dataset_manager
        self.hardware_info = detect_hardware()
        self.autotrain_agent = AutoTrainAgent()
        self.trainer_worker: Optional[YoloTrainerWorker] = None
        self.best_model_path: str = ""

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 顶部配置栏 (网格排列)
        config_group = QGroupBox("⚙️ 模型架构与超参数配置")
        config_layout = QGridLayout(config_group)

        # 1. 模型架构
        config_layout.addWidget(QLabel("预训练模型:"), 0, 0)
        self.combo_model = QComboBox()
        self.combo_model.addItems([
            "yolov8n.pt (Nano - 极速轻量)",
            "yolov8s.pt (Small - 平衡推荐)",
            "yolov8m.pt (Medium - 高精度)",
            "yolov8x.pt (XLarge - 极限精度)",
            "yolo11n.pt (YOLO11 Nano)",
            "yolo11s.pt (YOLO11 Small)",
            "yolo11m.pt (YOLO11 Medium)",
            "yolo11x.pt (YOLO11 XLarge)"
        ])
        config_layout.addWidget(self.combo_model, 0, 1)

        # 2. data.yaml 路径
        config_layout.addWidget(QLabel("数据集配置 (data.yaml):"), 0, 2)
        yaml_box = QHBoxLayout()
        self.lbl_yaml = QLabel("未选择 data.yaml")
        self.lbl_yaml.setStyleSheet("color: #00d4bb; font-weight: bold;")
        yaml_box.addWidget(self.lbl_yaml)
        self.btn_select_yaml = QPushButton("浏览...")
        self.btn_select_yaml.clicked.connect(self._on_browse_yaml)
        yaml_box.addWidget(self.btn_select_yaml)
        config_layout.addLayout(yaml_box, 0, 3)

        # 3. Epochs
        config_layout.addWidget(QLabel("训练轮数 (Epochs):"), 1, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(100)
        config_layout.addWidget(self.spin_epochs, 1, 1)

        # 4. Batch Size
        config_layout.addWidget(QLabel("批次大小 (Batch Size):"), 1, 2)
        self.combo_batch = QComboBox()
        self.combo_batch.addItems(["16", "32", "8", "4", "64", "-1 (Auto)"])
        config_layout.addWidget(self.combo_batch, 1, 3)

        # 5. Image Size
        config_layout.addWidget(QLabel("输入分辨率 (Image Size):"), 2, 0)
        self.combo_imgsz = QComboBox()
        self.combo_imgsz.addItems(["640", "512", "416", "320", "1024", "1280"])
        config_layout.addWidget(self.combo_imgsz, 2, 1)

        # 6. 计算设备
        config_layout.addWidget(QLabel("计算设备 (Device):"), 2, 2)
        self.combo_device = QComboBox()
        if self.hardware_info["is_cuda_available"]:
            for d in self.hardware_info["devices"]:
                self.combo_device.addItem(f"CUDA:{d['id']} ({d['name']} - {d['vram_gb']}GB)", d['id'])
        else:
            self.combo_device.addItem("CPU 处理器", "cpu")
        config_layout.addWidget(self.combo_device, 2, 3)

        # 智能推荐与操作按钮
        btn_bar = QHBoxLayout()
        self.btn_recommend = QPushButton("🤖 智能体自适应推荐参数 (AutoML)")
        self.btn_recommend.setStyleSheet("background-color: #3f37c9; color: white;")
        self.btn_recommend.clicked.connect(self._on_auto_recommend)
        btn_bar.addWidget(self.btn_recommend)

        btn_bar.addStretch()

        self.btn_start = QPushButton("🚀 开始训练")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.setMinimumWidth(120)
        self.btn_start.clicked.connect(self._on_start_training)
        btn_bar.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸️ 暂停")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause_training)
        btn_bar.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("⏹️ 终止训练")
        self.btn_stop.setObjectName("dangerButton")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_training)
        btn_bar.addWidget(self.btn_stop)

        config_layout.addLayout(btn_bar, 3, 0, 1, 4)
        main_layout.addWidget(config_group)

        # 进度状态条
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("训练就绪 | 等待启动...")
        main_layout.addWidget(self.progress_bar)

        # 中部大屏分割：左侧实时图表看板 + 右侧日志终端
        splitter = QSplitter(Qt.Horizontal)

        # 实时曲线
        self.metric_plots = MetricPlotsWidget()
        splitter.addWidget(self.metric_plots)

        # 日志终端
        self.log_console = LogConsoleWidget()
        splitter.addWidget(self.log_console)

        splitter.setSizes([550, 450])
        main_layout.addWidget(splitter, stretch=1)

    def auto_detect_yaml(self):
        """自动检测项目下的 data.yaml"""
        if not self.dm.project_dir:
            return
        split_yaml = os.path.join(self.dm.project_dir, "dataset_split", "data.yaml")
        root_yaml = os.path.join(self.dm.project_dir, "data.yaml")

        if os.path.exists(split_yaml):
            self.lbl_yaml.setText(split_yaml)
        elif os.path.exists(root_yaml):
            self.lbl_yaml.setText(root_yaml)

    def _on_browse_yaml(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "选择 data.yaml", filter="YAML Files (*.yaml *.yml)")
        if fpath:
            self.lbl_yaml.setText(fpath)

    def _on_auto_recommend(self):
        ctx = {
            "total_images": len(self.dm.image_files),
            "total_classes": len(self.dm.class_names),
            "is_cuda_available": self.hardware_info["is_cuda_available"],
            "gpu_vram_gb": self.hardware_info["primary_gpu_vram_gb"],
            "gpu_name": self.hardware_info["primary_gpu_name"]
        }
        res = self.autotrain_agent.run(ctx)

        self.spin_epochs.setValue(res["recommended_epochs"])
        self.combo_batch.setCurrentText(str(res["recommended_batch"]))

        # 匹配模型下拉
        for idx in range(self.combo_model.count()):
            if res["recommended_model"] in self.combo_model.itemText(idx):
                self.combo_model.setCurrentIndex(idx)
                break

        QMessageBox.information(
            self,
            "智能体推荐策略",
            f"🤖 决策建议：\n{res['reason']}\n\n"
            f"• 推荐模型: {res['recommended_model']}\n"
            f"• 推荐 Epochs: {res['recommended_epochs']}\n"
            f"• 推荐 Batch: {res['recommended_batch']}\n"
            f"• 推荐分辨率: {res['recommended_imgsz']}"
        )

    def _on_start_training(self):
        yaml_path = self.lbl_yaml.text()
        if not os.path.exists(yaml_path):
            QMessageBox.warning(self, "警告", "请先在【数据集管理】中划分数据集，或手动选择有效的 data.yaml 文件！")
            return

        model_token = self.combo_model.currentText().split()[0]
        batch_text = self.combo_batch.currentText()
        batch_size = -1 if "Auto" in batch_text else int(batch_text)
        imgsz = int(self.combo_imgsz.currentText())
        epochs = self.spin_epochs.value()
        device = self.combo_device.currentData() or "cpu"

        config = TrainConfig(
            data_yaml=yaml_path,
            model_name=model_token,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            device=str(device)
        )

        self.metric_plots.reset_plots()
        self.log_console.clear_logs()

        self.trainer_worker = YoloTrainerWorker(config)
        self.trainer_worker.epoch_finished.connect(self.metric_plots.update_metrics)
        self.trainer_worker.epoch_finished.connect(self._on_epoch_finished)
        self.trainer_worker.batch_progress.connect(self._on_batch_progress)
        self.trainer_worker.log_message.connect(self.log_console.append_log)
        self.trainer_worker.training_done.connect(self._on_training_done)
        self.trainer_worker.training_failed.connect(self._on_training_failed)

        self.trainer_worker.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setFormat("正在初始化训练...")

    def _on_pause_training(self):
        if self.trainer_worker:
            if self.btn_pause.text() == "⏸️ 暂停":
                self.trainer_worker.pause()
                self.btn_pause.setText("▶️ 恢复")
                self.log_console.append_log("⏸️ 用户已暂停训练")
            else:
                self.trainer_worker.resume()
                self.btn_pause.setText("⏸️ 暂停")
                self.log_console.append_log("▶️ 用户已恢复训练")

    def _on_stop_training(self):
        if self.trainer_worker:
            self.trainer_worker.stop()
            self.log_console.append_log("⏹️ 正在请求安全终止训练...")
            self.btn_stop.setEnabled(False)

    @Slot(int, dict)
    def _on_epoch_finished(self, epoch: int, metrics: dict):
        total = self.spin_epochs.value()
        pct = int((epoch / total) * 100)
        self.progress_bar.setValue(pct)
        map50 = metrics.get("mAP50", 0.0)
        box_loss = metrics.get("box_loss", 0.0)
        self.progress_bar.setFormat(f"Epoch {epoch}/{total} ({pct}%) | mAP50: {map50:.4f} | Box Loss: {box_loss:.4f}")

    @Slot(int, int, float, str)
    def _on_batch_progress(self, batch: int, total_b: int, loss: float, status: str):
        pass

    @Slot(str)
    def _on_training_done(self, best_path: str):
        self.best_model_path = best_path
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat(f"🎉 训练完成！权重: {os.path.basename(best_path)}")
        QMessageBox.information(self, "训练成功", f"🎉 模型训练顺利完成！\n最优权重已保存至:\n{best_path}\n\n您可直接前往【推理测试】页面验证模型！")

    @Slot(str)
    def _on_training_failed(self, error: str):
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setFormat("❌ 训练中断或失败")
        QMessageBox.critical(self, "训练失败", f"训练过程出错:\n{error}")
