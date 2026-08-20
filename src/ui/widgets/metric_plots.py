"""
实时训练指标动态图表控件 (Metric Plots Widget)
"""
from typing import Dict, List, Any
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Slot

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MetricPlotsWidget(QWidget):
    """训练过程动态指标监控看板 (Loss, mAP50, mAP50-95, Precision, Recall)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.epochs: List[int] = []
        self.box_loss: List[float] = []
        self.cls_loss: List[float] = []
        self.dfl_loss: List[float] = []
        self.map50: List[float] = []
        self.map50_95: List[float] = []
        self.precision: List[float] = []
        self.recall: List[float] = []

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建暗黑风格 Matplotlib 画布
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # 创建 2x2 子图
        self.ax_loss = self.fig.add_subplot(2, 2, 1)
        self.ax_map = self.fig.add_subplot(2, 2, 2)
        self.ax_pr = self.fig.add_subplot(2, 2, 3)
        self.ax_dfl = self.fig.add_subplot(2, 2, 4)

        self._style_axis(self.ax_loss, "Training Losses", "Loss")
        self._style_axis(self.ax_map, "mAP Validation", "mAP")
        self._style_axis(self.ax_pr, "Precision & Recall", "Value")
        self._style_axis(self.ax_dfl, "DFL Loss", "Loss")

        self.fig.tight_layout()
        self.canvas.draw()

    def _style_axis(self, ax, title: str, ylabel: str):
        ax.set_facecolor("#16161c")
        ax.set_title(title, color="#00d4bb", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", color="#a0a0b0", fontsize=8)
        ax.set_ylabel(ylabel, color="#a0a0b0", fontsize=8)
        ax.tick_params(colors="#808090", labelsize=8)
        ax.grid(True, linestyle="--", alpha=0.3, color="#404050")
        for spine in ax.spines.values():
            spine.set_color("#303040")

    @Slot(int, dict)
    def update_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """接收一轮 Epoch 结束后的指标并增量刷新图表"""
        self.epochs.append(epoch)
        self.box_loss.append(metrics.get("box_loss", 0.0))
        self.cls_loss.append(metrics.get("cls_loss", 0.0))
        self.dfl_loss.append(metrics.get("dfl_loss", 0.0))
        self.map50.append(metrics.get("mAP50", 0.0))
        self.map50_95.append(metrics.get("mAP50_95", 0.0))
        self.precision.append(metrics.get("precision", 0.0))
        self.recall.append(metrics.get("recall", 0.0))

        # 刷新 Loss 图
        self.ax_loss.clear()
        self._style_axis(self.ax_loss, "Training Losses", "Loss")
        if self.box_loss:
            self.ax_loss.plot(self.epochs, self.box_loss, label="Box Loss", color="#ff701f", linewidth=1.5)
        if self.cls_loss:
            self.ax_loss.plot(self.epochs, self.cls_loss, label="Cls Loss", color="#ff3838", linewidth=1.5)
        self.ax_loss.legend(loc="upper right", facecolor="#20202a", edgecolor="none", labelcolor="#e0e0e0", fontsize=7)

        # 刷新 mAP 图
        self.ax_map.clear()
        self._style_axis(self.ax_map, "mAP Metrics", "mAP")
        if self.map50:
            self.ax_map.plot(self.epochs, self.map50, label="mAP@50", color="#00d4bb", linewidth=1.8)
        if self.map50_95:
            self.ax_map.plot(self.epochs, self.map50_95, label="mAP@50-95", color="#6473ff", linewidth=1.5)
        self.ax_map.legend(loc="lower right", facecolor="#20202a", edgecolor="none", labelcolor="#e0e0e0", fontsize=7)

        # 刷新 Precision & Recall 图
        self.ax_pr.clear()
        self._style_axis(self.ax_pr, "Precision & Recall", "Score")
        if self.precision:
            self.ax_pr.plot(self.epochs, self.precision, label="Precision", color="#48f90a", linewidth=1.5)
        if self.recall:
            self.ax_pr.plot(self.epochs, self.recall, label="Recall", color="#cfd231", linewidth=1.5)
        self.ax_pr.legend(loc="lower right", facecolor="#20202a", edgecolor="none", labelcolor="#e0e0e0", fontsize=7)

        # 刷新 DFL Loss 图
        self.ax_dfl.clear()
        self._style_axis(self.ax_dfl, "DFL Loss", "Loss")
        if self.dfl_loss:
            self.ax_dfl.plot(self.epochs, self.dfl_loss, label="DFL Loss", color="#cb38ff", linewidth=1.5)
        self.ax_dfl.legend(loc="upper right", facecolor="#20202a", edgecolor="none", labelcolor="#e0e0e0", fontsize=7)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def reset_plots(self):
        """清空图表数据"""
        self.epochs.clear()
        self.box_loss.clear()
        self.cls_loss.clear()
        self.dfl_loss.clear()
        self.map50.clear()
        self.map50_95.clear()
        self.precision.clear()
        self.recall.clear()

        for ax, title, ylabel in [
            (self.ax_loss, "Training Losses", "Loss"),
            (self.ax_map, "mAP Validation", "mAP"),
            (self.ax_pr, "Precision & Recall", "Value"),
            (self.ax_dfl, "DFL Loss", "Loss")
        ]:
            ax.clear()
            self._style_axis(ax, title, ylabel)

        self.fig.tight_layout()
        self.canvas.draw_idle()
