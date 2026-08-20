"""
YOLO 训练引擎与异步子线程包装 (Trainer Engine)
"""
from typing import Dict, Any, Optional
import os
import sys
import threading
import time

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:
    # 纯 Python 测试环境 Fallback
    class Signal:
        def __init__(self, *args):
            self._callbacks = []

        def connect(self, cb):
            self._callbacks.append(cb)

        def emit(self, *args):
            for cb in self._callbacks:
                try:
                    cb(*args)
                except Exception:
                    pass

    class QThread(threading.Thread):
        def __init__(self):
            super().__init__()
            self.daemon = True


class TrainConfig:
    """训练参数配置实体"""
    def __init__(
        self,
        data_yaml: str,
        model_name: str = "yolov8n.pt",
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        device: str = "0",
        workers: int = 4,
        optimizer: str = "auto",
        lr0: float = 0.01,
        lrf: float = 0.01,
        project_name: str = "runs/detect",
        experiment_name: str = "train",
        resume: bool = False,
        amp: bool = True
    ):
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.imgsz = imgsz
        self.device = device
        self.workers = workers
        self.optimizer = optimizer
        self.lr0 = lr0
        self.lrf = lrf
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.resume = resume
        self.amp = amp


class YoloTrainerWorker(QThread):
    """在独立子线程中运行 Ultralytics 训练"""

    # 定义 Qt 回传信号
    epoch_finished = Signal(int, dict)          # epoch_index, metrics_dict
    batch_progress = Signal(int, int, float, str)  # current_batch, total_batches, batch_loss, status
    training_done = Signal(str)                  # best_model_weights_path
    training_failed = Signal(str)                # error_message
    log_message = Signal(str)                    # raw_log_text

    def __init__(self, config: TrainConfig):
        super().__init__()
        self.config = config
        self._is_running = True
        self._is_paused = False

    def stop(self):
        """中断训练"""
        self._is_running = False

    def pause(self):
        """暂停训练"""
        self._is_paused = True

    def resume(self):
        """恢复训练"""
        self._is_paused = False

    def run(self):
        try:
            self.log_message.emit(f"🚀 正在初始化 YOLO 训练环境: {self.config.model_name}...")
            self.log_message.emit(f"⚙️ 数据集配置文件: {self.config.data_yaml}")
            self.log_message.emit(f"⚙️ 目标设备: {self.config.device} | Epochs: {self.config.epochs} | Batch: {self.config.batch_size}")

            try:
                from ultralytics import YOLO
            except ImportError:
                raise RuntimeError("未安装 ultralytics 库，请先执行 `pip install ultralytics`")

            # 实例化 YOLO 模型
            model = YOLO(self.config.model_name)

            # 挂接 Ultralytics 回调
            def on_train_epoch_end(trainer):
                if not self._is_running:
                    trainer.stop = True
                    return

                epoch = trainer.epoch + 1
                # 提取训练与验证指标
                metrics = {}
                if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                    loss_items = trainer.loss_items.cpu().numpy() if hasattr(trainer.loss_items, "cpu") else trainer.loss_items
                    if len(loss_items) >= 3:
                        metrics["box_loss"] = float(loss_items[0])
                        metrics["cls_loss"] = float(loss_items[1])
                        metrics["dfl_loss"] = float(loss_items[2])

                # 提取验证集评估结果
                if hasattr(trainer, "metrics") and trainer.metrics:
                    m = trainer.metrics
                    metrics["precision"] = float(m.get("metrics/precision(B)", 0.0))
                    metrics["recall"] = float(m.get("metrics/recall(B)", 0.0))
                    metrics["mAP50"] = float(m.get("metrics/mAP50(B)", 0.0))
                    metrics["mAP50_95"] = float(m.get("metrics/mAP50-95(B)", 0.0))

                self.epoch_finished.emit(epoch, metrics)
                self.log_message.emit(
                    f"Epoch [{epoch}/{self.config.epochs}] - "
                    f"mAP50: {metrics.get('mAP50', 0.0):.4f} | "
                    f"Box Loss: {metrics.get('box_loss', 0.0):.4f}"
                )

            def on_train_batch_end(trainer):
                if not self._is_running:
                    trainer.stop = True
                    return

                # 处理暂停
                while self._is_paused and self._is_running:
                    time.sleep(0.5)

                batch = getattr(trainer, "batch", 0) + 1
                total_batches = getattr(trainer, "num_batches", 1)
                loss = 0.0
                if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                    loss = float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0.0

                self.batch_progress.emit(
                    batch,
                    total_batches,
                    loss,
                    f"Epoch {trainer.epoch + 1}/{self.config.epochs} [Batch {batch}/{total_batches}]"
                )

            model.add_callback("on_train_epoch_end", on_train_epoch_end)
            model.add_callback("on_train_batch_end", on_train_batch_end)

            # 启动训练
            results = model.train(
                data=self.config.data_yaml,
                epochs=self.config.epochs,
                batch=self.config.batch_size,
                imgsz=self.config.imgsz,
                device=self.config.device,
                workers=self.config.workers,
                optimizer=self.config.optimizer,
                lr0=self.config.lr0,
                lrf=self.config.lrf,
                project=self.config.project_name,
                name=self.config.experiment_name,
                resume=self.config.resume,
                amp=self.config.amp,
                verbose=True
            )

            # 获取最优权重路径
            best_weights = os.path.join(self.config.project_name, self.config.experiment_name, "weights", "best.pt")
            if not os.path.exists(best_weights):
                if hasattr(results, "save_dir"):
                    best_weights = os.path.join(str(results.save_dir), "weights", "best.pt")

            self.log_message.emit(f"🎉 训练顺利完成！最优权重已保存至: {best_weights}")
            self.training_done.emit(best_weights)

        except Exception as e:
            err_msg = str(e)
            self.log_message.emit(f"❌ 训练发生异常: {err_msg}")
            self.training_failed.emit(err_msg)
