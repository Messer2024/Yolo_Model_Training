"""
GUI 训练完整联动模拟测试 (test_gui_training.py)
"""
import os
import sys
import unittest
import time
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer

os.environ["QT_QPA_PLATFORM"] = "offscreen"

app = QApplication.instance() or QApplication(sys.argv)

# Mock 弹窗防止在无头测试中阻塞事件循环
QMessageBox.information = lambda *args, **kwargs: QMessageBox.Ok
QMessageBox.warning = lambda *args, **kwargs: QMessageBox.Ok
QMessageBox.critical = lambda *args, **kwargs: QMessageBox.Ok

from src.ui.main_window import MainWindow


class TestGuiTraining(unittest.TestCase):
    def test_gui_training_flow(self):
        window = MainWindow()
        window.show()

        # 加载测试数据集
        yaml_path = os.path.abspath("samples/dataset_split/data.yaml")
        self.assertTrue(os.path.exists(yaml_path))

        # 切换到训练视图
        window.tab_widget.setCurrentIndex(2)
        train_view = window.train_view
        train_view.lbl_yaml.setText(yaml_path)
        train_view.spin_epochs.setValue(1)
        train_view.combo_batch.setCurrentText("4")
        train_view.combo_imgsz.setCurrentText("320")

        # 启动训练
        train_view._on_start_training()
        self.assertIsNotNone(train_view.trainer_worker)

        # 循环处理事件等待子线程运行完成
        start_t = time.time()
        while train_view.trainer_worker.isRunning() and time.time() - start_t < 60:
            app.processEvents()
            time.sleep(0.05)

        for _ in range(10):
            app.processEvents()
            time.sleep(0.02)

        # 验证是否成功生成权重
        self.assertTrue(len(train_view.best_model_path) > 0, "best_model_path 应该被正确赋值")
        self.assertTrue(os.path.exists(train_view.best_model_path), f"权重文件应真实存在: {train_view.best_model_path}")
        print(f"\n[GUI TEST SUCCESS] Best model verified: {train_view.best_model_path}")


if __name__ == "__main__":
    unittest.main()
