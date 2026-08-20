"""
YOLO Studio - 应用程序主启动入口 (Application Entrypoint)
"""
import sys
import os

# 将项目根目录加入模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from src.ui.main_window import MainWindow
from src.utils.logger import logger


def main():
    # 启用高 DPI 缩放支持
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Studio")
    app.setOrganizationName("Antigravity")

    logger.info("正在启动 YOLO Studio 图形化工作站...")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
