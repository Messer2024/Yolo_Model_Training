"""
训练与系统日志终端控件 (Log Console Widget)
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QLabel, QCheckBox
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QTextCursor, QColor, QFont


class LogConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 顶部工具条
        toolbar = QHBoxLayout()
        title_label = QLabel("📋 实时运行日志 (Console Log)")
        title_label.setStyleSheet("font-weight: bold; color: #00d4bb;")
        toolbar.addWidget(title_label)

        toolbar.addStretch()

        self.auto_scroll_cb = QCheckBox("自动滚屏")
        self.auto_scroll_cb.setChecked(True)
        toolbar.addWidget(self.auto_scroll_cb)

        self.btn_clear = QPushButton("清空日志")
        self.btn_clear.clicked.connect(self.clear_logs)
        toolbar.addWidget(self.btn_clear)

        self.btn_copy = QPushButton("复制全部")
        self.btn_copy.clicked.connect(self.copy_all)
        toolbar.addWidget(self.btn_copy)

        layout.addLayout(toolbar)

        # 日志展示文本框
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setFont(QFont("Consolas", 10))
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #16161c;
                color: #dcdcdc;
                border: 1px solid #2d2d38;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.text_edit)

    @Slot(str)
    def append_log(self, text: str):
        """追加日志文本，根据内容着色"""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)

        color_hex = "#dcdcdc"
        if "❌" in text or "Error" in text or "error" in text or "失败" in text:
            color_hex = "#ff5555"
        elif "⚠️" in text or "Warning" in text or "warning" in text or "警告" in text:
            color_hex = "#ffb86c"
        elif "🎉" in text or "🚀" in text or "成功" in text or "完成" in text:
            color_hex = "#50fa7b"
        elif "Epoch" in text:
            color_hex = "#8be9fd"

        html_text = f"<span style='color:{color_hex};'>{text}</span><br>"
        cursor.insertHtml(html_text)

        if self.auto_scroll_cb.isChecked():
            self.text_edit.ensureCursorVisible()

    def clear_logs(self):
        self.text_edit.clear()

    def copy_all(self):
        self.text_edit.selectAll()
        self.text_edit.copy()
