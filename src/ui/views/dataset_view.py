"""
数据集管理与增强工作台视图 (Dataset View)
"""
from typing import Dict, Any
import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QGroupBox, QSlider, QSpinBox, QProgressBar,
    QCheckBox, QTextEdit, QMessageBox, QFileDialog, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt

from src.core.dataset_manager import DatasetManager
from agents.skills.augmentation_skill import AugmentationSkill
from src.utils.logger import logger


class DatasetView(QWidget):
    """数据集管理、健康度体检、划分与数据增强工作区"""

    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self.dm = dataset_manager
        self.aug_skill = AugmentationSkill()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 1. 顶部统计与健康度概览卡片
        summary_group = QGroupBox("📊 数据集总览与健康体检")
        summary_layout = QHBoxLayout(summary_group)

        self.lbl_stats = QLabel("项目未加载")
        self.lbl_stats.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffffff;")
        summary_layout.addWidget(self.lbl_stats)

        summary_layout.addStretch()

        self.lbl_health_badge = QLabel("健康度评分: --")
        self.lbl_health_badge.setStyleSheet("""
            background-color: #2b2b38;
            color: #00d4bb;
            font-size: 14px;
            font-weight: bold;
            padding: 8px 16px;
            border-radius: 6px;
            border: 1px solid #3d3d4e;
        """)
        summary_layout.addWidget(self.lbl_health_badge)

        self.btn_audit = QPushButton("🔍 深度体检 (Audit)")
        self.btn_audit.setObjectName("primaryButton")
        self.btn_audit.clicked.connect(self._on_audit_dataset)
        summary_layout.addWidget(self.btn_audit)

        main_layout.addWidget(summary_group)

        # 2. 中间区域：左侧类别分布表 + 右侧健康问题与改进建议
        mid_layout = QHBoxLayout()

        # 类别分布表
        class_group = QGroupBox("🏷️ 类别分布统计")
        class_layout = QVBoxLayout(class_group)
        self.table_classes = QTableWidget(0, 3)
        self.table_classes.setHorizontalHeaderLabels(["类别 ID", "类别名称", "标注目标数"])
        self.table_classes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        class_layout.addWidget(self.table_classes)
        mid_layout.addWidget(class_group, stretch=1)

        # 健康诊断与建议
        audit_group = QGroupBox("🩺 质量诊断报告与建议")
        audit_layout = QVBoxLayout(audit_group)
        self.txt_audit_report = QTextEdit()
        self.txt_audit_report.setReadOnly(True)
        self.txt_audit_report.setPlaceholderText("点击上方【深度体检】获取数据集质量诊断与优化建议...")
        audit_layout.addWidget(self.txt_audit_report)
        mid_layout.addWidget(audit_group, stretch=1)

        main_layout.addLayout(mid_layout)

        # 3. 底部配置：左侧数据集划分 + 右侧离线数据增强
        bottom_layout = QHBoxLayout()

        # 数据集切分
        split_group = QGroupBox("✂️ 自动划分训练集/验证集/测试集")
        split_layout = QGridLayout(split_group)

        split_layout.addWidget(QLabel("训练集 (Train %):"), 0, 0)
        self.spin_train = QSpinBox()
        self.spin_train.setRange(10, 95)
        self.spin_train.setValue(80)
        split_layout.addWidget(self.spin_train, 0, 1)

        split_layout.addWidget(QLabel("验证集 (Val %):"), 1, 0)
        self.spin_val = QSpinBox()
        self.spin_val.setRange(5, 50)
        self.spin_val.setValue(15)
        split_layout.addWidget(self.spin_val, 1, 1)

        split_layout.addWidget(QLabel("测试集 (Test %):"), 2, 0)
        self.spin_test = QSpinBox()
        self.spin_test.setRange(0, 30)
        self.spin_test.setValue(5)
        split_layout.addWidget(self.spin_test, 2, 1)

        self.btn_split = QPushButton("🚀 一键执行划分并生成 data.yaml")
        self.btn_split.setObjectName("primaryButton")
        self.btn_split.clicked.connect(self._on_split_dataset)
        split_layout.addWidget(self.btn_split, 3, 0, 1, 2)

        bottom_layout.addWidget(split_group, stretch=1)

        # 数据增强
        aug_group = QGroupBox("✨ 离线数据增强 (Data Augmentation)")
        aug_layout = QGridLayout(aug_group)

        self.cb_fliph = QCheckBox("水平翻转 (Horizontal Flip)")
        self.cb_fliph.setChecked(True)
        aug_layout.addWidget(self.cb_fliph, 0, 0)

        self.cb_flipv = QCheckBox("垂直翻转 (Vertical Flip)")
        aug_layout.addWidget(self.cb_flipv, 0, 1)

        self.cb_bright = QCheckBox("高光/高亮 (Brighten)")
        self.cb_bright.setChecked(True)
        aug_layout.addWidget(self.cb_bright, 1, 0)

        self.cb_dark = QCheckBox("低光/暗光 (Darken)")
        aug_layout.addWidget(self.cb_dark, 1, 1)

        self.cb_blur = QCheckBox("高斯模糊 (Blur)")
        aug_layout.addWidget(self.cb_blur, 2, 0)

        self.btn_run_aug = QPushButton("📦 生成增强样本扩充")
        self.btn_run_aug.clicked.connect(self._on_run_augmentation)
        aug_layout.addWidget(self.btn_run_aug, 3, 0, 1, 2)

        bottom_layout.addWidget(aug_group, stretch=1)

        main_layout.addLayout(bottom_layout)

    def refresh_view(self):
        """刷新数据集统计"""
        n_imgs = len(self.dm.image_files)
        total_boxes = sum(len(b) for b in self.dm.labels_map.values())
        self.lbl_stats.setText(f"📁 项目图像: {n_imgs} 张 | 标注目标总数: {total_boxes} 个 | 类别数: {len(self.dm.class_names)}")

        # 统计表格
        counts = {}
        for boxes in self.dm.labels_map.values():
            for b in boxes:
                counts[b.class_id] = counts.get(b.class_id, 0) + 1

        self.table_classes.setRowCount(len(self.dm.class_names))
        for i, name in enumerate(self.dm.class_names):
            cnt = counts.get(i, 0)
            self.table_classes.setItem(i, 0, QTableWidgetItem(str(i)))
            self.table_classes.setItem(i, 1, QTableWidgetItem(name))
            self.table_classes.setItem(i, 2, QTableWidgetItem(str(cnt)))

    def _on_audit_dataset(self):
        if not self.dm.image_files:
            QMessageBox.warning(self, "提示", "请先加载包含图像的项目！")
            return

        report = self.dm.audit_dataset()
        score = report.get("health_score", 0)
        self.lbl_health_badge.setText(f"健康度评分: {score} / 100")

        # 评分颜色
        if score >= 80:
            self.lbl_health_badge.setStyleSheet("background-color: #1b4332; color: #52b788; font-size: 14px; font-weight: bold; padding: 8px 16px; border-radius: 6px;")
        elif score >= 60:
            self.lbl_health_badge.setStyleSheet("background-color: #582f0e; color: #f7a072; font-size: 14px; font-weight: bold; padding: 8px 16px; border-radius: 6px;")
        else:
            self.lbl_health_badge.setStyleSheet("background-color: #4a0e17; color: #ff5a5f; font-size: 14px; font-weight: bold; padding: 8px 16px; border-radius: 6px;")

        # 生成诊断文本
        lines = [f"=== 数据集质量体检报告 (得分: {score}) ===", ""]
        lines.append(f"• 扫描图像总数: {report.get('total_images')}")
        lines.append(f"• 扫描目标总数: {report.get('total_boxes')}")
        lines.append(f"• 未标注空图数: {report.get('empty_images_count')}")
        lines.append("")

        issues = report.get("issues", [])
        if issues:
            lines.append("【⚠️ 发现的潜在问题】:")
            for iss in issues:
                lines.append(f"  - {iss}")
            lines.append("")

        suggestions = report.get("suggestions", [])
        if suggestions:
            lines.append("【💡 优化建议】:")
            for sug in suggestions:
                lines.append(f"  - {sug}")

        self.txt_audit_report.setText("\n".join(lines))

    def _on_split_dataset(self):
        if not self.dm.image_files:
            QMessageBox.warning(self, "提示", "请先加载包含图像的项目！")
            return

        tr = self.spin_train.value() / 100.0
        val = self.spin_val.value() / 100.0
        test = self.spin_test.value() / 100.0

        if abs((tr + val + test) - 1.0) > 0.01:
            QMessageBox.warning(self, "警告", f"训练、验证和测试比例之和必须为 100%！（当前合计: {(tr+val+test)*100:.0f}%）")
            return

        res = self.dm.split_dataset(train_ratio=tr, val_ratio=val, test_ratio=test)
        if res.get("success", False):
            QMessageBox.information(
                self,
                "划分成功",
                f"数据集划分完成！\n"
                f"训练集: {res.get('train_count')} 张\n"
                f"验证集: {res.get('val_count')} 张\n"
                f"测试集: {res.get('test_count')} 张\n\n"
                f"已生成 data.yaml: {res.get('yaml_path')}"
            )
        else:
            QMessageBox.critical(self, "失败", res.get("message", "划分失败"))

    def _on_run_augmentation(self):
        transforms = []
        if self.cb_fliph.isChecked(): transforms.append("flip_h")
        if self.cb_flipv.isChecked(): transforms.append("flip_v")
        if self.cb_bright.isChecked(): transforms.append("hsv_bright")
        if self.cb_dark.isChecked(): transforms.append("hsv_dark")
        if self.cb_blur.isChecked(): transforms.append("blur")

        if not transforms:
            QMessageBox.warning(self, "提示", "请至少勾选一种增强方式！")
            return

        if not self.dm.image_files:
            QMessageBox.warning(self, "提示", "项目无图片！")
            return

        # 执行增强
        import cv2
        count_created = 0
        for img_path in self.dm.image_files:
            boxes = [b.to_dict() for b in self.dm.labels_map.get(img_path, [])]
            if not boxes:
                continue

            results = self.aug_skill.execute(img_path, boxes, transforms)
            for aug_img, aug_boxes, suffix in results:
                base, ext = os.path.splitext(img_path)
                new_img_path = f"{base}_{suffix}{ext}"
                cv2.imwrite(new_img_path, aug_img)

                # 保存标注
                from src.core.annotation import BoundingBox
                b_objects = [BoundingBox.from_dict(d) for d in aug_boxes]
                self.dm.save_annotation(new_img_path, b_objects)
                count_created += 1

        # 重新扫描项目
        self.dm.load_project(self.dm.project_dir)
        self.refresh_view()
        QMessageBox.information(self, "增强完成", f"已成功生成 {count_created} 个增强样本并保存至项目！")
