"""
数据集管理器与划分单元测试 (test_dataset_manager.py)
"""
import os
import shutil
import tempfile
import unittest

from src.core.dataset_manager import DatasetManager
from src.core.annotation import BoundingBox


class TestDatasetManager(unittest.TestCase):
    def setUp(self):
        """创建一个包含测试图像与标注的临时目录"""
        self.temp_dir = tempfile.mkdtemp()

        # 创建 5 张测试图像文件 (使用简单占位文件)
        for i in range(5):
            img_path = os.path.join(self.temp_dir, f"test_{i:02d}.jpg")
            with open(img_path, "wb") as f:
                f.write(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb")

            # 为前 3 张写入标注
            if i < 3:
                txt_path = os.path.join(self.temp_dir, f"test_{i:02d}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("0 0.5 0.5 0.2 0.2\n")

        # 创建 classes.txt
        with open(os.path.join(self.temp_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("cat\ndog\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_manager_load(self):
        dm = DatasetManager(self.temp_dir)
        self.assertEqual(len(dm.image_files), 5)
        self.assertEqual(len(dm.class_names), 2)
        self.assertEqual(dm.class_names[0], "cat")
        self.assertEqual(dm.class_names[1], "dog")

        # 验证前 3 张有标注
        annotated_count = sum(1 for boxes in dm.labels_map.values() if len(boxes) > 0)
        self.assertEqual(annotated_count, 3)

    def test_dataset_split(self):
        dm = DatasetManager(self.temp_dir)
        res = dm.split_dataset(train_ratio=0.6, val_ratio=0.4, test_ratio=0.0)

        self.assertTrue(res["success"])
        self.assertEqual(res["train_count"], 3)
        self.assertEqual(res["val_count"], 2)
        self.assertTrue(os.path.exists(res["yaml_path"]))

    def test_dataset_audit(self):
        dm = DatasetManager(self.temp_dir)
        report = dm.audit_dataset()

        self.assertEqual(report["total_images"], 5)
        self.assertEqual(report["total_boxes"], 3)
        self.assertEqual(report["empty_images_count"], 2)
        self.assertTrue(0 <= report["health_score"] <= 100)


if __name__ == "__main__":
    unittest.main()
