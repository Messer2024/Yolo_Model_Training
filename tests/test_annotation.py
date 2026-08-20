"""
标注数据模型与几何计算单元测试 (test_annotation.py)
"""
import unittest
from src.core.annotation import BoundingBox, get_class_color, CLASS_PALETTE


class TestAnnotation(unittest.TestCase):
    def test_bounding_box_creation_and_clip(self):
        # 测试正常创建
        box = BoundingBox(class_id=0, x_center=0.5, y_center=0.5, width=0.2, height=0.4)
        self.assertEqual(box.class_id, 0)
        self.assertAlmostEqual(box.x_center, 0.5, places=5)
        self.assertAlmostEqual(box.y_center, 0.5, places=5)
        self.assertAlmostEqual(box.width, 0.2, places=5)
        self.assertAlmostEqual(box.height, 0.4, places=5)

        # 测试越界自动 clip
        out_box = BoundingBox(class_id=1, x_center=1.5, y_center=-0.2, width=0.4, height=0.4)
        self.assertLessEqual(out_box.x_center, 1.0)
        self.assertGreaterEqual(out_box.y_center, 0.0)

    def test_yolo_str_conversion(self):
        box = BoundingBox(class_id=2, x_center=0.123456, y_center=0.654321, width=0.111111, height=0.222222)
        yolo_str = box.to_yolo_str()
        self.assertTrue(yolo_str.startswith("2 0.123456 0.654321 0.111111 0.222222"))

        # 从字符串反向解析
        parsed = BoundingBox.from_yolo_str(yolo_str)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.class_id, 2)
        self.assertAlmostEqual(parsed.x_center, 0.123456, places=5)
        self.assertAlmostEqual(parsed.y_center, 0.654321, places=5)

    def test_xyxy_pixel_conversion(self):
        img_w, img_h = 1000, 500
        box = BoundingBox.from_xyxy(100, 50, 300, 250, img_w, img_h, class_id=0)
        self.assertAlmostEqual(box.x_center, 0.2, places=5)
        self.assertAlmostEqual(box.y_center, 0.3, places=5)
        self.assertAlmostEqual(box.width, 0.2, places=5)
        self.assertAlmostEqual(box.height, 0.4, places=5)

        # 反向转换
        x1, y1, x2, y2 = box.to_xyxy(img_w, img_h)
        self.assertEqual(x1, 100)
        self.assertEqual(y1, 50)
        self.assertEqual(x2, 300)
        self.assertEqual(y2, 250)

    def test_class_color_deterministic(self):
        c0 = get_class_color(0)
        c1 = get_class_color(1)
        self.assertIn(c0, CLASS_PALETTE)
        self.assertIn(c1, CLASS_PALETTE)
        self.assertNotEqual(c0, c1)


if __name__ == "__main__":
    unittest.main()
