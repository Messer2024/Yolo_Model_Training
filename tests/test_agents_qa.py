"""
测试与修复智能体及防抖画布单元测试 (test_agents_qa.py)
"""
import os
import sys
import unittest
import numpy as np

from agents.app_test_agent import AppTestAndAuditAgent
from agents.bug_fix_agent import BugFixAgent
from src.ui.widgets.video_canvas import ImageDisplayCanvas
from PySide6.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM"] = "offscreen"
app = QApplication.instance() or QApplication(sys.argv)


class TestAgentsQA(unittest.TestCase):
    def test_app_test_agent_execution(self):
        agent = AppTestAndAuditAgent()
        report = agent.run({"scope": "all"})

        self.assertIn("health_score", report)
        self.assertIn("total_tests", report)
        self.assertGreater(report["total_tests"], 0)
        self.assertEqual(report["status"], "HEALTHY")
        print(f"[QA TEST REPORT] Score: {report['health_score']}/100, Passed: {report['passed_tests']}/{report['total_tests']}")

    def test_bug_fix_agent_analysis(self):
        fix_agent = BugFixAgent()

        # 模拟 KeyError: 0 缺陷分析
        context = {
            "bug_reports": [
                {
                    "error_message": "KeyError: 0 in trainer.loss_items",
                    "module": "trainer"
                },
                {
                    "error_message": "Video player window expands recursively on each frame",
                    "module": "inference_view"
                }
            ]
        }

        result = fix_agent.run(context)
        self.assertEqual(result["total_analyzed"], 2)
        self.assertTrue(result["fix_plans"][0]["fix_applied"])
        self.assertEqual(result["fix_plans"][0]["diagnosis"]["bug_type"], "DICT_KEY_ERROR")
        self.assertEqual(result["fix_plans"][1]["diagnosis"]["bug_type"], "LAYOUT_RESIZE_FEEDBACK_LOOP")
        print(f"[BUG FIX AGENT] Successfully analyzed {result['total_analyzed']} bugs.")

    def test_video_canvas_no_inflation(self):
        canvas = ImageDisplayCanvas()
        init_size_hint = canvas.sizeHint()

        # 传入大尺寸图像
        large_img = np.zeros((1920, 1080, 3), dtype=np.uint8)
        canvas.set_bgr_image(large_img)

        # 验证 sizeHint 不会被大图撑大
        self.assertEqual(canvas.sizeHint().width(), init_size_hint.width())
        self.assertEqual(canvas.sizeHint().height(), init_size_hint.height())
        print("[CANVAS TEST] ImageDisplayCanvas maintains stable sizeHint without layout recursion.")


if __name__ == "__main__":
    unittest.main()
