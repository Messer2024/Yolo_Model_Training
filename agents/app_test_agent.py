"""
应用自动化巡检与质量测试智能体 (AppTestAndAuditAgent)
"""
from typing import Dict, Any, List
import time
import os

from agents.base_agent import BaseAgent
from agents.skills.gui_test_skill import GuiWorkflowTestSkill
from src.utils.logger import logger


class AppTestAndAuditAgent(BaseAgent):
    """
    负责对 YOLO Studio 软件进行自动化巡检测试、质量评估与 Bug 诊断
    生成结构化的测试与缺陷分析报告
    """

    def __init__(self):
        super().__init__(
            name="AppTestAndAuditAgent",
            role="自动化测试与质量巡检工程师 (QA & Stability Auditor)"
        )
        self.register_skill(GuiWorkflowTestSkill())

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行自动化巡检任务
        :param context: 巡检上下文，包含 scope ('all', 'annotation', 'dataset', 'inference', 'export') 等
        :return: 结构化测试报告与 Bug 列表
        """
        start_time = time.perf_counter()
        scope = context.get("scope", "all")
        sample_dir = context.get("sample_dir", os.path.abspath("samples/coco8"))

        logger.info(f"[{self.name}] 启动全流程自动化测试巡检，测试范围: {scope}")

        gui_skill = self.skills.get("gui_workflow_test_skill")
        test_results = gui_skill.execute(scope=scope, sample_dir=sample_dir) if gui_skill else {}

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        total = test_results.get("total_tests", 0)
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        errors = test_results.get("errors", [])

        # 计算系统健康度评分 (满分 100)
        if total > 0:
            pass_rate = passed / total
            health_score = int(pass_rate * 100)
        else:
            health_score = 0

        # 生成 Bug 诊断报告
        bug_reports: List[Dict[str, Any]] = []
        for err in errors:
            bug_reports.append({
                "module": err.get("module"),
                "severity": "HIGH" if "core" in err.get("module", "") else "MEDIUM",
                "error_message": err.get("error"),
                "detected_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })

        status_summary = "HEALTHY" if failed == 0 else "ISSUES_FOUND"

        report = {
            "agent_name": self.name,
            "status": status_summary,
            "health_score": health_score,
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed,
            "execution_time_ms": elapsed_ms,
            "bug_reports": bug_reports,
            "details": test_results.get("details", {}),
            "summary_text": f"测试完成！共执行 {total} 项测试，通过 {passed} 项，发现 {failed} 项异常，系统健康度评分: {health_score}/100"
        }

        logger.info(f"[{self.name}] 巡检完成: {report['summary_text']}")
        return report
