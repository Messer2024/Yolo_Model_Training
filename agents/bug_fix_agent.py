"""
自动化 Bug 修复与自愈智能体 (BugFixAgent)
"""
from typing import Dict, Any, List
import time

from agents.base_agent import BaseAgent
from agents.skills.bug_analyzer_skill import BugAnalyzerSkill
from src.utils.logger import logger


class BugFixAgent(BaseAgent):
    """
    负责接收 Bug 诊断报告或运行时异常信息
    分析缺陷根因、制定修复补丁并触发回归验证
    """

    def __init__(self):
        super().__init__(
            name="BugFixAgent",
            role="自动化缺陷修复与代码自愈工程师 (Auto-Fix & Self-Healing Engineer)"
        )
        self.register_skill(BugAnalyzerSkill())

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 Bug 分析与自愈策略制定
        :param context: 包含 'bug_reports' 列表或单个 'error_text'
        :return: 修复建议、补丁方案与自愈状态
        """
        start_time = time.perf_counter()
        analyzer = self.skills.get("bug_analyzer_skill")

        bug_reports = context.get("bug_reports", [])
        single_error = context.get("error_text")

        if single_error and not bug_reports:
            bug_reports = [{"error_message": single_error, "module": context.get("module", "general")}]

        logger.info(f"[{self.name}] 接收到 {len(bug_reports)} 个待修复缺陷项，开始分析...")

        fix_plans: List[Dict[str, Any]] = []
        for bug in bug_reports:
            err_msg = bug.get("error_message", "")
            mod = bug.get("module", "unknown")

            if analyzer:
                diag = analyzer.execute(error_text=err_msg, module_hint=mod)
            else:
                diag = {
                    "bug_type": "UNKNOWN",
                    "severity": "MEDIUM",
                    "root_cause": err_msg,
                    "recommended_action": "请检查相关模块代码并补充单元测试。"
                }

            fix_plans.append({
                "source_error": err_msg,
                "module": mod,
                "diagnosis": diag,
                "fix_applied": diag.get("auto_fixable", False),
                "resolution": diag.get("recommended_action")
            })

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        result = {
            "agent_name": self.name,
            "total_analyzed": len(bug_reports),
            "fix_plans": fix_plans,
            "execution_time_ms": elapsed_ms,
            "summary_text": f"已完成 {len(bug_reports)} 项缺陷分析并生成自愈方案！"
        }

        logger.info(f"[{self.name}] 分析与修复方案生成完毕")
        return result
