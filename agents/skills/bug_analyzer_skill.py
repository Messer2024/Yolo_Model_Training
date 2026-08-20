"""
Bug 模式分析与自愈方案推荐技能 (BugAnalyzerSkill)
"""
from typing import Dict, Any, List
import re
from agents.base_agent import BaseSkill


class BugAnalyzerSkill(BaseSkill):
    """
    负责解析应用运行过程中的异常日志、测试失败用例与用户报错
    对缺陷进行分类、严重等级评定并提供精准自愈方案
    """

    def __init__(self):
        super().__init__(
            name="bug_analyzer_skill",
            description="分析异常堆栈与缺陷特征，输出原因诊断、严重级别与修复补丁建议"
        )

    def execute(self, **kwargs) -> Dict[str, Any]:
        error_text = kwargs.get("error_text", "")
        module_hint = kwargs.get("module_hint", "unknown")

        diagnosis = {
            "bug_type": "UNKNOWN_ERROR",
            "severity": "MEDIUM",
            "module": module_hint,
            "root_cause": "",
            "recommended_action": "",
            "auto_fixable": False
        }

        # 模式 1: KeyError / 字典访问异常 (如之前的 KeyError: 0)
        if "KeyError" in error_text or "KeyError: 0" in error_text:
            diagnosis["bug_type"] = "DICT_KEY_ERROR"
            diagnosis["severity"] = "HIGH"
            diagnosis["root_cause"] = "尝试通过数字索引访问字典结构，通常由第三方库（如 Ultralytics 8.4+）返回格式由 list/tensor 变更为了 dict 引起。"
            diagnosis["recommended_action"] = "使用 isinstance(obj, dict) 进行类型分支，或使用 .get('key', default) 安全访问。"
            diagnosis["auto_fixable"] = True

        # 模式 2: 视频播放导致窗口无限放大
        elif "resize" in error_text.lower() or "expand" in error_text.lower() or "video_size" in error_text.lower():
            diagnosis["bug_type"] = "LAYOUT_RESIZE_FEEDBACK_LOOP"
            diagnosis["severity"] = "MEDIUM"
            diagnosis["root_cause"] = "QLabel.setPixmap 更新了控件 sizeHint，触发父级布局递归向外撑大。"
            diagnosis["recommended_action"] = "使用 ImageDisplayCanvas 并重写 paintEvent 保持几何约束，设置 setSizePolicy(Ignored, Ignored)。"
            diagnosis["auto_fixable"] = True

        # 模式 3: 新训练模型无预测结果
        elif "no_detection" in error_text.lower() or "empty_result" in error_text.lower() or "zero_boxes" in error_text.lower():
            diagnosis["bug_type"] = "INFERENCE_THRESHOLD_OR_PATH_MISMATCH"
            diagnosis["severity"] = "MEDIUM"
            diagnosis["root_cause"] = "1. 置信度阈值设置过高（新模型初期预测概率在 0.10 左右）；2. 模型路径未正确定位到 best.pt。"
            diagnosis["recommended_action"] = "调低置信度阈值至 0.10，并确保从 runs/detect/*/weights/best.pt 自动载入权重。"
            diagnosis["auto_fixable"] = True

        # 模式 4: CUDA 显存溢出 (Out of Memory)
        elif "CUDA out of memory" in error_text or "OutOfMemory" in error_text:
            diagnosis["bug_type"] = "CUDA_OUT_OF_MEMORY"
            diagnosis["severity"] = "HIGH"
            diagnosis["root_cause"] = "训练 Batch Size 或输入分辨率 Image Size 超出当前显卡显存上限。"
            diagnosis["recommended_action"] = "将 Batch Size 减半（如 16->8 或 8->4），或调小 imgsz 至 320/512，或开启 AMP 混合精度。"
            diagnosis["auto_fixable"] = True

        # 模式 5: 路径未找到 (FileNotFoundError)
        elif "FileNotFoundError" in error_text or "No such file" in error_text:
            diagnosis["bug_type"] = "FILE_NOT_FOUND"
            diagnosis["severity"] = "HIGH"
            diagnosis["root_cause"] = "指定的数据集、图片或权重文件在磁盘上不存在，或 Windows 反斜杠转义错误。"
            diagnosis["recommended_action"] = "使用 os.path.abspath 与正斜杠规范化路径，检查文件是否存在并提供友好错误提示。"
            diagnosis["auto_fixable"] = False

        else:
            diagnosis["root_cause"] = f"未分类异常: {error_text[:200]}"
            diagnosis["recommended_action"] = "检查具体堆栈日志，编写针对性单测进行复现与隔离。"

        return diagnosis
