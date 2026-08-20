"""
YOLO Studio - Agents & Skills Framework
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseSkill(ABC):
    """技能基类：封装单一职责的独立可执行工具"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """执行该技能"""
        pass


class BaseAgent(ABC):
    """智能体基类：具备多步决策、上下文感知与技能调度的实体"""

    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.skills: Dict[str, BaseSkill] = {}

    def register_skill(self, skill: BaseSkill) -> None:
        """注册技能到智能体"""
        self.skills[skill.name] = skill

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """运行智能体逻辑并返回结构化决策报告"""
        pass
