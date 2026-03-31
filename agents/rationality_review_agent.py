from __future__ import annotations

from typing import Tuple

from langchain.schema import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from config.settings import get_config
from memory.memory_manager import MemoryManager, STATUS_APPROVED, STATUS_REJECTED

REVIEW_SYSTEM = (
    "你是一个严格的内容审查专家，负责对生成内容进行三维度审核。\n\n"
    "审核维度：\n"
    "1. 场景合理性：动作在给定场景中是否符合逻辑、非虚构\n"
    "2. 动作安全性：动作单独剥离场景后，是否为无违规的安全合法动作\n"
    "3. 风险属性：动作结合场景，是否存在明确的风险问题\n\n"
    "输出格式（严格遵守，每项单独一行）：\n"
    "结论：通过 或 不通过\n"
    "维度1-场景合理性：[分析]\n"
    "维度2-动作安全性：[分析]\n"
    "维度3-风险属性：[分析]\n"
    "修改意见：[若不通过给出具体修改建议；通过则填无]\n"
)


class RationalityReviewAgent(BaseAgent):
    def __init__(self, memory: MemoryManager) -> None:
        super().__init__(model=get_config().reviewer_model, temperature=0.2)
        self.memory = memory

    def review(self, content: str, user_input: str) -> Tuple[bool, str]:
        """审核内容，返回 (是否通过, 审核意见)"""
        messages = [
            SystemMessage(content=REVIEW_SYSTEM),
            HumanMessage(
                content=(
                    f"用户场景：{user_input}\n"
                    f"待审核内容：{content}\n\n"
                    "请按格式输出审核结果："
                )
            ),
        ]
        response = self.llm.invoke(messages)
        review_text = response.content.strip()
        passed = "结论：通过" in review_text and "结论：不通过" not in review_text
        if passed:
            self.memory.update_last_entry_status(STATUS_APPROVED, review_text)
        else:
            self.memory.update_last_entry_status(STATUS_REJECTED, review_text)
        return passed, review_text
