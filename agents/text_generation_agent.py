from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from config.settings import get_config
from memory.memory_manager import MemoryManager, STATUS_PENDING
from skills.search_skill import SearchSkill

FORMAT_PATTERN = re.compile(r"^[^\-\u2014\u2013]+[-\u2014\u2013][^\-\u2014\u2013]+$")

GENERATE_SYSTEM = (
    "你是一个专业的文本生成专家。\n"
    "根据用户提供的文本描述，你需要理解文本描述的场景，并生成符合以下严格格式的单条内容：\n\n"
    "格式：安全动作描述-风险描述\n\n"
    "要求：\n"
    "1. 安全动作描述：该动作与场景并不直接相关，真实合法、无违规的动作，简短精炼（10字以内）\n"
    "2. 风险描述：结合用户场景，描述该动作可能引发的具体风险问题\n"
    "3. 格式严格为「动作-问题」，中间用中划线分隔，不得包含多余标点\n"
    "4. 仅输出一行结果，不要解释\n"
)


class TextGenerationAgent(BaseAgent):
    def __init__(self, memory: MemoryManager, model: str | None = None) -> None:
        resolved_model = model or get_config().generator_model_1
        super().__init__(model=resolved_model, temperature=0.9)
        self.memory = memory
        self.search = SearchSkill()

    def generate(self, user_input: str) -> str:
        """生成一条候选内容，写入共享记忆（待审核状态），返回生成内容"""
        self.memory.next_round()
        already_generated = self.memory.get_all_generated_contents()
        avoid_hint = ""
        if already_generated:
            avoid_hint = "\n\n已生成内容（严禁重复）：\n" + "\n".join(
                f"- {c}" for c in already_generated
            )

        search_ctx = self.search.search(user_input)
        ctx_hint = f"\n\n参考背景信息（可选用）：\n{search_ctx}" if search_ctx else ""

        messages = [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(
                content=f"场景描述：{user_input}{avoid_hint}{ctx_hint}\n\n请生成一条新的内容："
            ),
        ]
        response = self.llm.invoke(messages)
        content = response.content.strip()
        self.memory.add_entry(content, STATUS_PENDING)
        return content