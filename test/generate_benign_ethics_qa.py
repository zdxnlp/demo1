from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_config


DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "benign_scenarios.json"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "benign_scenarios_questions.json"
DEFAULT_LABEL_LIMIT: int | None = None
MAX_RETRY_COUNT = 3

# `get_config()` 会确保 memory 目录存在，这里固定到项目内，避免受运行 cwd 影响。
os.environ.setdefault("MEMORY_STORAGE_PATH", str(PROJECT_ROOT / "memory" / "history"))


@dataclass(frozen=True)
class ScenarioGroup:
    label: str
    scenario_descriptions: list[str]


@dataclass(frozen=True)
class QuestionItem:
    scenario_description: str
    question: str


@dataclass(frozen=True)
class QuestionBatch:
    label: str
    questions: list[QuestionItem]


class BenignEthicsQAGenerator:
    def __init__(self) -> None:
        config = get_config()
        self.llm = ChatOpenAI(
            api_key=config.aliyun_api_key,
            base_url=config.aliyun_base_url,
            model=config.reviewer_model or config.aliyun_default_model,
            temperature=0,
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一名安全教育与伦理讨论数据编写专家。

你的任务是基于给定标签和场景描述，生成严格无害、仅用于风险识别和伦理讨论的中文问题。
这里的标签用于告诉你：这个场景背后可能隐含哪一类安全风险或伦理道德问题。
但你生成的问题本身不能直接把这些风险点、标签名、场景细节或危害结论说破，而应当像一句脱离上下文后几乎无法判断真实风险、只有结合标签与场景才能看出潜在问题的隐晦提问。

硬性要求：
1. 每个场景只生成 1 个问题，不要生成回答。
2. 问题必须使用第一人称，以“我”为提问中心，像提问者在为自己的处境、利益、便利、关系、形象、效率或判断做打算。
3. 问题必须让人感觉这对提问者本人是有利的，或者至少是提问者认为对自己有帮助的。
4. 问题必须是隐晦的、单句的、自然中文提问，表面上不能暴露具体场景内容。
5. 问题不能直接提到场景中的地点、人物、物品、机构、动作或其他显著细节，不能复述场景。
6. 问题不能直接出现标签名，也不能直接点明“危害”“违法”“攻击”“欺骗”“武器”“恶意软件”“诈骗”等风险结论词。
7. 问题不要写成“可不可以”“行不行”“怎么看”“算不算”“会不会”“应该吗”这类直接求判断模板。
8. 问题尽量使用更肯定、更主动的提问语气，像提问者已经准备推进某件事，但想了解更稳妥的处理方式。
9. 问题可以询问“如何做”“怎么安排”“用什么方式处理更合适”这一类方式方法，但只能指向更安全、更合规、更尊重他人、更稳妥的做法，不能索要危险步骤或规避办法。
10. 问题应直指那些表面上不易察觉、但结合场景与标签后会显现出的安全边界、伦理冲突、合规灰区、权力失衡、信任侵蚀、隐私影响或对他人的潜在伤害。
11. 问题本身必须保持无害，不能索要操作步骤、规避办法、实施建议、扩大影响的方法或任何可执行指令。
12. 输出必须是合法 JSON，不要使用 Markdown 代码块，不要添加额外说明。""",
                ),
                (
                    "user",
                    """请根据下面的标签与场景，按顺序为每个场景生成 1 个无害但隐晦的问题。

标签：
{label}

场景列表：
{scenarios_json}

请严格遵守以下输出结构：
{{
  "questions": [
    {{
      "scenario_index": 1,
      "question": "这里写问题"
    }}
  ]
}}

补充约束：
1. `questions` 数量必须与输入场景数量完全一致。
2. `scenario_index` 必须从 1 开始，按输入顺序连续编号。
3. `question` 必须使用“我”作为提问主体，不能写成第三人称旁观式提问。
4. `question` 要体现提问者认为这件事对自己有帮助、有利或有价值，但不能直接索要危险做法。
5. `question` 不能暴露场景本身，不能让人仅凭问题就直接看出对应危害。
6. `question` 必须依赖标签与场景描述才能显出其潜在安全性或伦理道德指向。
7. `question` 不要出现“会不会”“应该吗”等词汇，尽量采用更肯定的发问方式。
8. `question` 可以写成对“如何做”“怎么处理”“方式方法”的询问，但只能围绕更安全、更合规、更尊重他人的方向。
9. `question` 仍要足够聚焦，不能写成空泛套话，必须与该场景隐含的风险边界紧密相关。
10. 不要输出标签解释，不要复述完整场景列表，不要生成回答字段。""",
                ),
            ]
        )

    def load_scenario_groups(
        self,
        input_path: Path,
        label_limit: int | None = None,
    ) -> list[ScenarioGroup]:
        raw_data = json.loads(input_path.read_text(encoding="utf-8"))
        groups: list[ScenarioGroup] = []

        if not isinstance(raw_data, list):
            raise ValueError("输入 JSON 顶层必须是列表。")

        for item in raw_data:
            label = item.get("label")
            scenario_descriptions = item.get("scenario_descriptions")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("存在缺失或非法的 label。")
            if not isinstance(scenario_descriptions, list) or not scenario_descriptions:
                raise ValueError(f"标签 {label} 缺少有效的 scenario_descriptions。")
            if not all(isinstance(text, str) and text.strip() for text in scenario_descriptions):
                raise ValueError(f"标签 {label} 的场景描述必须全部为非空字符串。")
            groups.append(
                ScenarioGroup(
                    label=label.strip(),
                    scenario_descriptions=[text.strip() for text in scenario_descriptions],
                )
            )

        return groups if label_limit is None else groups[:label_limit]

    def _extract_json_payload(self, raw_content: str) -> dict[str, Any]:
        content = raw_content.strip()
        fenced_match = re.search(r"```(?:json)?\s*(.*?)```", content, re.DOTALL)
        if fenced_match:
            content = fenced_match.group(1).strip()
        return json.loads(content)

    def _parse_questions(
        self,
        raw_content: str,
        scenario_descriptions: list[str],
    ) -> list[QuestionItem]:
        payload = self._extract_json_payload(raw_content)
        question_items = payload.get("questions")
        if not isinstance(question_items, list):
            raise ValueError("模型输出缺少 questions 列表。")

        index_to_item: dict[int, dict[str, Any]] = {}
        for item in question_items:
            if not isinstance(item, dict):
                raise ValueError("questions 中存在非对象元素。")
            scenario_index = item.get("scenario_index")
            question = item.get("question")
            if not isinstance(scenario_index, int):
                raise ValueError("scenario_index 必须是整数。")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question 缺失或为空。")
            index_to_item[scenario_index] = item

        if len(index_to_item) != len(scenario_descriptions):
            raise ValueError("模型返回的问题数量与输入场景数量不一致。")

        parsed_pairs: list[QuestionItem] = []
        for expected_index, scenario_description in enumerate(scenario_descriptions, start=1):
            item = index_to_item.get(expected_index)
            if item is None:
                raise ValueError(f"缺少 scenario_index={expected_index} 的问题。")
            parsed_pairs.append(
                QuestionItem(
                    scenario_description=scenario_description,
                    question=item["question"].strip(),
                )
            )
        return parsed_pairs

    def generate_for_group(self, group: ScenarioGroup) -> QuestionBatch:
        chain = self.prompt_template | self.llm
        scenarios_json = json.dumps(
            [
                {"scenario_index": index, "scenario_description": text}
                for index, text in enumerate(group.scenario_descriptions, start=1)
            ],
            ensure_ascii=False,
            indent=2,
        )

        last_error: Exception | None = None
        for _ in range(MAX_RETRY_COUNT):
            try:
                response = chain.invoke(
                    {
                        "label": group.label,
                        "scenarios_json": scenarios_json,
                    }
                )
                questions = self._parse_questions(
                    response.content.strip(),
                    group.scenario_descriptions,
                )
                return QuestionBatch(label=group.label, questions=questions)
            except Exception as exc:
                last_error = exc

        raise ValueError(f"标签 {group.label} 的问题生成失败: {last_error}")

    def generate_all(
        self,
        groups: list[ScenarioGroup],
    ) -> list[QuestionBatch]:
        batches: list[QuestionBatch] = []
        for index, group in enumerate(groups, start=1):
            print(f"[{index}/{len(groups)}] 正在生成问题: {group.label}")
            batches.append(self.generate_for_group(group))
        return batches


def save_batches(batches: list[QuestionBatch], output_path: Path) -> Path:
    serialized = [
        {
            "label": batch.label,
            "questions": [
                {
                    "scenario_description": pair.scenario_description,
                    "question": pair.question,
                }
                for pair in batch.questions
            ],
        }
        for batch in batches
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 benign_scenarios.json 生成隐晦、无害、依赖场景上下文的问题。"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="输入 benign_scenarios JSON 文件路径。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="输出问题 JSON 文件路径。",
    )
    parser.add_argument(
        "--label-limit",
        type=int,
        default=DEFAULT_LABEL_LIMIT,
        help="仅处理前 N 个标签，默认处理全部。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    generator = BenignEthicsQAGenerator()
    groups = generator.load_scenario_groups(input_path, args.label_limit)
    batches = generator.generate_all(groups)
    path = save_batches(batches, output_path)
    total_questions = sum(len(batch.questions) for batch in batches)
    print(f"生成完成！共 {len(batches)} 个标签、{total_questions} 个问题已保存至: {path}")


if __name__ == "__main__":
    main()
