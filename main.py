from __future__ import annotations

import sys

from config.settings import get_config
from memory.memory_manager import MemoryManager
from agents.text_generation_agent import FORMAT_PATTERN, TextGenerationAgent
from agents.rationality_review_agent import RationalityReviewAgent
from agents.selection_agent import SelectionAgent


def validate_input(text: str) -> str:
    """阶段 2：用户输入合法性校验"""
    text = text.strip()
    if not text:
        raise ValueError("输入内容不能为空，请提供场景描述。")
    if len(text) < 5:
        raise ValueError("输入内容过短，请提供更详细的场景描述（至少5个字符）。")
    if len(text) > 2000:
        raise ValueError("输入内容过长，请将场景描述控制在2000字符以内。")
    return text


def run_pipeline(user_input: str) -> str:
    """
    主流程控制器：
    阶段 2 - 输入预处理
    阶段 3/4/5 - 循环生成、审核、候选池构建
    阶段 6 - 最优结果选择
    阶段 7 - 结果归档输出
    """
    config = get_config()

    # 阶段 2：校验并初始化会话
    user_input = validate_input(user_input)
    memory = MemoryManager()
    memory.init_session(user_input)
    print(f"[会话 {memory.session_id}] 已启动，记忆文件：{memory.file_path}")

    generators = [
        TextGenerationAgent(memory, model=config.generator_model_1),
        TextGenerationAgent(memory, model=config.generator_model_2),
        TextGenerationAgent(memory, model=config.generator_model_3),
    ]
    reviewer = RationalityReviewAgent(memory)

    candidates = []
    total_attempts = 0
    max_retries = config.max_retry_count
    target = config.target_candidate_count

    # 阶段 3/4/5：循环生成直到候选池满足目标数量
    while len(candidates) < target and total_attempts < max_retries:
        total_attempts += 1
        # 轮流使用三个生成智能体
        generator = generators[(total_attempts - 1) % len(generators)]
        print(f"\n--- 第 {total_attempts} 次尝试（已通过 {len(candidates)}/{target}，使用生成器 {(total_attempts-1) % len(generators) + 1}）---")

        # 阶段 3：生成
        content = generator.generate(user_input)
        print(f"生成内容：{content}")

        # 格式校验，不符合直接回流
        if not FORMAT_PATTERN.match(content):
            print(f"[格式不符] 回流重新生成")
            memory.update_last_entry_status("格式不符-回流", "格式校验未通过，自动回流")
            continue

        # 阶段 4：审核
        passed, review_detail = reviewer.review(content, user_input)
        print(f"审核结论：{'通过' if passed else '不通过'}")
        if not passed:
            print(f"审核意见：{review_detail[:200]}...")
            continue

        # 阶段 5：加入候选池
        candidates.append(content)
        print(f"[候选池] 已收录第 {len(candidates)} 条：{content}")

    # 超重试次数仍未凑齐
    if len(candidates) < target:
        msg = (
            f"已达最大重试次数（{max_retries}），"
            f"仅收集到 {len(candidates)} 条通过审核的候选内容，"
            "无法完成本次任务，请调整场景描述后重试。"
        )
        memory.finalize(
            candidates[0] if candidates else "（无结果）",
            f"未达目标，仅有 {len(candidates)} 条候选",
        )
        raise RuntimeError(msg)

    # 阶段 6：选择最优结果
    selector = SelectionAgent(memory)
    best, score_info = selector.select(candidates, user_input)
    print(f"\n[最优结果] {best}")
    print(f"[评分依据]\n{score_info}")

    # 阶段 7：归档
    memory.finalize(best, score_info)
    print(f"\n[记忆归档] 已写入 {memory.file_path}")

    return best


def main() -> None:
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("请输入场景描述（回车确认）：")
        user_input = input("> ").strip()

    try:
        result = run_pipeline(user_input)
        print(f"\n{'='*60}")
        print(f"最终输出结果：{result}")
        print(f"{'='*60}")
    except (ValueError, RuntimeError) as e:
        print(f"\n[错误] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
