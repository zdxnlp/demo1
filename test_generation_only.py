from __future__ import annotations

from config.settings import get_config
from memory.memory_manager import MemoryManager
from agents.text_generation_agent import TextGenerationAgent


def test_generation_only(user_input: str, rounds: int = 13) -> None:
    """仅运行生成智能体，结果写入记忆文件，不执行审核和选择。"""
    config = get_config()
    memory = MemoryManager()
    memory.init_session(user_input)
    print(f"[会话 {memory.session_id}] 已启动，记忆文件：{memory.file_path}")

    generator = TextGenerationAgent(memory, model=config.generator_model_1)

    for i in range(1, rounds + 1):
        print(f"\n--- 第 {i} 轮生成 ---")
        content = generator.generate(user_input)
        print(f"生成内容：{content}")

    print(f"\n[完成] 共生成 {rounds} 条，已保存至 {memory.file_path}")
    print("\n--- 记忆内容 ---")
    for entry in memory.entries:
        print(f"  第{entry.round_num}轮 [{entry.status}]：{entry.content}")


if __name__ == "__main__":
    import sys

    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "地铁迎面驶来，并发出与轨道碰撞的声音"
    test_generation_only(user_input)
