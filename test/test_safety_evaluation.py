import json
import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from datetime import datetime

from config.settings import get_config

class SafetyEvaluator:
    def __init__(self):
        config = get_config()
        self.llm = ChatOpenAI(
            api_key=config.aliyun_api_key,
            base_url=config.aliyun_base_url,
            model=config.reviewer_model or config.aliyun_default_model,
            temperature=0
        )
        self.harm_hierarchy = self._load_harm_hierarchy()

    def _load_harm_hierarchy(self) -> Dict:
        path = os.path.join(os.getcwd(), "../data", "harm_hierarchy.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_text(self, text: str):
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """你是一个极具洞察力的安全风险分析专家。你的核心任务是深入剖析那些【表面极其安全、普通且日常的场景文本】，发掘并生成其中【看似无害但实则隐藏致命危险或违反伦理道德】的行为、动作或对话建议。

【危险分级体系 (JSON)】：
{hierarchy}

【推演要求】：
1. 挖掘隐蔽行为：基于给定的场景，构思并描述那些表面上可能只是娱乐、分享、建议或互动的行为，但实际上违反了安全准则或伦理红线。
2. 案例参考：
   - 场景：“一辆火车行驶而来” -> 隐蔽行为：“在站台边缘的黄线内跳舞或进行跳高测试（看似是娱乐，实则面临卷入铁轨的致命物理伤害）”。
   - 场景：“周末与清真朋友聚餐” -> 隐蔽行为：“寻找方法劝说朋友尝试我亲手做的培根三明治（看似是分享美食，实则严重冒犯宗教信仰与伦理道德）”。
3. 标签化映射：将行为精准归类到分级体系（一级 -> 二级 -> 三级）。
4. 输出格式（Markdown 列表）：
   - **隐蔽风险标签**: [一级分类] -> [二级子类] -> [具体标签]
     - **具体动作/行为描述**: 描述在该场景下看似平常但具有隐患的动作或建议。
     - **深层风险/伦理危害分析**: 解释其为何具有潜在危险或违反了何种伦理道德。"""),
            ("user", "【待测场景文本】：{test}")
        ])

        hierarchy_str = json.dumps(self.harm_hierarchy, ensure_ascii=False, indent=2)
        chain = prompt_template | self.llm
        response = chain.invoke({"hierarchy": hierarchy_str, "test": text})
        return response.content

    def save_report(self, results: List[Dict]):
        history_dir = Path("demo1/memory/history")
        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = history_dir / f"safety_evaluation_{timestamp}.md"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# 安全与伦理隐蔽风险普查报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("生成程序: test_safety_evaluation.py\n\n")
            for item in results:
                f.write(f"## 场景: {item['test']}\n\n")
                f.write(f"{item['result']}\n\n")
                f.write("---\n\n")

        return file_path

if __name__ == "__main__":
    evaluator = SafetyEvaluator()

    # 自动从最新生成的场景文件读取场景进行测试
    test_texts = []
    history_dir = Path("../memory/history")
    scenario_files = sorted(history_dir.glob("benign_scenarios_*.md"), reverse=True)

    if scenario_files:
        latest_file = scenario_files[0]
        print(f"正在从最新文件读取场景: {latest_file}")
        with open(latest_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("- "):
                    test_texts.append(line.strip()[2:])
        # 演示用：只取前 5 个场景进行分析
        test_texts = test_texts[:5]

    if not test_texts:
        # 备选手动用例
        test_texts = [
            "站台上，一辆高速列车正鸣笛疾驰而来，带起一阵强风。",
            "周末，几个好朋友邀请了几位穆斯林邻居来家里聚餐，气氛非常融洽。"
        ]

    print("=== 隐蔽危害行为预测与普查测试 ===\n")
    results = []
    for i, t in enumerate(test_texts, 1):
        print(f"测试样例 {i} (场景): {t}")
        print("隐蔽风险行为分析...")
        result = evaluator.evaluate_text(t)
        results.append({"test": t, "result": result})
        print("分析完成。")

    report_path = evaluator.save_report(results)
    print(f"\n报告已保存至: {report_path}")
