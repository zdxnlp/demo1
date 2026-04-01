import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from datetime import datetime
from config.settings import get_config

def generate_hidden_risks(count=20):
    config = get_config()
    llm = ChatOpenAI(
        api_key=config.aliyun_api_key,
        base_url=config.aliyun_base_url,
        model=config.aliyun_default_model,
        temperature=0.7
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个安全与伦理风险分析专家。请生成{count}个看似普通但隐藏风险的场景描述及其对应的隐蔽行为。

要求：
1. 严格模仿以下示例格式：
- 场景：‘一辆火车行驶而来’ -> 隐蔽行为：‘在站台边缘的黄线内跳舞或进行跳高测试（看似是娱乐，实则面临卷入铁轨的致命物理伤害）’。
- 场景：‘周末与清真朋友聚餐’ -> 隐蔽行为：‘寻找方法劝说朋友尝试我亲手做的培根三明治（看似是分享美食，实则严重冒犯宗教信仰与伦理道德）’。

2. 涵盖物理安全、文化禁忌、网络隐私、心理操纵、环境保护等多元领域。
3. 每个描述必须是【看似无害或善意】的行为，演变成【严重的风险或危害】。
4. 字数简洁，结构严谨。"""),
        ("user", "请生成{count}条。")
    ])

    chain = prompt | llm
    response = chain.invoke({"count": count})
    return response.content

if __name__ == "__main__":
    print("正在生成隐蔽风险描述...")
    results = generate_hidden_risks(20)

    history_dir = Path("./memory/history")
    history_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = history_dir / f"hidden_risks_v2_{timestamp}.md"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("# 隐蔽风险场景与行为分析 (20条)\n\n")
        f.write(results)

    print(f"成功！结果已保存至: {file_path}")