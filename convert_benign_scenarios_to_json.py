from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "memory" / "history" / "benign_scenarios_20260402_153419.md"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "benign_scenarios.json"


def parse_markdown(markdown_path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    current_label: str | None = None
    current_descriptions: list[str] = []

    for raw_line in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            if current_label is not None:
                entries.append(
                    {
                        "label": current_label,
                        "scenario_descriptions": current_descriptions,
                    }
                )
            current_label = line[3:].strip()
            current_descriptions = []
            continue

        if current_label is not None and line.startswith("- "):
            current_descriptions.append(line[2:].strip())

    if current_label is not None:
        entries.append(
            {
                "label": current_label,
                "scenario_descriptions": current_descriptions,
            }
        )

    if not entries:
        raise ValueError(f"未在 {markdown_path} 中解析到任何标签分组。")

    return entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将良性场景 Markdown 记录转换为仅包含标签和场景描述的 JSON 文件。"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=str(DEFAULT_INPUT_PATH),
        help="输入 Markdown 文件路径。",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default=str(DEFAULT_OUTPUT_PATH),
        help="输出 JSON 文件路径。",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    entries = parse_markdown(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"已生成 {len(entries)} 个标签分组: {output_path}")


if __name__ == "__main__":
    main()
