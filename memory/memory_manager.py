from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config.settings import get_config

# 内容状态常量
STATUS_PENDING = "待审核"
STATUS_APPROVED = "审核通过"
STATUS_REJECTED = "审核不通过"
STATUS_FINAL = "最终输出结果"


class MemoryEntry:
    def __init__(self, content: str, status: str, round_num: int, extra: str = ""):
        self.content = content
        self.status = status
        self.round_num = round_num
        self.extra = extra
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_markdown(self) -> str:
        lines = [
            f"### 第 {self.round_num} 轮 [{self.status}] {self.timestamp}",
            f"**内容：** {self.content}",
        ]
        if self.extra:
            lines.append(f"**备注：** {self.extra}")
        lines.append("")
        return "\n".join(lines)


class MemoryManager:
    def __init__(self, session_id: Optional[str] = None):
        config = get_config()
        self.storage_path = config.memory_storage_path
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.file_path: Path = self.storage_path / f"session_{self.session_id}.md"
        self.entries: List[MemoryEntry] = []
        self.user_input: str = ""
        self._round = 0

    def init_session(self, user_input: str) -> None:
        self.user_input = user_input
        self._round = 0
        header = (
            f"# 会话记忆 [{self.session_id}]\n"
            f"**创建时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"## 用户输入\n\n{user_input}\n\n"
            f"## 生成记录\n\n"
        )
        self.file_path.write_text(header, encoding="utf-8")

    def next_round(self) -> int:
        self._round += 1
        return self._round

    def add_entry(self, content: str, status: str, extra: str = "") -> MemoryEntry:
        entry = MemoryEntry(content, status, self._round, extra)
        self.entries.append(entry)
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(entry.to_markdown())
        return entry

    def update_last_entry_status(self, status: str, extra: str = "") -> None:
        if not self.entries:
            return
        last = self.entries[-1]
        last.status = status
        last.extra = extra
        self._rewrite_file()

    def get_approved_contents(self) -> List[str]:
        return [
            e.content
            for e in self.entries
            if e.status in (STATUS_APPROVED, STATUS_FINAL)
        ]

    def get_all_generated_contents(self) -> List[str]:
        return [e.content for e in self.entries]

    def finalize(self, final_content: str, score_info: str = "") -> None:
        for e in self.entries:
            if e.content == final_content:
                e.status = STATUS_FINAL
                e.extra = score_info
        footer = (
            f"## 最终输出结果\n\n"
            f"**{final_content}**\n\n"
            f"*评分依据：{score_info}*\n\n"
            f"---\n*会话归档时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        )
        self._rewrite_file()
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(footer)

    def _rewrite_file(self) -> None:
        header = (
            f"# 会话记忆 [{self.session_id}]\n"
            f"## 用户输入\n\n{self.user_input}\n\n"
            f"## 生成记录\n\n"
        )
        with self.file_path.open("w", encoding="utf-8") as f:
            f.write(header)
            for e in self.entries:
                f.write(e.to_markdown())
