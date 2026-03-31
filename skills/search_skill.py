from __future__ import annotations

from tavily import TavilyClient

from config.settings import get_config


class SearchSkill:
    def __init__(self) -> None:
        config = get_config()
        self._client = TavilyClient(api_key=config.tavily_api_key)

    def search(self, query: str, max_results: int = 3) -> str:
        """执行 Tavily 搜索，返回拼接后的文本摘要；失败时返回空字符串"""
        try:
            results = self._client.search(query=query, max_results=max_results)
            snippets = [
                r.get("content", "") for r in results.get("results", [])
            ]
            return "\n".join(s for s in snippets if s)
        except Exception:
            return ""