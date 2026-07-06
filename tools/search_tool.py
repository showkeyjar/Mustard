"""Real search tool using DuckDuckGo or fallback to built-in web search.

Tries duckduckgo-search first (pip install duckduckgo-search), then falls back
to a lightweight urllib-based search if that package is unavailable.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from carm.intent import IntentCategory
from carm.schemas import ToolResult

_DDGS_TIMEOUT_S = 5


class SearchTool:
    name = "search"
    capability_tags = [IntentCategory.SEARCH]

    def __init__(self, ddgs_timeout: int = _DDGS_TIMEOUT_S) -> None:
        self._ddgs = None
        self._ddgs_timeout = ddgs_timeout
        self._init_ddgs()

    def _init_ddgs(self) -> None:
        """Try to import duckduckgo-search for real search capability."""
        try:
            from ddgs import DDGS  # type: ignore[import-untyped]

            self._ddgs = DDGS()
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore[import-untyped]

                self._ddgs = DDGS()
            except ImportError:
                self._ddgs = None

    def execute(self, query: str, arguments: dict) -> ToolResult:
        top_k = arguments.get("top_k", 5)

        # Strategy 1: DuckDuckGo search (best quality, with timeout)
        if self._ddgs is not None:
            result = self._search_ddgs(query, top_k)
            if result is not None:
                return result

        # Strategy 2: Wikipedia API for factual queries
        result = self._search_wikipedia(query, top_k)
        if result is not None:
            return result

        # Strategy 3: Fallback with structured guidance
        return self._fallback_response(query, top_k)

    def _search_ddgs(self, query: str, top_k: int) -> ToolResult | None:
        """Search using duckduckgo-search library with timeout protection."""
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future: Future = pool.submit(self._ddgs.text, query, max_results=top_k)
                results = list(future.result(timeout=self._ddgs_timeout))
            if not results:
                return None

            snippets: list[str] = []
            for i, r in enumerate(results[:top_k], 1):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                snippet = f"{i}. {title}"
                if body:
                    snippet += f" — {body[:200]}"
                if href:
                    snippet += f" [来源: {href}]"
                snippets.append(snippet)

            summary = f"检索到 {len(snippets)} 条结果:\n" + "\n".join(snippets)
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result=summary,
                confidence=0.82,
                source="tool/search:duckduckgo",
            )
        except Exception:
            return None

    def _search_wikipedia(self, query: str, top_k: int) -> ToolResult | None:
        """Search Wikipedia API for factual queries (no extra dependencies)."""
        try:
            # Search for article titles
            search_url = "https://zh.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": str(top_k),
                    "format": "json",
                    "utf8": "1",
                }
            )
            req = urllib.request.Request(
                search_url, headers={"User-Agent": "MustardCARM/1.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                # Try English Wikipedia
                search_url_en = (
                    "https://en.wikipedia.org/w/api.php?"
                    + urllib.parse.urlencode(
                        {
                            "action": "query",
                            "list": "search",
                            "srsearch": query,
                            "srlimit": str(top_k),
                            "format": "json",
                            "utf8": "1",
                        }
                    )
                )
                req = urllib.request.Request(
                    search_url_en, headers={"User-Agent": "MustardCARM/1.0"}
                )
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                search_results = data.get("query", {}).get("search", [])

            if not search_results:
                return None

            snippets: list[str] = []
            for i, item in enumerate(search_results[:top_k], 1):
                title = item.get("title", "")
                # Strip HTML tags from snippet
                snippet_text = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                snippets.append(f"{i}. {title} — {snippet_text[:200]}")

            summary = f"检索到 {len(snippets)} 条百科结果:\n" + "\n".join(snippets)
            return ToolResult(
                ok=True,
                tool_name=self.name,
                result=summary,
                confidence=0.75,
                source="tool/search:wikipedia",
            )
        except Exception:
            return None

    def _fallback_response(self, query: str, top_k: int) -> ToolResult:
        """Fallback when no search backend is available — still provides structured guidance."""
        # Extract key concepts from the query to give a more useful fallback
        keywords = self._extract_keywords(query)
        guidance_parts: list[str] = []

        if any(token in query for token in ("比较", "对比", "区别", "优缺点", "vs")):
            guidance_parts.append(
                "这是一个比较类问题，建议从成本、性能、维护复杂度和生态角度分析。"
            )
        if any(token in query for token in ("计算", "预算", "多少", "价格")):
            guidance_parts.append("涉及数值计算，建议先明确计费口径再做精确比较。")
        if any(token in query for token in ("选型", "方案", "推荐")):
            guidance_parts.append("技术选型问题，建议收集具体场景约束后缩小范围。")

        if not guidance_parts:
            guidance_parts.append(
                f"无法检索外部信息。关键词: {', '.join(keywords[:5])}"
            )

        summary = (
            f"检索受限: 未能连接搜索服务（可安装 duckduckgo-search 包获得真实检索能力）。\n"
            + f"分析建议: {' '.join(guidance_parts)}"
        )

        return ToolResult(
            ok=True,
            tool_name=self.name,
            result=summary,
            confidence=0.35,
            source="tool/search:fallback",
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from Chinese/English text."""
        # Chinese segments
        chinese = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        # English words
        english = re.findall(r"[a-zA-Z]{2,}", text)
        return chinese + english
