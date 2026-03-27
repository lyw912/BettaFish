"""
UnifiedSearchDispatcher — 统一搜索调度器（Phase 1：仅 Tavily）

Phase 1 只接 Tavily，将子查询转换为 SourceItem 列表。
Phase 2 扩展：接入 BochaMultimodalSearch 和 InsightDB。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from loguru import logger

from .search import TavilyNewsAgency, TavilyResponse
from ..utils.config import settings

if TYPE_CHECKING:
    from ..graph.state import SourceItem, SubQueryItem


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """从 URL 中提取主域名（去 www. 前缀）。"""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace("www.", "").lower()
    except Exception:
        return ""


def _tavily_to_source_items(response: TavilyResponse, sq: Dict) -> List[Dict]:
    """将 Tavily 响应转为统一 SourceItem 格式的字典列表。"""
    items = []
    if not response or not response.results:
        return items

    for r in response.results:
        items.append(dict(
            source_id=str(uuid.uuid4()),
            url=r.url or "",
            title=r.title or "",
            source_api="tavily",
            platform=_extract_domain(r.url or ""),
            snippet=(r.content or "")[:500],
            full_content=r.raw_content,
            published_at=r.published_date,
            trust_score=0.0,
            stance_label=None,
            stance_confidence=0.0,
            sub_query_ref=sq["query"],
            rrf_score=r.score,
        ))
    return items


# ---------------------------------------------------------------------------
# 调度器
# ---------------------------------------------------------------------------

class UnifiedSearchDispatcher:
    """
    统一搜索调度器。

    Phase 1：只使用 Tavily。
    Phase 2 扩展点：在 _dispatch_one() 中增加 bocha / insight_db 分支。
    """

    def __init__(self):
        self._tavily: Optional[TavilyNewsAgency] = None

    @property
    def tavily(self) -> TavilyNewsAgency:
        if self._tavily is None:
            self._tavily = TavilyNewsAgency(api_key=settings.TAVILY_API_KEY)
        return self._tavily

    async def dispatch(self, sub_queries: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        并行调度所有子查询。

        Returns:
            (sources, errors)
        """
        tasks = [self._dispatch_one(sq) for sq in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        sources: List[Dict] = []
        errors: List[str] = []
        for sq, result in zip(sub_queries, results):
            if isinstance(result, Exception):
                msg = f"搜索失败 [{sq['query']}]: {result}"
                logger.warning(msg)
                errors.append(msg)
            elif isinstance(result, list):
                sources.extend(result)

        return sources, errors

    async def _dispatch_one(self, sq: Dict) -> List[Dict]:
        """
        对单个子查询选择后端并执行搜索。

        Phase 2 扩展点（在此添加 bocha / insight_db 分支）。
        """
        # Phase 2 扩展点:
        # target = sq.get("target_source", "any")
        # if target == "bocha":
        #     return await self._search_bocha(sq)
        # if target == "insight_db":
        #     return await self._search_insight_db(sq)

        return await self._search_tavily(sq)

    async def _search_tavily(self, sq: Dict) -> List[Dict]:
        """异步包装 Tavily 同步调用。"""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._call_tavily_sync, sq)
        return _tavily_to_source_items(response, sq)

    def _call_tavily_sync(self, sq: Dict) -> TavilyResponse:
        """根据子查询的优先级选择 Tavily 工具。"""
        query = sq["query"]
        priority = sq.get("priority", 3)

        if priority <= 2:
            return self.tavily.deep_search_news(query)
        else:
            return self.tavily.basic_search_news(query, max_results=7)
