"""
OutputAssemble 节点 — 基础版结构化输出（Phase 1）

Phase 1 仅计算 stance_distribution + 汇总 sources 列表。
opinion_clusters、knowledge_gaps、structured_summary 在 Phase 2/3 完善。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from loguru import logger

from ..state import QueryAgentOutput, QueryAgentState, SourceItem


# ---------------------------------------------------------------------------
# 覆盖度计算
# ---------------------------------------------------------------------------

_STANCE_THRESHOLDS: Dict[str, int] = {
    "support": 2,
    "oppose": 2,
    "official": 1,
    "neutral": 1,
}


def _compute_coverage_score(stance_counts: Dict[str, int]) -> float:
    """基于 STANCE_THRESHOLDS 计算覆盖度分数（0–1）。"""
    total_required = sum(_STANCE_THRESHOLDS.values())
    total_met = sum(
        min(stance_counts.get(s, 0), c)
        for s, c in _STANCE_THRESHOLDS.items()
    )
    return round(total_met / max(total_required, 1), 3)


# ---------------------------------------------------------------------------
# 节点函数
# ---------------------------------------------------------------------------

async def output_assemble_node(state: QueryAgentState) -> dict:
    """
    LangGraph 节点：组装最终结构化输出。

    Phase 1：
    - 使用 deduped_sources（stance_label 尚未填充，全部视为 "unclassified"）
    - 构造 stance_distribution、sources 列表
    - opinion_clusters / knowledge_gaps / structured_summary 留空，Phase 2/3 补全

    Phase 2 以后：
    - 使用 classified_sources（已有 stance_label + trust_score）
    - LLM 生成 opinion_clusters、knowledge_gaps、structured_summary
    """
    # Phase 2 开始后改用 classified_sources
    sources: List[SourceItem] = (
        state.get("classified_sources")
        or state.get("deduped_sources")
        or state.get("raw_sources")
        or []
    )

    query = state.get("original_query", "")
    total_raw = len(state.get("raw_sources", []))
    total_kept = len(sources)

    # 立场分布（Phase 1 时 stance_label 全为 None → 归入 "unclassified"）
    stance_counts = Counter(
        s.get("stance_label") or "unclassified" for s in sources
    )
    total = max(total_kept, 1)
    stance_distribution = {
        stance: round(count / total, 3)
        for stance, count in stance_counts.items()
    }

    coverage_score = _compute_coverage_score(dict(stance_counts))

    # 按 trust_score 降序排列（Phase 1 全为 0.0，顺序不变）
    sorted_sources = sorted(
        sources,
        key=lambda x: x.get("trust_score", 0.0),
        reverse=True,
    )

    output: QueryAgentOutput = {
        "original_query": query,
        "analysis_type": state.get("analysis_type", "general"),
        "search_iterations": state.get("search_iterations", 0),
        "total_sources_found": total_raw,
        "total_sources_kept": total_kept,
        "stance_distribution": stance_distribution,
        "opinion_clusters": [],          # Phase 2 填充
        "sources": sorted_sources,
        "knowledge_gaps": [],            # Phase 3 填充
        "coverage_score": coverage_score,
        "structured_summary": "",        # Phase 3 填充
        "trace_log": state.get("trace_log", []),
    }

    trace = (
        f"[OutputAssemble] 来源={total_kept}/{total_raw}, "
        f"立场分布={stance_distribution}, "
        f"覆盖度={coverage_score:.2f}"
    )
    logger.info(trace)

    return {
        "query_agent_output": output,
        "trace_log": [trace],
    }
