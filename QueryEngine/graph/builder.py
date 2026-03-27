"""
Query Agent LangGraph 子图构建

build_query_agent_graph() 返回编译好的 CompiledGraph，
可直接由 DeepSearchAgent.research_structured() 调用。

Phase 1 图结构（线性 + 条件边）：
  START
    → query_planner
    → unified_search
    → dedup_filter
    → output_assemble
    → END

Phase 2 扩展：在 dedup_filter → output_assemble 之间插入
  trust_scorer → stance_classify → coverage_check
并在 coverage_check 后增加条件边指向 gap_filler → unified_search。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    dedup_filter_node,
    output_assemble_node,
    query_planner_node,
    unified_search_node,
)
from .state import QueryAgentState


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def build_query_agent_graph():
    """
    构建并编译 Query Agent LangGraph 子图。

    Returns:
        CompiledGraph — 可调用 .ainvoke(state) 或 .invoke(state)
    """
    graph = StateGraph(QueryAgentState)

    # 注册节点
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("unified_search", unified_search_node)
    graph.add_node("dedup_filter", dedup_filter_node)
    graph.add_node("output_assemble", output_assemble_node)

    # Phase 1 线性流程
    graph.add_edge(START, "query_planner")
    graph.add_edge("query_planner", "unified_search")
    graph.add_edge("unified_search", "dedup_filter")
    graph.add_edge("dedup_filter", "output_assemble")
    graph.add_edge("output_assemble", END)

    # Phase 2 扩展点：
    # graph.add_node("trust_scorer", trust_scorer_node)
    # graph.add_node("stance_classify", stance_classify_node)
    # graph.add_node("coverage_check", coverage_check_node)
    # graph.add_node("gap_filler", gap_filler_node)
    #
    # graph.add_edge("dedup_filter", "trust_scorer")
    # graph.add_edge("trust_scorer", "stance_classify")
    # graph.add_edge("stance_classify", "coverage_check")
    # graph.add_conditional_edges(
    #     "coverage_check",
    #     coverage_router,
    #     {"sufficient": "output_assemble", "need_more": "gap_filler", "max_reached": "output_assemble"},
    # )
    # graph.add_edge("gap_filler", "unified_search")

    return graph.compile()
