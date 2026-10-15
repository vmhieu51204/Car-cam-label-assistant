"""Local vertical slice within the template's LangGraph layout."""

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.autolabel_state import AutoLabelState
from src.agents.nodes.autolabel import assess_node, detect_node


def build_autolabel_graph() -> CompiledStateGraph:
    """Build deterministic detection -> HCL orchestration without a remote LLM."""
    graph = StateGraph(AutoLabelState)
    graph.add_node("detect", detect_node)
    graph.add_node("assess", assess_node)
    graph.set_entry_point("detect")
    graph.add_edge("detect", "assess")
    graph.add_edge("assess", END)
    return graph.compile()


autolabel_graph = build_autolabel_graph()
