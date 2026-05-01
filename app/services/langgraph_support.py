from __future__ import annotations

LANGGRAPH_IMPORT_ERROR: str | None = None

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except ModuleNotFoundError as exc:
    END = None
    START = None
    StateGraph = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = str(exc)
