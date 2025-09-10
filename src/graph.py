from langgraph.graph import StateGraph, START, END
from src.state import TravelState
from src.agents.travel_agent import travel_agent

def build_travel_graph():
    graph = StateGraph(TravelState)
    graph.add_node("agent", travel_agent)
    graph.add_edge(START, "agent")
    graph.add_edge("agent", END)
    return graph.compile()

# Export app
APP = build_travel_graph()
