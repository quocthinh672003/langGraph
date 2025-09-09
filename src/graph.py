from langgraph.graph import StateGraph, START
from src.state import PlanExecute
from src.agents.planner import plan_step
from src.agents.executor import execute_step
from src.agents.replanner import replan_step, should_end


def build_graph():
    g = StateGraph(PlanExecute)
    g.add_node("planner", plan_step)
    g.add_node("agent", execute_step)
    g.add_node("replan", replan_step)

    g.add_edge(START, "planner")
    g.add_edge("planner", "agent")
    g.add_edge("agent", "replan")
    g.add_conditional_edges("replan", should_end, ["agent"])
    return g.compile()


APP = build_graph()  # Studio sẽ import biến này
