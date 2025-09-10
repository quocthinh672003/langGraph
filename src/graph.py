from langgraph.graph import StateGraph, START, END
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from src.agents.travel_agent import _step_parse, _step_search, _step_plan


def _get_llm() -> ChatOpenAI:
	"""Create a concise, deterministic LLM client for nodes."""
	return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, request_timeout=10, max_tokens=1700)


def node_parse(state: Any) -> Dict[str, Any]:
	llm = _get_llm()
	# Accept either raw string or dict with key 'input'
	if isinstance(state, str):
		user_input = state
		state_dict: Dict[str, Any] = {"input": user_input}
	else:
		state_dict = dict(state or {})
		user_input = state_dict.get("input", "")
	parsed = _step_parse(llm, user_input)
	return {**state_dict, **parsed}


def node_search(state: Dict[str, Any]) -> Dict[str, Any]:
	location = state.get("location", "")
	duration = state.get("duration", "")
	interests = state.get("interests", [])
	budget = state.get("budget", "")
	constraints = state.get("constraints", [])
	seed_queries = state.get("seed_queries", [])
	search_out = _step_search(location, duration, interests, budget, constraints, seed_queries)
	return {**state, **search_out}


def node_plan(state: Dict[str, Any]) -> Dict[str, Any]:
	llm = _get_llm()
	markdown = _step_plan(
		llm=llm,
		user_input=state.get("input", ""),
		task_type=state.get("task_type", "itinerary"),
		location=state.get("location", ""),
		interests=state.get("interests", []),
		budget=state.get("budget", ""),
		constraints=state.get("constraints", []),
		num_days=state.get("num_days", 0),
		curated_summary=state.get("curated_summary", ""),
		citations=state.get("citations", []),
	)
	return {**state, "response": markdown}


def build_travel_graph():
	graph = StateGraph(dict)
	graph.add_node("parse", node_parse)
	graph.add_node("search", node_search)
	graph.add_node("plan", node_plan)

	graph.add_edge(START, "parse")
	graph.add_edge("parse", "search")
	graph.add_edge("search", "plan")
	graph.add_edge("plan", END)
	return graph.compile()


# Export app
APP = build_travel_graph()
