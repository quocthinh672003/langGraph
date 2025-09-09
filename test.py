from langgraph.graph import StateGraph, START, END

def mock_llm(state):
    return {"role": "assistant", "content": "Hello, how are you?"}

graph = StateGraph(state)
graph.add_node("llm", mock_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)
graph.compile()

graph.invoke([{"role": "user", "content": "Hello, how are you?"}])