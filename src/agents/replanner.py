from typing import Literal
from langgraph.graph import END


def should_end(state) -> Literal["agent", END]:
    # Kết thúc khi không còn bước nào
    if not state.get("plan"):
        return END
    return "agent"


def replan_step(state):
    # Bỏ bước đầu vừa thực thi; khi hết thì tổng hợp response
    remaining = state["plan"][1:] if state["plan"] else []
    if remaining:
        return {"plan": remaining}
    # Tổng hợp Markdown từ past_steps
    days = []
    for idx, (task, content) in enumerate(state.get("past_steps", []), start=1):
        # content đã ở dạng Markdown cho từng ngày (executor tạo)
        days.append(content)
    final_md = "## Lịch trình du lịch\n" + "\n\n".join(days)
    return {"response": final_md}
