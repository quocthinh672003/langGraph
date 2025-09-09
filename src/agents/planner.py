from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class Steps(BaseModel):
    steps: List[str] = Field(
        description="Các bước thực hiện lần lượt, mỗi phần tử mô tả 1 ngày hoặc nhiệm vụ cụ thể."
    )


planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def plan_step(state):
    llm = planner_llm.with_structured_output(Steps)
    sys = "Bạn là planner. Hãy lập kế hoạch nhiều bước rõ ràng để tạo lịch trình du lịch tối ưu."
    user = f"""Yêu cầu người dùng (tiếng Việt):
{state["input"]}

Yêu cầu output:
- Trả về JSON Steps.steps là danh sách string, ví dụ:
["Ngày 1: ...", "Ngày 2: ...", "Ngày 3: ..."].
- Mỗi bước nêu rõ mục tiêu trong ngày (ưu tiên sở thích, ngân sách, hạn chế đi bộ)."""
    out = llm.invoke(
        [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    )
    return {"plan": out.steps}
