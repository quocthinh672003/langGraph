from typing import TypedDict, List, Tuple


class PlanExecute(TypedDict):
    input: str
    plan: List[str]  # danh sách bước/Ngày
    past_steps: List[Tuple[str, str]]  # (task, result)
    response: str  # Markdown cuối
