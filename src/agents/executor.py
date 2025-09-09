from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from src.tools.search import get_tools

exec_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
tools = get_tools()
agent_executor = create_react_agent(
    exec_llm,
    tools,
    prompt=(
        "Bạn là trợ lý du lịch. Khi thực thi 1 bước kế hoạch, bạn có thể dùng công cụ search để lấy dữ liệu thời gian thực. "
        "Hãy tạo output GÓI GỌN theo schema sau (Markdown, ngắn gọn, tiếng Việt):\n"
        "### Ngày X\n"
        "- **schedule**:\n"
        "- ... 2-4 mục, mỗi mục: thời gian ngắn + địa điểm + ghi chú\n"
        "- **transportation**: cách di chuyển hợp lý, ưu tiên ít đi bộ\n"
        "- **dining_suggestions**: 1-3 gợi ý quán ăn gần các điểm\n"
        "- **estimated_cost**: ~xxx,xxx VND\n"
        "Lưu ý: Chỉ in phần cho NGÀY đang thực thi. Dùng search để cập nhật giờ mở cửa/giá vé/địa điểm hot."
    ),
)


async def execute_step(state):
    plan = state["plan"]
    if not plan:
        return {}
    plan_str = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}

You are tasked with executing step 1: {task}."""
    resp = await agent_executor.ainvoke({"messages": [("user", task_formatted)]})
    content = resp["messages"][-1].content
    return {"past_steps": [(task, content)]}
