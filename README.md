# Travel Planning Agent với LangGraph

Hệ thống agent du lịch sử dụng kiến trúc Plan-and-Execute với LangGraph, có khả năng tạo lịch trình du lịch chi tiết từ yêu cầu tiếng Việt.

## Kiến trúc

- **Mô hình**: Planning (Plan-and-Execute) theo flow "planner → agent executor → replan/finish"
- **LLM**: OpenAI gpt-4o-mini
- **Orchestrator**: LangGraph (StateGraph)
- **Tool**: Tavily Search
- **Agent Executor**: ReAct agent với tool search
- **Quan sát**: LangSmith

## Cài đặt

1. Tạo virtual environment:

```bash
py -m venv .venv
.venv/Scripts/Activate.ps1
```

2. Cài đặt dependencies:

```bash
pip install -U langgraph "langgraph[cli]" langchain langchain-openai langchain-community tavily-python pydantic python-dotenv langsmith
```

3. Thiết lập environment variables:

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:TAVILY_API_KEY="tvly-..."
$env:LANGCHAIN_TRACING_V2="true"
$env:LANGSMITH_API_KEY="lsm-..."
$env:LANGSMITH_PROJECT="travel-agent"
```

Hoặc copy `env_example.txt` thành `.env` và điền API keys.

## Sử dụng

### Chạy test

```bash
python test.py
```

### Chạy LangGraph Studio

```bash
langgraph studio --graph src.graph:APP --host 127.0.0.1 --port 2024
```

Truy cập http://127.0.0.1:2024 để xem DAG và debug.

## Cấu trúc

- `src/state.py`: Định nghĩa state PlanExecute
- `src/tools/search.py`: Tavily search tool
- `src/agents/planner.py`: Agent lập kế hoạch
- `src/agents/executor.py`: Agent thực thi với ReAct
- `src/agents/replanner.py`: Logic replan và tổng hợp
- `src/graph.py`: LangGraph workflow chính

## Flow

1. **Planner**: Phân tích yêu cầu và tạo danh sách các bước/ngày
2. **Executor**: Thực thi từng bước với ReAct + Tavily search
3. **Replanner**: Cập nhật plan và tổng hợp kết quả cuối

## Output

Lịch trình Markdown chi tiết với:

- **schedule**: Lịch trình từng ngày
- **transportation**: Phương tiện di chuyển
- **dining_suggestions**: Gợi ý ăn uống
- **estimated_cost**: Chi phí ước tính
