# Travel Agent

Agent tạo lịch trình du lịch từ yêu cầu tiếng Việt.

## Cài đặt

```bash
pip install -r requirements.txt
```

Tạo file `.env`:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsm-...
LANGSMITH_PROJECT=travel-agent
```

## Chạy

```bash

# Studio
py -3.13 -m langgraph_cli dev
```

Mở http://127.0.0.1:2024

## Cách hoạt động

1. **Planner**: Tạo kế hoạch dựa trên yêu cầu
2. **Executor**: Tìm thông tin thời gian thực và tạo lịch trình
3. **Replanner**: Tổng hợp kết quả

## Input

```
"Hey, lên cho mình kế hoạch đi Đà Lạt 3 ngày 2 đêm với. Mình thích đi cà phê chill, chụp ảnh thiên nhiên. Ngân sách tầm trung thôi, và mình không thích đi bộ nhiều quá đâu nhé."
```

## Output

Lịch trình Markdown với schedule, transportation, dining, cost.