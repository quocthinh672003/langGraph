# Travel Agent (LangGraph)

Agent tạo lịch trình du lịch tiếng Việt, tích hợp tìm kiếm thời gian thực (Tavily), chạy trên LangGraph Dev Server + LangSmith Studio.

## Cài đặt nhanh

```powershell
py -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -U -r requirements.txt langgraph-cli "langgraph-cli[inmem]" langchain-tavily
```

Tạo file `.env`:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsm-...
LANGSMITH_PROJECT=travel-agent
```

## Chạy Dev Server + Studio

`langgraph.json` đã cấu hình graph id `travel-agent` → `src.graph:APP`.

```powershell
python -m langgraph_cli dev --config langgraph.json --host 127.0.0.1 --port 2024
```

Mở Studio (URL in ra trong terminal) và chọn assistant/graph: `travel-agent`.

## Kiến trúc hiện tại

- parse → search → plan (tất cả trong `src/agents/travel_agent.py`)
- Search: `langchain_tavily.TavilySearch` (đọc `TAVILY_API_KEY` từ `.env`)
- Orchestrator: `src/graph.py` (`APP`)

## Ví dụ yêu cầu (Đà Nẵng 1 tuần)

Input mẫu:

```
Lên cho mình kế hoạch đi Đà Nẵng 1 tuần với. Mình thích đi chơi chỗ giới trẻ, chụp ảnh địa điểm hot, ăn món ăn được nhiều review ngon. Ngân sách tầm trung thôi, và giúp mình sử dụng phương tiện nào tiện với.
```

Output (rút gọn minh hoạ):

```
# Lịch trình du lịch Đà Nẵng

## Thông tin chung
- Ngân sách: Trung bình
- Phong cách: Chụp ảnh, địa điểm hot, ẩm thực review cao
- Di chuyển: Thuê xe máy (150–200k/ngày) hoặc taxi/Grab (~10–15k/km)

---

NGÀY 1: Khám phá thành phố
# schedule
- 08:00–09:30: Cầu Rồng
- 10:00–12:00: Công viên APEC
- 14:00–16:00: ...

# transportation
- Thuê xe máy hoặc Grab

# dining_suggestions
- Bánh mì..., Mì Quảng..., Hải sản...

# estimated_cost
- Tổng ngày 1: ~xxxk VND (vé/ăn/xe)

... (tiếp tục đến ngày 7: Bà Nà Hills, Mỹ Khê, Sơn Trà, Ghềnh Bàng, văn hoá, mua sắm)
```

Lưu ý: Khi chạy thực, agent sẽ dùng search để cập nhật giờ mở cửa/giá vé/địa điểm hot/citimations mới nhất.
