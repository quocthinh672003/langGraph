from langchain_openai import ChatOpenAI
from src.tools.search import smart_search
from src.state import TravelState
from typing import List, Dict, Any
import re
import json
from langsmith import traceable


def _extract_json(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the first JSON object from LLM text; fallback on failure."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return fallback


def _rank_filter(
    items: List[Dict[str, Any]], interests: List[str]
) -> List[Dict[str, Any]]:
    """Lightweight ranking by interest keyword hits; returns top ~10 items."""
    if not items:
        return []
    keywords = [s.lower() for s in (interests or [])]

    def score(it: Dict[str, Any]) -> int:
        title = (it.get("title") or "").lower()
        content = (it.get("content") or "").lower()
        return sum(1 for kw in keywords if kw and (kw in title or kw in content))

    return sorted(items, key=score, reverse=True)[:10]


def _infer_days(duration_str: str) -> int:
    """Infer number of days from natural text (e.g., '1 tuần' -> 7)."""
    if not duration_str:
        return 0
    s = duration_str.lower()
    m = re.search(r"(\d+)\s*ngày", s)
    if m:
        return max(1, int(m.group(1)))
    if "tuần" in s:
        m2 = re.search(r"(\d+).*tuần", s)
        return (int(m2.group(1)) * 7) if m2 else 7
    return 0


def _get_max_tokens(num_days: int) -> int:
    if num_days <= 3:
        return 1200
    elif num_days <= 7:
        return 1700
    elif num_days <= 10:
        return 2200
    elif num_days <= 14:
        return 2700
    else:
        return 3200


def _normalize_budget_to_vnd(budget_str: str) -> Dict[str, int]:
    """Best-effort parse of budget string to VND range.

    Examples:
    - "5 triệu" -> {"min": 5_000_000, "max": 5_000_000}
    - "3-5 triệu" -> {"min": 3_000_000, "max": 5_000_000}
    - "tầm trung" -> {}
    """
    if not budget_str:
        return {}
    s = budget_str.lower().strip()
    # Replace separators
    s = s.replace(",", ".")
    # Pattern for ranges like 3-5 or 3 – 5
    import re as _re

    m = _re.search(
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(tr|triệu|m|million|vnd|k)?", s
    )
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        unit = m.group(3) or "tr"
        if unit in {"k"}:
            mult = 1_000
        elif unit in {"vnd"}:
            mult = 1
        else:
            mult = 1_000_000
        return {"min": int(a * mult), "max": int(b * mult)}
    m2 = _re.search(r"(\d+(?:\.\d+)?)\s*(tr|triệu|m|million)", s)
    if m2:
        v = float(m2.group(1))
        return {"min": int(v * 1_000_000), "max": int(v * 1_000_000)}
    m3 = _re.search(r"(\d{6,})", s)
    if m3:
        v = int(m3.group(1))
        return {"min": v, "max": v}
    return {}


@traceable(name="parse")
def _step_parse(llm: ChatOpenAI, user_input: str) -> Dict[str, Any]:
    """Parse user input and return structured fields including task_type."""
    analysis_prompt = f"""Phân tích yêu cầu: "{user_input}"
Trả JSON duy nhất:
{{
  "location": "...",
  "duration": "...",
  "interests": ["..."],
  "budget": "...",
  "constraints": ["..."],
  "search_queries": ["..."],
  "task_type": "itinerary|list_cafes|food_tour|photo_spots|compare|guide"
}}
"""
    system_msg = {
        "role": "system",
        "content": "Be concise, structured, and factual. Prefer lists. Do not invent details.",
    }
    analysis_response = llm.invoke(
        [
            system_msg,
            {"role": "user", "content": analysis_prompt},
        ]
    )
    analysis_data = _extract_json(analysis_response.content.strip(), {})

    intent = (analysis_data.get("task_type") or "itinerary").strip()
    if intent not in {
        "itinerary",
        "list_cafes",
        "food_tour",
        "photo_spots",
        "compare",
        "guide",
    }:
        intent = "itinerary"

    location = (analysis_data.get("location") or "").strip()
    duration = (analysis_data.get("duration") or "").strip()
    interests = analysis_data.get("interests") or []
    budget = (analysis_data.get("budget") or "").strip()
    budget_range = _normalize_budget_to_vnd(budget)
    constraints = analysis_data.get("constraints") or []
    seed_queries = analysis_data.get("search_queries") or []
    num_days = _infer_days(duration) if duration else 0

    return {
        "location": location,
        "duration": duration,
        "interests": interests,
        "budget": budget,
        "constraints": constraints,
        "seed_queries": seed_queries,
        "num_days": num_days,
        "task_type": intent,
        "budget_range": budget_range,
    }


@traceable(name="search")
def _step_search(
    location: str,
    duration: str,
    interests: List[str],
    budget: str,
    constraints: List[str],
    seed_queries: List[str],
    top_k: int = 6,
) -> Dict[str, Any]:
    """Run web search and return curated summary and citations."""
    payload = smart_search(
        location, duration, interests, budget, constraints, seed_queries
    )
    raw_items = payload.get("raw", [])
    citations = payload.get("citations", [])
    curated = _rank_filter(raw_items, interests)
    lines: List[str] = []
    for i, it in enumerate(curated[:top_k], start=1):
        title = it.get("title") or "Kết quả"
        content = (it.get("content") or "").strip()
        if len(content) > 120:
            content = content[:117] + "..."
        # Do not include URL in the summary line
        lines.append(f"{i}. {title} — {content}")
    summary = "\n".join(lines)
    if len(summary) > 1500:
        summary = summary[:1500] + "..."
    return {"curated_summary": summary, "citations": citations}


@traceable(name="plan")
def _step_plan(
    llm: ChatOpenAI,
    user_input: str,
    task_type: str,
    location: str,
    interests: List[str],
    budget: str,
    constraints: List[str],
    num_days: int,
    curated_summary: str,
    citations: List[str],
    price_mode: str = "relaxed",
) -> str:
    """Create the final Markdown output according to task type."""
    system_msg = {
        "role": "system",
        "content": "Be concise, structured, and factual. Prefer lists. Do not invent details.",
    }
    if task_type == "itinerary":
        if num_days <= 0:
            return (
                "Thiếu số ngày cho lịch trình. Vui lòng cho biết bạn đi bao nhiêu ngày."
            )
        plan_prompt = f"""Tạo lịch trình du lịch {location} (theo ngày) bằng Markdown.

Yêu cầu gốc: {user_input}
Sở thích: {", ".join(interests) if interests else "du lịch"}; Ngân sách: {budget or "không rõ"}; Hạn chế: {", ".join(constraints) if constraints else "không có"}

Nguyên tắc:
- Không dùng tên chung chung/placeholder (ví dụ “Khách sạn {location}”, “Homestay Mộng Mơ”). Nếu thiếu dữ liệu: ghi “Khu vực …, tham khảo thêm trên bản đồ”.
- Lưu trú: đề xuất 2–3 nơi THẬT (tên + khu vực + tầm giá) nếu có trong kết quả search.
- Chi phí: ghi theo khoảng (min–max) và nêu giả định phương tiện (ví dụ thuê xe máy 150–200k/ngày hoặc taxi/Grab ~10–15k/km). Không ghi “Vé: 0 VND”; nếu miễn phí ghi “miễn phí”.
- Gom điểm gần nhau theo khu vực; tránh nhảy xa; ưu tiên địa điểm hot, hợp giới trẻ; tránh filler.
- Ít đi bộ/di chuyển hợp lý: tối đa 2 khu vực/ngày; tránh đường vòng; loại điểm quá xa/trekking/khó tiếp cận (ví dụ các điểm đi bộ dài), ưu tiên lộ trình taxi/Grab ngắn.
- Ngân sách tổng phải ≤ ngân sách người dùng; nếu vượt, bắt buộc loại hoạt động/điểm vé cao và thay bằng điểm miễn phí/rẻ.
- Ngân sách/ngày ≈ ngân sách/{num_days}. Mỗi ngày chỉ 3–4 hoạt động phù hợp mức này, 2–3 gợi ý ăn uống ngắn gọn.
- Tránh lặp tên địa điểm/quán giữa các ngày; mỗi tên chỉ xuất hiện 1 lần trong toàn lịch.
- Chỉ dùng điểm có trong “Kết quả search”; nếu không chắc, ghi “khu vực …, tham khảo thêm trên bản đồ”.
- Giá món ăn (chế độ {price_mode}): nếu "strict" chỉ ghi khi có bằng chứng rõ từ search; nếu "relaxed" có thể ghi “dao động …–… VND”; nếu không chắc thì bỏ giá.
- Không giải thích, không in URL; chỉ in Markdown đúng template.

Kết quả search (rút gọn, thông tin thật):
{curated_summary}

# Lịch trình du lịch {location}

 Thông tin chung
- Ngân sách: [tổng + ngân sách/ngày và phân bổ hợp lý]
- Phong cách: [tóm tắt]
- Di chuyển: [gợi ý hạn chế đi bộ nếu có]

Viết lịch trình cho đủ {num_days} ngày theo định dạng sau (mỗi ngày 3–4 hoạt động, 2–3 gợi ý ăn uống, 1 dòng chi phí):

NGÀY X: [tên]
# Lịch trình
- hh:mm-hh:mm: [hoạt động + địa điểm]
- hh:mm-hh:mm: [hoạt động + địa điểm]
- hh:mm-hh:mm: [hoạt động + địa điểm]
# Di chuyển
- [phương án taxi/Grab/xe máy]
# Gợi ý ăn uống
- [quán 1]
- [quán 2]
# Chi phí ước tính
- Tổng ngày X: ~[khoảng] VND; chi tiết vé/ăn/xe
"""
    else:
        section_title = {
            "list_cafes": "Danh sách cà phê chill",
            "food_tour": "Food tour",
            "photo_spots": "Điểm chụp ảnh đẹp",
            "compare": "So sánh lựa chọn",
            "guide": "Gợi ý đi đâu làm gì",
        }.get(task_type, "Gợi ý tổng hợp")
        plan_prompt = f"""Tạo hướng dẫn du lịch {location} theo Markdown (không cần theo ngày).

Yêu cầu gốc: {user_input}
Sở thích: {", ".join(interests) if interests else "du lịch"}; Ngân sách: {budget or "không rõ"}; Hạn chế: {", ".join(constraints) if constraints else "không có"}

Kết quả search (rút gọn):
{curated_summary}

 {section_title}
- 8-12 gợi ý chất lượng, nhóm theo khu vực/vibe nếu hợp lý; tránh lặp tên.
- Mỗi mục: tên thật, mô tả ngắn, thời điểm ghé (giờ mở cửa nếu có), lưu ý; giá món theo chế độ {price_mode} (strict/relaxed) và không bịa khi không chắc.
- Không dùng tên placeholder.

 Gợi ý nơi ở (tầm trung)
- 2-3 lựa chọn THẬT (tên + khu vực + tầm giá); nếu thiếu: ghi khu vực + cách tìm trên bản đồ.

 Ẩm thực signature
- 3-5 món đặc trưng đúng vùng.

 Tips hữu ích
- 3-5 tips thực tế; chi phí ghi theo khoảng và nêu giả định khi cần.

Yêu cầu: Rõ ràng, súc tích; không dùng địa chỉ giả; ưu tiên giờ/giá 2024-2025."""
    resp = llm.invoke([system_msg, {"role": "user", "content": plan_prompt}])
    md = resp.content
    return md


def travel_agent(state: TravelState) -> TravelState:
    """Flexible travel agent with per-step LangSmith spans: parse → search → plan."""
    user_input = state["input"]

    # parse (gọn, riêng max_tokens thấp)
    llm_parse = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.1, request_timeout=10, max_tokens=700
    )
    parsed = _step_parse(llm_parse, user_input)
    location = parsed.get("location", "")
    duration = parsed.get("duration", "")
    interests = parsed.get("interests", [])
    budget = parsed.get("budget", "")
    constraints = parsed.get("constraints", [])
    seed_queries = parsed.get("search_queries") or [user_input]
    num_days = parsed.get("num_days", 0)
    task_type = parsed.get("task_type", "itinerary")
    budget_range = parsed.get("budget_range", {})

    # basic validation
    missing: List[str] = []
    if not location:
        missing.append("địa điểm")
    if task_type == "itinerary" and not duration:
        missing.append("thời gian (số ngày/đêm)")
    if missing:
        return {"response": "Vui lòng bổ sung: " + ", ".join(missing) + "."}

    # search with lightweight retry + summary cap
    try:
        search_out = _step_search(
            location, duration, interests, budget, constraints, seed_queries
        )
    except Exception:
        search_out = {"curated_summary": "", "citations": []}
    curated_summary = search_out.get("curated_summary", "")

    # plan (auto-scale theo loại tác vụ & số ngày)
    max_tokens = 1000 if task_type != "itinerary" else _get_max_tokens(num_days)
    llm_plan = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        request_timeout=15,
        max_tokens=max_tokens,
    )
    markdown = _step_plan(
        llm=llm_plan,
        user_input=user_input,
        task_type=task_type,
        location=location,
        interests=interests,
        budget=budget,
        constraints=constraints,
        num_days=num_days,
        curated_summary=curated_summary,
        citations=[],
        price_mode="relaxed",
    )
    return {"response": markdown}
