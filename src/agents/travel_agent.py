from langchain_openai import ChatOpenAI
from src.tools.search import smart_search
from src.state import TravelState
from typing import List, Dict, Any
import re
import json
from langsmith import traceable

# Configuration constants
MAX_CURATED_ITEMS = 6
MAX_SUMMARY_LENGTH = 1500
MAX_CONTENT_LENGTH = 120
FALLBACK_ITEMS_LIMIT = 4


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
    """Rank items by relevance to user interests."""
    if not items:
        return []
    
    # Clean and normalize keywords
    keywords = [kw.lower().strip() for kw in (interests or []) if kw and kw.strip()]

    def calculate_relevance_score(item: Dict[str, Any]) -> int:
        """Calculate relevance score based on keyword matches."""
        title = (item.get("title") or "").lower()
        content = (item.get("content") or "").lower()
        
        # Count keyword matches in title and content
        score = 0
        for keyword in keywords:
            if keyword in title:
                score += 2  # Title matches are more important
            if keyword in content:
                score += 1
        
        return score

    # Sort by relevance score (highest first) and limit results
    ranked_items = sorted(items, key=calculate_relevance_score, reverse=True)
    return ranked_items[:10]


def _infer_days(duration_str: str) -> int:
    """Infer number of days from natural text (e.g., '1 tuần' -> 7)."""
    if not duration_str:
        return 0
    s = duration_str.lower()
    
    # Pattern: "3 ngày 2 đêm" -> 3
    m = re.search(r"(\d+)\s*ngày", s)
    if m:
        return max(1, int(m.group(1)))
    
    # Pattern: "thứ Sáu đến Chủ Nhật" -> 3 ngày
    if "thứ" in s and "đến" in s:
        days_of_week = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]
        start_day = None
        end_day = None
        for day in days_of_week:
            if day in s:
                if start_day is None:
                    start_day = days_of_week.index(day)
                else:
                    end_day = days_of_week.index(day)
        if start_day is not None and end_day is not None:
            if end_day >= start_day:
                return end_day - start_day + 1
            else:  # Cross week
                return (7 - start_day) + end_day + 1
    
    # Pattern: "thứ Sáu" -> 1 ngày (fallback)
    if "thứ" in s:
        return 1
    
    # Pattern: "1 tuần" -> 7
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
    analysis_prompt = f"""Phân tích yêu cầu du lịch: "{user_input}"

QUAN TRỌNG: 
- ĐỌC KỸ TỪNG TỪ để xác định địa điểm chính xác
- Nếu text nói "Huế" thì location phải là "Huế", không phải "Hà Nội" hay nơi khác
- Phân tích thời gian cụ thể (ngày trong tuần, số ngày)
- Hiểu ngữ cảnh gia đình, nhóm bạn, cá nhân
- Tạo search queries về DU LỊCH, không phải về công nghệ

Trả JSON duy nhất:
{{
  "location": "tên địa điểm chính xác (VD: Huế, Đà Nẵng, Hà Nội)",
  "duration": "thời gian cụ thể (VD: 3 ngày 2 đêm, thứ Sáu đến Chủ Nhật)",
  "interests": ["sở thích từ yêu cầu"],
  "budget": "ngân sách được đề cập",
  "constraints": ["hạn chế, yêu cầu đặc biệt"],
  "search_queries": ["truy vấn tìm kiếm về DU LỊCH địa điểm đó"],
  "task_type": "itinerary|list_cafes|food_tour|photo_spots|compare|guide"
}}

Ví dụ phân tích:
- "Lên kế hoạch du lịch chi tiết tại Huế" → location: "Huế", search_queries: ["Huế địa điểm du lịch", "Huế ẩm thực", "Huế khách sạn"]
- "thứ Sáu đến Chủ Nhật" → duration: "3 ngày 2 đêm" 
- "gia đình 4 người" → constraints: ["phù hợp gia đình"]
- "8 triệu đồng" → budget: "8 triệu đồng"
- "Ngân sách: 8 triệu đồng" → budget: "8 triệu đồng"

NHẮC LẠI: 
- Nếu text nói "Huế" thì location phải là "Huế"!
- Nếu text nói "8 triệu đồng" thì budget phải là "8 triệu đồng"!
"""
    system_msg = {
        "role": "system",
        "content": "You are a travel planning expert. Read the input text carefully and extract information accurately. If the text mentions 'Huế', the location must be 'Huế'. If the text mentions '8 triệu đồng', the budget must be '8 triệu đồng'. Be precise and factual.",
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
    top_k: int = MAX_CURATED_ITEMS,
) -> Dict[str, Any]:
    """Run web search and return curated summary with fallback."""
    
    # Step 1: Get search results
    payload = smart_search(location, duration, interests, budget, constraints, seed_queries)
    raw_items = payload.get("raw", [])
    citations = payload.get("citations", [])
    
    # Step 2: Create curated summary from search results
    curated_summary = _create_curated_summary(raw_items, interests, top_k)
    
    # Step 3: Fallback if summary is empty
    if not curated_summary.strip():
        curated_summary = _create_fallback_summary(location, interests, budget)
    
    return {"curated_summary": curated_summary, "citations": citations}


def _create_curated_summary(raw_items: List[Dict[str, Any]], interests: List[str], top_k: int) -> str:
    """Create curated summary from search results."""
    if not raw_items:
        return ""
    
    # Rank and filter items by relevance
    curated = _rank_filter(raw_items, interests)
    
    # Build summary lines
    lines: List[str] = []
    for i, item in enumerate(curated[:top_k], start=1):
        title = item.get("title") or "Kết quả"
        content = (item.get("content") or "").strip()
        
        # Truncate long content
        if len(content) > MAX_CONTENT_LENGTH:
            content = content[:MAX_CONTENT_LENGTH-3] + "..."
        
        lines.append(f"{i}. {title} — {content}")
    
    summary = "\n".join(lines)
    
    # Truncate if too long
    if len(summary) > MAX_SUMMARY_LENGTH:
        summary = summary[:MAX_SUMMARY_LENGTH] + "..."
    
    return summary


def _create_fallback_summary(location: str, interests: List[str], budget: str) -> str:
    """Create fallback summary when search results are empty."""
    fallback_items = [
        f"1. {location} địa điểm du lịch nổi tiếng — Thành phố có nhiều di tích lịch sử và văn hóa đặc sắc",
        f"2. {location} ẩm thực địa phương — Nhiều món ăn truyền thống và quán ăn ngon",
        f"3. {location} khách sạn tầm trung — Nhiều lựa chọn lưu trú phù hợp ngân sách",
    ]
    
    # Add interest-specific fallbacks
    if interests:
        for interest in interests[:2]:  # Limit to 2 interests
            if "cà phê" in interest.lower():
                fallback_items.append(f"4. {location} quán cà phê chill — Nhiều quán cà phê với không gian đẹp")
            elif "chụp ảnh" in interest.lower():
                fallback_items.append(f"4. {location} điểm chụp ảnh đẹp — Nhiều góc chụp ảnh lý tưởng")
    
    return "\n".join(fallback_items[:FALLBACK_ITEMS_LIMIT])


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
        plan_prompt = f"""Tạo lịch trình du lịch {location} CHI TIẾT (theo ngày) bằng Markdown.

Yêu cầu gốc: {user_input}
Sở thích: {", ".join(interests) if interests else "du lịch"}; Ngân sách: {budget or "không rõ"}; Hạn chế: {", ".join(constraints) if constraints else "không có"}

QUAN TRỌNG VỀ NGÂN SÁCH:
- Ngân sách được đề cập: {budget or "không rõ"}
- Tổng chi phí phải ≤ ngân sách này
- Nếu ngân sách là "8 triệu đồng" thì tổng chi phí phải ≤ 8.000.000 VNĐ

QUAN TRỌNG - Tạo output CHI TIẾT như ví dụ:
- Thời gian cụ thể: "14:00", "15:30", "18:00" (không phải "08:00-10:00")
- Địa chỉ/quán ăn cụ thể: "Quán Bún Bò Huế (123 Nguyễn Huệ, Huế)"
- Chi phí chi tiết từng mục: "Tàu hỏa: 1.200.000 VNĐ", "Taxi: 100.000 VNĐ"
- Hoạt động phù hợp từng thành viên: "chèo SUP cho con trai", "cà phê sân vườn cho bố"
- Gợi ý nơi ở cụ thể: "Mộc Villa (có sân vườn, gần khu mua sắm)"

Nguyên tắc:
- Không dùng tên chung chung/placeholder. Nếu thiếu dữ liệu: ghi "Khu vực …, tham khảo thêm trên bản đồ".
- Lưu trú: đề xuất 2–3 nơi THẬT (tên + khu vực + tầm giá) nếu có trong kết quả search.
- Chi phí: ghi theo khoảng (min–max) và nêu giả định phương tiện. Không ghi "Vé: 0 VND"; nếu miễn phí ghi "miễn phí".
- Gom điểm gần nhau theo khu vực; tránh nhảy xa; ưu tiên địa điểm hot, hợp giới trẻ; tránh filler.
- Ít đi bộ/di chuyển hợp lý: tối đa 2 khu vực/ngày; tránh đường vòng; loại điểm quá xa/trekking/khó tiếp cận.
- Ngân sách tổng phải ≤ ngân sách người dùng; nếu vượt, bắt buộc loại hoạt động/điểm vé cao và thay bằng điểm miễn phí/rẻ.
- Ngân sách/ngày ≈ ngân sách/{num_days}. Mỗi ngày chỉ 3–4 hoạt động phù hợp mức này, 2–3 gợi ý ăn uống ngắn gọn.
- Tránh lặp tên địa điểm/quán giữa các ngày; mỗi tên chỉ xuất hiện 1 lần trong toàn lịch.
- Chỉ dùng điểm có trong "Kết quả search"; nếu không chắc, ghi "khu vực …, tham khảo thêm trên bản đồ".
- Giá món ăn (chế độ {price_mode}): nếu "strict" chỉ ghi khi có bằng chứng rõ từ search; nếu "relaxed" có thể ghi "dao động …–… VND"; nếu không chắc thì bỏ giá.
- Không giải thích, không in URL; chỉ in Markdown đúng template.

Kết quả search (rút gọn, thông tin thật):
{curated_summary}

# Lịch trình du lịch {location}

## Thông tin chung
- Ngân sách: [tổng + ngân sách/ngày và phân bổ hợp lý]
- Phong cách: [tóm tắt]
- Di chuyển: [gợi ý hạn chế đi bộ nếu có]

Viết lịch trình cho đủ {num_days} ngày theo định dạng CHI TIẾT sau:

## Ngày X: [tên]
### 📅 Schedule
- **[hh:mm]**: [hoạt động chi tiết + địa điểm cụ thể]
- **[hh:mm]**: [hoạt động chi tiết + địa điểm cụ thể]
- **[hh:mm]**: [hoạt động chi tiết + địa điểm cụ thể]

### 🚗 Transportation
- [phương án taxi/Grab/xe máy chi tiết]

### 🍽️ Dining Suggestions
- [bữa]: [tên quán cụ thể + địa chỉ/khu vực]
- [bữa]: [tên quán cụ thể + địa chỉ/khu vực]

### 💰 Estimated Cost
- [mục 1]: [số tiền cụ thể]
- [mục 2]: [số tiền cụ thể]
- [mục 3]: [số tiền cụ thể]
- **Tổng ngày X**: [tổng số tiền]

---

Viết đủ {num_days} ngày theo format trên. Cuối cùng thêm:

## Tổng hợp chi phí
- **Ngày 1**: [số tiền]
- **Ngày 2**: [số tiền]
- **Ngày 3**: [số tiền]
- **Tổng chi phí ăn uống, di chuyển**: [số tiền]
- **Chi phí [phương tiện]**: [số tiền]
- **Tổng chi phí toàn bộ**: [số tiền]

### LƯU Ý QUAN TRỌNG:
- Chi phí ước tính có thể thay đổi tùy thuộc vào lựa chọn thực tế và thời gian đặt chỗ.
- Nên đặt trước các hoạt động trải nghiệm và chỗ ăn uống để đảm bảo có chỗ.

Yêu cầu: Rõ ràng, súc tích; không dùng địa chỉ giả; ưu tiên giờ/giá 2024-2025."""
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
        model="gpt-4o-mini", temperature=0.1, request_timeout=10, max_tokens= 550
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
