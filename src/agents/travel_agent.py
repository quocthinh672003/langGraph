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
		start = text.find('{')
		end = text.rfind('}') + 1
		if start != -1 and end > start:
			return json.loads(text[start:end])
	except Exception:
		pass
	return fallback


def _rank_filter(items: List[Dict[str, Any]], interests: List[str]) -> List[Dict[str, Any]]:
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


@traceable(name="parse")
def _step_parse(llm: ChatOpenAI, user_input: str) -> Dict[str, Any]:
	"""Parse user input, classify intent, and generate search queries."""
	analysis_prompt = f"""Phân tích yêu cầu du lịch sau và trích xuất JSON:

Yêu cầu: "{user_input}"

Trả JSON với các trường:
{{
  "location": "địa điểm du lịch (không để trống, không bịa)",
  "duration": "thời gian (ví dụ: 3 ngày 2 đêm hay 1 tuần)",
  "interests": ["sở thích 1", "sở thích 2"],
  "budget": "mức ngân sách (thấp/trung bình/cao)",
  "constraints": ["hạn chế 1", "hạn chế 2"],
  "search_queries": ["truy vấn tiếng Việt 1", "truy vấn 2", "truy vấn 3"]
}}

Chỉ trả JSON hợp lệ, không thêm giải thích. Nếu thiếu thông tin quan trọng, để giá trị rỗng hoặc bỏ trường đó."""
	system_msg = {"role": "system", "content": "Be concise, structured, and factual. Prefer lists. Do not invent details."}
	analysis_response = llm.invoke([system_msg, {"role": "user", "content": analysis_prompt}])
	analysis_data = _extract_json(analysis_response.content.strip(), {})

	# quick intent classification (short prompt to save tokens)
	intent_prompt = f"""Trả JSON duy nhất: {{"task_type": one_of["itinerary","list_cafes","food_tour","photo_spots","compare","guide"]}}
Input: "{user_input}"""  # noqa: E501
	intent_resp = llm.invoke([system_msg, {"role": "user", "content": intent_prompt}])
	intent = _extract_json(intent_resp.content.strip(), {"task_type": "itinerary"}).get("task_type", "itinerary")
	if intent not in {"itinerary","list_cafes","food_tour","photo_spots","compare","guide"}:
		intent = "itinerary"

	location = (analysis_data.get("location") or "").strip()
	duration = (analysis_data.get("duration") or "").strip()
	interests = analysis_data.get("interests") or []
	budget = (analysis_data.get("budget") or "").strip()
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
	}


@traceable(name="search")
def _step_search(location: str, duration: str, interests: List[str], budget: str, constraints: List[str], seed_queries: List[str]) -> Dict[str, Any]:
	"""Run web search and return curated summary and citations."""
	payload = smart_search(location, duration, interests, budget, constraints, seed_queries)
	raw_items = payload.get("raw", [])
	citations = payload.get("citations", [])
	curated = _rank_filter(raw_items, interests)
	lines: List[str] = []
	for i, it in enumerate(curated[:8], start=1):
		title = it.get("title") or "Kết quả"
		url = it.get("url") or ""
		content = (it.get("content") or "").strip()
		if len(content) > 120:
			content = content[:117] + "..."
		lines.append(f"{i}. {title} — {content} {url}")
	return {"curated_summary": "\n".join(lines), "citations": citations}


@traceable(name="plan")
def _step_plan(llm: ChatOpenAI, user_input: str, task_type: str, location: str, interests: List[str], budget: str, constraints: List[str], num_days: int, curated_summary: str, citations: List[str]) -> str:
	"""Create the final Markdown output according to task type."""
	system_msg = {"role": "system", "content": "Be concise, structured, and factual. Prefer lists. Do not invent details."}
	if task_type == "itinerary":
		if num_days <= 0:
			return "Thiếu số ngày cho lịch trình. Vui lòng cho biết bạn đi bao nhiêu ngày."
		day_blocks = "\n\n---\n\n".join([
			f""" NGÀY {i+1}: [Tên]\n # schedule\n- **08:00-09:30**: [Hoạt động + địa điểm]\n- **10:00-12:00**: [Hoạt động + địa điểm]\n- **14:00-16:00**: [Hoạt động + địa điểm]\n- **16:30-18:00**: [Hoạt động + địa điểm]\n\n # transportation\n- [Phương tiện, lưu ý hạn chế đi bộ]\n\n # dining_suggestions\n- **Sáng**: [Quán + khu vực]\n- **Trưa**: [Quán + khu vực]\n- **Tối**: [Quán + khu vực]\n\n # estimated_cost\n- **Tổng ngày {i+1}**: ~[Số tiền] VND\n- **Chi tiết**: Vé ([số tiền]), ăn ([số tiền]), xe ([số tiền])"""
			for i in range(num_days)
		])
		plan_prompt = f"""Tạo lịch trình du lịch {location} (theo ngày) bằng Markdown.

Yêu cầu gốc: {user_input}
Sở thích: {', '.join(interests) if interests else 'du lịch'}; Ngân sách: {budget or 'không rõ'}; Hạn chế: {', '.join(constraints) if constraints else 'không có'}

Nguyên tắc:
- Gom điểm gần nhau theo khu vực; tránh nhảy xa.
- Ưu tiên địa điểm hot, hợp giới trẻ; tránh filler (ví dụ thư viện, công viên ít đặc sắc).
- Thêm món signature địa phương.
- Ngày cuối thêm mục "mua đặc sản mang về" (khô mực, tré, bánh khô mè...).

Kết quả search (rút gọn, thông tin thật):
{curated_summary}

# Lịch trình du lịch {location}

 Thông tin chung
- Ngân sách: [tóm tắt]
- Phong cách: [tóm tắt]
- Di chuyển: [gợi ý hạn chế đi bộ nếu có]

{day_blocks}

 Gợi ý nơi ở (tầm trung)
- [Khách sạn/Homestay 1] - [khoảng giá/đêm]
- [Khách sạn/Homestay 2] - [khoảng giá/đêm]
- [Khách sạn/Homestay 3] - [khoảng giá/đêm]

 Ẩm thực signature nên thử
- [Món 1]
- [Món 2]
- [Món 3]

 Tips & notes
- [Tip 1]
- [Tip 2]
- [Tip 3]

 Tổng chi phí ước tính ({num_days} ngày)
- Lưu trú: [khoảng]
- Ăn uống + cà phê: [khoảng]
- Di chuyển: [khoảng]
- Vé/hoạt động: [khoảng]
- Tổng: [khoảng]

 Nguồn tham khảo
- Liệt kê URL thật từ dữ liệu search (không bịa).

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
Sở thích: {', '.join(interests) if interests else 'du lịch'}; Ngân sách: {budget or 'không rõ'}; Hạn chế: {', '.join(constraints) if constraints else 'không có'}

Nguyên tắc:
- Gom theo khu vực/vibe; tránh backtracking; tránh filler.
- Ưu tiên địa điểm hot, giờ mở cửa/giá rõ ràng.
- Thêm món signature và gợi ý nơi ở tầm trung.

Kết quả search (rút gọn):
{curated_summary}

 {section_title}
- 8-12 gợi ý chất lượng, nhóm theo khu vực/vibe nếu hợp lý.
- Mỗi mục: tên, mô tả ngắn, thời điểm ghé (giờ mở cửa nếu có), lưu ý.

 Gợi ý nơi ở (tầm trung)
- 2-3 lựa chọn với khoảng giá/đêm (nếu có nguồn).

 Ẩm thực signature
- 3-5 món đặc trưng nên thử.

 Tips hữu ích
- 3-5 tips thực tế.

 Nguồn tham khảo
- Liệt kê URL thật; không bịa đặt.

Yêu cầu: Rõ ràng, súc tích; không dùng địa chỉ giả; ưu tiên giờ/giá 2024-2025."""
	resp = llm.invoke([system_msg, {"role": "user", "content": plan_prompt}])
	md = resp.content
	if citations and " Nguồn tham khảo" not in md:
		refs = "\n".join(f"- {u}" for u in citations)
		md += f"\n\n Nguồn tham khảo\n{refs}"
	return md


def travel_agent(state: TravelState) -> TravelState:
	"""Flexible travel agent with per-step LangSmith spans: parse → search → plan."""
	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, request_timeout=10, max_tokens=1700)
	user_input = state['input']

	# parse
	parsed = _step_parse(llm, user_input)
	location = parsed.get("location", "")
	duration = parsed.get("duration", "")
	interests = parsed.get("interests", [])
	budget = parsed.get("budget", "")
	constraints = parsed.get("constraints", [])
	seed_queries = parsed.get("seed_queries", [])
	num_days = parsed.get("num_days", 0)
	task_type = parsed.get("task_type", "itinerary")

	# basic validation
	missing: List[str] = []
	if not location:
		missing.append("địa điểm")
	if task_type == "itinerary" and not duration:
		missing.append("thời gian (số ngày/đêm)")
	if missing:
		return {"response": "Vui lòng bổ sung: " + ", ".join(missing) + "."}

	# search
	search_out = _step_search(location, duration, interests, budget, constraints, seed_queries)
	curated_summary = search_out.get("curated_summary", "")
	citations = search_out.get("citations", [])

	# plan
	markdown = _step_plan(
		llm=llm,
		user_input=user_input,
		task_type=task_type,
		location=location,
		interests=interests,
		budget=budget,
		constraints=constraints,
		num_days=num_days,
		curated_summary=curated_summary,
		citations=citations,
	)

	return {"response": markdown}
