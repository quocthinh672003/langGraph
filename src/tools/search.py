from langchain_tavily import TavilySearch
from typing import List, Dict, Any, Tuple, Optional
import re


def get_search_tool() -> TavilySearch:
	"""Return Tavily search tool with a small result cap for speed."""
	return TavilySearch(max_results=5)


def _normalize_result(result: Any) -> List[Dict[str, Any]]:
	"""Normalize Tavily outputs to a list of {title,url,content} dicts."""
	items: List[Dict[str, Any]] = []
	if isinstance(result, list):
		for r in result:
			title = r.get("title") if isinstance(r, dict) else None
			url = r.get("url") if isinstance(r, dict) else None
			content = r.get("content") or r.get("snippet") if isinstance(r, dict) else None
			if title or url or content:
				items.append({"title": title, "url": url, "content": content})
	elif isinstance(result, dict):
		res = result.get("results")
		if isinstance(res, list):
			for r in res:
				title = r.get("title")
				url = r.get("url")
				content = r.get("content") or r.get("snippet")
				if title or url or content:
					items.append({"title": title, "url": url, "content": content})
	else:
		# Fallback: extract URLs from raw text
		text = str(result)
		urls = re.findall(r"https?://\S+", text)
		for u in urls[:5]:
			items.append({"title": None, "url": u, "content": None})
	return items


def _summarize_items(items: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
	"""Build a compact summary and collect citation URLs."""
	lines: List[str] = []
	citations: List[str] = []
	for i, it in enumerate(items[:8], start=1):  # tighter cap for brevity
		title = it.get("title") or "Kết quả"
		url = it.get("url") or ""
		content = it.get("content") or ""
		short = content.strip()
		if len(short) > 140:
			short = short[:137] + "..."
		lines.append(f"{i}. {title} — {short} {url}")
		if url:
			citations.append(url)
	return ("\n".join(lines) if lines else ""), citations


def smart_search(
	location: str,
	duration: str,
	interests: List[str],
	budget: str,
	constraints: List[str],
	seed_queries: Optional[List[str]] = None,
) -> Dict[str, Any]:
	"""Run web search using LLM-provided queries or generic fallbacks.

	Returns a dict: {summary: str, citations: list[str], raw: list[items]}.
	"""
	# If destination is missing, skip search to avoid fabricating context
	if not location or not str(location).strip():
		return {"summary": "", "citations": [], "raw": []}

	search_tool = get_search_tool()

	queries: List[str] = []
	if seed_queries:
		queries.extend([q for q in seed_queries if isinstance(q, str) and q.strip()])

	# Generic fallbacks when the LLM doesn't provide queries
	if not queries:
		interests_str = " ".join(interests) if interests and interests != ["du lịch"] else ""
		queries.append(f"{location} {interests_str} địa điểm hot 2024 2025 giờ mở cửa giá vé".strip())
		queries.append(f"{location} quán ăn ngon review cao 2024 2025")
		queries.append(f"{location} quán cà phê đẹp chill 2024 2025")
		queries.append(f"{location} khách sạn tầm trung review tốt 2024 2025")

	# Use at most 4 queries to keep latency low
	queries = queries[:4]

	items: List[Dict[str, Any]] = []
	for q in queries:
		try:
			result = search_tool.invoke(q)
			items.extend(_normalize_result(result))
		except Exception as e:
			items.append({"title": "Lỗi", "url": None, "content": str(e)})

	summary, citations = _summarize_items(items)
	return {"summary": summary or f"Thông tin cơ bản về {location}", "citations": citations, "raw": items}


def search_specific_info(location: str, info_type: str) -> str:
	"""Backward-compatible focused search by info type; returns a concise summary string."""
	if not location or not str(location).strip():
		return ""
	search_tool = get_search_tool()

	specific_queries = {
		"giờ_mở_cửa": f"{location} giờ mở cửa địa điểm du lịch 2024 2025",
		"giá_vé": f"{location} giá vé tham quan 2024 2025",
		"quán_ăn": f"{location} quán ăn ngon review cao 2024 2025",
		"cà_phê": f"{location} quán cà phê đẹp chill 2024 2025",
		"chụp_ảnh": f"{location} địa điểm chụp ảnh đẹp 2024 2025",
		"thiên_nhiên": f"{location} cảnh đẹp thiên nhiên 2024 2025",
		"phương_tiện": f"{location} phương tiện di chuyển taxi xe máy",
		"khách_sạn": f"{location} khách sạn tầm trung review tốt 2024 2025",
	}

	query = specific_queries.get(info_type, f"{location} {info_type} 2024 2025")

	try:
		result = search_tool.invoke(query)
		items = _normalize_result(result)
		summary, _ = _summarize_items(items)
		return summary or ""
	except Exception:
		return ""
