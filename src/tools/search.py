from langchain_tavily import TavilySearch
from typing import List, Dict, Any, Optional
import re

# Config constants for easy tuning
MAX_QUERIES: int = 3
MAX_ITEMS: int = 6
SUMMARY_LEN: int = 110


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
            content = (
                r.get("content") or r.get("snippet") if isinstance(r, dict) else None
            )
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
    return items



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
        interests_str = (
            " ".join(interests) if interests and interests != ["du lịch"] else ""
        )
        queries.append(
            f"{location} {interests_str} địa điểm hot 2024 2025 giờ mở cửa giá vé".strip()
        )
        queries.append(f"{location} quán ăn ngon review cao 2024 2025")
        queries.append(f"{location} quán cà phê đẹp chill 2024 2025")
        queries.append(f"{location} khách sạn tầm trung review tốt 2024 2025")

    # Use at most MAX_QUERIES to keep latency low
    queries = queries[:MAX_QUERIES]

    items: List[Dict[str, Any]] = []
    for q in queries:
        try:
            result = search_tool.invoke(q)
            items.extend(_normalize_result(result))
        except Exception:
            # Skip errors silently to avoid polluting results
            continue

    return {
        "summary": f"Thông tin cơ bản về {location}",
        "citations": [],
        "raw": items,
    }


