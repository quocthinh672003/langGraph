from langchain_community.tools.tavily_search import TavilySearchResults


def get_tools():
    # 3 kết quả/ lần theo tutorial
    return [TavilySearchResults(max_results=3)]
