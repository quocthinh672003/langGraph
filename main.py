from dotenv import load_dotenv
from src.graph import APP
import os

load_dotenv()

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "travel-agent"

if __name__ == "__main__":
    # Test với input mẫu
    user_input = "Hey, lên cho mình kế hoạch đi Đà Lạt 3 ngày 2 đêm với. Mình thích đi cà phê chill, chụp ảnh thiên nhiên. Ngân sách tầm trung thôi, và mình không thích đi bộ nhiều quá đâu nhé."
    
    result = APP.invoke({"input": user_input})
    print("=" * 50)
    print("LỊCH TRÌNH DU LỊCH ĐÀ LẠT")
    print("=" * 50)
    print(result["response"])