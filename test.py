from src.graph import APP

if __name__ == "__main__":
    user_text = "Hey, lên cho mình kế hoạch đi Đà Lạt 3 ngày 2 đêm với. Mình thích đi cà phê chill, chụp ảnh thiên nhiên. Ngân sách tầm trung thôi, và mình không thích đi bộ nhiều quá nhé."
    out = APP.invoke({"input": user_text, "plan": [], "past_steps": [], "response": ""})
    print(out.get("response", ""))
