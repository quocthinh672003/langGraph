"""
Travel Agent Prompts - Optimized and organized
"""

# Parse prompt - Extract structured data from user input
PARSE_PROMPT = """Bạn là chuyên gia lập kế hoạch du lịch. ĐỌC KỸ TỪNG TỪ để xác định thông tin chính xác.

Nhiệm vụ: Phân tích yêu cầu du lịch và trích xuất thông tin cấu trúc.

QUAN TRỌNG - ĐỌC KỸ TỪNG TỪ:
- Nếu text nói "Huế" thì location phải là "Huế"
- Nếu text nói "8 triệu đồng" thì budget phải là "8 triệu đồng"
- Nếu text nói "thứ Sáu đến Chủ Nhật" thì duration phải là "thứ Sáu đến Chủ Nhật"
- Nếu text nói "chèo thuyền" thì interests phải có "chèo thuyền"

Trích xuất thông tin theo format JSON:
{{
    "location": "địa điểm chính xác",
    "duration": "thời gian chính xác", 
    "interests": ["sở thích 1", "sở thích 2"],
    "budget": "ngân sách chính xác",
    "constraints": ["hạn chế 1", "hạn chế 2"],
    "search_queries": ["từ khóa tìm kiếm 1", "từ khóa tìm kiếm 2"],
    "task_type": "itinerary|list_cafes|food_tour|photo_spots|compare|guide"
}}

Yêu cầu: {user_input}

Chỉ trả về JSON, không giải thích."""

# Search prompt - Generate search queries
SEARCH_PROMPT = """Tạo 3-5 từ khóa tìm kiếm tối ưu cho du lịch {location}.

Yêu cầu: {user_input}
Sở thích: {interests}
Ngân sách: {budget}
Hạn chế: {constraints}

Tạo từ khóa tìm kiếm:
- Địa điểm nổi tiếng, hot spots
- Ẩm thực địa phương, quán ăn ngon
- Hoạt động phù hợp sở thích
- Khách sạn, homestay tầm trung
- Tips du lịch, kinh nghiệm

Trả về danh sách từ khóa, mỗi từ khóa một dòng."""

# Plan prompt - Generate detailed itinerary
PLAN_PROMPT = """Tạo lịch trình du lịch {location} CHI TIẾT (theo ngày) bằng Markdown.

Yêu cầu gốc: {user_input}
Sở thích: {interests}; Ngân sách: {budget}; Hạn chế: {constraints}

QUAN TRỌNG VỀ NGÂN SÁCH:
- Ngân sách: {budget}
- Tổng chi phí phải ≤ ngân sách này
- Ưu tiên hoạt động phù hợp ngân sách (tầm trung/cao cấp/tiết kiệm)

QUAN TRỌNG VỀ THÀNH VIÊN:
- Đọc kỹ yêu cầu để hiểu từng thành viên và sở thích riêng
- Tạo hoạt động phù hợp (tuổi tác, sở thích, hạn chế)
- Ưu tiên hoạt động được đề cập cụ thể
- Tránh hoạt động không phù hợp

QUAN TRỌNG VỀ THỜI GIAN THỰC TẾ:
- Nếu đến 14:00 → Ngày 1 chỉ có buổi chiều (14:00-18:00)
- Nếu về 21:00 → Ngày cuối chỉ có buổi sáng/chiều (08:00-17:00)
- Không bắt đầu từ 08:00 nếu đến 14:00
- Tính toán thời gian di chuyển thực tế

Kết quả search: {curated_summary}

# Lịch trình du lịch {location}

## Thông tin chung
- Ngân sách: [tổng + phân bổ hợp lý]
- Phong cách: [tóm tắt]
- Di chuyển: [gợi ý hạn chế đi bộ nếu có]

Viết lịch trình cho đủ {num_days} ngày theo format:

## Ngày X: [tên]
### 📅 Schedule
- **[hh:mm]**: [hoạt động chi tiết + địa điểm cụ thể]
- **[hh:mm]**: [hoạt động chi tiết + địa điểm cụ thể]

### 🚗 Transportation
- [phương án taxi/Grab chi tiết]

### 🍽️ Dining Suggestions
- [bữa]: [tên quán cụ thể + địa chỉ/khu vực]

### 💰 Estimated Cost
- [mục 1]: [số tiền cụ thể]
- **Tổng ngày X**: [tổng số tiền]

---

Viết đủ {num_days} ngày. Cuối cùng thêm:

## Gợi ý nơi ở (tầm trung)
- [Tên khách sạn] - [khu vực] - [khoảng giá/đêm]

## Ẩm thực signature nên thử
- [Món 1] - [Quán địa phương cụ thể]
- [Món 2] - [Quán địa phương cụ thể]

## Tips & notes
- [Tip 1]
- [Tip 2]

## Tổng hợp chi phí
- **Ngày 1**: [số tiền]
- **Ngày 2**: [số tiền]
- **Tổng chi phí ăn uống**: [số tiền] (4 người × 3 bữa/ngày × 150k/người)
- **Tổng chi phí di chuyển**: [số tiền] (taxi/Grab cả ngày)
- **Tổng chi phí lưu trú**: [số tiền] (2 đêm × 4 người)
- **Tổng chi phí toàn bộ**: [số tiền] (phải gần với ngân sách)

### LƯU Ý QUAN TRỌNG:
- Chi phí ước tính có thể thay đổi tùy thuộc vào lựa chọn thực tế
- Nên đặt trước các hoạt động trải nghiệm và chỗ ăn uống

Yêu cầu: Rõ ràng, súc tích; không dùng địa chỉ giả; ưu tiên giờ/giá 2024-2025."""

# Non-itinerary prompt for other task types
NON_ITINERARY_PROMPT = """Tạo hướng dẫn du lịch {location} theo Markdown (không cần theo ngày).

Yêu cầu gốc: {user_input}
Sở thích: {interests}; Ngân sách: {budget}; Hạn chế: {constraints}

Kết quả search: {curated_summary}

# {section_title}
- 8-12 gợi ý chất lượng, nhóm theo khu vực/vibe nếu hợp lý
- Mỗi mục: tên thật, mô tả ngắn, thời điểm ghé, lưu ý
- Không dùng tên placeholder

## Gợi ý nơi ở (tầm trung)
- 2-3 lựa chọn THẬT (tên + khu vực + tầm giá)

## Ẩm thực signature
- 3-5 món đặc trưng đúng vùng

## Tips hữu ích
- 3-5 tips thực tế

Yêu cầu: Rõ ràng, súc tích; không dùng địa chỉ giả; ưu tiên giờ/giá 2024-2025."""

# Task type mapping
TASK_TYPE_MAPPING = {
    "list_cafes": "Danh sách cà phê chill",
    "food_tour": "Food tour",
    "photo_spots": "Điểm chụp ảnh đẹp",
    "compare": "So sánh lựa chọn",
    "guide": "Gợi ý đi đâu làm gì",
}
