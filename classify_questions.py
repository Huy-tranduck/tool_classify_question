"""
Script phân loại câu hỏi người dùng theo Use Case BRD - Trợ lý ảo Bộ Tài chính.
Sử dụng FPT Cloud API (Qwen3-32B) để phân loại từng batch câu hỏi.
"""

import csv
import json
import re
import time
import os
import requests
from datetime import datetime

# ============================================================
# CẤU HÌNH
# ============================================================
API_URL = "https://mkp-api.fptcloud.com/chat/completions"
API_KEY = ""  
MODEL = "Qwen3-32B"
BATCH_SIZE = 10  # Số câu hỏi mỗi batch gửi đến LLM
MAX_RETRIES = 3  # Số lần retry khi API lỗi
RETRY_DELAY = 5  # Giây chờ giữa các lần retry
REQUEST_DELAY = 1  # Giây chờ giữa các batch để tránh rate limit

INPUT_CSV = "log_chat_test.csv"
OUTPUT_CSV = "classified_questions.csv"
PROGRESS_FILE = "classification_progress.json"  # Lưu tiến trình để resume

# ============================================================
# SYSTEM PROMPT CHO LLM
# ============================================================
SYSTEM_PROMPT = """Bạn là chuyên gia phân loại câu hỏi cho hệ thống Trợ lý ảo Bộ Tài chính (FinGov).

## Nhiệm vụ
Với mỗi câu hỏi đầu vào, hãy phân loại vào đúng Use Case theo bảng phân loại bên dưới.
Sử dụng **TÊN USE CASE** (cột "Tên") khi trả kết quả.
Phân loại trực tiếp dựa trên nội dung câu hỏi, kể cả câu hỏi follow-up.

## Danh sách đơn vị trực thuộc Bộ Tài chính (dùng để nhận diện UC2/UC8)
Vụ Ngân sách nhà nước, Vụ Phát triển hạ tầng, Vụ Giám sát và Thẩm định đầu tư, Vụ Tài chính - Kinh tế ngành, Vụ Quốc phòng, an ninh, đặc biệt (Vụ I), Vụ Kinh tế địa phương và lãnh thổ, Vụ Quản lý quy hoạch, Vụ Các định chế tài chính, Vụ Tổ chức cán bộ, Vụ Pháp chế, Văn phòng, Cục Quản lý nợ và Kinh tế đối ngoại, Cục Quản lý công sản, Cục Quản lý đấu thầu, Cục Quản lý, giám sát chính sách thuế, phí và lệ phí, Cục Quản lý, giám sát bảo hiểm, Cục Quản lý, giám sát kế toán, kiểm toán, Cục Quản lý giá, Cục Phát triển doanh nghiệp nhà nước, Cục Phát triển doanh nghiệp tư nhân và kinh tế tập thể, Cục Đầu tư nước ngoài, Cục Kế hoạch - Tài chính, Cục Công nghệ thông tin và chuyển đổi số, Cục Thuế, Cục Hải quan, Cục Dự trữ Nhà nước, Cục Thống kê, Kho bạc Nhà nước, Ủy ban Chứng khoán Nhà nước, Viện Chiến lược và Chính sách kinh tế - tài chính, Báo Tài chính - Đầu tư, Tạp chí Kinh tế - Tài chính, Trường Bồi dưỡng cán bộ Kinh tế - Tài chính, Bảo hiểm xã hội Việt Nam, Học viện Tài chính, Trường Đại học Tài chính - Marketing, Học viện Chính sách và Phát triển, Trường Đại học Tài chính - Quản trị kinh doanh, Trường Đại học Tài chính - Kế toán, Trường Cao đẳng Kinh tế - Kế hoạch Đà Nẵng, Nhà xuất bản Kinh tế - Tài chính, Trung tâm Đổi mới sáng tạo Quốc gia.

## Thông tin lãnh đạo chủ chốt (dùng để nhận diện UC3)
- Lãnh đạo Chính phủ: Tổng bí thư & Chủ tịch nước: Tô Lâm; Thủ tướng: Lê Minh Hưng; Chủ tịch QH: Trần Thanh Mẫn; Thường trực BBT: Trần Cẩm Tú
- Bộ trưởng BTC: Ngô Văn Tuấn
- Thứ trưởng: Nguyễn Thị Bích Ngọc, Nguyễn Đức Chi, Cao Anh Tuấn, Lê Tấn Cận, Trần Quốc Phương, Nguyễn Đức Tâm, Tạ Anh Tuấn

## Bảng Use Case

| UC Code | Tên | Mô tả | Từ khóa gợi ý |
|---------|-----|-------|---------------|
| UC1 | Thông tin giới thiệu BTC | Định nghĩa, chức năng, cơ cấu, liên hệ Bộ Tài chính | "Bộ Tài chính", "chức năng", "trụ sở", "giới thiệu" |
| UC2 | Thông tin đơn vị trực thuộc | Cục, Vụ, đơn vị trực thuộc BTC (xem danh sách ở trên) | Tên Cục/Vụ cụ thể |
| UC2.1 | Tình huống nhiệm vụ đơn vị | Hỏi "liên hệ ai", "gọi đơn vị nào" theo tình huống | "liên hệ ai", "cơ quan nào phụ trách" |
| UC3 | Thông tin lãnh đạo | Họ tên, tiểu sử, chức vụ lãnh đạo CP/BTC/đơn vị | "lãnh đạo", "Bộ trưởng", "Thứ trưởng", "tiểu sử" |
| UC4.1 | Tra cứu TTHC | Mã, hồ sơ, lệ phí thủ tục | "mã thủ tục", "hồ sơ gồm", "lệ phí" |
| UC4.2 | Tình huống TTHC | Phân tích tình huống → chỉ thủ tục | tình huống cá nhân + "cần làm gì" |
| UC4.3 | Cập nhật TTHC | Quy định mới về thủ tục | "quy định mới", "thay đổi thủ tục" |
| UC4.4 | Hướng dẫn cổng DVC | Thao tác trên cổng dịch vụ công, eTax, ứng dụng thuế | "cổng DVC", "nộp hồ sơ online", "eTax" |
| UC5.1 | Tình huống chính sách | CS áp dụng cho trường hợp cụ thể | tình huống + "thuế", "ưu đãi" |
| UC5.2 | Tra cứu & cập nhật CS | CS hiện hành, thông tin CS mới | "chính sách hiện hành", "quy định về" |
| UC6.1 | Tra cứu văn bản PL | Tìm theo mã/tên/nội dung | mã VB cụ thể, "tóm tắt", "link tải" |
| UC6.2 | VB mới ban hành | VB mới, sửa đổi | "vừa ban hành", "mới ra" |
| UC7 | Tin tức & Nội dung khác | Giá, tỷ giá, tin hoạt động, chứng khoán | "giá vàng", "tỷ giá", "tin tức" |
| UC8 | Đơn vị thuộc Bộ (Tổng cục) | Ánh xạ về UC target | "Tổng cục Thuế", "Hải quan", "Kho bạc" |
| UC9.1 | Small talk | Chào hỏi, cảm ơn | "xin chào", "cảm ơn" |
| UC9.2 | Năng lực BOT | Hỏi bot làm gì, ai phát triển | "bạn là ai", "ai phát triển" |
| UC9.3 | Phản hồi/Lỗi | Phàn nàn sai, chậm | "sai rồi", "bị lỗi" |
| UC9.4 | Cảm xúc | Góp ý tích cực/tiêu cực | "bực mình", "rất tốt" |
| UC10 | Ngoài phạm vi (Từ chối) | Không liên quan BTC, thuộc Bộ khác | không liên quan tài chính |

## Quy tắc phân loại

1. **Ưu tiên ý định chính** của user khi phân loại
2. Nếu câu hỏi thuộc sub use case, ghi CẢ main UC và sub UC (dùng tên UC)
3. Nếu câu hỏi liên quan đến Tổng cục/Kho bạc (Đơn vị thuộc Bộ), ghi cả UC chính và UC target
4. Phân loại trực tiếp dựa trên nội dung câu hỏi — **kể cả câu follow-up**, không cần xử lý context
5. Khi không chắc chắn, chọn UC phù hợp nhất và ghi lý do

## Ví dụ phân loại

Input: "địa chỉ cục công nghệ thông tin chuyển đổi số"
→ {"main_use_case": "Thông tin đơn vị trực thuộc", "sub_use_case": null, "classification_reason": "Hỏi địa chỉ đơn vị trực thuộc BTC"}

Input: "cảm ơn bạn, tốt lắm"
→ {"main_use_case": "Small talk", "sub_use_case": null, "classification_reason": "Small talk - cảm ơn"}

Input: "liệt kê các lãnh đạo vụ tài chính kinh tế ngành"
→ {"main_use_case": "Thông tin lãnh đạo", "sub_use_case": null, "classification_reason": "Hỏi danh sách lãnh đạo đơn vị trực thuộc BTC"}

Input: "Thu nhập 35tr/tháng, 2 người phụ thuộc. Thuế TNCN bao nhiêu?"
→ {"main_use_case": "Tình huống chính sách", "sub_use_case": null, "classification_reason": "Tình huống cá nhân hỏi cách tính thuế TNCN"}

Input: "Mã ĐVQHNS chưa đồng bộ Tabmis Kho bạc thì sao?"
→ {"main_use_case": "Đơn vị thuộc Bộ (Tổng cục)", "sub_use_case": "Tình huống nhiệm vụ đơn vị", "classification_reason": "Liên quan Kho bạc, tình huống hỏi hỗ trợ kỹ thuật"}

Input: "chiến tranh thế giới hiện nay"
→ {"main_use_case": "Ngoài phạm vi (Từ chối)", "sub_use_case": null, "classification_reason": "Không liên quan BTC, từ chối"}

Input: "bạn được phát triển bởi ai?"
→ {"main_use_case": "Năng lực BOT", "sub_use_case": null, "classification_reason": "Hỏi về nguồn gốc/năng lực bot"}

Input: "Tôi muốn hỏi địa chỉ mail để gửi thư khiếu kiện công ty bảo hiểm"
→ {"main_use_case": "Thông tin đơn vị trực thuộc", "sub_use_case": "Tình huống nhiệm vụ đơn vị", "classification_reason": "Tình huống cần xác định đơn vị phụ trách (Cục QLGS bảo hiểm)"}
"""


# ============================================================
# HÀM ĐỌC CSV VÀ TRÍCH XUẤT TẤT CẢ TIN NHẮN
# ============================================================
def load_all_messages(csv_path: str) -> list[dict]:
    """Đọc toàn bộ CSV và trích xuất tất cả tin nhắn (của user và bot)."""
    messages = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sender = row.get("Sender Name", "").strip()
            messages.append({
                "time": row.get("Time (UTC)", ""),
                "conversation_id": row.get("Conversation ID", ""),
                "message_id": row.get("Message ID", ""),
                "sender_name": sender,
                "message_content": row.get("Message Content", "").strip(),
                "is_follow_up": row.get("is_follow_up", "False"),
                "is_follow_up_unmapped": row.get("is_follow_up_unmapped", "False"),
            })
    return messages


# ============================================================
# HÀM GỌI API LLM
# ============================================================
def call_llm_api(messages: list[dict], api_key: str = API_KEY) -> str:
    """Gọi FPT Cloud API và trả về text response (đã strip <think> tags)."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.1,  # Giảm temperature để phân loại nhất quán hơn
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                API_URL, headers=headers, data=json.dumps(data), timeout=120
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            # Xóa <think> tags từ model reasoning
            text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
            return text
        except requests.exceptions.RequestException as e:
            print(f"  ⚠ API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


# ============================================================
# HÀM XÂY DỰNG USER PROMPT CHO BATCH
# ============================================================
def build_batch_prompt(batch: list[dict]) -> str:
    """Tạo user prompt cho một batch câu hỏi."""
    prompt = "Hãy phân loại các câu hỏi sau vào Use Case tương ứng.\n"
    prompt += "Trả lời ĐÚNG ĐỊNH DẠNG JSON ARRAY, mỗi phần tử là một object.\n"
    prompt += "CHỈ trả về JSON array, KHÔNG kèm markdown code block hay text giải thích.\n\n"

    for i, q in enumerate(batch, 1):
        prompt += f"### Câu {i}\n"
        prompt += f'- conversation_id: "{q["conversation_id"]}"\n'
        prompt += f'- message_content: "{q["message_content"]}"\n'
        prompt += f'- is_follow_up: {q["is_follow_up"]}\n\n'

    prompt += """Trả về JSON array với format cho mỗi câu:
{
  "conversation_id": "<id>",
  "message_content": "<câu hỏi>",
  "is_follow_up": true/false,
  "main_use_case": "<Tên Use Case>",
  "sub_use_case": "<Tên Sub Use Case>" hoặc null,
  "classification_reason": "<Lý do phân loại ngắn gọn>"
}"""
    return prompt


# ============================================================
# HÀM PARSE KẾT QUẢ JSON TỪ LLM
# ============================================================
def parse_llm_response(response_text: str, batch: list[dict]) -> list[dict]:
    """Parse JSON response từ LLM, xử lý các trường hợp format lỗi."""
    # Thử tìm JSON array trong response
    # Loại bỏ markdown code block nếu có
    cleaned = re.sub(r"```json\s*", "", response_text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    try:
        results = json.loads(cleaned)
        if isinstance(results, list):
            return results
    except json.JSONDecodeError:
        pass

    # Thử tìm JSON array bằng regex
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            results = json.loads(match.group())
            if isinstance(results, list):
                return results
        except json.JSONDecodeError:
            pass

    # Thử tìm từng JSON object riêng lẻ
    objects = re.findall(r"\{[^{}]*\}", cleaned)
    if objects:
        results = []
        for obj_str in objects:
            try:
                obj = json.loads(obj_str)
                results.append(obj)
            except json.JSONDecodeError:
                continue
        if results:
            return results

    # Fallback: trả về danh sách với classification lỗi
    print(f"  ⚠ Không parse được JSON từ LLM response. Đánh dấu ERROR.")
    return [
        {
            "conversation_id": q["conversation_id"],
            "message_content": q["message_content"],
            "is_follow_up": q["is_follow_up"],
            "main_use_case": "ERROR",
            "sub_use_case": None,
            "classification_reason": "LLM response parse error",
        }
        for q in batch
    ]


# ============================================================
# LƯU / ĐỌC TIẾN TRÌNH
# ============================================================
def save_progress(processed_count: int, results: list[dict]):
    """Lưu tiến trình phân loại để có thể resume."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"processed_count": processed_count, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_progress() -> tuple[int, list[dict]]:
    """Đọc tiến trình đã lưu (nếu có)."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("processed_count", 0), data.get("results", [])
    return 0, []


# ============================================================
# HÀM XUẤT KẾT QUẢ RA CSV
# ============================================================
def export_results_csv(
    messages: list[dict], classifications: list[dict], output_path: str
):
    """Xuất kết quả phân loại ra file CSV, bao gồm cả tin nhắn của bot."""
    # Build lookup: conversation_id + message_content → classification
    classification_map = {}
    for c in classifications:
        key = (c.get("conversation_id", ""), c.get("message_content", ""))
        classification_map[key] = c

    fieldnames = [
        "time",
        "conversation_id",
        "message_id",
        "sender_name",
        "message_content",
        "is_follow_up",
        "is_follow_up_unmapped",
        "main_use_case",
        "sub_use_case",
        "classification_reason",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in messages:
            if q["sender_name"]:
                # Là user
                key = (q["conversation_id"], q["message_content"])
                c = classification_map.get(key, {})
                row = {
                    "time": q["time"],
                    "conversation_id": q["conversation_id"],
                    "message_id": q["message_id"],
                    "sender_name": q["sender_name"],
                    "message_content": q["message_content"],
                    "is_follow_up": q["is_follow_up"],
                    "is_follow_up_unmapped": q["is_follow_up_unmapped"],
                    "main_use_case": c.get("main_use_case", "UNMATCHED"),
                    "sub_use_case": c.get("sub_use_case") or "",
                    "classification_reason": c.get("classification_reason", ""),
                }
            else:
                # Là bot, không có classification
                row = {
                    "time": q["time"],
                    "conversation_id": q["conversation_id"],
                    "message_id": q["message_id"],
                    "sender_name": "",
                    "message_content": q["message_content"],
                    "is_follow_up": q["is_follow_up"],
                    "is_follow_up_unmapped": q["is_follow_up_unmapped"],
                    "main_use_case": "",
                    "sub_use_case": "",
                    "classification_reason": "",
                }
            writer.writerow(row)

    print(f"\n✅ Kết quả đã xuất ra: {output_path}")


# ============================================================
# HÀM CHÍNH
# ============================================================
def main():
    print("=" * 60)
    print("PHÂN LOẠI CÂU HỎI THEO USE CASE BRD - BỘ TÀI CHÍNH")
    print("=" * 60)

    # 1. Đọc dữ liệu
    print(f"\n📂 Đọc file CSV: {INPUT_CSV}")
    all_messages = load_all_messages(INPUT_CSV)
    
    # Chỉ lấy các tin nhắn của user để phân loại
    user_questions = [msg for msg in all_messages if msg["sender_name"]]
    print(f"   Tổng số tin nhắn: {len(all_messages)}, trong đó user: {len(user_questions)}")

    # 2. Kiểm tra tiến trình cũ
    start_idx, all_results = load_progress()
    if start_idx > 0:
        print(f"\n🔄 Tiếp tục từ câu {start_idx + 1}/{len(user_questions)} (đã xử lý {start_idx})")
    else:
        print(f"\n🚀 Bắt đầu phân loại {len(user_questions)} câu hỏi...")

    # 3. Phân loại theo batch
    total_batches = (len(user_questions) - start_idx + BATCH_SIZE - 1) // BATCH_SIZE
    batch_num = 0

    for i in range(start_idx, len(user_questions), BATCH_SIZE):
        batch = user_questions[i : i + BATCH_SIZE]
        batch_num += 1

        print(
            f"\n📦 Batch {batch_num}/{total_batches} "
            f"(câu {i + 1}-{min(i + BATCH_SIZE, len(user_questions))})"
        )

        # Xây dựng prompt
        user_prompt = build_batch_prompt(batch)

        # Gọi API
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response_text = call_llm_api(messages)
            batch_results = parse_llm_response(response_text, batch)

            # Đảm bảo số lượng kết quả khớp batch
            if len(batch_results) < len(batch):
                print(
                    f"  ⚠ LLM trả về {len(batch_results)} kết quả, "
                    f"cần {len(batch)}. Bổ sung ERROR cho phần thiếu."
                )
                for j in range(len(batch_results), len(batch)):
                    batch_results.append({
                        "conversation_id": batch[j]["conversation_id"],
                        "message_content": batch[j]["message_content"],
                        "is_follow_up": batch[j]["is_follow_up"],
                        "main_use_case": "ERROR",
                        "sub_use_case": None,
                        "classification_reason": "Missing from LLM response",
                    })

            all_results.extend(batch_results)

            # Hiển thị kết quả ngắn gọn
            for r in batch_results:
                uc = r.get("main_use_case", "?")
                sub = r.get("sub_use_case") or ""
                msg = (r.get("message_content", "")[:50] + "...") if len(r.get("message_content", "")) > 50 else r.get("message_content", "")
                uc_display = uc
                if sub:
                    uc_display += f" > {sub}"
                print(f"  ✓ [{uc_display}] {msg}")

        except Exception as e:
            print(f"  ❌ Batch {batch_num} thất bại: {e}")
            # Fallback cho cả batch
            for q in batch:
                all_results.append({
                    "conversation_id": q["conversation_id"],
                    "message_content": q["message_content"],
                    "is_follow_up": q["is_follow_up"],
                    "main_use_case": "ERROR",
                    "sub_use_case": None,
                    "classification_reason": f"API error: {str(e)[:100]}",
                })

        # Lưu tiến trình sau mỗi batch
        processed = i + len(batch)
        save_progress(processed, all_results)

        # Delay giữa các batch
        if i + BATCH_SIZE < len(user_questions):
            time.sleep(REQUEST_DELAY)

    # 4. Xuất kết quả
    export_results_csv(all_messages, all_results, OUTPUT_CSV)

    # 5. Thống kê tổng hợp
    print("\n📊 THỐNG KÊ PHÂN LOẠI:")
    print("-" * 40)
    uc_counts = {}
    for r in all_results:
        uc = r.get("main_use_case", "UNKNOWN")
        sub = r.get("sub_use_case") or ""
        key = f"{uc} > {sub}" if sub else uc
        uc_counts[key] = uc_counts.get(key, 0) + 1

    for uc, count in sorted(uc_counts.items()):
        pct = count / len(all_results) * 100
        print(f"  {uc:30s} : {count:4d} ({pct:5.1f}%)")

    print(f"\n  {'TỔNG':30s} : {len(all_results):4d}")

    # 6. Dọn file tiến trình
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        print(f"\n🧹 Đã xóa file tiến trình: {PROGRESS_FILE}")

    print("\n✅ HOÀN TẤT!")


# ============================================================
# CÁC HÀM CHO STREAMLIT DASHBOARD
# ============================================================
def check_api_key(api_key: str) -> bool:
    """Kiểm tra API Key có hợp lệ không."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 10
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def run_classification_pipeline(user_questions: list[dict], api_key: str, batch_size: int = BATCH_SIZE):
    """
    Generator để chạy phân loại và trả về tiến trình.
    Yields dict: {'processed': int, 'total': int, 'results': list[dict]}
    """
    total = len(user_questions)
    all_results = []
    
    for i in range(0, total, batch_size):
        batch = user_questions[i : i + batch_size]
        user_prompt = build_batch_prompt(batch)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response_text = call_llm_api(messages, api_key=api_key)
            batch_results = parse_llm_response(response_text, batch)
            
            if len(batch_results) < len(batch):
                for j in range(len(batch_results), len(batch)):
                    batch_results.append({
                        "conversation_id": batch[j]["conversation_id"],
                        "message_content": batch[j]["message_content"],
                        "is_follow_up": batch[j]["is_follow_up"],
                        "main_use_case": "ERROR",
                        "sub_use_case": None,
                        "classification_reason": "Missing from LLM response",
                    })
            all_results.extend(batch_results)
        except Exception as e:
            for q in batch:
                all_results.append({
                    "conversation_id": q["conversation_id"],
                    "message_content": q["message_content"],
                    "is_follow_up": q["is_follow_up"],
                    "main_use_case": "ERROR",
                    "sub_use_case": None,
                    "classification_reason": f"API error: {str(e)[:100]}",
                })
        
        yield {
            "processed": min(i + batch_size, total),
            "total": total,
            "results": all_results.copy()
        }
        
        if i + batch_size < total:
            time.sleep(REQUEST_DELAY)

if __name__ == "__main__":
    main()
