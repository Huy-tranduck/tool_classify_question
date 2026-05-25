"""
Script test riêng phần preprocess_data + đếm conversation/message.
Chạy: python test_preprocess.py <file.csv>
"""
import pandas as pd
import re
import sys
import os


def preprocess_data(df):
    remove_msg = """Chào bạn, tôi là trợ lý ảo của Bộ Tài chính. Rất vui được hỗ trợ bạn tra cứu thông tin về Bộ, các đơn vị trực thuộc Bộ, các thủ tục hành chính, hỏi đáp chính sách và tin tức mới nhất của ngành Tài chính. \n\nBạn cần hỗ trợ vấn đề gì hôm nay? \n\nLưu ý: Đây là phiên bản thử nghiệm từ ngày 01/04/2026. Trong quá trình sử dụng, nếu có điểm chưa hoàn thiện, mong bạn thông cảm và đóng góp ý kiến để hệ thống được cải thiện tốt hơn!"""

    total_raw = len(df)
    total_conv_raw = df['Conversation ID'].nunique()
    sender_raw = df['Sender Name'].fillna('').astype(str).str.strip()
    total_user_raw = ((sender_raw != '') & (sender_raw.str.lower() != 'nan')).sum()

    # 1. Drop các cột không cần thiết (giữ lại tất cả row kể cả live-chat)
    if 'Sender ID' in df.columns:
        df = df.drop(columns=['Sender ID', 'Channel', 'From'], errors='ignore')

    # 2. Loại bỏ row [get_started] và welcome message
    mask_get_started = df['Message Content'] == '[get_started]'
    mask_welcome = df['Message Content'] == remove_msg
    removed_system = (mask_get_started | mask_welcome).sum()
    df = df[~mask_get_started & ~mask_welcome]

    # 3. Loại bỏ conversation chỉ còn bot message (không có user message thực sự)
    #    Conversation hợp lệ = có ít nhất 1 message từ user (Sender Name không rỗng)
    sender_filled = df['Sender Name'].fillna('').astype(str).str.strip()
    convs_with_user = df.loc[
        (sender_filled != '') & (sender_filled.str.lower() != 'nan'),
        'Conversation ID'
    ].unique()

    convs_before = df['Conversation ID'].nunique()
    df = df[df['Conversation ID'].isin(convs_with_user)]
    convs_removed_empty = convs_before - df['Conversation ID'].nunique()

    # Sort
    df['Time_DT'] = pd.to_datetime(df['Time (UTC)'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df = df.sort_values(by=['Conversation ID', 'Time_DT']).drop(columns=['Time_DT']).reset_index(drop=True)

    # Follow-up detection
    question_pattern = re.compile(r'Bạn có muốn\s+(.*?)\s+không\?', re.IGNORECASE)
    short_answers = ['có', 'không', 'có nhé', 'không nhé', 'có ạ', 'không ạ', 'vâng', 'được', 'ok', 'yes', 'no']

    df['is_follow_up'] = False
    df['is_follow_up_unmapped'] = False

    for i in range(1, len(df)):
        current_content = str(df.loc[i, 'Message Content']).strip().lower()
        current_conv_id = df.loc[i, 'Conversation ID']
        prev_content = str(df.loc[i-1, 'Message Content'])
        prev_conv_id = df.loc[i-1, 'Conversation ID']
        prev_sender = str(df.loc[i-1, 'Sender Name'])

        if current_conv_id == prev_conv_id:
            if pd.isna(df.loc[i-1, 'Sender Name']) or prev_sender.strip() == '' or prev_sender.lower() == 'nan':
                if current_content in short_answers:
                    df.loc[i, 'is_follow_up'] = True
                    matches = question_pattern.findall(prev_content)
                    if matches:
                        topic = matches[-1].strip()
                        df.loc[i, 'Message Content'] = f"Tôi muốn {topic}?"
                else:
                    tokens = re.findall(r"\b\w+\b", current_content)
                    if len(tokens) == 1:
                        df.loc[i, 'is_follow_up_unmapped'] = True

    stats = {
        "total_raw": total_raw,
        "total_conv_raw": total_conv_raw,
        "total_user_raw": total_user_raw,
        "removed_system": removed_system,
        "convs_removed_empty": convs_removed_empty,
    }
    return df, stats


def main():
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        import glob
        pattern = "/home/huytranduck/Documents/FPT_FCI/BTC_AA/History_*_20260515_042152.csv"
        matches = glob.glob(pattern)
        csv_path = matches[0] if matches else "test_uc10_refusal.csv"

    if not os.path.exists(csv_path):
        print(f"❌ File không tồn tại: {csv_path}")
        sys.exit(1)

    print(f"📂 Đọc file: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print("  TRƯỚC KHI XỬ LÝ")
    print(f"{'='*60}")
    print(f"  Tổng số row          : {len(df)}")
    print(f"  Tổng conversation    : {df['Conversation ID'].nunique()}")
    sender_raw = df['Sender Name'].fillna('').astype(str).str.strip()
    user_raw = ((sender_raw != '') & (sender_raw.str.lower() != 'nan')).sum()
    print(f"  Tổng message user    : {user_raw}")
    if 'Channel' in df.columns:
        print(f"  Channels             : {df['Channel'].value_counts().to_dict()}")
    print(f"  Columns              : {list(df.columns)}")

    processed_df, stats = preprocess_data(df)

    # Map messages
    mapped_messages = []
    for _, row in processed_df.iterrows():
        sender_name = str(row.get("Sender Name", "")).strip()
        if sender_name.lower() == 'nan':
            sender_name = ""
        mapped_messages.append({
            "conversation_id": row.get("Conversation ID", ""),
            "sender_name": sender_name,
            "message_content": str(row.get("Message Content", "")).strip(),
            "is_follow_up": row.get("is_follow_up", False),
            "is_follow_up_unmapped": row.get("is_follow_up_unmapped", False),
        })

    user_questions = [msg for msg in mapped_messages if msg["sender_name"]]
    total_conversations = len(set(msg["conversation_id"] for msg in user_questions if msg["conversation_id"]))

    print(f"\n{'='*60}")
    print("  QUÁ TRÌNH LỌC")
    print(f"{'='*60}")
    print(f"  Row system bị loại    : {stats['removed_system']} ([get_started] + welcome)")
    print(f"  Conv rỗng bị loại     : {stats['convs_removed_empty']} (chỉ có [get_started]/welcome, không có user message)")
    print(f"  User msg bị loại      : {stats['total_user_raw'] - len(user_questions)} (thuộc conv rỗng)")
    print(f"  User msg trước lọc     : {stats['total_user_raw']}")
    print(f"  User msg sau lọc       : {len(user_questions)}")

    print(f"\n{'='*60}")
    print("  SAU KHI XỬ LÝ")
    print(f"{'='*60}")
    print(f"  Tổng message (sau lọc)  : {len(mapped_messages)}")
    print(f"  Tổng conversation       : {total_conversations}")
    print(f"  Tổng câu hỏi user       : {len(user_questions)}")

    follow_ups = sum(1 for m in mapped_messages if m["is_follow_up"])
    unmapped = sum(1 for m in mapped_messages if m["is_follow_up_unmapped"])
    print(f"  Follow-up (mapped)       : {follow_ups}")
    print(f"  Follow-up (unmapped)     : {unmapped}")

    # Preview
    print(f"\n{'='*60}")
    print("  PREVIEW 10 CÂU HỎI USER ĐẦU TIÊN")
    print(f"{'='*60}")
    for i, q in enumerate(user_questions[:10], 1):
        content = q["message_content"][:60] + "..." if len(q["message_content"]) > 60 else q["message_content"]
        fu = " [follow-up]" if q["is_follow_up"] else ""
        print(f"  {i:2d}. [{q['conversation_id'][:12]}...] {content}{fu}")

    print(f"\n✅ Test hoàn tất!")


if __name__ == "__main__":
    main()
