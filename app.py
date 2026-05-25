from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
import pandas as pd
import io
import os
import re
import time
import json
import uuid
import shutil
import asyncio
from datetime import datetime, timezone
from classify_questions import check_api_key, run_classification_pipeline

app = FastAPI()

import sys

# Support for PyInstaller
def get_base_path():
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        return sys._MEIPASS
    except Exception:
        return os.path.dirname(os.path.abspath(__file__))

def get_app_dir():
    """Get the directory of the executable or current script"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

# Global dict to store tasks (for simplicity)
tasks = {}

# History directory (always save relative to the executable, not temp folder)
HISTORY_DIR = os.path.join(get_app_dir(), "history")


# ─── History Helpers ───────────────────────────────────────────────────────────

def ensure_history_dir():
    """Create history directory if it doesn't exist."""
    os.makedirs(HISTORY_DIR, exist_ok=True)


def save_task_to_history(task_id: str, task: dict, original_filename: str):
    """Save a completed task's results to disk for persistence."""
    ensure_history_dir()
    task_dir = os.path.join(HISTORY_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)

    # Save classified CSV
    classified_path = os.path.join(task_dir, "classified_questions.csv")
    with open(classified_path, "w", encoding="utf-8-sig") as f:
        f.write(task["classified_csv"])

    # Save stats CSV
    stats_path = os.path.join(task_dir, "classification_stats.csv")
    with open(stats_path, "w", encoding="utf-8-sig") as f:
        f.write(task["stats_csv"])

    # Save metadata
    metadata = {
        "task_id": task_id,
        "original_filename": original_filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_questions": task.get("total", 0),
        "total_messages": len(task.get("mapped_messages", [])),
        "stats": task.get("stats_json", []),
    }
    metadata_path = os.path.join(task_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Update index
    _update_history_index(task_id, metadata)


def _update_history_index(task_id: str, metadata: dict):
    """Add or update an entry in the history index file."""
    index_path = os.path.join(HISTORY_DIR, "index.json")
    index = []
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
        except (json.JSONDecodeError, IOError):
            index = []

    # Remove existing entry with same task_id (if re-saving)
    index = [e for e in index if e.get("task_id") != task_id]

    # Add summary entry
    index.insert(0, {
        "task_id": task_id,
        "original_filename": metadata["original_filename"],
        "created_at": metadata["created_at"],
        "total_questions": metadata["total_questions"],
        "total_messages": metadata["total_messages"],
    })

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def load_history_index():
    """Load the history index from disk."""
    index_path = os.path.join(HISTORY_DIR, "index.json")
    if not os.path.exists(index_path):
        return []
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def load_history_entry(task_id: str):
    """Load full metadata for a single history entry."""
    metadata_path = os.path.join(HISTORY_DIR, task_id, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def delete_history_entry(task_id: str):
    """Delete a history entry from disk and index."""
    task_dir = os.path.join(HISTORY_DIR, task_id)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)

    # Update index
    index_path = os.path.join(HISTORY_DIR, "index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            index = [e for e in index if e.get("task_id") != task_id]
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
        except (json.JSONDecodeError, IOError):
            pass


# ─── Data Preprocessing ───────────────────────────────────────────────────────

def preprocess_data(df):
    remove_msg = """Chào bạn, tôi là trợ lý ảo của Bộ Tài chính. Rất vui được hỗ trợ bạn tra cứu thông tin về Bộ, các đơn vị trực thuộc Bộ, các thủ tục hành chính, hỏi đáp chính sách và tin tức mới nhất của ngành Tài chính. \n\nBạn cần hỗ trợ vấn đề gì hôm nay? \n\nLưu ý: Đây là phiên bản thử nghiệm từ ngày 01/04/2026. Trong quá trình sử dụng, nếu có điểm chưa hoàn thiện, mong bạn thông cảm và đóng góp ý kiến để hệ thống được cải thiện tốt hơn!"""

    if 'Sender ID' in df.columns:
        df = df.drop(columns=['Sender ID', 'Channel', 'From'], errors='ignore')
    
    df = df[(df['Message Content'] != '[get_started]') & (df['Message Content'] != remove_msg)]

    # Loại bỏ conversation chỉ còn bot message (không có user message thực sự)
    sender_filled = df['Sender Name'].fillna('').astype(str).str.strip()
    convs_with_user = df.loc[
        (sender_filled != '') & (sender_filled.str.lower() != 'nan'),
        'Conversation ID'
    ].unique()
    df = df[df['Conversation ID'].isin(convs_with_user)]

    df['Time_DT'] = pd.to_datetime(df['Time (UTC)'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df = df.sort_values(by=['Conversation ID', 'Time_DT']).drop(columns=['Time_DT']).reset_index(drop=True)
    
    question_pattern = re.compile(r'Bạn có muốn\s+(.*?)\s+không\?', re.IGNORECASE)

    short_answers = [
        'có', 'không', 'có nhé', 'không nhé', 
        'có ạ', 'không ạ', 'vâng', 'được', 'ok', 'yes', 'no'
    ]

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
    return df


# ─── Page Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(get_base_path(), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/viewer", response_class=HTMLResponse)
async def get_viewer():
    try:
        viewer_path = os.path.join(get_base_path(), "csv_viewer.html")
        with open(viewer_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Không tìm thấy file csv_viewer.html"


# ─── Classification API ───────────────────────────────────────────────────────

@app.post("/api/check_key")
async def api_check_key(api_key: str = Form(...)):
    is_valid = check_api_key(api_key)
    return {"valid": is_valid}

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), api_key: str = Form(...)):
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
        processed_df = preprocess_data(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi khi đọc file: {str(e)}")
    
    all_messages_raw = processed_df.to_dict('records')
    mapped_messages = []
    
    for row in all_messages_raw:
        sender_name = str(row.get("Sender Name", "")).strip()
        if sender_name.lower() == 'nan':
            sender_name = ""
            
        mapped_messages.append({
            "time": row.get("Time (UTC)", ""),
            "conversation_id": row.get("Conversation ID", ""),
            "message_id": row.get("Message ID", ""),
            "sender_name": sender_name,
            "message_content": str(row.get("Message Content", "")).strip(),
            "is_follow_up": row.get("is_follow_up", False),
            "is_follow_up_unmapped": row.get("is_follow_up_unmapped", False),
        })
    
    user_questions = [msg for msg in mapped_messages if msg["sender_name"]]
    
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "api_key": api_key,
        "mapped_messages": mapped_messages,
        "user_questions": user_questions,
        "progress": 0,
        "total": len(user_questions),
        "results": [],
        "stats": [],
        "original_filename": file.filename or "unknown.csv",
    }
    
    return {"task_id": task_id, "total_messages": len(mapped_messages), "user_questions_count": len(user_questions)}

async def classification_generator(task_id: str):
    task = tasks.get(task_id)
    if not task:
        yield f"data: {json.dumps({'error': 'Task not found'})}\n\n"
        return
        
    start_time = time.time()
    generator = run_classification_pipeline(task["user_questions"], task["api_key"])
    
    def get_next():
        try:
            return next(generator)
        except StopIteration:
            return None
            
    while True:
        progress_data = await asyncio.to_thread(get_next)
        if progress_data is None:
            break
            
        processed = progress_data['processed']
        total = progress_data['total']
        all_results = progress_data['results']
        
        task['progress'] = processed
        task['results'] = all_results
        
        elapsed = time.time() - start_time
        eta = (elapsed / processed) * (total - processed) if processed > 0 else 0
        
        yield f"data: {json.dumps({'status': 'processing', 'processed': processed, 'total': total, 'eta': round(eta, 1)})}\n\n"
        
    # Done processing, prepare final files
    all_results = task['results']
    mapped_messages = task['mapped_messages']
    
    classification_map = {}
    for c in all_results:
        key = (c.get("conversation_id", ""), c.get("message_content", ""))
        classification_map[key] = c
        
    final_rows = []
    for q in mapped_messages:
        row = {
            "Time (UTC)": q["time"],
            "Conversation ID": q["conversation_id"],
            "Message ID": q["message_id"],
            "Sender Name": q["sender_name"],
            "Message Content": q["message_content"],
            "is_follow_up": q["is_follow_up"],
            "is_follow_up_unmapped": q["is_follow_up_unmapped"]
        }
        if q["sender_name"]:
            key = (q["conversation_id"], q["message_content"])
            c = classification_map.get(key, {})
            row.update({
                "main_use_case": c.get("main_use_case", "UNMATCHED"),
                "sub_use_case": c.get("sub_use_case", ""),
                "classification_reason": c.get("classification_reason", "")
            })
        else:
            row.update({
                "main_use_case": "",
                "sub_use_case": "",
                "classification_reason": ""
            })
        final_rows.append(row)
        
    final_df = pd.DataFrame(final_rows)
    task['classified_csv'] = final_df.to_csv(index=False, encoding="utf-8-sig")
    
    uc_counts = {}
    for r in all_results:
        uc = r.get("main_use_case", "UNKNOWN")
        sub = r.get("sub_use_case") or ""
        key = f"{uc} > {sub}" if sub else uc
        uc_counts[key] = uc_counts.get(key, 0) + 1
        
    stats_df = pd.DataFrame([
        {"Use Case": k, "Số lượng": v, "Tỷ lệ (%)": round(v / len(all_results) * 100, 2)}
        for k, v in uc_counts.items()
    ]).sort_values(by="Số lượng", ascending=False)
    
    task['stats_csv'] = stats_df.to_csv(index=False, encoding="utf-8-sig")
    task['stats_json'] = stats_df.to_dict('records')
    task['status'] = 'done'
    
    # Persist to history on disk
    try:
        save_task_to_history(task_id, task, task.get("original_filename", "unknown.csv"))
    except Exception as e:
        print(f"[WARNING] Failed to save history for task {task_id}: {e}")
    
    yield f"data: {json.dumps({'status': 'done', 'stats': task['stats_json']})}\n\n"

@app.get("/api/stream/{task_id}")
async def stream_progress(task_id: str):
    return StreamingResponse(classification_generator(task_id), media_type="text/event-stream")

@app.get("/api/download/{task_id}/classified")
async def download_classified(task_id: str):
    task = tasks.get(task_id)
    if task and task['status'] == 'done':
        return StreamingResponse(io.StringIO(task['classified_csv']), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=classified_questions.csv"})
    
    # Fallback: try loading from history on disk
    csv_path = os.path.join(HISTORY_DIR, task_id, "classified_questions.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="classified_questions.csv")
    
    raise HTTPException(status_code=404, detail="File not ready")

@app.get("/api/download/{task_id}/stats")
async def download_stats(task_id: str):
    task = tasks.get(task_id)
    if task and task['status'] == 'done':
        return StreamingResponse(io.StringIO(task['stats_csv']), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=classification_stats.csv"})
    
    # Fallback: try loading from history on disk
    csv_path = os.path.join(HISTORY_DIR, task_id, "classification_stats.csv")
    if os.path.exists(csv_path):
        return FileResponse(csv_path, media_type="text/csv", filename="classification_stats.csv")
    
    raise HTTPException(status_code=404, detail="File not ready")


# ─── History API ───────────────────────────────────────────────────────────────

@app.get("/api/history")
async def api_history_list():
    """Return list of all past classification runs."""
    index = load_history_index()
    return JSONResponse(content=index)


@app.get("/api/history/{task_id}")
async def api_history_detail(task_id: str):
    """Return full metadata (including stats) for a single history entry."""
    entry = load_history_entry(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử này")
    return JSONResponse(content=entry)


@app.get("/api/history/{task_id}/classified")
async def api_history_download_classified(task_id: str):
    """Download classified CSV from history."""
    csv_path = os.path.join(HISTORY_DIR, task_id, "classified_questions.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="File không tồn tại")
    return FileResponse(csv_path, media_type="text/csv", filename="classified_questions.csv")


@app.get("/api/history/{task_id}/stats")
async def api_history_download_stats(task_id: str):
    """Download stats CSV from history."""
    csv_path = os.path.join(HISTORY_DIR, task_id, "classification_stats.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="File không tồn tại")
    return FileResponse(csv_path, media_type="text/csv", filename="classification_stats.csv")


@app.delete("/api/history/{task_id}")
async def api_history_delete(task_id: str):
    """Delete a history entry."""
    entry = load_history_entry(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Không tìm thấy lịch sử này")
    delete_history_entry(task_id)
    return {"success": True, "message": "Đã xóa thành công"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
