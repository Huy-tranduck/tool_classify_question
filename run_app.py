import uvicorn
import webbrowser
import threading
import time
import socket
import sys
import os

# Import the FastAPI app instance
from app import app

def get_free_port():
    """Find a free port if default is taken."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def open_browser(port):
    """Wait briefly for server to start, then open browser."""
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    print(f"Mở giao diện tại: {url}")
    webbrowser.open(url)

if __name__ == '__main__':
    # Add PyInstaller temp path to sys.path so modules can be imported
    if getattr(sys, 'frozen', False):
        sys.path.insert(0, sys._MEIPASS)
        
    port = 8000
    try:
        # Check if port 8000 is available
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
    except OSError:
        port = get_free_port()

    # Start browser thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    print("Khởi động máy chủ phân loại...")
    # Run Uvicorn server (blocking)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
