"""
run.py — EXE 진입점
실행하면 FastAPI 서버를 내장 기동하고 브라우저를 자동으로 연다.
"""
import sys
import os
import webbrowser
import threading
import time

# PyInstaller 번들 경로 보정
if getattr(sys, "frozen", False):
    os.chdir(sys._MEIPASS)
    # 번들 내 config/results 경로 설정
    os.makedirs(os.path.join(os.path.dirname(sys.executable), "results"), exist_ok=True)

PORT = 8000


def open_browser():
    """서버 시작 후 1.5초 뒤 브라우저 열기."""
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    # 브라우저 오픈 스레드
    threading.Thread(target=open_browser, daemon=True).start()

    # uvicorn 실행 (프로그래밍 방식)
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=PORT, log_level="info")
