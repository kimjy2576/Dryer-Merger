"""
run.py — EXE 진입점
실행하면 FastAPI 서버를 내장 기동하고 브라우저를 자동으로 연다.
업데이트 후 자동 재시작 지원.
"""
import sys
import os
import webbrowser
import threading
import time
import subprocess

# PyInstaller 번들 경로 보정
if getattr(sys, "frozen", False):
    os.chdir(sys._MEIPASS)
    os.makedirs(os.path.join(os.path.dirname(sys.executable), "results"), exist_ok=True)

PORT = 8000


def open_browser():
    """서버 시작 후 1.5초 뒤 브라우저 열기."""
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    first_run = True
    while True:
        if first_run:
            threading.Thread(target=open_browser, daemon=True).start()
            first_run = False

        print(f"\n{'='*40}")
        print(f"  Dryer Merger v5 — http://localhost:{PORT}")
        print(f"{'='*40}\n")

        import uvicorn
        uvicorn.run("server:app", host="127.0.0.1", port=PORT, log_level="info")

        # 서버가 종료되면 (업데이트 등) 재시작
        print("\n🔄 서버 재시작 중...\n")
        time.sleep(1)
