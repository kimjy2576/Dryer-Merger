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
    time.sleep(1.5)
    webbrowser.open(f"http://localhost:{PORT}")


if __name__ == "__main__":
    first_run = True
    while True:
        if first_run:
            threading.Thread(target=open_browser, daemon=True).start()
            first_run = False

        print(f"\n{'='*40}")
        print(f"  HPWD Data Manager — http://localhost:{PORT}")
        print(f"{'='*40}\n")

        # uvicorn을 자식 프로세스로 실행
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server:app",
             "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "info"],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        proc.wait()  # 서버 종료 대기

        print("\n🔄 서버 재시작 중...\n")
        time.sleep(2)
