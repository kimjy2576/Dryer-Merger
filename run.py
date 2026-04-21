"""
run.py — HPWD Data Manager 진입점
- python run.py: 개발 모드 (subprocess + 자동 재시작)
- EXE: PyInstaller 번들 (직접 실행)
"""
import sys
import os
import webbrowser
import threading
import time

PORT = 8000
IS_FROZEN = getattr(sys, "frozen", False)

# PyInstaller 번들 경로 보정
if IS_FROZEN:
    BASE = os.path.dirname(sys.executable)
    os.chdir(BASE)
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "config", "saves", "merge"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "config", "saves", "calc"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "config", "saves", "formula"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "config", "saves", "viewer"), exist_ok=True)
    # sys.path에 EXE 위치 추가 (모듈 import용)
    if BASE not in sys.path:
        sys.path.insert(0, BASE)


def open_browser():
    time.sleep(2.0)
    webbrowser.open(f"http://localhost:{PORT}")


def check_and_install_deps():
    """시작 시 주요 의존성 확인 및 자동 설치."""
    if IS_FROZEN:
        return  # EXE는 스킵
    required = [
        ("fastapi", "fastapi>=0.104"),
        ("uvicorn", "uvicorn[standard]>=0.24"),
        ("pandas", "pandas>=2.0"),
        ("CoolProp", "CoolProp>=6.4"),
        ("yaml", "PyYAML>=6.0"),
        ("openpyxl", "openpyxl>=3.1"),
        # 성능 개선용 (선택)
        ("pyarrow", "pyarrow>=15.0"),
        ("python_calamine", "python-calamine>=0.2"),
    ]
    missing = []
    for mod, spec in required:
        try:
            __import__(mod)
        except ImportError:
            missing.append(spec)
    if missing:
        print(f"\n[의존성 설치] 누락된 패키지 {len(missing)}개 설치 중...")
        print(f"   {', '.join(missing)}")
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("[의존성 설치] ✅ 완료\n")
        except Exception as e:
            print(f"[의존성 설치] ⚠️ 일부 실패: {e}")
            print("수동 설치: pip install -r requirements.txt\n")


if __name__ == "__main__":
    check_and_install_deps()

    print(f"\n{'='*50}")
    print(f"  HPWD Data Manager — http://localhost:{PORT}")
    print(f"  {'[EXE]' if IS_FROZEN else '[Python]'}")
    print(f"{'='*50}\n")

    threading.Thread(target=open_browser, daemon=True).start()

    if IS_FROZEN:
        # EXE: 직접 uvicorn 실행 (subprocess 불가)
        import uvicorn
        uvicorn.run("server:app", host="127.0.0.1", port=PORT, log_level="info")
    else:
        # 개발 모드: subprocess + 자동 재시작
        import subprocess
        while True:
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "server:app",
                 "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "info"],
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            proc.wait()
            print("\n🔄 서버 재시작 중...\n")
            time.sleep(2)
