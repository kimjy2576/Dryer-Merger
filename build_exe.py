"""
build_exe.py — HPWD Data Manager를 EXE로 빌드
사용법: python build_exe.py
결과: dist/HPWD_DataManager/ 폴더에 EXE 생성
"""
import subprocess
import sys
import os
import shutil

def main():
    print("=" * 50)
    print("  HPWD Data Manager — EXE 빌드")
    print("=" * 50)

    # 1. PyInstaller 설치 확인
    try:
        import PyInstaller
        print(f"✅ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("📦 PyInstaller 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "--break-system-packages"])

    # 2. 빌드 실행
    print("\n🔨 빌드 시작...\n")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "HPWD_DataManager",
        "--onedir",                    # 폴더 모드 (onefile보다 안정적)
        "--noconsole",                 # 콘솔 창 숨김 (선택: --console로 바꾸면 디버그용)
        "--noconfirm",                 # 기존 빌드 덮어쓰기
        "--add-data", f"static{os.pathsep}static",
        "--add-data", f"config{os.pathsep}config",
        "--hidden-import", "CoolProp",
        "--hidden-import", "CoolProp.CoolProp",
        "--hidden-import", "uvicorn",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.loops",
        "--hidden-import", "uvicorn.loops.auto",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--hidden-import", "uvicorn.lifespan.off",
        "--hidden-import", "fastapi",
        "--hidden-import", "starlette",
        "--hidden-import", "starlette.responses",
        "--hidden-import", "starlette.staticfiles",
        "--hidden-import", "pydantic",
        "--hidden-import", "yaml",
        "--hidden-import", "scipy",
        "--hidden-import", "scipy.interpolate",
        "--collect-all", "CoolProp",
        "--collect-submodules", "uvicorn",
        "run.py",
    ]

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print("\n❌ 빌드 실패!")
        return

    # 3. config/saves 폴더 복사 (세이브 슬롯용)
    dist_dir = os.path.join("dist", "HPWD_DataManager")
    saves_dir = os.path.join(dist_dir, "config", "saves")
    for sub in ["merge", "calc", "formula", "viewer"]:
        os.makedirs(os.path.join(saves_dir, sub), exist_ok=True)

    # results 폴더
    os.makedirs(os.path.join(dist_dir, "results"), exist_ok=True)

    # 4. 서버 모듈 복사 (PyInstaller가 못 잡을 수 있으므로)
    modules = [
        "server.py", "calculator.py", "preprocessor.py",
        "postprocessor.py", "performance.py", "properties.py",
        "io_handler.py", "pipeline.py",
    ]
    for m in modules:
        if os.path.exists(m):
            shutil.copy2(m, os.path.join(dist_dir, m))
            print(f"  📄 {m} 복사")

    print(f"\n✅ 빌드 완료!")
    print(f"📁 결과: {os.path.abspath(dist_dir)}")
    print(f"\n배포 방법:")
    print(f"  1. dist/HPWD_DataManager 폴더를 ZIP으로 압축")
    print(f"  2. 받은 사람은 압축 풀고 HPWD_DataManager.exe 더블클릭")
    print(f"  3. 브라우저에서 http://localhost:8000 자동 열림")


if __name__ == "__main__":
    main()
