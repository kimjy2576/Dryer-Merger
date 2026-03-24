"""
build_dist.py — PyArmor 난독화 → 배포용 repo 생성
사용법: python build_dist.py

결과: dist_release/ 폴더에 난독화된 코드 생성
      → 이 폴더를 배포용 repo에 푸시
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

# ── 설정 ──
SRC_DIR = Path(__file__).parent           # 원본 소스 위치
DIST_DIR = SRC_DIR / "dist_release"       # 난독화 결과
DIST_REPO = "https://github.com/kimjy2576/HPWD-DataManager.git"

# 난독화할 Python 파일
PY_FILES = [
    "server.py",
    "calculator.py",
    "preprocessor.py",
    "postprocessor.py",
    "performance.py",
    "properties.py",
    "io_handler.py",
    "pipeline.py",
    "run.py",
]

# 그대로 복사할 파일/폴더
COPY_FILES = [
    "static/index.html",
    "config/default_config.yaml",
    "config/__init__.py",
    "config/merge_settings_BR.json",
    "requirements.txt",
    "START.bat",
    "README.md",
    "build_exe.py",
]
COPY_DIRS = [
    "config/saves",
]


def run(cmd, **kw):
    print(f"  $ {cmd}")
    r = subprocess.run(cmd, shell=True, **kw)
    if r.returncode != 0:
        print(f"  ❌ 실패: {cmd}")
        sys.exit(1)
    return r


def main():
    print("=" * 55)
    print("  HPWD Data Manager — PyArmor 난독화 빌드")
    print("=" * 55)

    # 1. PyArmor 설치 확인
    try:
        import pyarmor
        print(f"\n✅ PyArmor {pyarmor.__version__}")
    except ImportError:
        print("\n📦 PyArmor 설치 중...")
        run(f"{sys.executable} -m pip install pyarmor")

    # 2. 기존 dist 정리
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)
    print(f"\n📁 출력: {DIST_DIR}")

    # 3. PyArmor 난독화
    print("\n🔒 Python 파일 난독화 중...")
    for py in PY_FILES:
        src = SRC_DIR / py
        if not src.exists():
            print(f"  ⚠️ 스킵 (없음): {py}")
            continue
        dst_dir = DIST_DIR
        run(f"{sys.executable} -m pyarmor gen --output \"{dst_dir}\" \"{src}\"")
        print(f"  ✅ {py}")

    # 4. 정적 파일 복사
    print("\n📄 정적 파일 복사 중...")
    for f in COPY_FILES:
        src = SRC_DIR / f
        dst = DIST_DIR / f
        if not src.exists():
            print(f"  ⚠️ 스킵 (없음): {f}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        print(f"  ✅ {f}")

    for d in COPY_DIRS:
        src = SRC_DIR / d
        dst = DIST_DIR / d
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(str(src), str(dst))
            print(f"  ✅ {d}/")

    # saves 빈 폴더 보장
    for sub in ["merge", "calc", "formula", "viewer"]:
        (DIST_DIR / "config" / "saves" / sub).mkdir(parents=True, exist_ok=True)
    # results 폴더
    (DIST_DIR / "results").mkdir(exist_ok=True)

    # 5. .gitkeep 추가 (빈 폴더 유지)
    for sub in ["merge", "calc", "formula", "viewer"]:
        (DIST_DIR / "config" / "saves" / sub / ".gitkeep").touch()
    (DIST_DIR / "results" / ".gitkeep").touch()

    # 6. 완료
    print(f"\n{'=' * 55}")
    print(f"  ✅ 난독화 빌드 완료!")
    print(f"  📁 결과: {DIST_DIR}")
    print(f"\n  다음 단계:")
    print(f"  1. cd dist_release")
    print(f"  2. git init")
    print(f"  3. git remote add origin {DIST_REPO}")
    print(f"  4. git add -A")
    print(f"  5. git commit -m \"release: vX.X\"")
    print(f"  6. git push -f origin main")
    print(f"\n  또는 이미 초기화된 경우:")
    print(f"  1. cd dist_release")
    print(f"  2. git add -A && git commit -m \"update\" && git push origin main")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
