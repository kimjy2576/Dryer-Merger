@echo off
chcp 65001 >nul 2>&1
title HPWD Data Manager - Setup

echo ══════════════════════════════════════════
echo   HPWD Data Manager - 설치 및 실행
echo ══════════════════════════════════════════
echo.

:: Python 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ╔══════════════════════════════════════════╗
    echo ║  Python이 설치되어 있지 않습니다!         ║
    echo ║                                          ║
    echo ║  Python 3.11 설치가 필요합니다.           ║
    echo ║                                          ║
    echo ║  설치 방법:                               ║
    echo ║  1. 아래 링크에서 다운로드                ║
    echo ║  2. 설치 시 "Add Python to PATH" 체크!   ║
    echo ║  3. 설치 후 이 파일을 다시 실행           ║
    echo ╚══════════════════════════════════════════╝
    echo.
    echo 다운로드 링크를 브라우저에서 엽니다...
    start https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo.
    pause
    exit /b 1
)

:: 버전 체크
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
echo [1/4] Python %PYVER% 확인

if %MINOR% LSS 10 (
    echo.
    echo [경고] Python 3.10 이상이 필요합니다. (현재: %PYVER%^)
    echo Python 3.11 설치를 권장합니다.
    echo https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo.
    pause
    exit /b 1
)
if %MINOR% GTR 12 (
    echo.
    echo [경고] Python 3.13+는 CoolProp 호환 문제가 있을 수 있습니다.
    echo Python 3.11 사용을 권장합니다.
    echo 계속하려면 아무 키나 누르세요...
    pause >nul
)
echo.

:: 패키지 설치
echo [2/4] 필수 패키지 확인 중...
python -c "import fastapi, uvicorn, pandas, numpy, CoolProp, yaml, scipy" >nul 2>&1
if %errorlevel% neq 0 (
    echo       패키지 설치 중... (최초 1회, 2-3분 소요)
    echo.
    pip install fastapi uvicorn[standard] pyyaml pandas numpy CoolProp scipy --quiet 2>nul
    if %errorlevel% neq 0 (
        echo.
        echo [오류] 패키지 설치 실패!
        echo 수동 설치: pip install fastapi uvicorn pyyaml pandas numpy CoolProp scipy
        pause
        exit /b 1
    )
    echo       설치 완료!
) else (
    echo       모든 패키지 설치됨 (스킵)
)
echo.

:: 업데이트 확인 (GitHub 최신 버전)
echo [3/4] 최신 버전 확인...
echo       (업데이트는 프로그램 내 [업데이트] 버튼 사용)
echo.

:: 실행
echo [4/4] HPWD Data Manager 시작!
echo.
echo ══════════════════════════════════════════
echo   브라우저에서 자동으로 열립니다.
echo   http://localhost:8000
echo.
echo   종료: 이 창을 닫거나 Ctrl+C
echo ══════════════════════════════════════════
echo.
python run.py
pause
