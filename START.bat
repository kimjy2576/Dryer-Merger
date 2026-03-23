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
    echo [오류] Python이 설치되어 있지 않습니다.
    echo.
    echo 아래 링크에서 Python을 설치해주세요:
    echo   https://www.python.org/downloads/
    echo.
    echo 설치 시 반드시 "Add Python to PATH" 체크!
    echo.
    pause
    exit /b 1
)

echo [1/3] Python 확인 완료
python --version
echo.

:: 패키지 설치
echo [2/3] 필수 패키지 설치 중...
pip install fastapi uvicorn pyyaml pandas numpy CoolProp scipy --quiet --break-system-packages 2>nul || pip install fastapi uvicorn pyyaml pandas numpy CoolProp scipy --quiet
echo.

:: 실행
echo [3/3] HPWD Data Manager 시작!
echo.
echo ═══════════════════════════════════════════
echo   브라우저에서 자동으로 열립니다.
echo   종료하려면 이 창을 닫으세요.
echo ═══════════════════════════════════════════
echo.
python run.py
pause
