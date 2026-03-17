@echo off
echo ============================================
echo  DryerMerger EXE 빌드
echo ============================================
echo.
echo [1/3] 의존성 설치 중...
pip install -r requirements.txt pyinstaller --quiet

echo [2/3] EXE 빌드 중... (2~3분 소요)
pyinstaller DryerMerger.spec --noconfirm --clean

echo [3/3] 완료!
echo.
echo 실행 파일: dist\DryerMerger\DryerMerger.exe
echo 더블클릭하면 브라우저가 자동으로 열립니다.
echo.
pause
