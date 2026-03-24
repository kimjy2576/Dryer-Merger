# HPWD Data Manager

히트펌프 건조기 실험 데이터 통합 분석 프로그램

---

## 설치 방법 (최초 1회)

### 1단계: Python 설치

아래 링크에서 Python 3.11을 다운로드하여 설치합니다.

> https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

⚠️ **설치 시 반드시 "Add Python to PATH" 체크!**

![Python PATH 설정](https://docs.python.org/3/_images/win_installer.png)

### 2단계: 프로그램 다운로드

CMD(명령 프롬프트)를 열고 아래 명령어를 **한 줄씩** 입력합니다.

```
cd /d %USERPROFILE%\Downloads
```
```
curl -L -o dryer.zip https://github.com/kimjy2576/Dryer-Merger/archive/refs/heads/main.zip
```
```
powershell -command "Expand-Archive -Force dryer.zip ."
```

### 3단계: 패키지 설치 (최초 1회, 2~3분 소요)

```
cd Dryer-Merger-main
```
```
pip install fastapi uvicorn[standard] pyyaml pandas numpy CoolProp scipy openpyxl scikit-learn
```

### 4단계: 실행

```
python run.py
```

브라우저에서 `http://localhost:8000` 이 자동으로 열립니다.

---

## 실행 방법 (2회차부터)

CMD에서 아래 명령어만 입력하면 됩니다.

```
cd /d %USERPROFILE%\Downloads\Dryer-Merger-main
python run.py
```

또는 폴더 안의 **`START.bat`** 을 더블클릭해도 됩니다.

---

## 업데이트 방법

### 방법 1: 프로그램 내 업데이트 (권장)

프로그램 상단의 **[🔄 업데이트]** 버튼을 클릭하면 자동으로 최신 버전이 적용됩니다.

### 방법 2: 수동 업데이트 (화면이 안 나올 때)

CMD에서 아래 명령어를 한 줄씩 입력합니다.

```
cd /d %USERPROFILE%\Downloads
```
```
rmdir /s /q Dryer-Merger-main 2>nul
```
```
curl -L -o dryer.zip https://github.com/kimjy2576/Dryer-Merger/archive/refs/heads/main.zip
```
```
powershell -command "Expand-Archive -Force dryer.zip ."
```
```
cd Dryer-Merger-main
python run.py
```

---

## 프로그램 사용 순서

```
Merge → Calculation → Visualization
```

### 1. Merge 탭

- 실험 데이터 폴더 경로를 입력하고 **[경로 탐색]** 클릭
- 케이스를 선택하고 **[변수 스캔]** → 변수 설정 확인
- **[Merge 실행]** → 완료 후 **[🧮 Calculation으로 →]** 클릭

### 2. Calculation 탭

- Merged 파일이 자동 선택됨
- 변수 매핑 확인 (자동 매핑 또는 수동 설정)
- 파일별 실험 입력값 입력 (선택사항: 건조부하, 초기/최종 무게)
- **[Calculation 실행]** → 완료 후 **[📊 Viewer에서 보기 →]** 클릭

### 3. Visualization 탭

- 개별선도 / 습공기 선도 / P-h 선도 확인
- 카테고리별 탭: 압축기, 증발기, 응축기, 유량, 성능, 전력, 열전달, 제어
- 그래프 설정: 선 두께, 크기, 간격 조절 가능
- 케이스 색상 변경, 이름 편집 가능

---

## 설정 저장

각 탭에서 **[⭐ 디폴트 저장]** / **[⭐ 디폴트 불러오기]** / **[💾 세이브 슬롯]** 으로 설정을 저장하고 불러올 수 있습니다.

저장된 설정은 프로그램 재시작 후에도 유지됩니다.

---

## 종료 방법

- CMD 창을 닫거나
- CMD에서 `Ctrl + C` 입력

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| `python`이 인식되지 않음 | Python 재설치 시 "Add to PATH" 체크 |
| 패키지 설치 오류 | `pip install --upgrade pip` 후 재시도 |
| CoolProp 설치 실패 | Python 3.11로 재설치 (3.13은 미지원) |
| 화면이 빈 페이지 | 수동 업데이트 방법으로 재설치 |
| 포트 충돌 (8000) | 기존 프로그램 종료 후 재시도 |

---

## 시스템 요구사항

- Windows 10 이상
- Python 3.10 ~ 3.12 (3.11 권장)
- 브라우저 (Chrome/Edge 권장)

---

## 담당자

건조기선행연구팀 김진영 선임
