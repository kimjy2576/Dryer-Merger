# HPWD Data Manager

히트펌프 건조기 실험 데이터 통합 분석 프로그램

BR (BlackRose) + MX100 + AMS 데이터를 병합하고, 열역학 계산을 수행한 뒤, 개별선도 / 습공기 선도 / P-h 선도로 시각화합니다.

---

## 설치 방법 (최초 1회)

### 1단계: Python 설치

아래 링크에서 Python 3.11을 다운로드하여 설치합니다.

> https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

⚠️ **설치 시 반드시 "Add Python to PATH" 체크!**

※ Windows Store 버전 Python은 권한 문제가 있으므로 반드시 위 링크의 공식 버전을 설치하세요.

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
pip install fastapi uvicorn[standard] pyyaml pandas numpy CoolProp scipy openpyxl xlrd scikit-learn
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

프로그램 상단의 날짜 옆 **[🔄 업데이트]** 버튼을 클릭하면 자동으로 최신 버전이 적용됩니다.

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
Merge → Calculation → Formula (선택) → Visualization
```

### 1. Merge 탭

실험 데이터(BR + MX100 + AMS)를 시간축 기준으로 병합합니다.

1. 실험 데이터 상위 폴더 경로를 입력하고 **[경로 탐색]** 클릭
2. 케이스(폴더)를 선택
3. **[변수 스캔]** → 변수 설정 확인 (포함/제외, W&B, 이름 변경)
4. **[Merge 실행]** → 완료 후 **[🧮 Calculation으로 →]** 클릭

지원 데이터 형식:
- BR: `.csv` (BlackRose 로그)
- MX100: `.xls` (요코가와 MX100, 25행 헤더)
- AMS: `_ams.csv`

변수명에 공백이 있어도 자동 매칭됩니다. (예: `HP Fan Q Current` ↔ `HP_Fan_Q_Current`)

### 2. Calculation 탭

병합된 데이터에 열역학 계산을 수행합니다.

1. Merged 파일이 자동 선택됨 (또는 수동 선택/삭제 가능)
2. 변수 매핑 확인 (자동 매핑)
3. 파일별 실험 입력값 입력 (선택사항: 건조부하, 초기/최종 무게)
4. **[Calculation 실행]** → 완료 후 **[📊 Viewer에서 보기 →]** 클릭

주요 계산 항목: 압력(psat 역산), COP, SMER, 열교환량, 엔탈피/엔트로피, 응축수량, 잔여수분율 등

### 3. Formula 탭 (선택)

커스텀 수식을 적용하여 추가 변수를 생성합니다.

### 4. Visualization 탭

3가지 차트로 데이터를 시각화합니다.

**개별선도:**
- 카테고리별 탭: 압축기 / 증발기 / 응축기 / 유량 / 성능지표 / 소비전력 / 열전달 / 제어지표
- 변수 선택, Y축 범위 설정, 레이아웃(행×열) 조절
- 스무딩(SMA/EMA), 구간 통계
- Y축 한글 레이블 (80+ 변수)

**습공기 선도:**
- 증발기/응축기 입출구 상태점 표시
- Auto Fit / 고정 모드

**P-h 선도:**
- 냉매 포화선 + 사이클 오버레이
- 압축기 입출구 / 응축기 / 증발기 상태점

**공통 기능:**
- 케이스 색상 변경 (color picker)
- 케이스 이름 편집 / 삭제
- 그래프 설정: 선 두께, 크기(높이/너비), 간격 조절
- 파일 선택 삭제

---

## 설정 저장

각 탭에서 설정을 저장하고 불러올 수 있습니다.

- **[⭐ 디폴트 저장]**: 다음 실행 시 자동으로 불러옴
- **[⭐ 디폴트 불러오기]**: 저장된 디폴트를 수동으로 불러옴
- **[💾 세이브 슬롯]**: 여러 설정을 이름별로 저장/불러오기/삭제

Merge, Calculation, Formula, Visualization 각각 독립적으로 저장됩니다.

---

## Setting 탭

냉매 종류, 대기압, 실험 환경 등 기본 설정을 변경할 수 있습니다.

---

## 종료 방법

- CMD 창을 닫거나
- CMD에서 `Ctrl + C` 입력

---

## 문제 해결

| 증상 | 해결 |
|------|------|
| `python`이 인식되지 않음 | Python 공식 버전 재설치 시 "Add to PATH" 체크. Windows Store 버전은 사용 불가 |
| `pip`가 인식되지 않음 | `python -m pip install ...` 으로 대체 |
| 패키지 설치 오류 | `python -m pip install --upgrade pip` 후 재시도 |
| CoolProp 설치 실패 | Python 3.11로 재설치 (3.13은 미지원) |
| MX100 변수가 안 잡힘 | `pip install xlrd` 실행 |
| 화면이 빈 페이지 | 수동 업데이트 방법으로 재설치 |
| 포트 충돌 (8000) | 기존 프로그램 종료 후 재시도 |
| 토출 압력이 9.99로 고정 | Merge 재실행 (outlier_thresholds 수정됨) |

---

## 시스템 요구사항

- Windows 10 이상
- Python 3.10 ~ 3.12 (3.11 권장)
- 브라우저 (Chrome / Edge 권장)

---

## 담당자

건조기선행연구팀 김진영 선임
