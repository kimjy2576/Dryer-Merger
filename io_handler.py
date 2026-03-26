"""
io_handler.py — 파일 탐색, 읽기, 쓰기, 이름 변환
"""
import os
import glob
import pandas as pd


# ──────────────────────────────────────────────
#  파일 탐색
# ──────────────────────────────────────────────
def find_files(data_path: str) -> dict:
    """
    data_path 안의 AMS / BlackRose / MX100 파일 경로 리스트를 반환한다.
    Returns:
        {"ams": [...], "br": [...], "mx100": [...]}
    """
    return {
        "ams":   sorted(glob.glob(os.path.join(data_path, "*_ams.csv"))),
        "br":    sorted(glob.glob(os.path.join(data_path, "*_br.csv"))),
        "mx100": sorted(glob.glob(os.path.join(data_path, "*_temp.*"))),
    }


# ──────────────────────────────────────────────
#  파일명 자동 변환  (_br / _temp 접미사 부여)
# ──────────────────────────────────────────────
def rename_files_in_folder(folder_path: str) -> None:
    """
    폴더 내 .csv → _br.csv, .xls → _temp.xls 로 이름 변경.
    - 이미 BR/_br 또는 MX100/_temp 파일이 존재하면 변환하지 않음
    - _merged, _calc, _formula, _result 파일은 절대 변환하지 않음
    """
    files = os.listdir(folder_path)
    files_lower = [f.lower() for f in files]

    # 이미 분류된 파일이 있으면 변환 불필요
    has_br = any(f.endswith("_br.csv") for f in files_lower)
    has_mx = any(f.endswith("_temp.xls") or f.endswith("_temp.xlsx") for f in files_lower)

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        if not os.path.isfile(filepath):
            continue

        name, ext = os.path.splitext(filename)
        nl = name.lower()

        # 생성 파일은 절대 변환하지 않음
        if any(tag in nl for tag in ["merged", "calc", "formula", "result", ".cache"]):
            continue

        new_name = None
        if ext.lower() == ".csv" and not has_br and not nl.endswith("_br") and "_ams" not in nl:
            new_name = f"{name}_br{ext}"
        elif ext.lower() in (".xls", ".xlsx") and not has_mx and not nl.endswith("_temp"):
            new_name = f"{name}_temp{ext}"

        if new_name:
            os.replace(filepath, os.path.join(folder_path, new_name))


# ──────────────────────────────────────────────
#  파일 읽기
# ──────────────────────────────────────────────
def read_blackrose(filepath: str) -> pd.DataFrame:
    """BlackRose CSV 읽기 (cp949 / utf-8 자동 시도)."""
    for enc in ("cp949", "utf-8"):
        try:
            df = pd.read_csv(filepath, encoding=enc, skiprows=[0])
            df.columns = [c.strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"BlackRose 파일 인코딩 인식 실패: {filepath}")


def read_ams(filepath: str) -> pd.DataFrame:
    """AMS CSV 읽기."""
    df = pd.read_csv(filepath, encoding="utf-8", skiprows=[0])
    df.columns = [c.strip() for c in df.columns]
    return df


def read_mx100_single_header(filepath: str) -> pd.DataFrame:
    """MX100 Excel 읽기 (단일 헤더)."""
    return pd.read_excel(filepath, skiprows=24, header=0)


def read_mx100_dual_header(filepath: str) -> pd.DataFrame:
    """
    MX100 Excel 읽기 (이중 헤더).
    윗줄이 Unnamed이면 아랫줄(Date/Time)을 사용하고,
    Date+Time 열을 결합하여 단일 Time 열로 변환한다.
    """
    df = pd.read_excel(filepath, skiprows=24, header=[0, 1])

    new_columns = []
    for col in df.columns:
        if "Unnamed" in str(col[0]):
            new_columns.append(col[1])
        else:
            new_columns.append(col[0])
    df.columns = new_columns

    if "Date" in df.columns and "Time" in df.columns:
        df["Time"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
        df.drop(columns=["Date"], inplace=True, errors="ignore")

    return df


def read_source_files(
    ams_path: str | None,
    br_path: str | None,
    mx100_path: str | None,
    data_time: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    data_time 에 맞춰 필요한 파일만 읽고, 나머지는 빈 DataFrame 을 반환한다.
    """
    df_ams = pd.DataFrame()
    df_br  = pd.DataFrame()
    df_mx  = pd.DataFrame()

    if data_time == "BR":
        if not br_path:
            raise FileNotFoundError("BlackRose 파일이 없습니다.")
        df_br = read_blackrose(br_path)
        if mx100_path:
            df_mx = read_mx100_single_header(mx100_path)

    elif data_time == "AMS":
        if not ams_path:
            raise FileNotFoundError("AMS 파일이 없습니다.")
        df_ams = read_ams(ams_path)

    elif data_time == "MX100":
        if not mx100_path:
            raise FileNotFoundError("MX100 파일이 없습니다.")
        df_mx = read_mx100_dual_header(mx100_path)

    return df_ams, df_br, df_mx


# ──────────────────────────────────────────────
#  파일 저장
# ──────────────────────────────────────────────
def save_merged(
    df: pd.DataFrame,
    data_path: str,
    case_name: str,
    numbering: int,
) -> str:
    """병합 결과를 CSV로 저장하고 저장 경로를 반환한다."""
    parts = case_name.split("_")[:9]
    prefix = "_".join(parts) + f"_{numbering}"
    out_path = os.path.join(data_path, f"{prefix}_merged.csv")
    df.to_csv(out_path, index=False)
    return out_path
