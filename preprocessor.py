"""
preprocessor.py — 개별 데이터소스 전처리, 시간 동기화, 병합

[최적화]
- BlackRose 시간 파싱: .apply(lambda) → str accessor 벡터화
- sync_and_merge: pd.merge_asof 활용, 불필요 copy 제거
- to_numeric: apply 대신 infer_objects 또는 컬럼별 astype 활용
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser


# ══════════════════════════════════════════════
#  시간 파싱 유틸리티
# ══════════════════════════════════════════════
def parse_flexible_datetime(value):
    """다양한 형식의 날짜/시간 값을 datetime으로 변환한다."""
    if pd.isna(value) or value == "":
        return pd.NaT
    if isinstance(value, (datetime, pd.Timestamp)):
        return value
    if isinstance(value, (int, float)):
        return datetime(1899, 12, 30) + timedelta(days=value)

    s = str(value).strip()
    if "00:00:00" in s:
        s = s.replace("00:00:00", "").strip()
    if ":" not in s and "." in s and len(s) <= 8:
        s = s.replace(".", ":")

    try:
        return dateutil_parser.parse(s)
    except Exception:
        return pd.NaT


def _align_datetime_to_base(df: pd.DataFrame, base_date) -> pd.DataFrame:
    """
    Time 열을 base_date 기준으로 통일.
    [v2] 자정 넘김을 '연속 단조감소 탐지'로 판별 (hour 비교 제거).
    """
    if pd.isna(base_date):
        return df

    # Time 컬럼이 없으면 LogTime 등에서 폴백
    if "Time" not in df.columns:
        for alt in ["LogTime", "logtime", "LOGTIME", "time", "DateTime", "datetime", "Timestamp"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Time"})
                break
    if "Time" not in df.columns:
        return df

    temp_dt = pd.to_datetime(df["Time"], errors="coerce")
    time_comp = temp_dt - temp_dt.dt.normalize()
    base_midnight = pd.Timestamp(base_date).normalize()
    df["Time"] = base_midnight + time_comp

    # 자정 넘김 탐지: 시간이 감소하는 지점에서 +1일
    # (예: 23:59:59 → 00:00:00 에서 diff가 음수)
    if len(df) > 1:
        time_diff = df["Time"].diff()
        midnight_crossings = time_diff < pd.Timedelta(0)
        # 자정 넘김 이후 모든 행에 +1일씩 누적
        day_offset = midnight_crossings.cumsum()
        df["Time"] = df["Time"] + pd.to_timedelta(day_offset, unit="D")

    return df


# ══════════════════════════════════════════════
#  BlackRose 전처리
# ══════════════════════════════════════════════
def _fuzzy_col_match(df_columns, wanted_columns):
    """공백/언더스코어를 동일시하여 컬럼 매칭. 매칭된 실제 컬럼명 리스트 반환."""
    # 정규화 맵: "HP Fan Q Current" → "hp_fan_q_current"
    norm = lambda s: s.strip().replace(" ", "_").lower()
    col_map = {norm(c): c for c in df_columns}  # 정규화 → 실제
    result = []
    for w in wanted_columns:
        if w in df_columns:
            result.append(w)  # 정확 매칭 우선
        elif norm(w) in col_map:
            result.append(col_map[norm(w)])  # 퍼지 매칭
    return result


def preprocess_blackrose(
    df_raw: pd.DataFrame,
    selected_columns: list,
    column_mapping: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    BlackRose 데이터 전처리.
    """
    if df_raw.empty:
        dummy = pd.DataFrame({"Time": [0]})
        return dummy.copy(), dummy.copy()

    df = df_raw.copy()
    df.columns = [col.strip() for col in df.columns]

    # LogTime 폴백: 다양한 시간 컬럼명 지원
    time_col = None
    for candidate in ["LogTime", "logtime", "Log Time", "Time", "DateTime", "Timestamp"]:
        if candidate in df.columns:
            time_col = candidate; break
    if not time_col:
        first = df.columns[0]
        try:
            pd.to_datetime(df[first].iloc[0])
            time_col = first
        except: pass
    if not time_col:
        print(f"[preprocess_br] 시간 컬럼 못 찾음. 컬럼: {df.columns[:10].tolist()}")
        dummy = pd.DataFrame({"Time": [0]})
        return dummy.copy(), dummy.copy()

    time_str = df[time_col].astype(str)
    df["Time"] = time_str.str.split(" ").str[1].str.split(".").str[0]

    # 필터링: 컬럼 없으면 전체 데이터 사용
    if "SW_ProtectionCount" in df.columns:
        df_main = df[df["SW_ProtectionCount"] > 0].copy()
    else:
        df_main = df.copy()
    if "MainProcess" in df.columns:
        df_sub = df[df["MainProcess"] == 10].copy()
    else:
        df_sub = pd.DataFrame(columns=df.columns)

    df_main.rename(columns=column_mapping, inplace=True)

    add_columns = ["Time", "SubProcess", "RemainTime"]
    matched = _fuzzy_col_match(df_main.columns, selected_columns)
    df_main = df_main[matched]
    df_add  = df_sub[[c for c in add_columns if c in df_sub.columns]]

    return df_main, df_add


# ══════════════════════════════════════════════
#  AMS 전처리
# ══════════════════════════════════════════════
def preprocess_ams(
    df_raw: pd.DataFrame,
    column_mapping: dict,
    scale_factors: dict,
    subprocess_mapping: dict,
    selected_columns: list,
) -> pd.DataFrame:
    """AMS 데이터 전처리."""
    if df_raw.empty:
        return pd.DataFrame({"Time": [0]})

    df = df_raw.copy()
    df.rename(columns=column_mapping, inplace=True)

    # 스케일 변환 — 벡터화
    for col, factor in scale_factors.items():
        if col in df.columns:
            df[col] = df[col] * factor

    if "SubProcess" in df.columns:
        df["SubProcess"] = df["SubProcess"].replace(subprocess_mapping)

    available = _fuzzy_col_match(df.columns, selected_columns)
    df = df[available].copy()
    df["Time"] = df["Time"].astype(str).str.slice(0, 8)
    return df


# ══════════════════════════════════════════════
#  MX100 전처리
# ══════════════════════════════════════════════
def preprocess_mx100(
    df_raw: pd.DataFrame,
    useless_columns: list,
) -> pd.DataFrame:
    """MX100 데이터 전처리."""
    if df_raw.empty:
        return pd.DataFrame({"Time": [0]})

    df = df_raw.copy()
    df.drop(index=0, inplace=True, errors="ignore")
    cols_to_drop = [df.columns[i] for i in useless_columns if i < len(df.columns)]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    df.rename(columns={df.columns[0]: "Time"}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]

    # [최적화] apply(pd.to_numeric) → 개별 컬럼 astype 시도
    num_cols = df.columns.difference(["Time"])
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ══════════════════════════════════════════════
#  시간 동기화 & 병합
# ══════════════════════════════════════════════
def sync_and_merge(
    dfs: list[pd.DataFrame],
    data_time: str,
    df_br: pd.DataFrame,
    df_ams: pd.DataFrame,
    df_mx100: pd.DataFrame,
) -> pd.DataFrame:
    """
    여러 DataFrame을 기준 시간축(1초 간격)으로 리샘플링·병합.
    [v3 최적화] reindex를 한번에 처리, ref_df 이중 변환 제거.
    """
    ref_map = {"BR": df_br, "AMS": df_ams, "MX100": df_mx100}
    ref_df = ref_map[data_time]
    if ref_df.empty or "Time" not in ref_df.columns:
        return pd.DataFrame({"Time": [0]})

    start_time = parse_flexible_datetime(ref_df["Time"].iloc[0])
    print(f"  [sync] 기준시간: {start_time}, data_time={data_time}")

    # 중복 제거 + 시간 변환
    clean_dfs = []
    for i, df in enumerate(dfs):
        if df.empty or "Time" not in df.columns:
            continue
        df = df.copy()
        df = df.drop_duplicates(subset="Time", keep="first").reset_index(drop=True)
        _align_datetime_to_base(df, start_time)

        # Time을 datetime으로 강제
        if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
            df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

        # NaT/중복 제거
        df = df.dropna(subset=["Time"]).reset_index(drop=True)
        df = df[~df["Time"].duplicated(keep="first")].reset_index(drop=True)

        if len(df) > 0:
            print(f"  [sync] df[{i}]: {len(df)}행 {len(df.columns)}열, Time={df['Time'].iloc[0]}~{df['Time'].iloc[-1]}")
            clean_dfs.append(df)

    if not clean_dfs:
        return pd.DataFrame({"Time": [0]})

    # end_time: 첫 번째 clean_df에서
    end_time = clean_dfs[0]["Time"].iloc[-1]

    # 1초 간격 시간축
    try:
        time_index = pd.date_range(start=start_time, end=end_time, freq="s", name="Time")
        print(f"  [sync] time_index: {len(time_index)}초")
    except Exception as e:
        print(f"  [sync] date_range 실패: {e}")
        return pd.DataFrame({"Time": [0]})

    # reindex (각 df별 try/except)
    resampled = []
    for di, df in enumerate(clean_dfs):
        try:
            if not df["Time"].is_monotonic_increasing:
                df = df.sort_values("Time").reset_index(drop=True)
            rs = df.set_index("Time").reindex(time_index, method=None)
            rs = rs.ffill(limit=30).bfill(limit=30)
            resampled.append(rs)
        except Exception as e:
            print(f"  [sync] reindex 실패 df[{di}]: {e}")
            try:
                df = df.sort_values("Time")
                dummy = pd.DataFrame({"Time": time_index})
                merged_one = pd.merge_asof(dummy, df, on="Time", direction="nearest", tolerance=pd.Timedelta("30s"))
                resampled.append(merged_one.set_index("Time"))
                print(f"  [sync] df[{di}] merge_asof 폴백 성공")
            except Exception as e2:
                print(f"  [sync] df[{di}] 폴백도 실패: {e2}")

    if not resampled:
        return pd.DataFrame({"Time": [0]})

    # 병합 (중복 컬럼 제거)
    merged = pd.concat(resampled, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    merged = merged.reset_index()

    # 숫자 변환 + 선형 보간 (NaN 있는 컬럼만, 배치 처리)
    num_cols = merged.columns.difference(["Time"])
    non_numeric = [c for c in num_cols if not pd.api.types.is_numeric_dtype(merged[c])]
    if non_numeric:
        merged[non_numeric] = merged[non_numeric].apply(pd.to_numeric, errors="coerce")

    has_nan = merged[num_cols].isnull().any()
    nan_cols = has_nan[has_nan].index.tolist()
    if nan_cols:
        merged[nan_cols] = merged[nan_cols].interpolate(
            method="linear", limit_direction="both")

    # 단조성 검증
    if not merged["Time"].is_monotonic_increasing:
        merged = merged.sort_values("Time").reset_index(drop=True)

    print(f"  [sync] 최종: {len(merged)}행×{len(merged.columns)}열")
    return merged


# ══════════════════════════════════════════════
#  시간 열 부가 (Time_sec, Time_min)
# ══════════════════════════════════════════════
def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Time 열 기준으로 Time_sec, Time_min 열 추가."""
    t0 = df["Time"].iloc[0]
    time_sec = (df["Time"] - t0).dt.total_seconds()
    time_sec = time_sec - time_sec.iloc[0]
    df.insert(1, "Time_sec", time_sec)
    df.insert(2, "Time_min", (time_sec / 60).round(2))
    return df
