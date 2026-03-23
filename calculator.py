"""
calculator.py — Stage 1: 병합 데이터의 기본 파생 컬럼 계산
- 압력 추정 (온도 → Psat)
- fTemp 건조도 감지
- 발산값 보정
- MX100 누락 컬럼 유도
"""
import numpy as np
import pandas as pd
from properties import get_props


def run_stage1(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stage 1 계산 실행."""
    setup = cfg["processing"].get("setup_type", "BR")
    env = cfg["environment"]
    props = get_props(env["refrigerant"], env["patm"], env.get("backend", "HEOS"))

    if setup == "MX100":
        df = _derive_missing_mx100(df)
    elif setup == "BR":
        df = _clamp_divergent(df)
        df["fTemp_calc"] = _calc_ftemp(df, cfg["ftemp"])
        df = _calc_pressures(df, props, env["patm"])
        df = _calc_discharge_superheat(df, props)

    return df


def _col(df, *names):
    """여러 이름 중 존재하는 첫 컬럼 반환."""
    for n in names:
        if n in df.columns: return df[n].values
    raise KeyError(f"컬럼 없음: {names}")

def _calc_ftemp(df, cfg_ft):
    k1, k2, k3 = cfg_ft["k1"], cfg_ft["k2"], cfg_ft["k3"]
    delta = cfg_ft["auto_delta"]
    win = int(np.round(cfg_ft["moving_avg_window_min"] * 60, 0))
    raw = 10 * (k1 * _col(df, "Heatpump_DuctOutTemp", "T_Air_Cond_Out")
                - k2 * _col(df, "Heatpump_DuctInTemp", "T_Air_Eva_In")
                + k3 * _col(df, "Heatpump_EvaInTemp", "T_Eva_In")) + delta / 10
    return pd.Series(raw, index=df.index).rolling(window=win).mean().bfill().round(0)


def _calc_pressures(df, props, patm):
    print(f"  [_calc_pressures] 컬럼: P_Comp_Out={'Y' if 'P_Comp_Out' in df.columns else 'N'}, "
          f"P_Cond_Out={'Y' if 'P_Cond_Out' in df.columns else 'N'}")

    # ── 고압 (토출압) ──
    if "P_Comp_Out" not in df.columns and "P_Cond_Out" not in df.columns:
        # P_Comp_Out이 없으면 냉매 온도로 psat 역산
        # T_Cond_M1만 2상 영역이므로 psat 역산 유효
        if "T_Cond_M1" in df.columns:
            p = props.psat_t_barg(df["T_Cond_M1"].values)
            df["P_Comp_Out"] = p; df["P_Cond_Out"] = p
            print(f"  [P_Comp_Out] ← psat(T_Cond_M1), 범위: {p.min():.2f}~{p.max():.2f} barg")
        else:
            # T_Cond_In/T_Comp_Out → 과열 상태라 psat이 약간 높게 나옴 (근사)
            for c in ["T_Cond_In", "T_Comp_Out", "T_Comp_Body"]:
                if c in df.columns:
                    p = props.psat_t_barg(df[c].values)
                    df["P_Comp_Out"] = p; df["P_Cond_Out"] = p
                    print(f"  ⚠️ [P_Comp_Out] ← psat({c}), 과열 근사! 범위: {p.min():.2f}~{p.max():.2f} barg")
                    break
            else:
                print(f"  ❌ [P_Comp_Out] 계산 불가 — 냉매 온도 없음")
    else:
        if "P_Comp_Out" in df.columns:
            print(f"  [P_Comp_Out] ← 센서값, 범위: {df['P_Comp_Out'].min():.2f}~{df['P_Comp_Out'].max():.2f}")
        if "P_Cond_Out" not in df.columns and "T_Cond_M1" in df.columns:
            df["P_Cond_Out"] = props.psat_t_barg(df["T_Cond_M1"].values)

    # ── 저압 (흡입압) ──
    if "P_Comp_In" not in df.columns and "P_Eva_In" not in df.columns:
        for c in ["T_Eva_In", "Heatpump_EvaInTemp"]:
            if c in df.columns:
                p = props.psat_t_barg(df[c].values)
                df["P_Comp_In"] = p; df["P_Eva_In"] = p
                print(f"  [P_Comp_In] ← psat({c}), 범위: {p.min():.2f}~{p.max():.2f} barg")
                break
    else:
        if "P_Comp_In" in df.columns:
            print(f"  [P_Comp_In] ← 센서값, 범위: {df['P_Comp_In'].min():.2f}~{df['P_Comp_In'].max():.2f}")
        if "P_Eva_In" not in df.columns:
            for c in ["T_Eva_In", "Heatpump_EvaInTemp"]:
                if c in df.columns:
                    df["P_Eva_In"] = props.psat_t_barg(df[c].values); break

    if "T_Air_Eva_In" not in df.columns:
        if "Heatpump_DuctInTemp" in df.columns:
            df["T_Air_Eva_In"] = df["Heatpump_DuctInTemp"].values
    if "T_Air_Cond_Out" not in df.columns:
        if "Heatpump_DuctOutTemp" in df.columns:
            df["T_Air_Cond_Out"] = df["Heatpump_DuctOutTemp"].values
    return df


def _calc_discharge_superheat(df, props):
    if "P_Comp_Out" in df.columns:
        t_dew = props.tsat_p(df["P_Comp_Out"].values)
        df["T_SH_CompOut"] = df["Heatpump_CompTemp"].values - t_dew
    else:
        df["T_SH_CompOut"] = 0
    return df


def _clamp_divergent(df):
    if "Heatpump_CompTemp" in df.columns:
        df.loc[df["Heatpump_CompTemp"] >= 200, "Heatpump_CompTemp"] = -1
    if "HP_SuperHeatDegree" in df.columns:
        bad = df["HP_SuperHeatDegree"] >= 1000
        if bad.any() and "Heatpump_EvaOutTemp" in df.columns:
            df.loc[bad, "HP_SuperHeatDegree"] = (
                df.loc[bad, "Heatpump_EvaOutTemp"] - df.loc[bad, "Heatpump_EvaInTemp"])
    return df


def _derive_missing_mx100(df):
    _map = {
        "T_Comp_In": "T_Eva_Out", "T_Comp_Out": "T_Cond_In",
        "Heatpump_CompTemp": "T_Comp_Out",
        "Heatpump_EvaInTemp": "T_Eva_In", "Heatpump_EvaOutTemp": "T_Eva_Out",
        "Heatpump_DuctInTemp": "T_Air_Eva_In", "Heatpump_DuctOutTemp": "T_Air_Cond_Out",
    }
    for tgt, src in _map.items():
        if tgt not in df.columns and src in df.columns:
            df[tgt] = df[src]
    if "HP_SuperHeatDegree" not in df.columns and "T_Eva_Out" in df.columns:
        df["HP_SuperHeatDegree"] = df["T_Eva_Out"] - df["T_Eva_In"]
    if "Heatpump_DryMotionInfo" not in df.columns:
        df["Heatpump_DryMotionInfo"] = 0
    if "Hz_Comp" in df.columns:
        df["HP_CompCurrentHz"] = df["Hz_Comp"] * 120 / 6 / 60
    return df
