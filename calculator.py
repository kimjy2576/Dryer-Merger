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
    setup = cfg["processing"]["setup_type"]
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


def _calc_ftemp(df, cfg_ft):
    k1, k2, k3 = cfg_ft["k1"], cfg_ft["k2"], cfg_ft["k3"]
    delta = cfg_ft["auto_delta"]
    win = int(np.round(cfg_ft["moving_avg_window_min"] * 60, 0))
    raw = 10 * (k1 * df["Heatpump_DuctOutTemp"].values
                - k2 * df["Heatpump_DuctInTemp"].values
                + k3 * df["Heatpump_EvaInTemp"].values) + delta / 10
    return pd.Series(raw, index=df.index).rolling(window=win).mean().bfill().round(0)


def _calc_pressures(df, props, patm):
    if "P_Comp_Out" not in df.columns and "P_Cond_Out" not in df.columns:
        ref_col = "T_Cond_M1" if "T_Cond_M1" in df.columns else "Heatpump_DuctOutTemp"
        p = props.psat_t_barg(df[ref_col].values)
        df["P_Comp_Out"] = p
        df["P_Cond_Out"] = p
    elif "P_Cond_Out" not in df.columns:
        ref_col = "T_Cond_M1" if "T_Cond_M1" in df.columns else "Heatpump_DuctOutTemp"
        df["P_Cond_Out"] = props.psat_t_barg(df[ref_col].values)

    if "P_Comp_In" not in df.columns and "P_Eva_In" not in df.columns:
        p = props.psat_t_barg(df["Heatpump_EvaInTemp"].values)
        df["P_Comp_In"] = p
        df["P_Eva_In"] = p
    elif "P_Eva_In" not in df.columns:
        df["P_Eva_In"] = props.psat_t_barg(df["Heatpump_EvaInTemp"].values)

    if "T_Air_Eva_In" not in df.columns:
        df["T_Air_Eva_In"] = np.round(df["Heatpump_DuctInTemp"].values * 0.8 + 9, 2)
    if "T_Air_Cond_Out" not in df.columns:
        df["T_Air_Cond_Out"] = np.round(df["Heatpump_DuctOutTemp"].values * 0.8 + 9, 2)
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
