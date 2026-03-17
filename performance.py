"""
performance.py — Stage 2: 열역학 성능 계산
merged CSV → 응축기/증발기/압축기 물성 → 질량유량 → 성능지표 → 단일 출력 CSV

모든 하드코딩 제거 → cfg 딕셔너리에서 파라미터 참조.
CoolProp 호출 → properties.py LUT로 대체.
"""
import numpy as np
import pandas as pd
from properties import (
    get_props, psat_water, abs_humidity, rh_from_ah,
    h_moist_air, v_moist_air,
)


def run_stage2(df: pd.DataFrame, cfg: dict, exp: dict | None = None) -> pd.DataFrame:
    """
    Stage 2 성능 계산 메인 엔트리.

    Parameters:
        df   : Stage 1 완료된 merged DataFrame
        cfg  : YAML 설정
        exp  : 실험 입력값 {"load_kg": float, "imc_kg": float, "fmc_kg": float}
               None이면 RMC 계산 생략
    Returns:
        df_calc : 계산 결과 DataFrame (df 컬럼 + 계산 컬럼)
    """
    calc_cfg = cfg["calculation"]
    env = cfg["environment"]
    patm = env["patm"]
    props = get_props(env["refrigerant"], patm, env.get("backend", "HEOS"))
    rh_eva_out = calc_cfg["rh_eva_out_assumed"]
    time_interval = cfg["processing"]["time_interval"]

    # ── 0. 전처리: 센서 오프셋 + 누락 컬럼 ──
    df = _apply_sensor_offsets(df, calc_cfg.get("sensor_offsets", {}))
    df = _ensure_columns(df)
    df = _apply_power_corrections(df, calc_cfg)

    # ── 1. 응축기 공기측 ──
    air_cond = _calc_condenser_air(df, rh_eva_out, patm)

    # ── 2. 응축기 냉매측 ──
    ref_cond = _calc_condenser_ref(df, props)

    # ── 3. 증발기 냉매측 ──
    ref_eva = _calc_evaporator_ref(df, props, ref_cond)

    # ── 4. 압축기 냉매측 ──
    ref_comp = _calc_compressor_ref(df, props)

    # ── 5. 질량유량 & 열전달량 ──
    flow = _calc_mass_flow(df, cfg, air_cond, ref_cond, ref_eva, ref_comp, time_interval)

    # ── 6. 증발기 공기측 + 성능지표 ──
    perf = _calc_evaporator_air_and_performance(
        df, props, patm, rh_eva_out, air_cond, flow, time_interval, exp
    )

    # ── 7. 에너지 적산 ──
    df = _calc_energy_integration(df, calc_cfg, time_interval)

    # ── 8. 결과 DataFrame 조립 ──
    df_calc = _assemble_output(df, air_cond, ref_cond, ref_eva, ref_comp, flow, perf)

    # ── 9. 후처리 LPF ──
    lpf_cols = calc_cfg.get("calc_lpf_columns", [])
    tau = calc_cfg.get("calc_lpf_tau", 10)
    if lpf_cols:
        valid = [c for c in lpf_cols if c in df_calc.columns]
        if valid:
            alpha = 1.0 / (tau + 1.0)
            df_calc[valid] = df_calc[valid].ewm(alpha=alpha, adjust=False).mean().round(2)

    return df_calc


# ════════════════════════════════════════════════
#  0. 전처리
# ════════════════════════════════════════════════
def _apply_sensor_offsets(df, offsets: dict):
    """센서 오프셋 보정 (YAML에서 읽은 값만큼 가감)."""
    for col, offset in offsets.items():
        if col in df.columns and offset != 0:
            df[col] = df[col] + offset
    return df


def _ensure_columns(df):
    """누락 컬럼 대체."""
    defaults = {
        "Heatpump_EvaInTemp": "T_Eva_In",
        "Heatpump_EvaOutTemp": "T_Eva_Out",
        "Heatpump_DuctInTemp": "T_Air_Eva_In",
        "Heatpump_DuctOutTemp": "T_Air_Cond_Out",
        "Heatpump_CompTemp": "T_Comp_Out",
    }
    for tgt, src in defaults.items():
        if tgt not in df.columns and src in df.columns:
            df[tgt] = df[src]

    for col in ["Po_Comp", "Po_Fan", "Po_WD"]:
        if col not in df.columns:
            if "d_currentABS_IqeRef" in df.columns:
                df[col] = df["d_currentABS_IqeRef"] * 1e9
            else:
                df[col] = 0

    if "Heatpump_DryMotionInfo" not in df.columns:
        df["Heatpump_DryMotionInfo"] = 0
    if "HP_CompCurrentHz" not in df.columns:
        df["HP_CompCurrentHz"] = 0
    if "T_Comp_In" not in df.columns and "Heatpump_EvaOutTemp" in df.columns:
        df["T_Comp_In"] = df["Heatpump_EvaOutTemp"]
    if "T_Comp_Out" not in df.columns and "T_Cond_In" in df.columns:
        df["T_Comp_Out"] = df["T_Cond_In"]
    return df


def _apply_power_corrections(df, calc_cfg):
    """DAQ PC별 전력 보정."""
    pc = calc_cfg.get("selected_pc", "none")
    corrections = calc_cfg.get("power_corrections", {}).get(pc, {})
    for col, val in corrections.items():
        if col in df.columns and val != 0:
            df[col] = df[col] + val
    return df


# ════════════════════════════════════════════════
#  1. 응축기 공기측
# ════════════════════════════════════════════════
def _calc_condenser_air(df, rh_eva_out, patm):
    T_in = df["T_Air_Eva_Out"].values.astype(float) if "T_Air_Eva_Out" in df.columns \
        else df["Heatpump_DuctInTemp"].values.astype(float)
    T_out = df["Heatpump_DuctOutTemp"].values.astype(float)

    # 입구 (= 증발기 출구)
    AH_in = abs_humidity(T_in, rh_eva_out, patm)
    h_in = h_moist_air(T_in, AH_in)
    v_in = v_moist_air(T_in, AH_in, patm)

    # 출구 (AH 보존, 현열 교환)
    AH_out = AH_in
    RH_out = rh_from_ah(AH_out, T_out, patm)
    h_out = h_moist_air(T_out, AH_out)
    v_out = v_moist_air(T_out, AH_out, patm)

    return {
        "AH_in": AH_in, "AH_out": AH_out,
        "RH_in": np.full_like(T_in, rh_eva_out), "RH_out": RH_out,
        "h_in": h_in, "h_out": h_out,
        "v_in": v_in, "v_out": v_out,
    }


# ════════════════════════════════════════════════
#  2. 응축기 냉매측
# ════════════════════════════════════════════════
def _calc_condenser_ref(df, props):
    T_in = df["T_Cond_In"].values.astype(float) if "T_Cond_In" in df.columns \
        else df["T_Comp_Out"].values.astype(float) if "T_Comp_Out" in df.columns \
        else df["Heatpump_CompTemp"].values.astype(float)
    T_out = df["T_Cond_Out"].values.astype(float) if "T_Cond_Out" in df.columns \
        else df["Heatpump_DuctOutTemp"].values.astype(float)
    T_m1 = df["T_Cond_M1"].values.astype(float) if "T_Cond_M1" in df.columns else T_out
    P_in = df["P_Comp_Out"].values.astype(float)
    P_out = df["P_Cond_Out"].values.astype(float) if "P_Cond_Out" in df.columns else P_in

    h_in = props.h_tp_superheat(T_in, P_in)
    s_in = props.s_tp_superheat(T_in, P_in)
    v_in = props.v_tp_superheat(T_in, P_in)

    h_out = props.h_tp_subcool(T_out, P_in)
    s_out = props.s_tp_subcool(T_out, P_in)
    v_out = props.v_tp_subcool(T_out, P_in)

    dh = h_in - h_out
    T_dew = props.tsat_p((P_in + P_out) / 2)
    T_sc = T_m1 - T_out

    return {
        "h_in": h_in, "h_out": h_out, "s_in": s_in, "s_out": s_out,
        "v_in": v_in, "v_out": v_out, "dh": dh,
        "T_dew": T_dew, "T_subcool": T_sc,
    }


# ════════════════════════════════════════════════
#  3. 증발기 냉매측
# ════════════════════════════════════════════════
def _calc_evaporator_ref(df, props, ref_cond):
    P_in = df["P_Eva_In"].values.astype(float) if "P_Eva_In" in df.columns \
        else df["P_Comp_In"].values.astype(float)
    P_out = df["P_Comp_In"].values.astype(float)
    T_out = df["Heatpump_EvaOutTemp"].values.astype(float)

    # 입구: 과냉 출구와 동일 (등엔탈피 팽창)
    h_in = ref_cond["h_out"].copy()
    if "T_Subcooler_Out" in df.columns:
        T_sub = df["T_Subcooler_Out"].values.astype(float)
        P_high = df["P_Comp_Out"].values.astype(float)
        h_in = props.h_tp_subcool(T_sub, P_high)

    h_out = props.h_tp_superheat(T_out, P_out)
    s_out = props.s_tp_superheat(T_out, P_out)
    v_out = props.v_tp_superheat(T_out, P_out)

    dh = h_out - h_in
    T_boil = props.tsat_p((P_in + P_out) / 2)
    T_sh = T_out - T_boil

    h_lat = props.h_latent(P_in)
    h_liq = props.h_liq(P_in)
    quality = np.where(h_lat > 0, (h_in - h_liq) / h_lat, 0)

    return {
        "h_in": h_in, "h_out": h_out,
        "s_in": ref_cond["s_out"], "s_out": s_out,
        "v_in": ref_cond["v_out"], "v_out": v_out,
        "dh": dh, "T_boil": T_boil, "T_superheat": T_sh, "quality": quality,
    }


# ════════════════════════════════════════════════
#  4. 압축기 냉매측
# ════════════════════════════════════════════════
def _calc_compressor_ref(df, props):
    T_in = df["T_Comp_In"].values.astype(float) if "T_Comp_In" in df.columns \
        else df["Heatpump_EvaOutTemp"].values.astype(float)
    P_in = df["P_Comp_In"].values.astype(float)
    T_out = df["T_Cond_In"].values.astype(float) if "T_Cond_In" in df.columns \
        else df["Heatpump_CompTemp"].values.astype(float)
    P_out = df["P_Comp_Out"].values.astype(float)

    h_in = props.h_tp_superheat(T_in, P_in)
    h_out = props.h_tp_superheat(T_out, P_out)
    s_in = props.s_tp_superheat(T_in, P_in)
    s_out = props.s_tp_superheat(T_out, P_out)
    v_in = props.v_tp_superheat(T_in, P_in)
    v_out = props.v_tp_superheat(T_out, P_out)
    rho_in = props.rho_tp(T_in, P_in)

    dh = h_out - h_in
    pr = np.where(P_in > 0, P_out / P_in, 1)

    return {
        "h_in": h_in, "h_out": h_out, "s_in": s_in, "s_out": s_out,
        "v_in": v_in, "v_out": v_out, "rho_in": rho_in,
        "dh": dh, "pr": pr,
    }


# ════════════════════════════════════════════════
#  5. 질량유량 & 열전달량
# ════════════════════════════════════════════════
def _calc_mass_flow(df, cfg, air_cond, ref_cond, ref_eva, ref_comp, dt):
    calc_cfg = cfg["calculation"]
    af = calc_cfg.get("airflow_estimation", {})

    dh_air = air_cond["h_out"] - air_cond["h_in"]
    safe_dh = np.where(np.abs(dh_air) < 1e-9, 1e-6, dh_air)
    v_out = air_cond["v_out"]
    AH_out = air_cond["AH_out"]

    # 풍량 결정
    if af.get("method") == "fan_power" and "Po_Fan" in df.columns:
        cmm = df["Po_Fan"].values.astype(float) / af["coeff_a"] * af["coeff_b"]
    else:
        cmm = np.full(len(df), 1.0)

    # 열전달량 (공기 기준)
    Q_cond = cmm * (60 * safe_dh * 1000) / (v_out * 3600)

    # 냉매 유량 역산
    safe_dh_ref = np.where(ref_cond["dh"] == 0, 1e-6, ref_cond["dh"])
    mdot_ref = Q_cond / (safe_dh_ref * 1000) * 3600

    # 증발기 열량
    Q_eva = mdot_ref * ref_eva["dh"] * 1000 / 3600

    # 건공기 / 습공기 유량
    v_da = v_out / (1 + AH_out)
    mdot_mair = cmm * 60 / np.where(v_da == 0, 1e-6, v_da)
    mdot_dair = mdot_mair / (1 + AH_out)

    # 필터링 (경험적)
    mdot_filtered = _filter_mass_flow(mdot_ref, df["Time_min"].values, ref_cond["dh"])
    Q_eva_filtered = mdot_filtered / 3600 * ref_eva["dh"] * 1000

    return {
        "cmm_cond": cmm, "mdot_mair": mdot_mair, "mdot_dair": mdot_dair,
        "mdot_ref": mdot_ref, "mdot_ref_filtered": mdot_filtered,
        "Q_cond": Q_cond, "Q_eva": Q_eva, "Q_eva_filtered": Q_eva_filtered,
    }


def _filter_mass_flow(raw, time_min, dh_cond):
    """경험 기반 유량 필터링."""
    safe = np.where(dh_cond == 0, 1e-6, dh_cond)
    flow = raw.copy()
    prev = 0.0
    for i in range(len(flow)):
        v = flow[i]
        if v < 0:
            v = 0
        elif time_min[i] < 5 and v > 10:
            v = 0
        elif abs(v) > 50:
            v = prev
        else:
            prev = v
        flow[i] = v
    return flow


# ════════════════════════════════════════════════
#  6. 증발기 공기측 + 성능지표
# ════════════════════════════════════════════════
def _calc_evaporator_air_and_performance(df, props, patm, rh_eva_out, air_cond, flow, dt, exp):
    T_eva_in = df["Heatpump_DuctInTemp"].values.astype(float)
    T_eva_out = df["T_Air_Eva_Out"].values.astype(float) if "T_Air_Eva_Out" in df.columns \
        else df["Heatpump_DuctInTemp"].values.astype(float) - 5
    AH_cond_in = air_cond["AH_in"]
    mdot_dair = flow["mdot_dair"]
    Q_eva = flow["Q_eva_filtered"]
    Po_comp = df["Po_Comp"].values.astype(float)
    Po_wd = df["Po_WD"].values.astype(float)
    time_min = df["Time_min"].values.astype(float)

    # AH / RH 역산 (증발기 입구)
    safe_flow = np.where(mdot_dair == 0, 1.0, mdot_dair)
    num = (Q_eva / safe_flow * 3.6) - 1.006 * (T_eva_in - T_eva_out)
    den = 2501 + 1.86 * (T_eva_in - T_eva_out)
    AH_eva_in = np.where(mdot_dair == 0, 1e-6, num / den + AH_cond_in)
    AH_eva_in = np.maximum(AH_eva_in, 1e-8)
    RH_eva_in = rh_from_ah(AH_eva_in, T_eva_in, patm)

    h_eva_in = h_moist_air(T_eva_in, AH_eva_in)

    # 현열 / 잠열
    Q_sen_da = mdot_dair * 1.006 * (T_eva_in - T_eva_out) / 3.6
    Q_sen_w = mdot_dair * (AH_cond_in * 1.805) * (T_eva_in - T_eva_out) / 3.6
    Q_sen = Q_sen_da + Q_sen_w
    Q_lat = Q_eva - Q_sen

    SFH = np.where(Q_eva != 0, Q_sen / Q_eva, 0)
    LFH = np.where(Q_eva != 0, Q_lat / Q_eva, 0)

    # COP
    COP_cool = np.where(Po_comp != 0, Q_eva / Po_comp, 0)
    COP_heat = np.where(Po_comp != 0, flow["Q_cond"] / Po_comp, 0)
    COP_sys_cool = np.where(Po_wd != 0, Q_eva / Po_wd, 0)
    COP_sys_heat = np.where(Po_wd != 0, flow["Q_cond"] / Po_wd, 0)

    # 응축수량
    cold_start = df.loc[df["Heatpump_DryMotionInfo"] == 2, "Time_min"]
    cold_t = cold_start.min() if not cold_start.empty else 99999
    mask_cold = time_min >= cold_t
    water_gs = np.where(mask_cold, 0,
                        (AH_eva_in - AH_cond_in) * mdot_dair * 1000 / 3600)
    water_gm = water_gs * 60
    water_kg = np.cumsum(water_gm * (dt / 60) / 1000)

    # RMC 보정
    rmc = np.zeros_like(water_kg)
    if exp and exp.get("imc_kg") and exp.get("load_kg"):
        rmc = _calc_rmc(water_kg, exp["imc_kg"], exp["fmc_kg"], exp["load_kg"])

    # 건조 COP / 압축기 효율
    h_cond_out = air_cond["h_out"]
    h_eva_in_air = h_eva_in
    Q_dry = mdot_dair * (h_eva_in_air - h_cond_out) / 3600
    COP_dry = np.where(Po_comp != 0, Q_dry / Po_comp, 0)

    # 압축기 효율
    comp_work = flow["mdot_ref_filtered"] * flow.get("dh_comp", 0) * 1000 / 3600 \
        if "dh_comp" not in flow else \
        flow["mdot_ref_filtered"] * 0  # fallback

    # SMER
    SMER = np.where(Q_dry != 0, water_gs / (Q_dry / 1000), 0)

    # 풍량 (증발기측)
    v_da_eva = v_moist_air(T_eva_in, AH_eva_in, patm)
    cmm_eva = mdot_dair * v_da_eva / 60

    return {
        "AH_eva_in": AH_eva_in, "RH_eva_in": RH_eva_in,
        "h_eva_in_air": h_eva_in, "Q_sen": Q_sen, "Q_lat": Q_lat,
        "SFH": SFH, "LFH": LFH,
        "COP_cool": COP_cool, "COP_heat": COP_heat,
        "COP_sys_cool": COP_sys_cool, "COP_sys_heat": COP_sys_heat,
        "water_gs": water_gs, "water_gm": water_gm, "water_kg": water_kg,
        "rmc": rmc, "Q_dry": Q_dry, "COP_dry": COP_dry, "SMER": SMER,
        "cmm_eva": cmm_eva,
    }


def _calc_rmc(water_kg, imc, fmc, load):
    """RMC 보정 (가중치 + 단조증가 + 스무딩)."""
    actual_delta = imc - fmc
    calc_delta = water_kg[-1] - water_kg[0]
    wf = actual_delta / calc_delta if calc_delta != 0 else 1.0

    adj = (water_kg - water_kg[0]) * wf
    current_mass = imc - adj
    rmc = ((current_mass / load) - 1) * 100

    # 스무딩
    win = min(15, len(rmc) // 4)
    if win > 1:
        kernel = np.ones(win) / win
        raw_water = imc - (rmc / 100 + 1) * load
        smooth = np.convolve(raw_water, kernel, mode="same")
        pad = min(5, len(smooth) // 4)
        if pad > 0 and len(smooth) > pad * 2:
            smooth[:pad] = np.linspace(raw_water[0], smooth[pad], pad)
            smooth[-pad:] = np.linspace(smooth[-pad], raw_water[-1], pad)
        smooth = np.maximum.accumulate(smooth)
        smooth = np.minimum(smooth, raw_water[-1])
        smooth[-1] = raw_water[-1]
        rmc = ((imc - smooth) / load - 1) * 100

    return np.round(rmc, 2)


# ════════════════════════════════════════════════
#  7. 에너지 적산
# ════════════════════════════════════════════════
def _calc_energy_integration(df, calc_cfg, dt):
    factor = dt / 3600

    # 드럼 전력 추정
    if "Po_Drum" not in df.columns:
        dp = calc_cfg.get("drum_power", {})
        on_w, off_w = dp.get("on_watts", 25), dp.get("off_watts", 10)
        on_s, off_s = dp.get("on_seconds", 56), dp.get("off_seconds", 4)
        cycle = on_s + off_s
        df["Po_Drum_calc"] = np.where(df["Time_sec"] % cycle < on_s, on_w, off_w)

    # 에너지 적산
    for col in ["Po_WD", "Po_Comp", "Po_Fan", "Po_Drum_calc"]:
        if col in df.columns:
            e_col = col.replace("Po_", "E_")
            df[e_col] = (df[col] * factor).cumsum()
    return df


# ════════════════════════════════════════════════
#  8. 출력 DataFrame 조립
# ════════════════════════════════════════════════
def _assemble_output(df, air_cond, ref_cond, ref_eva, ref_comp, flow, perf):
    """모든 계산 결과를 단일 DataFrame으로 조립."""
    out = df.copy()

    # 응축기 공기
    out["AH_Cond_In"] = air_cond["AH_in"]
    out["AH_Cond_Out"] = air_cond["AH_out"]
    out["RH_Cond_In"] = air_cond["RH_in"]
    out["RH_Cond_Out"] = air_cond["RH_out"]
    out["h_Cond_In_MAir"] = air_cond["h_in"]
    out["h_Cond_Out_MAir"] = air_cond["h_out"]

    # 응축기 냉매
    out["h_Cond_In_ref"] = ref_cond["h_in"]
    out["h_Cond_Out_ref"] = ref_cond["h_out"]
    out["s_Cond_In_ref"] = ref_cond["s_in"]
    out["s_Cond_Out_ref"] = ref_cond["s_out"]
    out["v_Cond_In_ref"] = ref_cond["v_in"]
    out["v_Cond_Out_ref"] = ref_cond["v_out"]
    out["T_Dew_ref"] = ref_cond["T_dew"]
    out["T_subcooling"] = ref_cond["T_subcool"]

    # 증발기 냉매
    out["h_Eva_In_ref"] = ref_eva["h_in"]
    out["h_Eva_Out_ref"] = ref_eva["h_out"]
    out["s_Eva_In_ref"] = ref_eva["s_in"]
    out["s_Eva_Out_ref"] = ref_eva["s_out"]
    out["v_Eva_In_ref"] = ref_eva["v_in"]
    out["v_Eva_Out_ref"] = ref_eva["v_out"]
    out["T_Boiling_ref"] = ref_eva["T_boil"]
    out["T_Superheating"] = ref_eva["T_superheat"]
    out["Quality_x"] = ref_eva["quality"]

    # 압축기 냉매
    out["h_Comp_In_ref"] = ref_comp["h_in"]
    out["h_Comp_Out_ref"] = ref_comp["h_out"]
    out["s_Comp_In_ref"] = ref_comp["s_in"]
    out["s_Comp_Out_ref"] = ref_comp["s_out"]
    out["v_Comp_In_ref"] = ref_comp["v_in"]
    out["v_Comp_Out_ref"] = ref_comp["v_out"]
    out["Ratio_Compression"] = ref_comp["pr"]

    # 유량
    out["Flow_air_Cond_CMM"] = flow["cmm_cond"]
    out["Flow_Mair_Cond_kgH"] = flow["mdot_mair"]
    out["Flow_Dair_kgH"] = flow["mdot_dair"]
    out["Flow_ref_kgH"] = flow["mdot_ref"]
    out["Flow_ref_kgH_recalc"] = flow["mdot_ref_filtered"]

    # 열전달량
    out["Qrefr_Cond_recalc"] = flow["Q_cond"]
    out["Qrefr_Eva_recalc"] = flow["Q_eva_filtered"]
    out["Q_sen"] = perf["Q_sen"]
    out["Q_lat"] = perf["Q_lat"]
    out["Q_Dry"] = perf["Q_dry"]

    # 성능
    out["Factor_SFH"] = perf["SFH"]
    out["Factor_LFH"] = perf["LFH"]
    out["COP_cooling"] = perf["COP_cool"]
    out["COP_heating"] = perf["COP_heat"]
    out["COP_sys_cooling"] = perf["COP_sys_cool"]
    out["COP_sys_heating"] = perf["COP_sys_heat"]
    out["COP_Dry"] = perf["COP_dry"]
    out["SMER"] = perf["SMER"]

    # 습도
    out["AH_Eva_In"] = perf["AH_eva_in"]
    out["RH_Eva_In"] = perf["RH_eva_in"]
    out["h_Eva_In_MAir"] = perf["h_eva_in_air"]
    out["Flow_air_Eva_CMM"] = perf["cmm_eva"]

    # 수분
    out["Water_calc_gM"] = perf["water_gm"]
    out["Water_calc_kg"] = perf["water_kg"]
    out["RMC_calc"] = perf["rmc"]

    return out
