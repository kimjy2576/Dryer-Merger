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
    각 계산 블록의 필수 컬럼을 사전 검증하여, 누락 시 해당 블록을 스킵한다.
    스킵된 블록 정보는 df.attrs["skipped_blocks"]에 저장.
    """
    calc_cfg = cfg["calculation"]
    env = cfg["environment"]
    patm = env["patm"]
    props = get_props(env["refrigerant"], patm, env.get("backend", "HEOS"))
    rh_eva_out = calc_cfg["rh_eva_out_assumed"]
    time_interval = cfg["processing"]["time_interval"]
    n = len(df)

    # ── 0. 전처리 ──
    df = _apply_sensor_offsets(df, calc_cfg.get("sensor_offsets", {}))
    df = _ensure_columns(df)
    df = _apply_power_corrections(df, calc_cfg)

    # ── 블록별 의존성 정의 ──
    BLOCKS = {
        "condenser_air":  {"required": [["T_Air_Eva_Out", "T_Air_Eva_In"], ["T_Air_Cond_Out"]],
                           "label": "응축기 공기측"},
        "condenser_ref":  {"required": [["T_Cond_In", "T_Comp_Out", "T_Comp_Body"], ["T_Cond_Out"], ["P_Comp_Out"]],
                           "label": "응축기 냉매측"},
        "evaporator_ref": {"required": [["P_Comp_In"], ["T_Eva_Out"]],
                           "label": "증발기 냉매측", "depends": ["condenser_ref"]},
        "compressor_ref": {"required": [["T_Comp_In", "T_Eva_Out"], ["P_Comp_In"],
                                         ["T_Cond_In", "T_Comp_Body"], ["P_Comp_Out"]],
                           "label": "압축기 냉매측"},
        "mass_flow":      {"label": "질량유량 & 열전달",
                           "depends": ["condenser_air", "condenser_ref", "evaporator_ref", "compressor_ref"]},
        "performance":    {"required": [["T_Air_Eva_In"], ["Po_Comp"], ["Po_WD"]],
                           "label": "성능지표",
                           "depends": ["mass_flow"]},
    }

    def _check_block(name):
        """블록의 필수 컬럼 체크. 각 그룹은 OR 관계 (하나라도 있으면 OK)."""
        info = BLOCKS[name]
        # 의존 블록이 스킵됐으면 자동 스킵
        for dep in info.get("depends", []):
            if dep in skipped:
                return False, f"{BLOCKS[dep]['label']} 스킵됨"
        # 컬럼 체크
        for group in info.get("required", []):
            if not any(c in df.columns for c in group):
                return False, f"필요 컬럼 없음: {' 또는 '.join(group)}"
        return True, ""

    skipped = {}  # {block_name: reason}
    computed = {} # 계산 결과 저장
    has_rh = "RH_Eva_In" in df.columns

    # ── 2. 응축기 냉매측 (공기 무관, 1회만) ──
    ok, reason = _check_block("condenser_ref")
    if ok:
        computed["ref_cond"] = _calc_condenser_ref(df, props)
    else:
        skipped["condenser_ref"] = reason
        computed["ref_cond"] = _empty_ref(n)

    # ── 3. 증발기 냉매측 (공기 무관, 1회만) ──
    ok, reason = _check_block("evaporator_ref")
    if ok:
        computed["ref_eva"] = _calc_evaporator_ref(df, props, computed["ref_cond"])
    else:
        skipped["evaporator_ref"] = reason
        computed["ref_eva"] = _empty_ref_eva(n)

    # ── 4. 압축기 냉매측 (공기 무관, 1회만) ──
    ok, reason = _check_block("compressor_ref")
    if ok:
        computed["ref_comp"] = _calc_compressor_ref(df, props)
    else:
        skipped["compressor_ref"] = reason
        computed["ref_comp"] = _empty_ref_comp(n)

    # ── 1+5+6: 공기측 수렴 반복 ──
    ok_air, reason_air = _check_block("condenser_air")
    ok_flow, reason_flow = _check_block("mass_flow")
    ok_perf, reason_perf = _check_block("performance")

    if ok_air and ok_flow and ok_perf and not has_rh:
        # ━━━ RH 미측정: 수렴 반복 ━━━
        iter_cfg = calc_cfg.get("rh_iteration", {})
        max_iter = iter_cfg.get("max_iter", 15)
        tol = iter_cfg.get("tolerance", 0.005)       # RH 수렴 임계값
        damping = iter_cfg.get("damping", 0.4)        # 감쇠 계수 (안정성)

        T_eva_out = df["T_Air_Eva_Out"].values.astype(float) if "T_Air_Eva_Out" in df.columns \
            else df["T_Air_Eva_In"].values.astype(float)

        # 초기 가정: 스칼라 → 벡터
        rh_guess = np.full(n, rh_eva_out)
        converge_info = {"iterations": 0, "residual": 1.0, "converged": False}

        for it in range(max_iter):
            # 1. 응축기 공기측 (rh_guess 벡터 사용)
            computed["air_cond"] = _calc_condenser_air(df, rh_guess, patm)

            # 5. 질량유량 & 열전달
            computed["flow"] = _calc_mass_flow(
                df, cfg, computed["air_cond"], computed["ref_cond"],
                computed["ref_eva"], computed["ref_comp"], time_interval)

            # 6. 성능지표 (에너지 밸런스로 AH_eva_in 역산)
            computed["perf"] = _calc_evaporator_air_and_performance(
                df, props, patm, rh_eva_out, computed["air_cond"],
                computed["flow"], time_interval, exp)

            # ── 수렴 체크: AH_eva_in에서 AH_eva_out 역산 ──
            AH_eva_in = computed["perf"]["AH_eva_in"]
            AH_cond_in = computed["air_cond"]["AH_in"]
            mdot_dair = computed["flow"]["mdot_dair"]
            water_gs = computed["perf"]["water_gs"]

            # 증발기 출구 AH = 입구 AH - 제습량/건공기유량
            safe_mdot = np.where(mdot_dair == 0, 1.0, mdot_dair)
            AH_eva_out_new = AH_eva_in - water_gs / 1000.0 * 3600.0 / safe_mdot
            AH_eva_out_new = np.maximum(AH_eva_out_new, 1e-8)

            # 새 RH_eva_out 계산
            rh_new = rh_from_ah(AH_eva_out_new, T_eva_out, patm)
            rh_new = np.clip(rh_new, 0.3, 1.0)  # 물리적 범위 제한

            # 잔차
            residual = np.nanmean(np.abs(rh_new - rh_guess))
            converge_info["iterations"] = it + 1
            converge_info["residual"] = float(residual)

            if residual < tol:
                converge_info["converged"] = True
                break

            # 감쇠 업데이트
            rh_guess = damping * rh_new + (1.0 - damping) * rh_guess

        # 수렴 정보 저장
        computed["perf"]["converge_info"] = converge_info

    elif ok_air and ok_flow and ok_perf and has_rh:
        # ━━━ RH 측정 있음: 반복 불필요, 1회 계산 ━━━
        computed["air_cond"] = _calc_condenser_air(df, rh_eva_out, patm)
        computed["flow"] = _calc_mass_flow(
            df, cfg, computed["air_cond"], computed["ref_cond"],
            computed["ref_eva"], computed["ref_comp"], time_interval)
        computed["perf"] = _calc_evaporator_air_and_performance(
            df, props, patm, rh_eva_out, computed["air_cond"],
            computed["flow"], time_interval, exp)

    else:
        # ━━━ 필수 컬럼 부족: 스킵 ━━━
        if not ok_air:
            skipped["condenser_air"] = reason_air
            computed["air_cond"] = _empty_air_cond(n)
        else:
            computed["air_cond"] = _calc_condenser_air(df, rh_eva_out, patm)
        if not ok_flow:
            skipped["mass_flow"] = reason_flow
            computed["flow"] = _empty_flow(n)
        else:
            computed["flow"] = _empty_flow(n)  # 의존 블록 스킵 시
        if not ok_perf:
            skipped["performance"] = reason_perf
        computed["perf"] = _empty_perf(n)

    # ── 7. 에너지 적산 ──
    df = _calc_energy_integration(df, calc_cfg, time_interval)

    # ── 7.5. IMC/FMC 기반 유량 보정 → 전체 재계산 ──
    imc_flow, imc_perf, imc_info = None, None, None
    use_imc = calc_cfg.get("use_imc_correction", True)
    if (use_imc and exp and exp.get("imc_kg")
            and computed.get("flow") and computed.get("perf")
            and computed["perf"].get("water_kg") is not None
            and len(computed["perf"]["water_kg"]) > 0):
        imc_flow, imc_perf, imc_info = _apply_imc_full(
            df, cfg, props, patm, rh_eva_out,
            computed["air_cond"], computed["ref_cond"], computed["ref_eva"],
            computed["flow"], computed["perf"], time_interval, exp)

    # ── 8. 결과 조립 ──
    df_calc = _assemble_output(df, computed["air_cond"], computed["ref_cond"],
                                computed["ref_eva"], computed["ref_comp"],
                                computed["flow"], computed["perf"],
                                imc_flow, imc_perf, cfg)

    # ── 9. 후처리 LPF ──
    lpf_cols = calc_cfg.get("calc_lpf_columns", [])
    tau = calc_cfg.get("calc_lpf_tau", 10)
    if lpf_cols:
        valid = [c for c in lpf_cols if c in df_calc.columns]
        if valid:
            alpha = 1.0 / (tau + 1.0)
            df_calc[valid] = df_calc[valid].ewm(alpha=alpha, adjust=False).mean().round(2)

    # 스킵 정보를 attrs에 저장 (서버에서 로그 출력용)
    df_calc.attrs["skipped_blocks"] = skipped

    # 수렴 반복 정보
    converge_info = computed.get("perf", {}).get("converge_info")
    if converge_info:
        df_calc.attrs["converge_info"] = converge_info

    if imc_info:
        df_calc.attrs["imc_correction"] = imc_info

    return df_calc# ════════════════════════════════════════════════
#  빈 결과 생성기 (스킵된 블록 대체)
# ════════════════════════════════════════════════
def _empty_air_cond(n):
    z = np.zeros(n)
    return {"AH_in": z, "AH_out": z, "RH_in": z, "RH_out": z,
            "h_in": z, "h_out": z, "v_in": np.ones(n), "v_out": np.ones(n)}

def _empty_ref(n):
    z = np.zeros(n)
    return {"h_in": z, "h_out": z, "s_in": z, "s_out": z,
            "v_in": z, "v_out": z, "dh": z, "T_dew": z, "T_subcool": z}

def _empty_ref_eva(n):
    z = np.zeros(n)
    return {"h_in": z, "h_out": z, "s_in": z, "s_out": z,
            "v_in": z, "v_out": z, "dh": z, "T_boil": z, "T_superheat": z, "quality": z}

def _empty_ref_comp(n):
    z = np.zeros(n)
    return {"h_in": z, "h_out": z, "s_in": z, "s_out": z,
            "v_in": z, "v_out": z, "rho_in": z, "dh": z, "pr": np.ones(n)}

def _empty_flow(n):
    z = np.zeros(n)
    return {"cmm_cond": z, "mdot_mair": z, "mdot_dair": z,
            "mdot_ref": z, "mdot_ref_filtered": z,
            "Q_cond": z, "Q_cond_filtered": z,
            "Q_eva": z, "Q_eva_filtered": z, "eta_vol": z}

def _empty_perf(n):
    z = np.zeros(n)
    return {"AH_eva_in": z, "RH_eva_in": z, "h_eva_in_air": z,
            "Q_sen": z, "Q_lat": z, "SFH": z, "LFH": z,
            "COP_cool": z, "COP_heat": z, "COP_sys_cool": z, "COP_sys_heat": z,
            "water_gs": z, "water_gm": z, "water_kg": z,
            "rmc": z, "Q_drum_sen": z, "Q_drum_lat": z, "Q_drum_total": z,
            "Drum_SFH": z, "Drum_LFH": z, "SMER": z, "cmm_eva": z,
            "rh_mode": "skipped"}


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
    """레거시 컬럼명을 통일 명명 규칙으로 변환 + 누락 컬럼 대체."""
    # ── 레거시 → 통일명 자동 변환 ──
    rename_map = {
        "Heatpump_EvaOutTemp": "T_Eva_Out",
        "Heatpump_DuctInTemp": "T_Air_Eva_In",
        "Heatpump_DuctOutTemp": "T_Air_Cond_Out",
        "Heatpump_CompTemp": "T_Comp_Body",
        "HP_CompCurrentHz": "Ctrl_Comp_Hz",
        "Heatpump_DryMotionInfo": "Ctrl_DryMotion",
        "T_Cond_M1": "T_Cond_Mid",
        # 추가 호환
        "Heatpump_EvaInTemp": "T_Eva_In",
        "T_Air_Eva_In": "T_Air_Eva_In",  # 이미 통일명이면 그대로
    }
    actual_renames = {}
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            actual_renames[old] = new
    if actual_renames:
        df = df.rename(columns=actual_renames)

    # ── 폴백: 없는 컬럼은 대체 컬럼에서 복제 ──
    fallbacks = [
        ("T_Comp_In",      ["T_Eva_Out"]),
        ("T_Comp_Out",     ["T_Cond_In", "T_Comp_Body"]),
        ("T_Air_Eva_In",   ["T_Eva_In"]),
        ("T_Air_Cond_Out", ["T_Air_Eva_In"]),  # 최후 fallback
    ]
    for tgt, srcs in fallbacks:
        if tgt not in df.columns:
            for src in srcs:
                if src in df.columns:
                    df[tgt] = df[src]
                    break

    # ── 전력 컬럼 보정 ──
    for col in ["Po_Comp", "Po_Fan", "Po_WD"]:
        if col not in df.columns:
            df[col] = 0

    # ── 제어 컬럼 기본값 ──
    if "Ctrl_DryMotion" not in df.columns:
        df["Ctrl_DryMotion"] = 0
    if "Ctrl_Comp_Hz" not in df.columns:
        df["Ctrl_Comp_Hz"] = 0

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
        else df["T_Air_Eva_In"].values.astype(float)
    T_out = df["T_Air_Cond_Out"].values.astype(float)

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
        else df["T_Comp_Body"].values.astype(float)
    T_out = df["T_Cond_Out"].values.astype(float) if "T_Cond_Out" in df.columns \
        else df["T_Air_Cond_Out"].values.astype(float)
    T_m1 = df["T_Cond_Mid"].values.astype(float) if "T_Cond_Mid" in df.columns else T_out
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
    T_out = df["T_Eva_Out"].values.astype(float)

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
        else df["T_Eva_Out"].values.astype(float)
    P_in = df["P_Comp_In"].values.astype(float)
    T_out = df["T_Cond_In"].values.astype(float) if "T_Cond_In" in df.columns \
        else df["T_Comp_Body"].values.astype(float)
    P_out = df["P_Comp_Out"].values.astype(float)

    h_in = props.h_tp_superheat(T_in, P_in)
    h_out = props.h_tp_superheat(T_out, P_out)
    s_in = props.s_tp_superheat(T_in, P_in)
    s_out = props.s_tp_superheat(T_out, P_out)
    v_in = props.v_tp_superheat(T_in, P_in)
    v_out = props.v_tp_superheat(T_out, P_out)
    rho_in = props.rho_tp(T_in, P_in)

    # 비열비 κ = Cp/Cv (흡입 조건)
    cp_in = props.cp_tp(T_in, P_in)
    cv_in = props.cv_tp(T_in, P_in)
    kappa = np.where(cv_in > 0, cp_in / cv_in, 1.2)

    dh = h_out - h_in
    pr = np.where(P_in > 0, P_out / P_in, 1)

    # 절대압 기준 압축비 (체적효율 계산용)
    patm_bar = props.patm / 100  # kPa → bar (≈1.01325)
    P_in_abs = P_in + patm_bar
    P_out_abs = P_out + patm_bar
    pr_abs = np.where(P_in_abs > 0, P_out_abs / P_in_abs, 1)

    return {
        "h_in": h_in, "h_out": h_out, "s_in": s_in, "s_out": s_out,
        "v_in": v_in, "v_out": v_out, "rho_in": rho_in,
        "dh": dh, "pr": pr, "pr_abs": pr_abs, "kappa": kappa,
    }


# ════════════════════════════════════════════════
#  5. 질량유량 & 열전달량
# ════════════════════════════════════════════════
def _calc_mass_flow(df, cfg, air_cond, ref_cond, ref_eva, ref_comp, dt):
    """
    냉매 질량유량 계산.
    
    체적효율 기반 (기본):
      ṁ_ref = V_comp × N × η_vol × ρ_suc × 3600
      η_vol = 1 - C_v × (PR^(1/κ) - 1)
    
    팬 전력 기반 (폴백):
      CMM = Po_Fan / coeff_a × coeff_b → Q_cond → ṁ_ref
    """
    calc_cfg = cfg["calculation"]
    af = calc_cfg.get("airflow_estimation", {})

    dh_air = air_cond["h_out"] - air_cond["h_in"]
    safe_dh_air = np.where(np.abs(dh_air) < 1e-9, 1e-6, dh_air)
    v_out = air_cond["v_out"]
    AH_out = air_cond["AH_out"]

    # ── 냉매 유량 계산 방법 결정 ──
    V_cc = calc_cfg.get("compressor_volume_cc", 0)
    vol_cfg = calc_cfg.get("volumetric_efficiency", {})
    vol_method = vol_cfg.get("method", "calculated")
    C_v = vol_cfg.get("clearance_volume_ratio", 0.03)
    fixed_eta = vol_cfg.get("fixed_value", 0.85)
    # HP_CompCurrentHz → Ctrl_Comp_Hz로 리네임될 수 있음
    hz_col = None
    for c in ["Ctrl_Comp_Hz", "HP_CompCurrentHz"]:
        if c in df.columns:
            hz_col = c; break
    use_volumetric = V_cc > 0 and hz_col is not None and ref_comp.get("rho_in") is not None

    if use_volumetric:
        # ── 체적효율 기반 ──
        V_m3 = V_cc / 1e6  # cc → m³
        N_hz = df[hz_col].values.astype(float)  # Hz = rev/s
        rho_suc = ref_comp["rho_in"]   # kg/m³
        pr_abs = ref_comp["pr_abs"]    # 절대압 기준 압축비
        kappa = ref_comp["kappa"]      # 비열비

        if vol_method == "fixed":
            # 고정 체적효율
            eta_vol = np.full(len(df), fixed_eta)
            print(f"  [mass_flow] 체적효율 고정: η_vol={fixed_eta}")
        else:
            # 클리어런스 기반: η_vol = 1 - C_v × (PR^(1/κ) - 1)
            safe_kappa = np.where(kappa <= 0, 1.2, kappa)
            eta_vol = 1 - C_v * (np.power(np.maximum(pr_abs, 1), 1/safe_kappa) - 1)
            eta_vol = np.clip(eta_vol, 0.0, 1.0)
            print(f"  [mass_flow] 체적효율 계산: C_v={C_v}, η_vol 평균={np.nanmean(eta_vol):.3f}")

        # ṁ_ref [kg/h] = V [m³] × N [rev/s] × η_vol × ρ [kg/m³] × 3600
        mdot_ref = V_m3 * N_hz * eta_vol * rho_suc * 3600
        mdot_ref = np.maximum(mdot_ref, 0)

        # 열량 계산
        Q_cond = mdot_ref / 3600 * ref_cond["dh"] * 1000
        Q_eva = mdot_ref / 3600 * ref_eva["dh"] * 1000

        # 풍량 역산: Q_cond = CMM × (60 × Δh_air × 1000) / (v_out × 3600)
        cmm = Q_cond * v_out * 3600 / (60 * safe_dh_air * 1000)
        cmm = np.maximum(cmm, 0)

        print(f"  [mass_flow] V={V_cc}cc, ṁ_ref 평균={np.nanmean(mdot_ref):.2f} kg/h")
    else:
        # ── 팬 전력 기반 (폴백) ──
        if af.get("method") == "fan_power" and "Po_Fan" in df.columns:
            cmm = df["Po_Fan"].values.astype(float) / af["coeff_a"] * af["coeff_b"]
        else:
            cmm = np.full(len(df), 1.0)

        Q_cond = cmm * (60 * safe_dh_air * 1000) / (v_out * 3600)
        safe_dh_ref = np.where(ref_cond["dh"] == 0, 1e-6, ref_cond["dh"])
        mdot_ref = Q_cond / (safe_dh_ref * 1000) * 3600
        Q_eva = mdot_ref / 3600 * ref_eva["dh"] * 1000

        eta_vol = np.zeros(len(df))
        print(f"  [mass_flow] 팬 전력 기반 (폴백): coeff_a={af.get('coeff_a')}, coeff_b={af.get('coeff_b')}")

    # 건공기 / 습공기 유량
    v_da = v_out / (1 + AH_out)
    mdot_mair = cmm * 60 / np.where(v_da == 0, 1e-6, v_da)
    mdot_dair = mdot_mair / (1 + AH_out)

    # 필터링 (경험적)
    mdot_filtered = _filter_mass_flow(mdot_ref, df["Time_min"].values, ref_cond["dh"])
    Q_eva_filtered = mdot_filtered / 3600 * ref_eva["dh"] * 1000
    Q_cond_filtered = mdot_filtered / 3600 * ref_cond["dh"] * 1000

    return {
        "cmm_cond": cmm, "mdot_mair": mdot_mair, "mdot_dair": mdot_dair,
        "mdot_ref": mdot_ref, "mdot_ref_filtered": mdot_filtered,
        "Q_cond": Q_cond, "Q_cond_filtered": Q_cond_filtered,
        "Q_eva": Q_eva, "Q_eva_filtered": Q_eva_filtered,
        "eta_vol": eta_vol,
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


def _apply_imc_full(df, cfg, props, patm, rh_eva_out,
                    air_cond, ref_cond, ref_eva,
                    flow_raw, perf_raw, time_interval, exp):
    """
    IMC/FMC 기반 전체 재계산.
    
    1. scale = (IMC-FMC) / water_kg_total
    2. ṁ_ref × scale → 보정 유량
    3. 보정 유량으로 flow 전체 재계산 (Q, CMM, mdot_air 등)
    4. 보정 flow로 perf 전체 재계산 (AH, COP, SMER, water, RMC, drum 등)
    → 모든 파생 변수가 일관되게 보정됨
    """
    total_real = exp["imc_kg"] - (exp.get("fmc_kg") or 0)
    total_calc = perf_raw["water_kg"][-1] if len(perf_raw["water_kg"]) > 0 else 0

    if total_real <= 0 or total_calc <= 0:
        return None, None, {"converged": False, "reason": "IMC-FMC 또는 계산 응축수 <= 0"}

    scale = total_real / total_calc
    info = {"scale_factor": float(scale),
            "total_real_kg": float(total_real),
            "total_calc_kg": float(total_calc)}

    # ── 보정 유량 ──
    mdot_ref_corr = flow_raw["mdot_ref"] * scale
    mdot_ref_corr = np.maximum(mdot_ref_corr, 0)

    # ── flow 전체 재계산 ──
    Q_eva_corr = mdot_ref_corr / 3600 * ref_eva["dh"] * 1000
    Q_cond_corr = mdot_ref_corr / 3600 * ref_cond["dh"] * 1000

    # CMM 역산: Q_cond = CMM × (60 × Δh_air × 1000) / (v_out × 3600)
    dh_air = air_cond["h_out"] - air_cond["h_in"]
    safe_dh_air = np.where(np.abs(dh_air) < 1e-9, 1e-6, dh_air)
    v_out = air_cond["v_out"]
    AH_out = air_cond["AH_out"]
    cmm_corr = Q_cond_corr * v_out * 3600 / (60 * safe_dh_air * 1000)
    cmm_corr = np.maximum(cmm_corr, 0)

    # 공기 유량 재계산
    v_da = v_out / (1 + AH_out)
    mdot_mair_corr = cmm_corr * 60 / np.where(v_da == 0, 1e-6, v_da)
    mdot_dair_corr = mdot_mair_corr / (1 + AH_out)

    flow_corr = {
        "cmm_cond": cmm_corr,
        "mdot_mair": mdot_mair_corr,
        "mdot_dair": mdot_dair_corr,
        "mdot_ref": mdot_ref_corr,
        "mdot_ref_filtered": mdot_ref_corr,
        "Q_cond": Q_cond_corr,
        "Q_cond_filtered": Q_cond_corr,
        "Q_eva": Q_eva_corr,
        "Q_eva_filtered": Q_eva_corr,
        "eta_vol": flow_raw.get("eta_vol", np.zeros(len(df))),
    }

    # ── perf 전체 재계산 ──
    perf_corr = _calc_evaporator_air_and_performance(
        df, props, patm, rh_eva_out, air_cond,
        flow_corr, time_interval, exp)

    return flow_corr, perf_corr, info


# ════════════════════════════════════════════════
#  6. 증발기 공기측 + 성능지표
# ════════════════════════════════════════════════
def _calc_evaporator_air_and_performance(df, props, patm, rh_eva_out, air_cond, flow, dt, exp):
    T_eva_in = df["T_Air_Eva_In"].values.astype(float)
    T_eva_out = df["T_Air_Eva_Out"].values.astype(float) if "T_Air_Eva_Out" in df.columns \
        else df["T_Air_Eva_In"].values.astype(float) - 5
    AH_cond_in = air_cond["AH_in"]
    mdot_dair = flow["mdot_dair"]
    Q_eva = flow["Q_eva_filtered"]
    Po_comp = df["Po_Comp"].values.astype(float)
    Po_wd = df["Po_WD"].values.astype(float)
    time_min = df["Time_min"].values.astype(float)

    # ── RH 측정 유/무 분기 ──
    has_rh = "RH_Eva_In" in df.columns
    rh_mode = "measured" if has_rh else "calculated"

    if has_rh:
        # ── A. RH 측정 있음: 직접 계산 ──
        RH_eva_in = df["RH_Eva_In"].values.astype(float)
        # RH가 0~1 범위인지 확인 (% 단위면 변환)
        if np.nanmean(RH_eva_in) > 1.5:
            RH_eva_in = RH_eva_in / 100.0
        RH_eva_in = np.clip(RH_eva_in, 0.01, 1.0)
        AH_eva_in = abs_humidity(T_eva_in, RH_eva_in, patm)
    else:
        # ── B. RH 측정 없음: 에너지 밸런스 역산 ──
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
    cold_start = df.loc[df["Ctrl_DryMotion"] == 2, "Time_min"]
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

    # ── 드럼 열전달 (응축기출구 → 드럼 → 증발기입구) ──
    # 드럼 입구 = 응축기 공기 출구
    T_drum_in = df["T_Air_Cond_Out"].values.astype(float) if "T_Air_Cond_Out" in df.columns \
        else T_eva_in  # fallback
    AH_drum_in = air_cond["AH_out"]          # 응축기 출구 AH (현열교환이므로 AH_in과 동일)
    h_drum_in = air_cond["h_out"]             # 응축기 출구 엔탈피

    # 드럼 출구 = 증발기 공기 입구
    T_drum_out = T_eva_in
    AH_drum_out = AH_eva_in
    h_drum_out = h_eva_in

    # 드럼 현열 [W]: 고온 공기가 세탁물에 열 전달 (공기 냉각)
    # Q = mdot_da [kg/h] × Cp [kJ/(kg·K)] × ΔT [K] / 3.6 → [W]
    Cp_da = 1.006  # 건공기 정압비열 [kJ/(kg·K)]
    Q_drum_sen = mdot_dair * Cp_da * (T_drum_in - T_drum_out) / 3.6

    # 드럼 잠열 [W]: 세탁물 수분이 증발하여 공기가 흡수
    # Q = mdot_da [kg/h] × ΔAH [kg/kg] × hfg [kJ/kg] / 3.6 → [W]
    hfg = 2501.0  # 물 증발잠열 [kJ/kg] @25°C
    Q_drum_lat = mdot_dair * (AH_drum_out - AH_drum_in) * hfg / 3.6

    # 드럼 총 열전달 [W]: 엔탈피 기반
    Q_drum_total = mdot_dair * (h_drum_out - h_drum_in) / 3.6

    # 드럼 현잠열비
    Q_drum_sum = Q_drum_sen + Q_drum_lat
    safe_drum = np.where(np.abs(Q_drum_sum) < 1e-6, 1e-6, Q_drum_sum)
    Drum_SFH = np.where(np.abs(Q_drum_sum) > 1e-6, Q_drum_sen / safe_drum, 0)
    Drum_LFH = np.where(np.abs(Q_drum_sum) > 1e-6, Q_drum_lat / safe_drum, 0)

    # SMER [kg/kWh]: 총 전력 대비 제습량
    # water_gs [g/s] * 3600 / 1000 → [kg/h], Po_WD [W] / 1000 → [kW]
    SMER = np.where(Po_wd > 0, water_gs * 3.6 / Po_wd, 0)

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
        "rmc": rmc,
        "Q_drum_sen": Q_drum_sen, "Q_drum_lat": Q_drum_lat,
        "Q_drum_total": Q_drum_total, "Drum_SFH": Drum_SFH, "Drum_LFH": Drum_LFH,
        "SMER": SMER, "cmm_eva": cmm_eva, "rh_mode": rh_mode,
        # IMC 보정용 참조값
        "_AH_cond_in": AH_cond_in, "_Po_comp": Po_comp, "_Po_wd": Po_wd,
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
def _assemble_output(df, air_cond, ref_cond, ref_eva, ref_comp, flow, perf,
                     imc_flow=None, imc_perf=None, cfg=None):
    """
    모든 계산 결과를 단일 DataFrame으로 조립.
    IMC 보정 시: 기본 변수명 = 보정값, _raw 접미사 = 비보정 원본
    IMC 비활성: 기본 변수명 = 비보정값, _raw 없음
    """
    out = df.copy()
    has_imc = imc_flow is not None and imc_perf is not None

    # 사용할 flow/perf (IMC 있으면 보정값이 기본)
    f = imc_flow if has_imc else flow
    p = imc_perf if has_imc else perf

    # 응축기 공기 (IMC 무관 — 온도/RH 기반)
    out["AH_Cond_In"] = air_cond["AH_in"]
    out["AH_Cond_Out"] = air_cond["AH_out"]
    out["RH_Cond_In"] = air_cond["RH_in"]
    out["RH_Cond_Out"] = air_cond["RH_out"]
    out["h_Cond_In_MAir"] = air_cond["h_in"]
    out["h_Cond_Out_MAir"] = air_cond["h_out"]

    # 응축기 냉매 (IMC 무관 — P, T 기반)
    out["h_Cond_In_ref"] = ref_cond["h_in"]
    out["h_Cond_Out_ref"] = ref_cond["h_out"]
    out["s_Cond_In_ref"] = ref_cond["s_in"]
    out["s_Cond_Out_ref"] = ref_cond["s_out"]
    out["v_Cond_In_ref"] = ref_cond["v_in"]
    out["v_Cond_Out_ref"] = ref_cond["v_out"]
    out["T_Dew_ref"] = ref_cond["T_dew"]
    out["T_subcooling"] = ref_cond["T_subcool"]

    # 증발기 냉매 (IMC 무관)
    out["h_Eva_In_ref"] = ref_eva["h_in"]
    out["h_Eva_Out_ref"] = ref_eva["h_out"]
    out["s_Eva_In_ref"] = ref_eva["s_in"]
    out["s_Eva_Out_ref"] = ref_eva["s_out"]
    out["v_Eva_In_ref"] = ref_eva["v_in"]
    out["v_Eva_Out_ref"] = ref_eva["v_out"]
    out["T_Boiling_ref"] = ref_eva["T_boil"]
    out["T_Superheating"] = ref_eva["T_superheat"]
    out["Quality_x"] = ref_eva["quality"]

    # 압축기 냉매 (IMC 무관)
    out["h_Comp_In_ref"] = ref_comp["h_in"]
    out["h_Comp_Out_ref"] = ref_comp["h_out"]
    out["s_Comp_In_ref"] = ref_comp["s_in"]
    out["s_Comp_Out_ref"] = ref_comp["s_out"]
    out["v_Comp_In_ref"] = ref_comp["v_in"]
    out["v_Comp_Out_ref"] = ref_comp["v_out"]
    out["Ratio_Compression"] = ref_comp["pr"]

    # ── 유량 (기본 = 보정 또는 원본) ──
    out["Flow_air_Cond_CMM"] = f["cmm_cond"]
    out["Flow_Mair_Cond_kgH"] = f["mdot_mair"]
    out["Flow_Dair_kgH"] = f["mdot_dair"]
    out["Flow_ref_kgH"] = f["mdot_ref"]
    out["Flow_ref_kgH_recalc"] = f["mdot_ref_filtered"]
    out["Eta_vol"] = f.get("eta_vol", np.zeros(len(df)))

    # ── 열전달량 ──
    out["Qrefr_Cond_recalc"] = f["Q_cond_filtered"]
    out["Qrefr_Eva_recalc"] = f["Q_eva_filtered"]
    out["Qrefr_Cond"] = f["Q_cond"]
    out["Qrefr_Eva"] = f["Q_eva"]
    out["Q_sen_eva"] = p["Q_sen"]
    out["Q_lat_eva"] = p["Q_lat"]

    # ── 드럼 열전달 ──
    out["Q_Drum_Sen"] = p["Q_drum_sen"]
    out["Q_Drum_Lat"] = p["Q_drum_lat"]
    out["Q_Drum_Total"] = p["Q_drum_total"]

    # ── 성능 ──
    out["Factor_SFH"] = p["SFH"]
    out["Factor_LFH"] = p["LFH"]
    out["Drum_SFH"] = p["Drum_SFH"]
    out["Drum_LFH"] = p["Drum_LFH"]
    out["COP_cooling"] = p["COP_cool"]
    out["COP_heating"] = p["COP_heat"]
    out["COP_sys_cooling"] = p["COP_sys_cool"]
    out["COP_sys_heating"] = p["COP_sys_heat"]
    out["SMER"] = p["SMER"]

    # ── 습도 ──
    out["AH_Eva_In"] = p["AH_eva_in"]
    out["RH_Eva_In"] = p["RH_eva_in"]
    out["h_Eva_In_MAir"] = p["h_eva_in_air"]
    out["Flow_air_Eva_CMM"] = p["cmm_eva"]

    # ── 수분 ──
    out["Water_calc_gM"] = p["water_gm"]
    out["Water_calc_kg"] = p["water_kg"]
    out["RMC_calc"] = p["rmc"]

    # ── 원본 프로그램 호환 변수 ──
    # 온도 _ref 접미사 (플로팅 호환)
    for col, alias in [("T_Comp_In","T_Comp_In_ref"),("T_Comp_Out","T_Comp_Out_ref"),
                        ("T_Cond_In","T_Cond_In_ref"),("T_Cond_Out","T_Cond_Out_ref"),
                        ("T_Eva_In","T_Eva_In_ref"),("T_Eva_Out","T_Eva_Out_ref")]:
        if col in out.columns:
            out[alias] = out[col]

    # RH 실측값 (센서 데이터 있으면)
    for rh_col in ["RH_Eva_In_measure","RH_Air_Eva_In"]:
        if rh_col in out.columns: break
    else:
        if "RH_Eva_In" in out.columns:
            out["RH_Eva_In_measure"] = out["RH_Eva_In"]  # fallback

    # 등온 압축 유량: ṁ = V × N × ρ_suc × 3600 (η_vol=1)
    if "Ctrl_Comp_Hz" in out.columns:
        V_cc = 7.5  # config에서 가져올 수도 있지만 간단히
        try: V_cc = cfg["calculation"].get("compressor_volume_cc", 7.5)
        except: pass
        V_m3 = V_cc / 1e6
        N = out["Ctrl_Comp_Hz"].values.astype(float)
        rho = ref_comp["rho_in"]
        out["Flow_ref_kgH_isothermal"] = V_m3 * N * rho * 3600

    # 재보정 응축수량 (별칭)
    out["Water_calc_gM_recal"] = p["water_gm"]

    # 건조 열량: Q_Dry = mdot_da × (h_eva_in - h_cond_out) / 3.6
    mdot_da = f["mdot_dair"]
    h_eva_in_air = p["AH_eva_in"]  # 실제로는 h_eva_in_air 필요
    h_cond_out_air = air_cond["h_out"]
    h_eva_in_air_val = p.get("h_eva_in_air", np.zeros(len(df)))
    Q_dry = mdot_da * (h_eva_in_air_val - h_cond_out_air) / 3.6
    out["Q_Dry"] = Q_dry

    # 건조 COP
    Po_comp = df["Po_Comp"].values.astype(float) if "Po_Comp" in df.columns else np.ones(len(df))
    Po_wd = df["Po_WD"].values.astype(float) if "Po_WD" in df.columns else np.ones(len(df))
    out["COP_Dry"] = np.where(Po_comp > 0, Q_dry / Po_comp, 0)
    out["COP_sys_Dry"] = np.where(Po_wd > 0, Q_dry / Po_wd, 0)

    # SMER 변형 (제습능력/전력)
    water_gs = p.get("water_gs", np.zeros(len(df)))
    out["P_Dehumidity_PoComp"] = np.where(Po_comp > 0, water_gs * 3.6 / Po_comp, 0)
    out["P_Dehumidity_PoWD"] = np.where(Po_wd > 0, water_gs * 3.6 / Po_wd, 0)

    # 압축기 효율: η_comp = (h_out_is - h_in) / (h_out - h_in)
    h_in_c = ref_comp["h_in"]
    h_out_c = ref_comp["h_out"]
    dh_actual = h_out_c - h_in_c
    # 등엔트로피 출구 엔탈피 근사: h_out_is ≈ h_in + v_in × (P_out - P_in) × 100
    # 더 정확한 방법: s_in에서 P_out 조건으로 계산 (여기서는 근사)
    v_in_c = ref_comp["v_in"]
    P_in_barg = df["P_Comp_In"].values.astype(float) if "P_Comp_In" in df.columns else np.zeros(len(df))
    P_out_barg = df["P_Comp_Out"].values.astype(float) if "P_Comp_Out" in df.columns else np.zeros(len(df))
    dP_kPa = (P_out_barg - P_in_barg) * 100  # barg → kPa
    dh_isentropic = v_in_c * dP_kPa  # 근사 [kJ/kg]
    out["Eff_Comp"] = np.where(np.abs(dh_actual) > 0.1, dh_isentropic / dh_actual, 0)
    out["Eff_Comp"] = np.clip(out["Eff_Comp"], 0, 1.5)

    # 열교환기 유효도: ε = Q / Q_max
    T_air_in_eva = df["T_Air_Eva_In"].values.astype(float) if "T_Air_Eva_In" in df.columns else np.zeros(len(df))
    T_air_out_eva = df["T_Air_Eva_Out"].values.astype(float) if "T_Air_Eva_Out" in df.columns else np.zeros(len(df))
    T_boil = ref_eva.get("T_boil", np.zeros(len(df)))
    T_dew = ref_cond.get("T_dew", np.zeros(len(df)))
    T_air_cond_out = df["T_Air_Cond_Out"].values.astype(float) if "T_Air_Cond_Out" in df.columns else np.zeros(len(df))
    # 증발기: ε = (T_air_in - T_air_out) / (T_air_in - T_boil)
    dT_max_eva = T_air_in_eva - T_boil
    out["Eff_Evap"] = np.where(np.abs(dT_max_eva) > 0.1, (T_air_in_eva - T_air_out_eva) / dT_max_eva, 0)
    out["Eff_Evap"] = np.clip(out["Eff_Evap"], 0, 1.5)
    # 응축기: ε = (T_air_out - T_air_in) / (T_dew - T_air_in)
    T_air_in_cond = T_air_out_eva  # 응축기 입구 = 증발기 출구
    dT_max_cond = T_dew - T_air_in_cond
    out["Eff_Cond"] = np.where(np.abs(dT_max_cond) > 0.1, (T_air_cond_out - T_air_in_cond) / dT_max_cond, 0)
    out["Eff_Cond"] = np.clip(out["Eff_Cond"], 0, 1.5)

    # 열손실
    Q_cond_ref = f["Q_cond"]
    Q_eva_ref = f["Q_eva"]
    # Q_Comp_loss = Po_Comp - ṁ_ref × Δh_comp (= 압축기 방열)
    mdot_ref_hs = f["mdot_ref"] / 3600  # kg/s
    Q_comp_work = mdot_ref_hs * dh_actual * 1000  # W
    out["Q_Comp_loss"] = Po_comp - Q_comp_work
    # Q_HP_loss = (Q_cond + Po_Comp) - Q_eva - Po_Comp = Q_cond - Q_eva (시스템 열손실)
    out["Q_HP_loss"] = Q_cond_ref + Po_comp - Q_eva_ref - Po_comp
    # 간단히: Q_HP_loss = Q_cond - Q_eva - 배관 등 열손실
    out["Q_HP_loss"] = Q_cond_ref - Q_eva_ref - Q_comp_work

    # ── IMC 보정 시: 원본을 _raw로 저장 ──
    if has_imc:
        out["Flow_air_Cond_CMM_raw"] = flow["cmm_cond"]
        out["Flow_Mair_Cond_kgH_raw"] = flow["mdot_mair"]
        out["Flow_Dair_kgH_raw"] = flow["mdot_dair"]
        out["Flow_ref_kgH_raw"] = flow["mdot_ref"]
        out["Qrefr_Cond_raw"] = flow["Q_cond"]
        out["Qrefr_Eva_raw"] = flow["Q_eva"]
        out["Q_sen_eva_raw"] = perf["Q_sen"]
        out["Q_lat_eva_raw"] = perf["Q_lat"]
        out["Q_Drum_Sen_raw"] = perf["Q_drum_sen"]
        out["Q_Drum_Lat_raw"] = perf["Q_drum_lat"]
        out["Q_Drum_Total_raw"] = perf["Q_drum_total"]
        out["Factor_SFH_raw"] = perf["SFH"]
        out["Factor_LFH_raw"] = perf["LFH"]
        out["COP_cooling_raw"] = perf["COP_cool"]
        out["COP_heating_raw"] = perf["COP_heat"]
        out["COP_sys_cooling_raw"] = perf["COP_sys_cool"]
        out["COP_sys_heating_raw"] = perf["COP_sys_heat"]
        out["SMER_raw"] = perf["SMER"]
        out["AH_Eva_In_raw"] = perf["AH_eva_in"]
        out["RH_Eva_In_raw"] = perf["RH_eva_in"]
        out["Flow_air_Eva_CMM_raw"] = perf["cmm_eva"]
        out["Water_calc_gM_raw"] = perf["water_gm"]
        out["Water_calc_kg_raw"] = perf["water_kg"]
        out["RMC_calc_raw"] = perf["rmc"]

    return out
