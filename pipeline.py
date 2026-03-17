"""
pipeline.py — 통합 파이프라인
Stage 1: 다중 소스 데이터 병합 + 기본 파생
Stage 2: 열역학 성능 계산
→ 단일 CSV 출력
"""
import os
import time

from config import load_config, get_column_mapping, get_selected_columns
from io_handler import (
    rename_files_in_folder, find_files, read_source_files, save_merged,
)
from preprocessor import (
    preprocess_blackrose, preprocess_ams, preprocess_mx100,
    sync_and_merge, add_time_columns,
)
from calculator import run_stage1
from postprocessor import run_postprocessing
from performance import run_stage2
from properties import get_props


def process_single(
    ams_path, br_path, mx100_path,
    cfg: dict, case_name: str, data_path: str, numbering: int,
    exp: dict | None = None,
) -> str:
    """
    단일 파일 세트 전체 파이프라인 실행.

    Parameters:
        exp: {"load_kg": float, "imc_kg": float, "fmc_kg": float} or None
    Returns:
        저장된 파일 경로
    """
    t0 = time.perf_counter()
    proc = cfg["processing"]
    data_time = proc["data_time"]
    calc_enabled = cfg.get("calculation", {}).get("enabled", False)

    # ── Stage 1: 병합 ──
    df_ams, df_br, df_mx = read_source_files(ams_path, br_path, mx100_path, data_time)

    df_br_main, df_br_add = preprocess_blackrose(
        df_br,
        selected_columns=get_selected_columns(cfg, "br"),
        column_mapping=get_column_mapping(cfg, "blackrose"),
    )
    df_ams_proc = preprocess_ams(
        df_ams,
        column_mapping=get_column_mapping(cfg, "ams"),
        scale_factors=cfg.get("ams_scale_factors", {}),
        subprocess_mapping=cfg.get("subprocess_mapping", {}),
        selected_columns=get_selected_columns(cfg, "ams"),
    )
    df_mx_proc = preprocess_mx100(df_mx, useless_columns=cfg["mx100"]["useless_columns"])

    merged = sync_and_merge(
        dfs=[df_br_main, df_br_add, df_mx_proc, df_ams_proc],
        data_time=data_time,
        df_br=df_br_main, df_ams=df_ams_proc, df_mx100=df_mx_proc,
    )
    merged = add_time_columns(merged)
    merged = run_stage1(merged, cfg)
    merged = run_postprocessing(merged, cfg)

    # ── Stage 2: 성능 계산 (옵션) ──
    if calc_enabled:
        result = run_stage2(merged, cfg, exp)
    else:
        result = merged

    # ── 저장 ──
    out_path = save_merged(result, data_path, case_name, numbering)
    elapsed = time.perf_counter() - t0
    print(f"    → {os.path.basename(out_path)} ({elapsed:.1f}초, {len(result)}행×{len(result.columns)}열)")
    return out_path


def process_case(
    case_name: str, category_path: str, cfg: dict,
    exp_list: list[dict] | None = None,
) -> list[str]:
    """케이스 폴더 내 모든 파일 세트 처리."""
    data_path = os.path.join(category_path, case_name)
    rename_files_in_folder(data_path)

    files = find_files(data_path)
    data_time = cfg["processing"]["data_time"]
    ref_key = {"BR": "br", "AMS": "ams", "MX100": "mx100"}[data_time]
    ref_files = files[ref_key]

    if not ref_files:
        print(f"[경고] {case_name}: 기준 파일 없음")
        return []

    outputs = []
    for i in range(len(ref_files)):
        print(f"  {case_name} — {i+1}/{len(ref_files)}")
        exp = exp_list[i] if exp_list and i < len(exp_list) else None

        out = process_single(
            ams_path=files["ams"][i] if i < len(files["ams"]) else None,
            br_path=files["br"][i] if i < len(files["br"]) else None,
            mx100_path=files["mx100"][i] if i < len(files["mx100"]) else None,
            cfg=cfg, case_name=case_name, data_path=data_path,
            numbering=i+1, exp=exp,
        )
        outputs.append(out)

    return outputs


def run_pipeline(
    category_path: str, case_names: list[str], cfg: dict,
    exp_map: dict[str, list[dict]] | None = None,
) -> dict[str, list[str]]:
    """
    전체 파이프라인 실행.

    Parameters:
        exp_map: {case_name: [{"load_kg": ..., "imc_kg": ..., "fmc_kg": ...}, ...]}
    """
    t_start = time.perf_counter()
    env = cfg["environment"]
    print(f"{'='*60}")
    print(f"  통합 분석 시작: {env['refrigerant']} / {cfg['processing']['setup_type']}")
    print(f"  Stage 2 (성능계산): {'ON' if cfg.get('calculation',{}).get('enabled') else 'OFF'}")
    print(f"{'='*60}")

    # LUT 사전 초기화
    get_props(env["refrigerant"], env["patm"], env.get("backend", "HEOS"))

    all_outputs = {}
    for case in case_names:
        exp_list = exp_map.get(case) if exp_map else None
        outputs = process_case(case, category_path, cfg, exp_list)
        all_outputs[case] = outputs
        print(f"  {case} 완료 ({len(outputs)}건)")

    total = sum(len(v) for v in all_outputs.values())
    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  전체 완료: {total}개 파일 ({elapsed:.1f}초)")
    print(f"{'='*60}")
    return all_outputs
