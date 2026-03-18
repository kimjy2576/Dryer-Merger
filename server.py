"""
server.py — FastAPI 백엔드 (v5 — Merge / Calculation 분리)
"""
import os, sys, uuid, time, traceback
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import load_config

app = FastAPI(title="Dryer Merger", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}

BASE_DIR = Path(__file__).parent
if getattr(sys, "frozen", False):
    EXE_DIR = Path(sys.executable).parent
else:
    EXE_DIR = BASE_DIR
RESULT_DIR = EXE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

sessions: dict[str, dict] = {}
DEFAULT_CFG = load_config()


# ══════════════════════════════════════════════
#  Pydantic 모델
# ══════════════════════════════════════════════
class BrowseRequest(BaseModel):
    path: str

class SelectRequest(BaseModel):
    category_path: str
    case_names: list[str]

class MergeRequest(BaseModel):
    session_id: str
    config: Optional[dict] = None
    variable_settings: Optional[dict] = None  # {"col_name": {"include": bool, "weight": float, "bias": float}, ...}

class CalcRequest(BaseModel):
    session_id: str
    config: Optional[dict] = None
    experimental: Optional[dict] = None    # {"load_kg", "imc_kg", "fmc_kg"}
    source_files: Optional[list[str]] = None  # 특정 merged 파일 지정 (없으면 세션의 merge 결과 사용)
    variable_mapping: Optional[dict] = None   # {"calc_var": "merged_col", ...}


# ══════════════════════════════════════════════
#  설정 API
# ══════════════════════════════════════════════
@app.get("/api/config")
def get_config():
    return DEFAULT_CFG

@app.put("/api/config")
def update_config(body: dict):
    global DEFAULT_CFG
    DEFAULT_CFG = body
    yaml_path = BASE_DIR / "config" / "default_config.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(DEFAULT_CFG, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return {"status": "ok"}


# ══════════════════════════════════════════════
#  경로 탐색 API
# ══════════════════════════════════════════════
@app.post("/api/browse")
def browse_directory(req: BrowseRequest):
    p = Path(req.path)
    if not p.exists(): raise HTTPException(404, f"경로 없음: {req.path}")
    if not p.is_dir(): raise HTTPException(400, f"디렉토리 아님: {req.path}")
    items = []
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith("."): continue
            items.append({"name": item.name, "is_dir": item.is_dir(),
                          "size": item.stat().st_size if item.is_file() else None})
    except PermissionError:
        raise HTTPException(403, f"접근 권한 없음: {req.path}")
    return {"path": str(p.resolve()), "items": items,
            "cases": [i["name"] for i in items if i["is_dir"]]}


@app.post("/api/select")
def select_cases(req: SelectRequest):
    category = Path(req.category_path)
    if not category.exists(): raise HTTPException(404, f"경로 없음: {req.category_path}")
    sid = str(uuid.uuid4())[:8]
    case_files = {}
    for case in req.case_names:
        case_dir = category / case
        if not case_dir.is_dir(): continue
        case_files[case] = _classify_files(case_dir)
    if not case_files: raise HTTPException(400, "유효한 케이스 없음")
    sessions[sid] = {
        "status": "ready", "progress": 0, "log": [], "error": None,
        "category_path": str(category), "case_files": case_files,
        "merge_results": [], "calc_results": [],
    }
    return {"session_id": sid, "cases": case_files}


def _classify_files(folder: Path) -> dict:
    files = {"br": [], "ams": [], "mx100": []}
    for f in sorted(folder.iterdir()):
        if not f.is_file(): continue
        n = f.name.lower()
        if "_br.csv" in n or (n.endswith(".csv") and "br" in n):
            files["br"].append(f.name)
        elif "_ams.csv" in n or (n.endswith(".csv") and "ams" in n):
            files["ams"].append(f.name)
        elif n.endswith((".xls", ".xlsx")) or "_temp." in n:
            files["mx100"].append(f.name)
        elif n.endswith(".csv") and "_merged" not in n and "_calc" not in n and "_result" not in n:
            files["br"].append(f.name)
    return files


# ══════════════════════════════════════════════
#  컬럼 스캔 API (Merge 전 변수 설정용)
# ══════════════════════════════════════════════
@app.post("/api/scan-columns")
def scan_columns(req: BrowseRequest):
    """
    세션의 첫 번째 케이스 폴더에서 BR/AMS/MX100 파일을 읽어
    사용 가능한 전체 컬럼 목록을 반환한다.
    """
    sid = req.path  # session_id를 path 필드로 받음
    if sid not in sessions:
        raise HTTPException(404, "세션 없음")
    s = sessions[sid]
    category = Path(s["category_path"])
    case_files = s["case_files"]

    all_columns = {}  # {"col_name": {"source": "BR"|"AMS"|"MX100", "dtype": str}}
    dt = DEFAULT_CFG["processing"]["data_time"]

    # 첫 번째 케이스에서 샘플 읽기
    for case_name, cf in case_files.items():
        case_dir = category / case_name
        # BR
        if cf["br"]:
            try:
                fp = case_dir / cf["br"][0]
                for enc in ("cp949", "utf-8"):
                    try:
                        df = pd.read_csv(fp, encoding=enc, skiprows=[0], nrows=5)
                        for c in df.columns:
                            if c not in all_columns:
                                all_columns[c] = {"source": "BR", "dtype": str(df[c].dtype)}
                        break
                    except: continue
            except: pass
        # AMS
        if cf["ams"]:
            try:
                fp = case_dir / cf["ams"][0]
                df = pd.read_csv(fp, encoding="utf-8", skiprows=[0], nrows=5)
                for c in df.columns:
                    if c not in all_columns:
                        all_columns[c] = {"source": "AMS", "dtype": str(df[c].dtype)}
            except: pass
        # MX100
        if cf["mx100"]:
            try:
                fp = case_dir / cf["mx100"][0]
                df = pd.read_excel(fp, skiprows=24, header=0, nrows=5)
                for c in df.columns:
                    if c not in all_columns:
                        all_columns[c] = {"source": "MX100", "dtype": str(df[c].dtype)}
            except: pass
        break  # 첫 번째 케이스만

    return {
        "total": len(all_columns),
        "columns": all_columns,
    }


# ══════════════════════════════════════════════
#  Merge API (Stage 1 분리)
# ══════════════════════════════════════════════
@app.post("/api/merge")
async def start_merge(req: MergeRequest, background_tasks: BackgroundTasks):
    sid = req.session_id
    if sid not in sessions: raise HTTPException(404, "세션 없음")
    s = sessions[sid]
    if s["status"] == "processing": raise HTTPException(409, "처리 중")
    cfg = req.config or DEFAULT_CFG
    s.update(status="processing", progress=0, log=[], merge_results=[], error=None)
    background_tasks.add_task(_run_merge, sid, cfg, req.variable_settings)
    return {"status": "started", "mode": "merge"}


def _run_merge(sid: str, cfg: dict, var_settings: dict | None = None):
    s = sessions[sid]
    category = Path(s["category_path"])
    case_files = s["case_files"]
    dt = cfg["processing"]["data_time"]

    try:
        _log(sid, "═══ Merge 시작 ═══")
        from preprocessor import (
            preprocess_blackrose, preprocess_ams, preprocess_mx100,
            sync_and_merge, add_time_columns,
        )
        from calculator import run_stage1
        from postprocessor import run_postprocessing
        from config import get_column_mapping, get_selected_columns
        from io_handler import rename_files_in_folder

        # 변수 설정 로그
        if var_settings:
            included = [k for k, v in var_settings.items() if v.get("include", True)]
            excluded = [k for k, v in var_settings.items() if not v.get("include", True)]
            wb_applied = [k for k, v in var_settings.items()
                         if v.get("include", True) and (v.get("weight", 1.0) != 1.0 or v.get("bias", 0.0) != 0.0)]
            _log(sid, f"  변수 설정: {len(included)}개 포함, {len(excluded)}개 제외, {len(wb_applied)}개 W&B 보정")

        total = sum(len(cf[{"BR":"br","AMS":"ams","MX100":"mx100"}[dt]]) for cf in case_files.values())
        done, results = 0, []

        for case_name, cf in case_files.items():
            case_dir = category / case_name
            rename_files_in_folder(str(case_dir))
            cf = _classify_files(case_dir)
            ref_key = {"BR":"br","AMS":"ams","MX100":"mx100"}[dt]
            n = len(cf[ref_key])

            for i in range(n):
                t0 = time.perf_counter()
                _log(sid, f"[{case_name}] {i+1}/{n} 병합 중...")

                bp = str(case_dir / cf["br"][i]) if i < len(cf["br"]) else None
                ap = str(case_dir / cf["ams"][i]) if i < len(cf["ams"]) else None
                mp = str(case_dir / cf["mx100"][i]) if i < len(cf["mx100"]) else None

                df_ams, df_br, df_mx = _read(ap, bp, mp, dt)
                df_br_main, df_br_add = preprocess_blackrose(
                    df_br, get_selected_columns(cfg, "br"), get_column_mapping(cfg, "blackrose"))
                df_ams_proc = preprocess_ams(
                    df_ams, get_column_mapping(cfg, "ams"),
                    cfg.get("ams_scale_factors", {}), cfg.get("subprocess_mapping", {}),
                    get_selected_columns(cfg, "ams"))
                df_mx_proc = preprocess_mx100(df_mx, cfg["mx100"]["useless_columns"])

                merged = sync_and_merge(
                    [df_br_main, df_br_add, df_mx_proc, df_ams_proc],
                    dt, df_br_main, df_ams_proc, df_mx_proc)
                merged = add_time_columns(merged)
                merged = run_stage1(merged, cfg)
                merged = run_postprocessing(merged, cfg)

                # ── 변수 설정 적용 (W&B + 포함/제외) ──
                if var_settings:
                    merged = _apply_variable_settings(merged, var_settings)

                # _merged.csv 저장
                parts = case_name.split("_")[:9]
                prefix = "_".join(parts)
                out_name = f"{prefix}_{i+1}_merged.csv"
                merged.to_csv(RESULT_DIR / out_name, index=False)
                merged.to_csv(case_dir / out_name, index=False)
                results.append(out_name)

                el = time.perf_counter() - t0
                _log(sid, f"  → {out_name} ({el:.1f}초, {len(merged)}행×{len(merged.columns)}열)")
                done += 1
                s["progress"] = int(done / max(total, 1) * 100)

        s.update(merge_results=results, status="done", progress=100)
        _log(sid, f"\n═══ Merge 완료: {len(results)}개 파일 ═══")

    except Exception as e:
        s.update(status="error", error=str(e))
        _log(sid, f"에러: {e}\n{traceback.format_exc()}")


def _apply_variable_settings(df: pd.DataFrame, var_settings: dict) -> pd.DataFrame:
    """
    변수별 Weight & Bias 적용 + 제외 컬럼 삭제.
    var_settings: {"col_name": {"include": bool, "weight": float, "bias": float}}
    수식: col = col * weight + bias
    """
    # 1. W&B 적용 (include=True인 것만)
    for col, s in var_settings.items():
        if col not in df.columns:
            continue
        if not s.get("include", True):
            continue
        w = s.get("weight", 1.0)
        b = s.get("bias", 0.0)
        if w != 1.0 or b != 0.0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] * w + b

    # 2. 제외 컬럼 삭제 (Time, Time_sec, Time_min은 항상 유지)
    protected = {"Time", "Time_sec", "Time_min"}
    drop_cols = [col for col, s in var_settings.items()
                 if not s.get("include", True) and col in df.columns and col not in protected]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ══════════════════════════════════════════════
#  Calculation API (Stage 2 분리)
# ══════════════════════════════════════════════
@app.post("/api/calculate")
async def start_calculation(req: CalcRequest, background_tasks: BackgroundTasks):
    sid = req.session_id
    if sid not in sessions: raise HTTPException(404, "세션 없음")
    s = sessions[sid]
    if s["status"] == "processing": raise HTTPException(409, "처리 중")

    cfg = req.config or DEFAULT_CFG
    # 소스 파일: 명시적 지정 or 세션의 merge 결과
    source = req.source_files or s.get("merge_results", [])
    if not source: raise HTTPException(400, "Merge 결과가 없습니다. Merge를 먼저 실행하세요.")

    s.update(status="processing", progress=0, log=[], calc_results=[], error=None, skipped_info={})
    background_tasks.add_task(_run_calc, sid, cfg, source, req.experimental, req.variable_mapping)
    return {"status": "started", "mode": "calculate", "source_count": len(source)}


def _run_calc(sid: str, cfg: dict, source_files: list[str],
              exp: dict | None, var_map: dict | None):
    s = sessions[sid]

    try:
        _log(sid, "═══ Calculation 시작 ═══")
        env = cfg["environment"]
        _log(sid, f"냉매: {env['refrigerant']} ({env.get('backend','HEOS')})")

        from properties import get_props
        get_props(env["refrigerant"], env["patm"], env.get("backend", "HEOS"))
        from performance import run_stage2

        total = len(source_files)
        results = []

        for i, fn in enumerate(source_files):
            t0 = time.perf_counter()
            _log(sid, f"[{i+1}/{total}] {fn} 계산 중...")

            # merged CSV 읽기
            src_path = RESULT_DIR / fn
            if not src_path.exists():
                _log(sid, f"  [경고] 파일 없음: {fn}, 스킵")
                continue
            df = pd.read_csv(src_path)
            _log(sid, f"  입력: {len(df)}행 × {len(df.columns)}열")

            # 변수 매핑 적용 (있으면)
            if var_map:
                rename_dict = {}
                for calc_var, merged_col in var_map.items():
                    if merged_col in df.columns and calc_var != merged_col:
                        rename_dict[merged_col] = calc_var
                if rename_dict:
                    df = df.rename(columns=rename_dict)
                    _log(sid, f"  변수 매핑: {len(rename_dict)}개 적용")

            # Stage 2 실행
            df_calc = run_stage2(df, cfg, exp)
            _log(sid, f"  계산 완료: {len(df_calc.columns)}열")

            # 스킵된 블록 로그
            skipped = df_calc.attrs.get("skipped_blocks", {})
            if skipped:
                _log(sid, f"  [경고] {len(skipped)}개 블록 스킵:")
                for blk, reason in skipped.items():
                    _log(sid, f"    - {blk}: {reason}")
                # 스킵 정보를 세션에 저장 (프론트엔드 팝업용)
                if "skipped_info" not in s:
                    s["skipped_info"] = {}
                s["skipped_info"][fn] = skipped

            # _calc.csv 저장
            out_name = fn.replace("_merged.csv", "_calc.csv")
            df_calc.to_csv(RESULT_DIR / out_name, index=False)

            # 원본 폴더에도 저장 (세션에 category_path가 있으면)
            cat_path = s.get("category_path")
            if cat_path:
                # 케이스 폴더 추정 (파일명에서)
                for case_name in s.get("case_files", {}):
                    if case_name.replace(" ", "_") in fn or fn.startswith(case_name[:10]):
                        case_dir = Path(cat_path) / case_name
                        if case_dir.exists():
                            df_calc.to_csv(case_dir / out_name, index=False)
                            _log(sid, f"  → {case_dir / out_name}")
                        break

            results.append(out_name)
            el = time.perf_counter() - t0
            _log(sid, f"  → {out_name} ({el:.1f}초)")
            s["progress"] = int((i + 1) / total * 100)

        s.update(calc_results=results, status="done", progress=100)
        _log(sid, f"\n═══ Calculation 완료: {len(results)}개 파일 ═══")

    except Exception as e:
        s.update(status="error", error=str(e))
        _log(sid, f"에러: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════
#  컬럼 조회 API (변수 매핑용)
# ══════════════════════════════════════════════
@app.get("/api/columns/{fn}")
def get_columns(fn: str):
    """merged CSV의 컬럼 목록 반환 (Calc 변수 매핑 UI용)."""
    p = RESULT_DIR / fn
    if not p.exists(): raise HTTPException(404, f"파일 없음: {fn}")
    df = pd.read_csv(p, nrows=1)
    return {
        "filename": fn,
        "columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
    }


# ══════════════════════════════════════════════
#  Calc 필요 변수 정의 + 매핑 API
# ══════════════════════════════════════════════
CALC_VARIABLES = [
    # 압력 (필수)
    {"key": "P_Comp_Out",          "label": "압축기 토출 압력",    "unit": "barg",  "category": "압력",     "required": True,  "default_match": ["P_Comp_Out", "P_Cond_In"]},
    {"key": "P_Comp_In",           "label": "압축기 흡입 압력",    "unit": "barg",  "category": "압력",     "required": True,  "default_match": ["P_Comp_In", "P_Eva_Out"]},
    {"key": "P_Cond_Out",          "label": "응축기 출구 압력",    "unit": "barg",  "category": "압력",     "required": False, "default_match": ["P_Cond_Out"]},
    {"key": "P_Eva_In",            "label": "증발기 입구 압력",    "unit": "barg",  "category": "압력",     "required": False, "default_match": ["P_Eva_In"]},
    # 온도 (필수)
    {"key": "T_Cond_In",           "label": "응축기 냉매 입구 온도", "unit": "°C", "category": "온도",     "required": True,  "default_match": ["T_Cond_In", "T_Comp_Out", "Heatpump_CompTemp"]},
    {"key": "T_Cond_Out",          "label": "응축기 냉매 출구 온도", "unit": "°C", "category": "온도",     "required": True,  "default_match": ["T_Cond_Out"]},
    {"key": "T_Cond_M1",           "label": "응축기 중간 온도",     "unit": "°C",  "category": "온도",     "required": False, "default_match": ["T_Cond_M1"]},
    {"key": "Heatpump_CompTemp",   "label": "압축기 온도",         "unit": "°C",  "category": "온도",     "required": False, "default_match": ["Heatpump_CompTemp", "T_Comp_Out"]},
    {"key": "Heatpump_EvaOutTemp", "label": "증발기 냉매 출구 온도", "unit": "°C", "category": "온도",     "required": True,  "default_match": ["Heatpump_EvaOutTemp", "T_Eva_Out"]},
    {"key": "T_Comp_In",           "label": "압축기 흡입 온도",     "unit": "°C",  "category": "온도",     "required": False, "default_match": ["T_Comp_In", "Heatpump_EvaOutTemp"]},
    {"key": "T_Comp_Out",          "label": "압축기 토출 온도",     "unit": "°C",  "category": "온도",     "required": False, "default_match": ["T_Comp_Out", "T_Cond_In"]},
    {"key": "T_Subcooler_Out",     "label": "서브쿨러 출구 온도",   "unit": "°C",  "category": "온도",     "required": False, "default_match": ["T_Subcooler_Out"]},
    # 공기 온도 (필수)
    {"key": "T_Air_Eva_Out",       "label": "증발기 공기 출구 온도", "unit": "°C", "category": "공기",     "required": True,  "default_match": ["T_Air_Eva_Out"]},
    {"key": "Heatpump_DuctInTemp", "label": "덕트 입구 온도",      "unit": "°C",  "category": "공기",     "required": True,  "default_match": ["Heatpump_DuctInTemp", "T_Air_Eva_In"]},
    {"key": "Heatpump_DuctOutTemp","label": "덕트 출구 온도",      "unit": "°C",  "category": "공기",     "required": True,  "default_match": ["Heatpump_DuctOutTemp", "T_Air_Cond_Out"]},
    # 전력 (필수)
    {"key": "Po_WD",               "label": "총 전력",            "unit": "W",    "category": "전력",     "required": True,  "default_match": ["Po_WD"]},
    {"key": "Po_Comp",             "label": "압축기 전력",         "unit": "W",    "category": "전력",     "required": True,  "default_match": ["Po_Comp"]},
    {"key": "Po_Fan",              "label": "팬 전력",            "unit": "W",    "category": "전력",     "required": True,  "default_match": ["Po_Fan"]},
    # 제어 (선택)
    {"key": "HP_CompCurrentHz",    "label": "압축기 주파수",       "unit": "Hz",   "category": "제어",     "required": False, "default_match": ["HP_CompCurrentHz"]},
    {"key": "Heatpump_DryMotionInfo","label": "건조 모션 정보",    "unit": "-",    "category": "제어",     "required": False, "default_match": ["Heatpump_DryMotionInfo"]},
    # 시간
    {"key": "Time_min",            "label": "시간 (분)",          "unit": "min",  "category": "시간",     "required": True,  "default_match": ["Time_min"]},
    {"key": "Time_sec",            "label": "시간 (초)",          "unit": "sec",  "category": "시간",     "required": True,  "default_match": ["Time_sec"]},
]

@app.get("/api/calc-variables")
def get_calc_variables():
    """Calculation에 필요한 변수 목록 반환."""
    return {"variables": CALC_VARIABLES}


@app.post("/api/auto-map")
def auto_map(req: BrowseRequest):
    """
    merged CSV 파일명을 받아서 CALC_VARIABLES와 자동 매칭 결과를 반환.
    exact match → default_match 순서로 탐색.
    """
    fn = req.path
    p = RESULT_DIR / fn
    if not p.exists(): raise HTTPException(404, f"파일 없음: {fn}")
    df = pd.read_csv(p, nrows=1)
    merged_cols = df.columns.tolist()

    mapping = {}  # {calc_var: matched_merged_col or null}
    for v in CALC_VARIABLES:
        matched = None
        # 1. exact match
        if v["key"] in merged_cols:
            matched = v["key"]
        else:
            # 2. default_match 리스트에서 순서대로 탐색
            for dm in v.get("default_match", []):
                if dm in merged_cols:
                    matched = dm
                    break
        mapping[v["key"]] = matched

    unmatched = [v["key"] for v in CALC_VARIABLES if v["required"] and not mapping.get(v["key"])]

    return {
        "filename": fn,
        "merged_columns": merged_cols,
        "mapping": mapping,
        "unmatched_required": unmatched,
        "total_matched": sum(1 for v in mapping.values() if v),
        "total_required": sum(1 for v in CALC_VARIABLES if v["required"]),
    }


# ══════════════════════════════════════════════
#  공통 유틸
# ══════════════════════════════════════════════
def _read(ap, bp, mp, dt):
    df_a, df_b, df_m = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if bp and os.path.exists(bp):
        for enc in ("cp949", "utf-8"):
            try: df_b = pd.read_csv(bp, encoding=enc, skiprows=[0]); break
            except: continue
    if ap and os.path.exists(ap):
        df_a = pd.read_csv(ap, encoding="utf-8", skiprows=[0])
    if mp and os.path.exists(mp):
        try: df_m = pd.read_excel(mp, skiprows=24, header=0)
        except:
            try:
                df_m = pd.read_excel(mp, skiprows=24, header=[0,1])
                cols = [c[1] if "Unnamed" in str(c[0]) else c[0] for c in df_m.columns]
                df_m.columns = cols
                if "Date" in df_m.columns and "Time" in df_m.columns:
                    df_m["Time"] = df_m["Date"].astype(str)+" "+df_m["Time"].astype(str)
                    df_m.drop(columns=["Date"], inplace=True, errors="ignore")
            except: pass
    return df_a, df_b, df_m

def _log(sid, msg):
    if sid in sessions: sessions[sid]["log"].append(msg)


# ══════════════════════════════════════════════
#  상태 / 결과 API
# ══════════════════════════════════════════════
@app.get("/api/status/{sid}")
def get_status(sid: str):
    if sid not in sessions: raise HTTPException(404)
    s = sessions[sid]
    return {k: s.get(k) for k in ("status","progress","log","merge_results","calc_results","error","skipped_info")}

@app.get("/api/results")
def list_results():
    files = sorted(RESULT_DIR.glob("*.csv"), key=os.path.getmtime, reverse=True)
    return [{"name": f.name, "size": f.stat().st_size,
             "type": "merged" if "_merged" in f.name else "calc" if "_calc" in f.name else "other"}
            for f in files]

@app.get("/api/results/{fn}")
def download(fn: str):
    p = RESULT_DIR / fn
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, media_type="text/csv", filename=fn)

@app.get("/api/preview/{fn}")
def preview(fn: str, max_rows: int = 3000):
    p = RESULT_DIR / fn
    if not p.exists(): raise HTTPException(404)
    df = pd.read_csv(p, nrows=max_rows)
    if "Time" in df.columns: df["Time"] = df["Time"].astype(str)
    df = df.where(pd.notnull(df), None)
    return {
        "columns": df.columns.tolist(),
        "data": df.values.tolist(),
        "total_rows": sum(1 for _ in open(p)) - 1,
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
    }

@app.delete("/api/session/{sid}")
def delete_session(sid: str):
    sessions.pop(sid, None)
    return {"status": "deleted"}


# ══════════════════════════════════════════════
#  프론트엔드 서빙
# ══════════════════════════════════════════════
STATIC = Path(sys._MEIPASS) / "static" if getattr(sys, "frozen", False) else BASE_DIR / "static"
@app.get("/")
def index():
    f = STATIC / "index.html"
    return HTMLResponse(f.read_text("utf-8")) if f.exists() else HTMLResponse("<h1>Dryer Merger v5</h1>")

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
