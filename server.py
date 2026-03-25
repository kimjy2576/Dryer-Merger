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

app = FastAPI(title="HPWD Data Manager", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
    EXE_DIR = BASE_DIR
else:
    BASE_DIR = Path(__file__).parent
    EXE_DIR = BASE_DIR
STATIC = BASE_DIR / "static"
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
    session_id: Optional[str] = None
    config: Optional[dict] = None
    experimental: Optional[dict] = None    # {"load_kg", "imc_kg", "fmc_kg"}
    source_files: Optional[list[str]] = None  # 특정 merged 파일 지정
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


@app.get("/api/saves/{category}")
def list_saves(category: str):
    """세이브 슬롯 목록. category: merge/calc/formula"""
    import json
    d = BASE_DIR / "config" / "saves" / category
    d.mkdir(parents=True, exist_ok=True)
    slots = []
    for f in sorted(d.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            meta = json.loads(f.read_text("utf-8"))
            slots.append({"name": f.stem, "timestamp": meta.get("timestamp", ""), "size": f.stat().st_size})
        except:
            slots.append({"name": f.stem, "timestamp": "", "size": f.stat().st_size})
    return {"slots": slots}


@app.post("/api/saves/{category}/{name}")
def save_slot(category: str, name: str, req: dict):
    """세이브 슬롯에 저장."""
    import json
    d = BASE_DIR / "config" / "saves" / category
    d.mkdir(parents=True, exist_ok=True)
    req["timestamp"] = pd.Timestamp.now().isoformat()
    req["slot_name"] = name
    (d / f"{name}.json").write_text(json.dumps(req, ensure_ascii=False, indent=2), "utf-8")
    return {"saved": name, "category": category}


@app.get("/api/saves/{category}/{name}")
def load_slot(category: str, name: str):
    """세이브 슬롯에서 불러오기."""
    import json
    p = BASE_DIR / "config" / "saves" / category / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, f"슬롯 없음: {name}")
    return json.loads(p.read_text("utf-8"))


@app.delete("/api/saves/{category}/{name}")
def delete_slot(category: str, name: str):
    """세이브 슬롯 삭제."""
    p = BASE_DIR / "config" / "saves" / category / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, f"슬롯 없음: {name}")
    p.unlink()
    return {"deleted": name}


@app.get("/api/default-merge-settings")
def get_default_merge_settings():
    """기본 Merge 변수 설정 반환."""
    p = BASE_DIR / "config" / "merge_settings_BR.json"
    if not p.exists():
        return {"variable_settings": {}}
    import json
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/save-default-merge-settings")
def save_default_merge_settings(req: dict):
    """현재 Merge 변수 설정을 디폴트로 저장."""
    import json
    p = BASE_DIR / "config" / "merge_settings_BR.json"
    data = {"version": 1, "type": "merge_settings",
            "timestamp": pd.Timestamp.now().isoformat(),
            "variable_settings": req.get("variable_settings", {})}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return {"saved": len(data["variable_settings"]), "path": str(p)}


@app.get("/api/default-calc-mapping")
def get_default_calc_mapping():
    """기본 Calculation 변수 매핑 반환."""
    import json
    p = BASE_DIR / "config" / "calc_mapping_default.json"
    if not p.exists():
        return {"mapping": {}}
    return json.loads(p.read_text("utf-8"))


@app.post("/api/save-default-calc-mapping")
def save_default_calc_mapping(req: dict):
    """현재 Calculation 변수 매핑을 디폴트로 저장."""
    import json
    p = BASE_DIR / "config" / "calc_mapping_default.json"
    data = {"version": 1, "type": "calc_mapping",
            "timestamp": pd.Timestamp.now().isoformat(),
            "mapping": req.get("mapping", {})}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return {"saved": len(data["mapping"]), "path": str(p)}


@app.get("/api/default-formula-settings")
def get_default_formula_settings():
    """기본 Formula 수식 설정 반환."""
    import json
    p = BASE_DIR / "config" / "formula_default.json"
    if not p.exists():
        return {"overrides": {}, "custom": []}
    return json.loads(p.read_text("utf-8"))


@app.post("/api/save-default-formula-settings")
def save_default_formula_settings(req: dict):
    """현재 Formula 수식을 디폴트로 저장."""
    import json
    p = BASE_DIR / "config" / "formula_default.json"
    data = {"version": 1, "type": "formula_settings",
            "timestamp": pd.Timestamp.now().isoformat(),
            "overrides": req.get("overrides", {}),
            "custom": req.get("custom", [])}
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    return {"saved_overrides": len(data["overrides"]),
            "saved_custom": len(data["custom"]), "path": str(p)}


@app.get("/api/default-viewer-settings")
def get_default_viewer_settings():
    """기본 Viewer 그래프 설정 반환."""
    import json
    p = BASE_DIR / "config" / "viewer_default.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text("utf-8"))


@app.post("/api/save-default-viewer-settings")
def save_default_viewer_settings(req: dict):
    """현재 Viewer 설정을 디폴트로 저장."""
    import json
    p = BASE_DIR / "config" / "viewer_default.json"
    req["timestamp"] = pd.Timestamp.now().isoformat()
    p.write_text(json.dumps(req, ensure_ascii=False, indent=2), "utf-8")
    return {"saved": True, "path": str(p)}


@app.get("/api/validate-refrigerant/{name}")
def validate_ref(name: str, backend: str = "HEOS"):
    """냉매명이 CoolProp/REFPROP에서 유효한지 검증."""
    try:
        from properties import validate_refrigerant
        return validate_refrigerant(name, backend)
    except Exception as e:
        return {"valid": False, "input": name, "coolprop_name": name,
                "error": f"검증 중 오류: {str(e)}"}


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
        case_files[case] = _classify_files(case_dir, DEFAULT_CFG.get("file_rules"))
    if not case_files: raise HTTPException(400, "유효한 케이스 없음")
    sessions[sid] = {
        "status": "ready", "progress": 0, "log": [], "error": None,
        "category_path": str(category), "case_files": case_files,
        "merge_results": [], "calc_results": [],
    }
    return {"session_id": sid, "cases": case_files}


def _classify_files(folder: Path, file_rules: dict = None) -> dict:
    """파일 식별 규칙에 따라 소스별 분류."""
    if not file_rules:
        file_rules = {
            "mx100": {"extensions": [".xls", ".xlsx"], "include_patterns": []},
            "nidaq": {"extensions": [".tsv"], "include_patterns": []},
            "ams":   {"extensions": [".csv"], "include_patterns": ["_ams"]},
            "br":    {"extensions": [".csv"], "include_patterns": []},
        }
    files = {k: [] for k in file_rules}
    order = [k for k in file_rules if k != "br"] + (["br"] if "br" in file_rules else [])
    for f in sorted(folder.iterdir()):
        if not f.is_file(): continue
        n = f.name.lower()
        if any(tag in n for tag in ["merged", "calc", "result", "formula", ".cache"]):
            continue
        for src in order:
            rule = file_rules[src]
            exts = rule.get("extensions", [])
            inc = rule.get("include_patterns", [])
            exc = rule.get("exclude_patterns", [])
            if exts and not any(n.endswith(e) for e in exts):
                continue
            if inc and not any(p.lower() in n for p in inc):
                continue
            if exc and any(p.lower() in n for p in exc):
                continue
            files[src].append(f.name)
            break
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

    all_columns = {}
    dt = DEFAULT_CFG["processing"]["data_time"]
    fr = DEFAULT_CFG.get("file_rules", {})

    # 첫 번째 케이스에서 샘플 읽기
    for case_name, cf in case_files.items():
        case_dir = category / case_name
        # 최신 파일 목록으로 재분류
        cf = _classify_files(case_dir, fr)

        def _scan_csv(src_key, label, skip=1, enc_list=("cp949","utf-8")):
            if not cf.get(src_key): return
            fp = case_dir / cf[src_key][0]
            rule = fr.get(src_key, {})
            skip_r = rule.get("skip_rows", skip)
            enc_l = [rule.get("encoding")] if rule.get("encoding") else list(enc_list)
            for enc in enc_l:
                try:
                    skiparg = list(range(skip_r)) if skip_r > 0 else None
                    df = pd.read_csv(fp, encoding=enc, skiprows=skiparg, nrows=5)
                    df.columns = [c.strip() for c in df.columns]
                    for c in df.columns:
                        if c not in all_columns:
                            all_columns[c] = {"source": label, "dtype": str(df[c].dtype)}
                    break
                except: continue

        # BR
        _scan_csv("br", "BR", skip=1, enc_list=("cp949","utf-8"))
        # AMS
        _scan_csv("ams", "AMS", skip=1, enc_list=("utf-8",))
        # NI DAQ
        _scan_csv("nidaq", "NIDAQ", skip=0, enc_list=("utf-8",))
        # MX100 (Excel)
        if cf.get("mx100"):
            try:
                fp = case_dir / cf["mx100"][0]
                skip_r = fr.get("mx100", {}).get("skip_rows", 24)
                df = pd.read_excel(fp, skiprows=skip_r, header=0, nrows=5)
                df.columns = [c.strip() for c in df.columns]
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

        # 사용 소스 확인
        sources = cfg["processing"].get("data_sources", ["BR", "AMS", "MX100"])
        use_br = "BR" in sources
        use_ams = "AMS" in sources
        use_mx = "MX100" in sources
        use_ni = "NIDAQ" in sources
        _log(sid, f"  사용 소스: {', '.join(sources)} / 기준 시간축: {dt}")

        # 변수 설정 로그
        if var_settings:
            included = [k for k, v in var_settings.items() if v.get("include", True)]
            excluded = [k for k, v in var_settings.items() if not v.get("include", True)]
            wb_applied = [k for k, v in var_settings.items()
                         if v.get("include", True) and (v.get("weight", 1.0) != 1.0 or v.get("bias", 0.0) != 0.0)]
            renamed = [k for k, v in var_settings.items()
                       if v.get("include", True) and v.get("rename") and v.get("rename") != k]
            _log(sid, f"  변수 설정: {len(included)}개 포함, {len(excluded)}개 제외, {len(wb_applied)}개 W&B, {len(renamed)}개 이름변경")

        ref_map = {"BR":"br","AMS":"ams","MX100":"mx100","NIDAQ":"nidaq"}
        total = sum(len(cf.get(ref_map.get(dt,"br"), [])) for cf in case_files.values())
        done, results = 0, []

        for case_name, cf in case_files.items():
            case_dir = category / case_name
            rename_files_in_folder(str(case_dir))
            cf = _classify_files(case_dir, cfg.get("file_rules"))
            ref_key = {"BR":"br","AMS":"ams","MX100":"mx100","NIDAQ":"nidaq"}.get(dt,"br")
            n = len(cf.get(ref_key, []))

            for i in range(n):
                t0 = time.perf_counter()
                _log(sid, f"[{case_name}] {i+1}/{n} 병합 중...")

                # 선택된 소스만 읽기
                bp = str(case_dir / cf["br"][i]) if use_br and cf.get("br") and i < len(cf["br"]) else None
                ap = str(case_dir / cf["ams"][i]) if use_ams and cf.get("ams") and i < len(cf["ams"]) else None
                mp = str(case_dir / cf["mx100"][i]) if use_mx and cf.get("mx100") and i < len(cf["mx100"]) else None
                np_ = str(case_dir / cf["nidaq"][i]) if use_ni and cf.get("nidaq") and i < len(cf["nidaq"]) else None

                t1 = time.perf_counter()
                df_ams, df_br, df_mx, df_ni = _read(ap, bp, mp, dt, np_=np_, file_rules=cfg.get("file_rules"))
                _log(sid, f"  📖 읽기: {time.perf_counter()-t1:.1f}s (BR={len(df_br)}행 MX100={len(df_mx)}행 NIDAQ={len(df_ni)}행)")

                t1 = time.perf_counter()
                # 선택된 소스만 전처리
                if use_br and not df_br.empty:
                    df_br_main, df_br_add = preprocess_blackrose(
                        df_br, get_selected_columns(cfg, "br"), get_column_mapping(cfg, "blackrose"))
                else:
                    df_br_main, df_br_add = pd.DataFrame(), pd.DataFrame()

                if use_ams and not df_ams.empty:
                    df_ams_proc = preprocess_ams(
                        df_ams, get_column_mapping(cfg, "ams"),
                        cfg.get("ams_scale_factors", {}), cfg.get("subprocess_mapping", {}),
                        get_selected_columns(cfg, "ams"))
                else:
                    df_ams_proc = pd.DataFrame()

                if use_mx and not df_mx.empty:
                    df_mx_proc = preprocess_mx100(df_mx, cfg["mx100"]["useless_columns"])
                else:
                    df_mx_proc = pd.DataFrame()

                # NI DAQ 전처리 (Time 컬럼 + 숫자 변환)
                if use_ni and not df_ni.empty:
                    df_ni_proc = df_ni.copy()
                    # Time 컬럼 확인/변환
                    if "Time" not in df_ni_proc.columns:
                        for alt in ["time","DateTime","Timestamp","Date_Time"]:
                            if alt in df_ni_proc.columns:
                                df_ni_proc.rename(columns={alt:"Time"}, inplace=True); break
                        else:
                            df_ni_proc.rename(columns={df_ni_proc.columns[0]:"Time"}, inplace=True)
                    num_cols = df_ni_proc.columns.difference(["Time"])
                    for col in num_cols:
                        df_ni_proc[col] = pd.to_numeric(df_ni_proc[col], errors="coerce")
                else:
                    df_ni_proc = pd.DataFrame()
                _log(sid, f"  ⚙️ 전처리: {time.perf_counter()-t1:.1f}s")

                t1 = time.perf_counter()
                merged = sync_and_merge(
                    [df_br_main, df_br_add, df_mx_proc, df_ams_proc, df_ni_proc],
                    dt, df_br_main, df_ams_proc, df_mx_proc)
                _log(sid, f"  🔗 시간동기화: {time.perf_counter()-t1:.1f}s ({len(merged)}행)")

                # 디버그: 압력/온도 컬럼 존재 여부
                pcols = [c for c in merged.columns if 'P_Comp' in c or 'P_Cond' in c or 'P_Eva' in c]
                tcols = [c for c in merged.columns if c.startswith('T_Cond') or c.startswith('T_Eva')]
                _log(sid, f"  🔍 병합 후 압력컬럼: {pcols or '없음'}")
                _log(sid, f"  🔍 병합 후 냉매온도: {tcols or '없음'}")
                if 'P_Comp_Out' in merged.columns:
                    v = merged['P_Comp_Out'].dropna()
                    _log(sid, f"  🔍 P_Comp_Out 원본값: min={v.min():.3f}, max={v.max():.3f}, mean={v.mean():.3f}")
                if 'T_Cond_M1' in merged.columns:
                    v = merged['T_Cond_M1'].dropna()
                    _log(sid, f"  🔍 T_Cond_M1 값: min={v.min():.1f}, max={v.max():.1f}")

                t1 = time.perf_counter()
                merged = add_time_columns(merged)
                merged = run_stage1(merged, cfg)
                if 'P_Comp_Out' in merged.columns:
                    v = merged['P_Comp_Out'].dropna()
                    _log(sid, f"  🔍 Stage1 후 P_Comp_Out: min={v.min():.3f}, max={v.max():.3f}")
                _log(sid, f"  📐 Stage1: {time.perf_counter()-t1:.1f}s")

                t1 = time.perf_counter()
                merged = run_postprocessing(merged, cfg)
                _log(sid, f"  🔧 후처리: {time.perf_counter()-t1:.1f}s")

                # ── 변수 설정 적용 (W&B + 포함/제외) ──
                if var_settings:
                    merged = _apply_variable_settings(merged, var_settings)

                # _merged.csv 저장 (한번 쓰고 복사)
                t1 = time.perf_counter()
                parts = case_name.split("_")[:9]
                prefix = "_".join(parts)
                out_name = f"{prefix}_{i+1}_merged.csv"
                result_path = RESULT_DIR / out_name
                merged.to_csv(result_path, index=False)
                import shutil
                try: shutil.copy2(str(result_path), str(case_dir / out_name))
                except: pass
                results.append(out_name)
                _log(sid, f"  💾 저장: {time.perf_counter()-t1:.1f}s")

                el = time.perf_counter() - t0
                _log(sid, f"  → {out_name} (총 {el:.1f}초, {len(merged)}행×{len(merged.columns)}열)")
                done += 1
                s["progress"] = int(done / max(total, 1) * 100)

        s.update(merge_results=results, status="done", progress=100)
        _log(sid, f"\n═══ Merge 완료: {len(results)}개 파일 ═══")

    except Exception as e:
        s.update(status="error", error=str(e))
        _log(sid, f"에러: {e}\n{traceback.format_exc()}")


def _apply_variable_settings(df: pd.DataFrame, var_settings: dict) -> pd.DataFrame:
    """
    변수별 Weight & Bias 적용 + 이름 변경 + 제외 컬럼 삭제.
    공백/언더스코어를 동일시하여 매칭.
    """
    # 퍼지 매칭 맵: 설정 키 → 실제 컬럼명
    norm = lambda s: s.strip().replace(" ", "_").lower()
    col_map = {norm(c): c for c in df.columns}
    def resolve(key):
        if key in df.columns: return key
        return col_map.get(norm(key))

    # 1. W&B 적용 (include=True인 것만)
    for col, s in var_settings.items():
        actual = resolve(col)
        if not actual: continue
        if not s.get("include", True): continue
        w = s.get("weight", 1.0)
        b = s.get("bias", 0.0)
        if w != 1.0 or b != 0.0:
            if pd.api.types.is_numeric_dtype(df[actual]):
                df[actual] = df[actual] * w + b

    # 2. 이름 변경
    rename_map = {}
    for col, s in var_settings.items():
        actual = resolve(col)
        if not actual: continue
        if not s.get("include", True): continue
        rn = s.get("rename", "")
        if rn and rn != actual and rn not in df.columns:
            rename_map[actual] = rn
    if rename_map:
        df = df.rename(columns=rename_map)

    # 3. 제외 컬럼 삭제 (Time, Time_sec, Time_min은 항상 유지)
    protected = {"Time", "Time_sec", "Time_min"}
    drop_cols = []
    for col, s in var_settings.items():
        actual = resolve(col)
        if not actual: continue
        if not s.get("include", True) and actual not in protected:
            drop_cols.append(actual)
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    return df


# ══════════════════════════════════════════════
#  로컬 Merged 파일 임포트 API
# ══════════════════════════════════════════════
class ImportRequest(BaseModel):
    file_paths: list[str]  # 로컬 CSV 파일 절대 경로 리스트
    add_merged_suffix: bool = True  # _merged 접미사 자동 추가 여부

@app.post("/api/import-merged")
def import_merged(req: ImportRequest):
    """로컬 경로의 CSV 파일을 results 폴더로 복사 + 출처 폴더 기록."""
    import shutil, json
    imported = []
    errors = []
    # 출처 메타데이터 로드/생성
    meta_path = RESULT_DIR / "_sources.json"
    sources = {}
    if meta_path.exists():
        try: sources = json.loads(meta_path.read_text("utf-8"))
        except: pass

    for fp in req.file_paths:
        p = Path(fp)
        if not p.exists():
            errors.append(f"파일 없음: {fp}")
            continue
        if not p.suffix.lower() == '.csv':
            errors.append(f"CSV 아님: {fp}")
            continue
        dest_name = p.name
        if req.add_merged_suffix and "_merged" not in dest_name and "_calc" not in dest_name:
            dest_name = p.stem + "_merged.csv"
        dest = RESULT_DIR / dest_name
        try:
            shutil.copy2(str(p), str(dest))
            imported.append(dest_name)
            # 출처 폴더 기록
            sources[dest_name] = str(p.parent.resolve())
        except Exception as e:
            errors.append(f"{p.name}: {str(e)}")

    # 메타데이터 저장
    try: meta_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), "utf-8")
    except: pass

    return {"imported": imported, "errors": errors, "count": len(imported)}


@app.post("/api/browse-files")
def browse_files(req: BrowseRequest):
    """폴더 내 CSV 파일 목록 반환 (로컬 파일 선택용)."""
    p = Path(req.path)
    if not p.exists(): raise HTTPException(404, f"경로 없음: {req.path}")
    if not p.is_dir(): raise HTTPException(400, f"디렉토리 아님: {req.path}")
    files = []
    try:
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() == '.csv':
                files.append({"name": f.name, "path": str(f.resolve()),
                              "size": f.stat().st_size,
                              "has_merged": "_merged" in f.name.lower(),
                              "has_calc": "_calc" in f.name.lower()})
    except PermissionError:
        raise HTTPException(403, f"접근 권한 없음: {req.path}")
    return {"path": str(p.resolve()), "files": files}


class ScanCasesRequest(BaseModel):
    category_path: str
    case_names: list[str]

@app.post("/api/scan-case-csvs")
def scan_case_csvs(req: ScanCasesRequest):
    """선택된 케이스 폴더들에서 CSV 파일을 일괄 수집."""
    category = Path(req.category_path)
    if not category.exists():
        raise HTTPException(404, f"경로 없음: {req.category_path}")
    result = {}  # {case_name: [{name, path, size, has_merged}]}
    total = 0
    for case in req.case_names:
        case_dir = category / case
        if not case_dir.is_dir():
            continue
        csvs = []
        for f in sorted(case_dir.iterdir()):
            if f.is_file() and f.suffix.lower() == '.csv':
                csvs.append({"name": f.name, "path": str(f.resolve()),
                             "size": f.stat().st_size,
                             "has_merged": "_merged" in f.name.lower(),
                             "has_calc": "_calc" in f.name.lower(),
                             "case": case})
                total += 1
        result[case] = csvs
    return {"category": str(category.resolve()), "cases": result, "total_files": total}


# ══════════════════════════════════════════════
#  Calculation API (Stage 2 분리)
# ══════════════════════════════════════════════
@app.post("/api/calculate")
async def start_calculation(req: CalcRequest, background_tasks: BackgroundTasks):
    sid = req.session_id
    # 세션 없으면 임시 세션 자동 생성
    if not sid or sid not in sessions:
        sid = str(uuid.uuid4())[:8]
        sessions[sid] = {
            "status": "ready", "progress": 0, "log": [], "error": None,
            "category_path": "", "case_files": {},
            "merge_results": [], "calc_results": [],
        }
    s = sessions[sid]
    if s["status"] == "processing": raise HTTPException(409, "처리 중")

    cfg = req.config or DEFAULT_CFG
    source = req.source_files or s.get("merge_results", [])
    if not source: raise HTTPException(400, "Merged 파일을 선택하세요.")

    s.update(status="processing", progress=0, log=[], calc_results=[], error=None, skipped_info={})
    background_tasks.add_task(_run_calc, sid, cfg, source, req.experimental, req.variable_mapping)
    return {"status": "started", "mode": "calculate", "source_count": len(source), "session_id": sid}


def _run_calc(sid: str, cfg: dict, source_files: list[str],
              exp: dict | None, var_map: dict | None):
    s = sessions[sid]

    try:
        _log(sid, "═══ Calculation 시작 ═══")
        env = cfg["environment"]
        from properties import resolve_refrigerant
        cp_name = resolve_refrigerant(env["refrigerant"])
        display = f"{env['refrigerant']}" if env["refrigerant"] == cp_name else f"{env['refrigerant']} → {cp_name}"
        _log(sid, f"냉매: {display} ({env.get('backend','HEOS')})")

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

            # 압력 재계산 (Merge에서 잘못 계산됐을 수 있으므로)
            from calculator import _calc_pressures
            from properties import get_props as _gp
            _props = _gp(env["refrigerant"], env["patm"], env.get("backend", "HEOS"))
            df = _calc_pressures(df, _props, env["patm"])

            # Stage 2 실행 (파일별 실험값 지원)
            file_exp = exp.get(fn, exp) if isinstance(exp, dict) and fn in exp else exp
            df_calc = run_stage2(df, cfg, file_exp)
            _log(sid, f"  계산 완료: {len(df_calc.columns)}열")
            rh_mode = "측정값 사용" if "RH_Eva_In" in df.columns else "에너지밸런스 역산 (수렴 반복)"
            _log(sid, f"  RH 모드: {rh_mode}")

            # 수렴 정보 로그
            skipped = df_calc.attrs.get("skipped_blocks", {})
            if skipped:
                _log(sid, f"  [경고] {len(skipped)}개 블록 스킵:")
                for blk, reason in skipped.items():
                    _log(sid, f"    - {blk}: {reason}")
                # 스킵 정보를 세션에 저장 (프론트엔드 팝업용)
                if "skipped_info" not in s:
                    s["skipped_info"] = {}
                s["skipped_info"][fn] = skipped

            # 수렴 반복 정보 로그
            conv = df_calc.attrs.get("converge_info")
            if conv:
                status = "수렴" if conv["converged"] else "미수렴"
                _log(sid, f"  수렴 반복: {conv['iterations']}회, 잔차={conv['residual']:.6f} ({status})")

            # _calc.csv 저장
            out_name = fn.replace("_merged.csv", "_calc.csv")
            df_calc.to_csv(RESULT_DIR / out_name, index=False)

            # 원본 폴더에도 저장
            cat_path = s.get("category_path")
            if cat_path:
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
    {"key": "P_Comp_Out",      "label": "압축기 토출 압력",      "unit": "barg", "category": "압력", "required": False,  "default_match": ["P_Comp_Out", "P_Cond_In"]},
    {"key": "P_Comp_In",       "label": "압축기 흡입 압력",      "unit": "barg", "category": "압력", "required": False,  "default_match": ["P_Comp_In", "P_Eva_Out"]},
    {"key": "P_Cond_Out",      "label": "응축기 출구 압력",      "unit": "barg", "category": "압력", "required": False, "default_match": ["P_Cond_Out"]},
    {"key": "P_Eva_In",        "label": "증발기 입구 압력",      "unit": "barg", "category": "압력", "required": False, "default_match": ["P_Eva_In"]},
    # 냉매 온도
    {"key": "T_Cond_In",       "label": "응축기 냉매 입구 온도",  "unit": "°C",  "category": "냉매 온도", "required": True,  "default_match": ["T_Cond_In", "T_Comp_Out", "Heatpump_CompTemp"]},
    {"key": "T_Cond_Out",      "label": "응축기 냉매 출구 온도",  "unit": "°C",  "category": "냉매 온도", "required": True,  "default_match": ["T_Cond_Out"]},
    {"key": "T_Cond_Mid",      "label": "응축기 냉매 중간 온도",  "unit": "°C",  "category": "냉매 온도", "required": False, "default_match": ["T_Cond_Mid", "T_Cond_M1"]},
    {"key": "T_Comp_Body",     "label": "압축기 본체 온도",      "unit": "°C",  "category": "냉매 온도", "required": False, "default_match": ["T_Comp_Body", "Heatpump_CompTemp", "T_Comp_Out"]},
    {"key": "T_Eva_Out",       "label": "증발기 냉매 출구 온도",  "unit": "°C",  "category": "냉매 온도", "required": True,  "default_match": ["T_Eva_Out", "Heatpump_EvaOutTemp"]},
    {"key": "T_Comp_In",       "label": "압축기 흡입 온도",      "unit": "°C",  "category": "냉매 온도", "required": False, "default_match": ["T_Comp_In", "Heatpump_EvaOutTemp", "T_Eva_Out"]},
    {"key": "T_Comp_Out",      "label": "압축기 토출 온도",      "unit": "°C",  "category": "냉매 온도", "required": False, "default_match": ["T_Comp_Out", "T_Cond_In"]},
    {"key": "T_Subcooler_Out", "label": "서브쿨러 출구 온도",    "unit": "°C",  "category": "냉매 온도", "required": False, "default_match": ["T_Subcooler_Out"]},
    # 공기 온도
    {"key": "T_Air_Eva_In",    "label": "증발기 공기 입구 온도",  "unit": "°C",  "category": "공기 온도", "required": True,  "default_match": ["T_Air_Eva_In", "Heatpump_DuctInTemp"]},
    {"key": "T_Air_Eva_Out",   "label": "증발기 공기 출구 온도",  "unit": "°C",  "category": "공기 온도", "required": True,  "default_match": ["T_Air_Eva_Out"]},
    {"key": "T_Air_Cond_Out",  "label": "응축기 공기 출구 온도",  "unit": "°C",  "category": "공기 온도", "required": True,  "default_match": ["T_Air_Cond_Out", "Heatpump_DuctOutTemp"]},
    {"key": "RH_Eva_In",       "label": "증발기 입구 RH (측정)", "unit": "-",   "category": "공기 온도", "required": False, "default_match": ["RH_Eva_In", "RH_Eva_In_measure"]},
    # 전력
    {"key": "Po_WD",           "label": "총 전력",             "unit": "W",   "category": "전력", "required": True,  "default_match": ["Po_WD"]},
    {"key": "Po_Comp",         "label": "압축기 전력",          "unit": "W",   "category": "전력", "required": True,  "default_match": ["Po_Comp"]},
    {"key": "Po_Fan",          "label": "팬 전력",             "unit": "W",   "category": "전력", "required": True,  "default_match": ["Po_Fan"]},
    # 제어
    {"key": "Ctrl_Comp_Hz",    "label": "압축기 주파수",        "unit": "Hz",  "category": "제어", "required": False, "default_match": ["Ctrl_Comp_Hz", "HP_CompCurrentHz"]},
    {"key": "Ctrl_DryMotion",  "label": "건조 모션 정보",       "unit": "-",   "category": "제어", "required": False, "default_match": ["Ctrl_DryMotion", "Heatpump_DryMotionInfo"]},
    # 시간
    {"key": "Time_min",        "label": "시간 (분)",           "unit": "min", "category": "시간", "required": True,  "default_match": ["Time_min"]},
    {"key": "Time_sec",        "label": "시간 (초)",           "unit": "sec", "category": "시간", "required": True,  "default_match": ["Time_sec"]},
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
def _read(ap, bp, mp, dt, np_=None, file_rules=None):
    df_a, df_b, df_m, df_n = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    fr = file_rules or {}
    if bp and os.path.exists(bp):
        enc_l = [fr.get("br",{}).get("encoding","cp949")] if fr.get("br",{}).get("encoding") else ["cp949","utf-8"]
        skip = fr.get("br",{}).get("skip_rows",1)
        for enc in enc_l:
            try: df_b = pd.read_csv(bp, encoding=enc, skiprows=list(range(skip)) if skip else None); df_b.columns=[c.strip() for c in df_b.columns]; break
            except: continue
    if ap and os.path.exists(ap):
        enc = fr.get("ams",{}).get("encoding","utf-8")
        skip = fr.get("ams",{}).get("skip_rows",1)
        try: df_a = pd.read_csv(ap, encoding=enc, skiprows=list(range(skip)) if skip else None); df_a.columns=[c.strip() for c in df_a.columns]
        except: pass
    if mp and os.path.exists(mp):
        skip = fr.get("mx100",{}).get("skip_rows",24)
        try:
            df_m = pd.read_excel(mp, skiprows=skip, header=0)
            df_m.columns = [c.strip() for c in df_m.columns]
        except Exception as e1:
            try:
                df_m = pd.read_excel(mp, skiprows=skip, header=[0,1])
                cols = [c[1] if "Unnamed" in str(c[0]) else c[0] for c in df_m.columns]
                df_m.columns = [c.strip() for c in cols]
                if "Date" in df_m.columns and "Time" in df_m.columns:
                    df_m["Time"] = df_m["Date"].astype(str)+" "+df_m["Time"].astype(str)
                    df_m.drop(columns=["Date"], inplace=True, errors="ignore")
            except Exception as e2:
                print(f"[MX100] 읽기 실패: {e1} / {e2}")
    if np_ and os.path.exists(np_):
        enc = fr.get("nidaq",{}).get("encoding","utf-8")
        skip = fr.get("nidaq",{}).get("skip_rows",0)
        try:
            sep = '\t' if np_.lower().endswith('.tsv') else ','
            df_n = pd.read_csv(np_, encoding=enc, sep=sep, skiprows=list(range(skip)) if skip else None)
            df_n.columns = [c.strip() for c in df_n.columns]
        except Exception as e:
            print(f"[NIDAQ] 읽기 실패: {e}")
    return df_a, df_b, df_m, df_n

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
    import json
    files = sorted(RESULT_DIR.glob("*.csv"), key=os.path.getmtime, reverse=True)
    # 출처 메타데이터
    meta_path = RESULT_DIR / "_sources.json"
    sources = {}
    if meta_path.exists():
        try: sources = json.loads(meta_path.read_text("utf-8"))
        except: pass
    return [{"name": f.name, "size": f.stat().st_size,
             "type": "merged" if "_merged" in f.name else "formula" if "_formula" in f.name else "calc" if "_calc" in f.name else "other",
             "source_dir": sources.get(f.name, "")}
            for f in files]

@app.get("/api/results/{fn:path}")
def download(fn: str):
    from urllib.parse import unquote
    fn = unquote(fn)
    p = RESULT_DIR / fn
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, media_type="text/csv", filename=fn)


@app.delete("/api/results/{fn:path}")
def delete_result(fn: str):
    """결과 파일 삭제 + 출처 메타데이터 정리."""
    import json
    from urllib.parse import unquote
    fn = unquote(fn)
    p = RESULT_DIR / fn
    if not p.exists(): raise HTTPException(404, f"파일 없음: {fn}")
    p.unlink()
    # 메타데이터에서도 제거
    meta_path = RESULT_DIR / "_sources.json"
    if meta_path.exists():
        try:
            sources = json.loads(meta_path.read_text("utf-8"))
            sources.pop(fn, None)
            meta_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2), "utf-8")
        except: pass
    return {"deleted": fn}

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
# (STATIC은 상단에서 정의됨)
# ══════════════════════════════════════════════
#  Viewer 데이터 API
# ══════════════════════════════════════════════
@app.get("/api/viewer-data/{fn:path}")
def viewer_data(fn: str, max_rows: int = 7200):
    """calc/merged CSV를 컬럼별 배열 JSON으로 반환 (Viewer용)."""
    from urllib.parse import unquote
    fn = unquote(fn)
    p = RESULT_DIR / fn
    if not p.exists():
        print(f"  [viewer-data] 404: {fn}, RESULT_DIR={RESULT_DIR}")
        print(f"  [viewer-data] 존재하는 파일: {[f.name for f in RESULT_DIR.iterdir() if f.is_file()][:10]}")
        raise HTTPException(404, f"파일 없음: {fn}")
    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise HTTPException(400, f"CSV 읽기 실패: {str(e)}")
    if len(df) > max_rows:
        step = max(1, len(df) // max_rows)
        df = df.iloc[::step].reset_index(drop=True)
    
    # 숫자 변환 시도 (문자열로 읽힌 컬럼 복구)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and c != "Time":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # NaN → None 처리 + 숫자 컬럼만
    cols = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].round(4)
            cols[c] = [None if pd.isna(v) else float(v) for v in s]
    return {"filename": fn, "rows": len(df), "columns": list(cols.keys()), "data": cols}


@app.get("/api/ref-saturation/{name}")
def ref_saturation(name: str):
    """냉매 포화선 데이터 반환 (P-h 선도 배경용)."""
    try:
        from properties import resolve_refrigerant
        cp_name = resolve_refrigerant(name)
        from CoolProp.CoolProp import PropsSI
        Tc = PropsSI("Tcrit", cp_name) - 273.15
        Pc = PropsSI("Pcrit", cp_name) / 1e5
        hc = PropsSI("H", "T", Tc + 273.15, "P", Pc * 1e5, cp_name) / 1000

        Tmin = max(-50, PropsSI("Tmin", cp_name) - 273.15 + 1)
        temps = np.linspace(Tmin, Tc - 0.5, 200)
        hl, hv, P = [], [], []
        for t in temps:
            try:
                tk = t + 273.15
                p = PropsSI("P", "T", tk, "Q", 0, cp_name) / 1e5
                h_l = PropsSI("H", "T", tk, "Q", 0, cp_name) / 1000
                h_v = PropsSI("H", "T", tk, "Q", 1, cp_name) / 1000
                P.append(round(p, 4)); hl.append(round(h_l, 2)); hv.append(round(h_v, 2))
            except:
                pass
        return {"name": name, "coolprop_name": cp_name,
                "critical": {"T": round(Tc, 2), "P": round(Pc, 3), "h": round(hc, 2)},
                "sat": {"P": P, "hl": hl, "hv": hv}}
    except Exception as e:
        raise HTTPException(400, f"냉매 포화선 생성 실패: {str(e)}")


@app.post("/api/compute-enthalpy")
def compute_enthalpy(req: dict):
    """
    T + P 배열로부터 냉매 엔탈피 일괄 계산 (P-h 선도용).
    merged CSV에 h 컬럼이 없을 때 프론트엔드에서 호출.
    입력: {T_Comp_In:[], T_Comp_Out:[], T_Cond_Out:[], P_Comp_In:[], P_Comp_Out:[], refrigerant, patm}
    출력: {h_Comp_In:[], h_Comp_Out:[], h_Cond_Out:[]}
    """
    try:
        from properties import resolve_refrigerant
        from CoolProp.CoolProp import PropsSI

        ref = resolve_refrigerant(req.get("refrigerant", "R290"))
        patm = req.get("patm", 101.325)  # kPa

        def calc_h(T_arr, P_arr_barg):
            """T[°C] + P[barg] → h[kJ/kg] 배열."""
            if not T_arr or not P_arr_barg:
                return None
            result = []
            for t, p in zip(T_arr, P_arr_barg):
                if t is None or p is None:
                    result.append(None)
                    continue
                try:
                    p_pa = (p + patm / 100) * 1e5  # barg → Pa
                    h = PropsSI("H", "T", t + 273.15, "P", p_pa, ref) / 1000  # J/kg → kJ/kg
                    result.append(round(h, 2))
                except:
                    result.append(None)
            return result

        out = {}
        T_ci = req.get("T_Comp_In")
        T_co = req.get("T_Comp_Out")
        T_cd = req.get("T_Cond_Out")
        P_in = req.get("P_Comp_In")
        P_out = req.get("P_Comp_Out")

        if T_ci and P_in:
            out["h_Comp_In"] = calc_h(T_ci, P_in)
        if T_co and P_out:
            out["h_Comp_Out"] = calc_h(T_co, P_out)
        if T_cd and P_out:
            out["h_Cond_Out"] = calc_h(T_cd, P_out)

        return out
    except Exception as e:
        raise HTTPException(400, f"엔탈피 계산 실패: {str(e)}")


# ══════════════════════════════════════════════
#  커스텀 수식 엔진
# ══════════════════════════════════════════════
@app.post("/api/eval-formulas")
def eval_formulas(req: dict):
    """
    CSV 파일에 수식을 적용하여 새 컬럼 추가.
    입력: {filename: str, formulas: [{name, expr, enabled}]}
    pandas df.eval() 사용 — 기존 컬럼 참조 가능.
    """
    fn = req.get("filename")
    formulas = req.get("formulas", [])
    if not fn:
        raise HTTPException(400, "filename 필요")

    p = RESULT_DIR / fn
    if not p.exists():
        raise HTTPException(404, f"파일 없음: {fn}")

    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise HTTPException(400, f"CSV 읽기 실패: {str(e)}")

    results = []
    errors = []
    for f in formulas:
        name = f.get("name", "").strip()
        expr = f.get("expr", "").strip()
        if not name or not expr or not f.get("enabled", True):
            continue
        try:
            # 안전성: 위험 키워드 차단
            dangerous = ["import", "exec", "eval", "__", "open", "os.", "sys.", "subprocess"]
            if any(d in expr.lower() for d in dangerous):
                errors.append({"name": name, "error": "허용되지 않는 키워드"})
                continue
            df[name] = df.eval(expr)
            results.append({"name": name, "expr": expr, "rows": int(df[name].notna().sum())})
        except Exception as e:
            errors.append({"name": name, "error": str(e)})

    # 결과 저장
    out_name = fn.replace(".csv", "_formula.csv")
    df.to_csv(RESULT_DIR / out_name, index=False)

    return {
        "output": out_name,
        "applied": results,
        "errors": errors,
        "total_columns": len(df.columns),
        "total_rows": len(df),
    }


@app.post("/api/preview-formula")
def preview_formula(req: dict):
    """수식 1개를 미리보기 (처음 10행만)."""
    fn = req.get("filename")
    expr = req.get("expr", "").strip()
    if not fn or not expr:
        raise HTTPException(400, "filename, expr 필요")

    p = RESULT_DIR / fn
    if not p.exists():
        raise HTTPException(404, f"파일 없음: {fn}")

    try:
        df = pd.read_csv(p, nrows=100)
        dangerous = ["import", "exec", "eval", "__", "open", "os.", "sys.", "subprocess"]
        if any(d in expr.lower() for d in dangerous):
            raise HTTPException(400, "허용되지 않는 키워드")
        result = df.eval(expr)
        sample = [None if pd.isna(v) else round(float(v), 4) for v in result.head(10)]
        return {"preview": sample, "dtype": str(result.dtype)}
    except Exception as e:
        raise HTTPException(400, f"수식 오류: {str(e)}")


@app.get("/api/formula-columns/{fn}")
def formula_columns(fn: str):
    """수식에서 참조할 수 있는 컬럼 목록."""
    p = RESULT_DIR / fn
    if not p.exists():
        raise HTTPException(404)
    df = pd.read_csv(p, nrows=1)
    return {"columns": df.columns.tolist(), "numeric": df.select_dtypes(include=[np.number]).columns.tolist()}


@app.get("/api/version")
def get_version():
    """현재 버전 정보 — 최신 파일 수정 날짜 기반."""
    import subprocess
    # 1순위: git 커밋 날짜
    try:
        date_str = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"],
            cwd=str(BASE_DIR), stderr=subprocess.DEVNULL
        ).decode().strip()[:10]  # "2026-03-23"
        commit = subprocess.check_output(
            ["git", "log", "-1", "--format=%h"],
            cwd=str(BASE_DIR), stderr=subprocess.DEVNULL
        ).decode().strip()
        return {"date": date_str, "commit": commit, "base_dir": str(BASE_DIR)}
    except:
        pass
    # 2순위: server.py 수정 날짜
    try:
        import datetime
        mtime = os.path.getmtime(BASE_DIR / "server.py")
        date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        return {"date": date_str, "commit": "", "base_dir": str(BASE_DIR)}
    except:
        return {"date": "unknown", "commit": "", "base_dir": str(BASE_DIR)}


@app.post("/api/update")
def self_update():
    """GitHub에서 최신 코드 다운로드 + 서버 재시작."""
    import subprocess, zipfile, io, shutil
    repo_url = "https://github.com/kimjy2576/Dryer-Merger/archive/refs/heads/main.zip"

    try:
        # 1. 다운로드
        import urllib.request
        resp = urllib.request.urlopen(repo_url, timeout=30)
        zip_data = resp.read()

        # 2. 임시 폴더에 압축 해제
        tmp = BASE_DIR / "_update_tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            zf.extractall(tmp)

        # 3. 압축 해제된 폴더 찾기
        extracted = list(tmp.iterdir())
        src = extracted[0] if len(extracted) == 1 and extracted[0].is_dir() else tmp

        # 4. 파일 덮어쓰기 (config/saves는 보존)
        preserve = {"config/saves", "results", "_update_tmp"}
        updated = []
        for item in src.rglob("*"):
            rel = item.relative_to(src)
            # 보존 폴더 스킵
            if any(str(rel).startswith(p) for p in preserve):
                continue
            dest = BASE_DIR / rel
            if item.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(item), str(dest))
                updated.append(str(rel))

        # 5. 임시 폴더 삭제
        shutil.rmtree(tmp, ignore_errors=True)

        # 6. 서버 재시작 (백그라운드)
        import threading
        def restart():
            import time; time.sleep(1)
            os._exit(0)  # uvicorn이 재시작
        threading.Thread(target=restart, daemon=True).start()

        return {"status": "updated", "files": len(updated), "message": "서버 재시작 중... 3초 후 새로고침하세요."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
def index():
    f = STATIC / "index.html"
    return HTMLResponse(f.read_text("utf-8")) if f.exists() else HTMLResponse("<h1>HPWD Data Manager</h1>")

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
