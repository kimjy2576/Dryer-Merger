"""
server.py — FastAPI 백엔드 (v4.1 — 서버 경로 탐색 방식)
파일 업로드 없이 서버 로컬/네트워크 경로에서 직접 데이터를 읽음.
"""
import os, uuid, time, traceback
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

app = FastAPI(title="Dryer Merger", version="4.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    return {"status": "ok"}

BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

sessions: dict[str, dict] = {}
DEFAULT_CFG = load_config()


# ══════════════════════════════════════════════
#  Pydantic 모델
# ══════════════════════════════════════════════
class BrowseRequest(BaseModel):
    path: str

class SelectRequest(BaseModel):
    category_path: str           # 상위 폴더 경로
    case_names: list[str]        # 선택한 케이스 폴더명 리스트

class ProcessRequest(BaseModel):
    session_id: str
    config: Optional[dict] = None
    experimental: Optional[dict] = None  # {"load_kg", "imc_kg", "fmc_kg"}


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
#  경로 탐색 API (업로드 대체)
# ══════════════════════════════════════════════
@app.post("/api/browse")
def browse_directory(req: BrowseRequest):
    """
    지정 경로 내의 폴더/파일 목록을 반환한다.
    폴더 선택 UI용.
    """
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(404, f"경로를 찾을 수 없습니다: {req.path}")
    if not p.is_dir():
        raise HTTPException(400, f"디렉토리가 아닙니다: {req.path}")

    items = []
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue
            items.append({
                "name": item.name,
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None,
            })
    except PermissionError:
        raise HTTPException(403, f"접근 권한이 없습니다: {req.path}")

    # 서브폴더(케이스) 목록
    cases = [i["name"] for i in items if i["is_dir"]]

    return {
        "path": str(p.resolve()),
        "items": items,
        "cases": cases,
    }


@app.post("/api/select")
def select_cases(req: SelectRequest):
    """
    케이스 폴더를 선택하고 세션을 생성한다.
    각 케이스 폴더 내 BR/AMS/MX100 파일을 자동 탐지.
    """
    category = Path(req.category_path)
    if not category.exists():
        raise HTTPException(404, f"경로 없음: {req.category_path}")

    sid = str(uuid.uuid4())[:8]
    case_files = {}

    for case in req.case_names:
        case_dir = category / case
        if not case_dir.is_dir():
            continue
        case_files[case] = _classify_files(case_dir)

    if not case_files:
        raise HTTPException(400, "유효한 케이스 폴더가 없습니다.")

    sessions[sid] = {
        "status": "ready",
        "progress": 0,
        "log": [],
        "results": [],
        "error": None,
        "category_path": str(category),
        "case_files": case_files,
    }

    return {
        "session_id": sid,
        "cases": case_files,
    }


def _classify_files(folder: Path) -> dict:
    """폴더 내 파일을 BR/AMS/MX100으로 분류."""
    files = {"br": [], "ams": [], "mx100": []}
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        n = f.name.lower()
        if "_br.csv" in n or (n.endswith(".csv") and "br" in n):
            files["br"].append(f.name)
        elif "_ams.csv" in n or (n.endswith(".csv") and "ams" in n):
            files["ams"].append(f.name)
        elif n.endswith((".xls", ".xlsx")) or "_temp." in n:
            files["mx100"].append(f.name)
        elif n.endswith(".csv") and "_merged" not in n and "_calc" not in n:
            files["br"].append(f.name)
    return files


# ══════════════════════════════════════════════
#  처리 API
# ══════════════════════════════════════════════
@app.post("/api/process")
async def start_processing(req: ProcessRequest, background_tasks: BackgroundTasks):
    sid = req.session_id
    if sid not in sessions:
        raise HTTPException(404, "세션 없음")
    s = sessions[sid]
    if s["status"] == "processing":
        raise HTTPException(409, "처리 중")

    cfg = req.config or DEFAULT_CFG
    s.update(status="processing", progress=0, log=[], results=[], error=None)
    background_tasks.add_task(_run, sid, cfg, req.experimental)
    return {"status": "started"}


def _run(sid: str, cfg: dict, exp: dict | None):
    s = sessions[sid]
    category = Path(s["category_path"])
    case_files = s["case_files"]
    dt = cfg["processing"]["data_time"]
    calc_on = cfg.get("calculation", {}).get("enabled", False)

    try:
        _log(sid, "통합 파이프라인 시작...")
        env = cfg["environment"]
        _log(sid, f"냉매: {env['refrigerant']} / Stage 2: {'ON' if calc_on else 'OFF'}")

        from properties import get_props
        get_props(env["refrigerant"], env["patm"], env.get("backend", "HEOS"))

        from preprocessor import (
            preprocess_blackrose, preprocess_ams, preprocess_mx100,
            sync_and_merge, add_time_columns,
        )
        from calculator import run_stage1
        from postprocessor import run_postprocessing
        from performance import run_stage2
        from config import get_column_mapping, get_selected_columns
        from io_handler import rename_files_in_folder

        # 전체 파일 수 산정
        total_files = sum(
            len(cf[{"BR":"br","AMS":"ams","MX100":"mx100"}[dt]])
            for cf in case_files.values()
        )
        done = 0
        results = []

        for case_name, cf in case_files.items():
            case_dir = category / case_name
            rename_files_in_folder(str(case_dir))
            # 파일 재분류 (rename 후)
            cf = _classify_files(case_dir)

            ref_key = {"BR": "br", "AMS": "ams", "MX100": "mx100"}[dt]
            n = len(cf[ref_key])

            for i in range(n):
                t0 = time.perf_counter()
                _log(sid, f"[{case_name}] {i+1}/{n} 처리 중...")

                # 경로 직접 구성 (업로드 없이 원본 경로에서 읽기)
                bp = str(case_dir / cf["br"][i]) if i < len(cf["br"]) else None
                ap = str(case_dir / cf["ams"][i]) if i < len(cf["ams"]) else None
                mp = str(case_dir / cf["mx100"][i]) if i < len(cf["mx100"]) else None

                df_ams, df_br, df_mx = _read(ap, bp, mp, dt)

                df_br_main, df_br_add = preprocess_blackrose(
                    df_br, get_selected_columns(cfg, "br"),
                    get_column_mapping(cfg, "blackrose"))
                df_ams_proc = preprocess_ams(
                    df_ams, get_column_mapping(cfg, "ams"),
                    cfg.get("ams_scale_factors", {}),
                    cfg.get("subprocess_mapping", {}),
                    get_selected_columns(cfg, "ams"))
                df_mx_proc = preprocess_mx100(df_mx, cfg["mx100"]["useless_columns"])

                merged = sync_and_merge(
                    [df_br_main, df_br_add, df_mx_proc, df_ams_proc],
                    dt, df_br_main, df_ams_proc, df_mx_proc)
                merged = add_time_columns(merged)
                merged = run_stage1(merged, cfg)
                merged = run_postprocessing(merged, cfg)
                _log(sid, f"  Stage 1: {len(merged)}행")

                if calc_on:
                    _log(sid, "  Stage 2 실행 중...")
                    merged = run_stage2(merged, cfg, exp)
                    _log(sid, f"  Stage 2: {len(merged.columns)}열")

                # 결과 저장 (results 폴더 + 원본 폴더 양쪽)
                parts = case_name.split("_")[:9]
                prefix = "_".join(parts)
                out_name = f"{prefix}_{i+1}_result.csv"

                # 서버 results 폴더
                (RESULT_DIR / out_name).parent.mkdir(exist_ok=True)
                merged.to_csv(RESULT_DIR / out_name, index=False)

                # 원본 케이스 폴더에도 저장
                merged.to_csv(case_dir / out_name, index=False)

                results.append(out_name)
                el = time.perf_counter() - t0
                _log(sid, f"  → {out_name} ({el:.1f}초, {len(merged)}행×{len(merged.columns)}열)")
                _log(sid, f"  → 저장: {case_dir / out_name}")

                done += 1
                s["progress"] = int(done / max(total_files, 1) * 100)

        s.update(results=results, status="done", progress=100)
        _log(sid, f"\n전체 완료: {len(results)}개 파일 생성")

    except Exception as e:
        s.update(status="error", error=str(e))
        _log(sid, f"에러: {e}\n{traceback.format_exc()}")


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
    return {k: s[k] for k in ("status","progress","log","results","error")}

@app.get("/api/results")
def list_results():
    files = sorted(RESULT_DIR.glob("*.csv"), key=os.path.getmtime, reverse=True)
    return [{"name": f.name, "size": f.stat().st_size} for f in files]

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


# ══════════════════════════════════════════════
#  프론트엔드 서빙
# ══════════════════════════════════════════════
STATIC = BASE_DIR / "static"
@app.get("/")
def index():
    f = STATIC / "index.html"
    return HTMLResponse(f.read_text("utf-8")) if f.exists() else HTMLResponse("<h1>Dryer Merger v4.1</h1>")

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
