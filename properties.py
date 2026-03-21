"""
properties.py — 통합 물성 계산 라이브러리 v2

[개선사항]
1. 단상 물성: 2D 그리드 캐시 (T×P → h,s,v,rho,cp,cv)
   → CoolProp 루프 호출 제거, scipy RegularGridInterpolator 사용
2. 습공기: 고온 정밀도 개선 (Cp_v 온도 의존성 반영)
3. psat_water: 0°C 이하 얼음 포화압 분기 (Murphy & Koop 2005)
4. 2상 경계: 포화온도 판별 후 분기 (단순 클램핑 제거)
"""
import numpy as np
from CoolProp.CoolProp import PropsSI
from scipy.interpolate import RegularGridInterpolator
import pandas as _pd


# ════════════════════════════════════════════════
#  냉매 별칭 매핑 (사용자 입력 → CoolProp 이름)
# ════════════════════════════════════════════════
REFRIGERANT_ALIASES = {
    "R290": "Propane",
    "R600a": "IsoButane",
    "R600": "n-Butane",
    "R717": "Ammonia",
    "R744": "CarbonDioxide",
    "R718": "Water",
    "R1270": "Propylene",
    "R22": "R22",
    "R407C": "R407C",
    "R404A": "R404A",
    "R507A": "R507A",
    "R410A": "R410A",
    "R134a": "R134a",
    "R32": "R32",
    "R1234yf": "R1234yf",
    "R1234ze(E)": "R1234ze(E)",
    "R152a": "R152a",
    "R513A": "R513A",
    "R454C": "R454C",
    "R454A": "R454A",
    "R455A": "R455A",
    "Propane": "Propane",
}


def resolve_refrigerant(name: str) -> str:
    """사용자 입력 냉매명을 CoolProp 이름으로 변환."""
    if name in REFRIGERANT_ALIASES:
        return REFRIGERANT_ALIASES[name]
    return name  # 별칭 없으면 그대로 전달 (CoolProp이 직접 판단)


def validate_refrigerant(name: str, backend: str = "HEOS") -> dict:
    """냉매명이 유효한지 검증. 성공 시 CoolProp 이름 + 임계점 반환."""
    cp_name = resolve_refrigerant(name)
    try:
        Tc = PropsSI("Tcrit", f"{backend}::{cp_name}") - 273.15
        Pc = PropsSI("Pcrit", f"{backend}::{cp_name}") / 1e5
        return {"valid": True, "input": name, "coolprop_name": cp_name,
                "Tc": round(Tc, 1), "Pc": round(Pc, 1)}
    except Exception as e:
        return {"valid": False, "input": name, "coolprop_name": cp_name,
                "error": str(e)}


class FluidProperties:
    """
    냉매 물성 고속 계산기.
    - 포화선: 1D LUT + np.interp
    - 단상: 2D 그리드 캐시 + RegularGridInterpolator
    - 초기화 ~1-2초, 이후 배열 즉시 반환 (CoolProp 루프 0회)
    """

    def __init__(self, refrigerant: str, patm: float = 101.325, backend: str = "HEOS"):
        self.refrigerant_input = refrigerant
        self.refrigerant = resolve_refrigerant(refrigerant)
        self.patm = patm
        self.backend = backend
        self._ref = f"{backend}::{self.refrigerant}"
        self._build_sat_tables()
        self._build_2d_cache()

    # ──── 1D 포화선 LUT ────
    def _build_sat_tables(self):
        ref = self._ref
        # 임계온도 이하로 제한
        Tc = PropsSI("Tcrit", ref) - 273.15  # °C
        t_max = min(130, Tc - 0.5)  # 임계점 0.5°C 아래까지
        self._t_range = np.linspace(-40, t_max, int((t_max + 40) * 10) + 1)
        t_k = self._t_range + 273.15
        self._psat_from_t = np.array([
            PropsSI("P", "T", tk, "Q", 0, ref) / 1000 for tk in t_k])

        self._p_barg = np.linspace(-1.0, 45.0, 4601)
        p_pa = (self._p_barg * 100 + self.patm) * 1000

        keys = ["tsat", "h_liq", "h_vap", "s_liq", "s_vap", "v_liq", "v_vap"]
        self._sat = {k: np.zeros(len(self._p_barg)) for k in keys}
        for i, pp in enumerate(p_pa):
            try:
                self._sat["tsat"][i] = PropsSI("T", "P", pp, "Q", 1, ref)
                self._sat["h_liq"][i] = PropsSI("H", "P", pp, "Q", 0, ref) / 1000
                self._sat["h_vap"][i] = PropsSI("H", "P", pp, "Q", 1, ref) / 1000
                self._sat["s_liq"][i] = PropsSI("S", "P", pp, "Q", 0, ref) / 1000
                self._sat["s_vap"][i] = PropsSI("S", "P", pp, "Q", 1, ref) / 1000
                dl = PropsSI("D", "P", pp, "Q", 0, ref)
                dv = PropsSI("D", "P", pp, "Q", 1, ref)
                self._sat["v_liq"][i] = 1/dl if dl > 0 else 0
                self._sat["v_vap"][i] = 1/dv if dv > 0 else 0
            except Exception:
                pass

    # ──── 2D 단상 물성 그리드 캐시 ────
    def _build_2d_cache(self):
        """
        T [-40..150°C, 1°C] × P [-1..45 barg, 0.5bar]
        → h, s, v, rho, cp, cv 6종 RegularGridInterpolator
        ~18000 포인트, ~1.5초
        """
        ref = self._ref
        self._grid_t = np.arange(-40, 151, 1.0)
        self._grid_p = np.arange(-1.0, 45.5, 0.5)
        nt, npg = len(self._grid_t), len(self._grid_p)

        prop_keys = ["H", "S", "D", "C", "CVMASS"]
        scales    = [1000, 1000, 1, 1000, 1000]
        names     = ["h", "s", "rho", "cp", "cv"]
        grids = {n: np.full((nt, npg), np.nan) for n in names}
        grids["v"] = np.full((nt, npg), np.nan)

        for i, tc in enumerate(self._grid_t):
            tk = tc + 273.15
            for j, pb in enumerate(self._grid_p):
                ppa = (pb * 100 + self.patm) * 1000
                try:
                    for pk, sc, nm in zip(prop_keys, scales, names):
                        grids[nm][i, j] = PropsSI(pk, "T", tk, "P", ppa, ref) / sc
                    grids["v"][i, j] = 1.0 / grids["rho"][i, j] if grids["rho"][i, j] > 0 else np.nan
                except Exception:
                    pass

        self._interps = {}
        for nm in ["h", "s", "v", "rho", "cp", "cv"]:
            g = grids[nm]
            # NaN → ffill/bfill (2상 영역 등 계산 불가 포인트 채우기)
            df_g = _pd.DataFrame(g)
            g = df_g.ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1).values
            self._interps[nm] = RegularGridInterpolator(
                (self._grid_t, self._grid_p), g,
                method="linear", bounds_error=False, fill_value=None)

    # ──── 포화선 보간 ────
    def psat_t(self, t_c):
        return np.interp(np.asarray(t_c, dtype=float), self._t_range, self._psat_from_t)

    def psat_t_barg(self, t_c):
        return np.round((self.psat_t(t_c) - self.patm) / 100, 2)

    def tsat_p(self, p_barg):
        return np.interp(np.asarray(p_barg, dtype=float), self._p_barg, self._sat["tsat"]) - 273.15

    def h_liq(self, p): return np.interp(np.asarray(p, dtype=float), self._p_barg, self._sat["h_liq"])
    def h_vap(self, p): return np.interp(np.asarray(p, dtype=float), self._p_barg, self._sat["h_vap"])
    def h_latent(self, p): return self.h_vap(p) - self.h_liq(p)
    def s_liq(self, p): return np.interp(np.asarray(p, dtype=float), self._p_barg, self._sat["s_liq"])
    def s_vap(self, p): return np.interp(np.asarray(p, dtype=float), self._p_barg, self._sat["s_vap"])
    def v_liq(self, p): return np.interp(np.asarray(p, dtype=float), self._p_barg, self._sat["v_liq"])
    def v_vap(self, p): return np.interp(np.asarray(p, dtype=float), self._p_barg, self._sat["v_vap"])

    # ──── 2D 단상 물성 (벡터 즉시 반환) ────
    def _q2d(self, name, t_c, p_barg):
        t = np.atleast_1d(np.asarray(t_c, dtype=float))
        p = np.atleast_1d(np.asarray(p_barg, dtype=float))
        pts = np.column_stack([t.ravel(), p.ravel()])
        return self._interps[name](pts).reshape(t.shape)

    def h_tp(self, t_c, p_barg):  return self._q2d("h", t_c, p_barg)
    def s_tp(self, t_c, p_barg):  return self._q2d("s", t_c, p_barg)
    def v_tp(self, t_c, p_barg):  return self._q2d("v", t_c, p_barg)
    def rho_tp(self, t_c, p_barg): return self._q2d("rho", t_c, p_barg)
    def cp_tp(self, t_c, p_barg): return self._q2d("cp", t_c, p_barg)
    def cv_tp(self, t_c, p_barg): return self._q2d("cv", t_c, p_barg)

    # ──── 상태 판별 + 보정 (포화온도 기반) ────
    def h_tp_superheat(self, t_c, p_barg):
        """과열 증기 엔탈피. T < Tsat이면 포화증기 반환."""
        t = np.asarray(t_c, dtype=float)
        t_sat = self.tsat_p(p_barg)
        h_calc = self.h_tp(t_c, p_barg)
        h_sat = self.h_vap(p_barg)
        return np.where(t < t_sat, h_sat, np.maximum(h_calc, h_sat))

    def s_tp_superheat(self, t_c, p_barg):
        t = np.asarray(t_c, dtype=float)
        t_sat = self.tsat_p(p_barg)
        return np.where(t < t_sat, self.s_vap(p_barg),
                        np.maximum(self.s_tp(t_c, p_barg), self.s_vap(p_barg)))

    def v_tp_superheat(self, t_c, p_barg):
        t = np.asarray(t_c, dtype=float)
        t_sat = self.tsat_p(p_barg)
        return np.where(t < t_sat, self.v_vap(p_barg),
                        np.maximum(self.v_tp(t_c, p_barg), self.v_vap(p_barg)))

    def h_tp_subcool(self, t_c, p_barg):
        """과냉 액체 엔탈피. T > Tsat이면 포화액 반환 (불완전 응축 방어)."""
        t = np.asarray(t_c, dtype=float)
        t_sat = self.tsat_p(p_barg)
        h_calc = self.h_tp(t_c, p_barg)
        h_sat = self.h_liq(p_barg)
        return np.where(t > t_sat, h_sat, np.minimum(h_calc, h_sat))

    def s_tp_subcool(self, t_c, p_barg):
        t = np.asarray(t_c, dtype=float)
        t_sat = self.tsat_p(p_barg)
        return np.where(t > t_sat, self.s_liq(p_barg),
                        np.minimum(self.s_tp(t_c, p_barg), self.s_liq(p_barg)))

    def v_tp_subcool(self, t_c, p_barg):
        t = np.asarray(t_c, dtype=float)
        t_sat = self.tsat_p(p_barg)
        return np.where(t > t_sat, self.v_liq(p_barg),
                        np.minimum(self.v_tp(t_c, p_barg), self.v_liq(p_barg)))


# ════════════════════════════════════════
#  물 포화압력 — lazy 초기화 (import 시 CoolProp 호출 제거)
# ════════════════════════════════════════
_WATER_T = None
_WATER_PSAT_LIQ = None

def _init_water_table():
    global _WATER_T, _WATER_PSAT_LIQ
    if _WATER_T is not None:
        return
    _WATER_T = np.linspace(-0.01, 110, 2201)
    _WATER_PSAT_LIQ = np.array([
        PropsSI("P", "T", max(t, 0.01) + 273.15, "Q", 0, "Water") / 1000 for t in _WATER_T])

def _psat_ice(t_c):
    """얼음 위 포화 수증기압 [kPa] (Murphy & Koop 2005)."""
    T = np.asarray(t_c, dtype=float) + 273.15
    ln_p = 9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T
    return np.exp(ln_p) / 1000

def psat_water(t_c):
    """물/얼음 위 포화 수증기압 [kPa]. 0°C 미만은 얼음 포화압."""
    _init_water_table()
    t = np.asarray(t_c, dtype=float)
    p_liq = np.interp(t, _WATER_T, _WATER_PSAT_LIQ)
    p_ice = _psat_ice(t)
    return np.where(t < 0, p_ice, p_liq)


# ════════════════════════════════════════
#  습공기 물성 — 고온 보정
# ════════════════════════════════════════
def abs_humidity(t_c, rh, patm=101.325):
    psat = psat_water(t_c)
    pw = psat * np.asarray(rh, dtype=float)
    pa = patm - pw
    pa = np.where(pa <= 0, 1e-6, pa)
    return 0.62198 * pw / pa

def rh_from_ah(ah, t_c, patm=101.325):
    psat = psat_water(t_c)
    pw = np.asarray(ah) * patm / (0.62198 + np.asarray(ah))
    return np.clip(pw / np.where(psat == 0, 1e-6, psat), 0.0, 1.0)

def h_moist_air(t_c, ah):
    """습공기 엔탈피 [kJ/kg_da] — Cp_v 온도 의존성 반영."""
    t = np.asarray(t_c, dtype=float)
    w = np.asarray(ah, dtype=float)
    cpv = 1.82 + 0.00079 * t  # 수증기 비열 온도 보정 (ASHRAE)
    return 1.006 * t + w * (2501 + cpv * t)

def v_moist_air(t_c, ah, patm=101.325):
    t_k = np.asarray(t_c, dtype=float) + 273.15
    w = np.asarray(ah, dtype=float)
    return (8.3145 / 28.9645 * t_k) * (1 + 1.6078 * w) / (patm * 1000) * 1000


# ════════════════════════════════════════
#  전역 캐시
# ════════════════════════════════════════
_cache: dict[str, FluidProperties] = {}

def get_props(refrigerant, patm=101.325, backend="HEOS"):
    key = f"{backend}::{refrigerant}_{patm}"
    if key not in _cache:
        print(f"  물성 2D 캐시 초기화: {refrigerant} ... ", end="", flush=True)
        _cache[key] = FluidProperties(refrigerant, patm, backend)
        print("완료")
    return _cache[key]
