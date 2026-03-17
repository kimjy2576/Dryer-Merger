"""
config_loader.py — YAML 설정 파일 로드 및 접근 유틸리티
"""
import yaml
import os
import sys


def _get_base_path():
    """PyInstaller 번들 환경에서도 올바른 경로를 반환한다."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "config")
    return os.path.dirname(os.path.abspath(__file__))


DEFAULT_CONFIG_PATH = os.path.join(_get_base_path(), "default_config.yaml")


def load_config(path: str = None) -> dict:
    """YAML 설정 파일을 읽어 딕셔너리로 반환한다."""
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_column_mapping(cfg: dict, source: str) -> dict:
    """source ('ams' | 'blackrose') 에 해당하는 컬럼 매핑 딕셔너리를 반환한다."""
    return cfg.get("column_mappings", {}).get(source, {})


def get_selected_columns(cfg: dict, setup_type: str) -> list:
    """setup_type ('br' | 'ams') 에 해당하는 선택 컬럼 리스트를 반환한다."""
    return cfg.get("selected_columns", {}).get(setup_type.lower(), [])


def get_outlier_thresholds(cfg: dict) -> dict:
    """접두사별 이상치 임계값 딕셔너리를 반환한다.  예: {'Po_': [0, 1000], ...}"""
    return cfg.get("outlier_thresholds", {})
