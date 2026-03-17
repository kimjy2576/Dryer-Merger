# -*- mode: python ; coding: utf-8 -*-
"""
DryerMerger.spec — PyInstaller 빌드 스펙

빌드: pyinstaller DryerMerger.spec
결과: dist/DryerMerger/ 폴더 (DryerMerger.exe 포함)
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# CoolProp 데이터 파일
coolprop_datas = collect_data_files('CoolProp')

a = Analysis(
    ['run.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('config/default_config.yaml', 'config'),
        ('static/index.html', 'static'),
    ] + coolprop_datas,
    hiddenimports=[
        'server',
        'config',
        'properties',
        'calculator',
        'performance',
        'preprocessor',
        'postprocessor',
        'pipeline',
        'io_handler',
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'CoolProp',
        'CoolProp.CoolProp',
        'scipy.interpolate',
        'scipy.ndimage',
        'sklearn.ensemble',
        'yaml',
        'pandas',
        'numpy',
        'openpyxl',
        'dateutil',
        'dateutil.parser',
    ] + collect_submodules('CoolProp')
      + collect_submodules('uvicorn'),
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'test', 'unittest', 'matplotlib'],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DryerMerger',
    debug=False,
    strip=False,
    upx=True,
    console=True,       # 콘솔 표시 (로그 확인용, 배포 시 False)
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name='DryerMerger',
)
