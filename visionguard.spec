# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import customtkinter

block_cipher = None

# Locate CustomTkinter directory for theme/asset bundling
ctk_dir = Path(customtkinter.__file__).parent

datas = [
    ('config.yaml', '.'),
    ('firmware', 'firmware'),
    ('data/database/.gitkeep', 'data/database'),
    ('data/images/.gitkeep', 'data/images'),
    ('data/logs/.gitkeep', 'data/logs'),
    ('data/models/.gitkeep', 'data/models'),
    (str(ctk_dir), 'customtkinter'),
]

hiddenimports = [
    'PIL._tkinter_finder',
    'customtkinter',
    'serial',
    'serial.tools',
    'serial.tools.list_ports',
    'onnxruntime',
    'insightface',
    'cv2',
    'yaml',
    'joblib',
    'scipy.spatial.transform',
    'scipy.spatial.transform._rotation',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionGuardAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Set to False if no console window is desired
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionGuardAI',
)
