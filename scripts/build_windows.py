"""
build_windows.py
================
Build and packaging script to bundle VisionGuard AI desktop application
into a standalone Windows distribution directory or executable using PyInstaller.

Usage:
    python scripts/build_windows.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT_DIR / "dist"
BUILD_DIR = ROOT_DIR / "build"


def check_prerequisites():
    try:
        import PyInstaller  # noqa: F401
        print("✓ PyInstaller is installed.")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])


def clean_previous_builds():
    print("Cleaning previous build artifacts...")
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)


def build_executable():
    spec_file = ROOT_DIR / "visionguard.spec"
    print(f"Building application with PyInstaller using {spec_file}...")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        str(spec_file),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT_DIR))
    print(f"\n✓ Build successful! Executable and assets located at:\n  {DIST_DIR / 'VisionGuardAI'}")


if __name__ == "__main__":
    check_prerequisites()
    clean_previous_builds()
    build_executable()
