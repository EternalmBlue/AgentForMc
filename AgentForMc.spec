# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata


ROOT = Path(SPECPATH).resolve()

datas = [
    (str(ROOT / "config.toml"), "."),
    (str(ROOT / ".env.example"), "."),
    (
        str(ROOT / "packaging" / "pyinstaller" / "empty_data_marker.txt"),
        "data",
    ),
]
binaries = []
hiddenimports = []


def collect_package(import_name):
    try:
        package_datas, package_binaries, package_hiddenimports = collect_all(import_name)
    except Exception as exc:
        print(f"WARNING: could not collect PyInstaller package data for {import_name}: {exc}")
        return
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)


def collect_metadata(distribution_name):
    try:
        datas.extend(copy_metadata(distribution_name))
    except Exception:
        pass


for package_name in [
    "agent_for_mc",
    "grpc",
    "google.protobuf",
    "lancedb",
    "pyarrow",
    "deepagents",
    "langchain",
    "langchain_core",
    "langchain_deepseek",
    "langchain_openai",
    "langgraph",
    "langsmith",
    "opentelemetry",
]:
    collect_package(package_name)

for distribution_name in [
    "grpcio",
    "protobuf",
    "lancedb",
    "pyarrow",
    "deepagents",
    "langchain",
    "langchain-core",
    "langchain-deepseek",
    "langchain-openai",
    "langgraph",
    "langsmith",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "opentelemetry-exporter-otlp-proto-http",
    "opentelemetry-instrumentation",
    "opentelemetry-instrumentation-requests",
]:
    collect_metadata(distribution_name)

hiddenimports.extend(
    collect_submodules("agent_for_mc")
    + collect_submodules("google.protobuf")
    + collect_submodules("grpc")
)

a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AgentForMc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
