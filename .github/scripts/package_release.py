from __future__ import annotations

import argparse
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path


def _chmod_executable(path: Path) -> None:
    current_mode = path.stat().st_mode
    path.chmod(
        current_mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )


def _add_zip_entry(zip_file: zipfile.ZipFile, path: Path, arcname: Path) -> None:
    if path.is_dir():
        zip_file.write(path, str(arcname).replace("\\", "/") + "/")
        return
    zip_file.write(path, str(arcname).replace("\\", "/"))


def _create_zip(source_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        _add_zip_entry(zip_file, source_dir, Path(source_dir.name))
        for path in sorted(source_dir.rglob("*")):
            _add_zip_entry(zip_file, path, Path(source_dir.name) / path.relative_to(source_dir))


def _create_tar_gz(source_dir: Path, archive_path: Path) -> None:
    with tarfile.open(archive_path, "w:gz") as tar_file:
        tar_file.add(source_dir, arcname=source_dir.name)


def _write_github_output(*, asset_name: str, asset_path: Path) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if not output_path:
        return
    with Path(output_path).open("a", encoding="utf-8") as handle:
        handle.write(f"asset_name={asset_name}\n")
        handle.write(f"asset_path={asset_path.resolve()}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Package AgentForMc release assets")
    parser.add_argument("--version", required=True)
    parser.add_argument("--os-name", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--archive", choices=["zip", "tar.gz"], required=True)
    parser.add_argument("--windows", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    executable_name = "AgentForMc.exe" if args.windows else "AgentForMc"
    built_executable = root / "dist" / executable_name
    if not built_executable.is_file():
        raise FileNotFoundError(f"Built executable not found: {built_executable}")

    package_name = f"AgentForMc-{args.version}-{args.os_name}-{args.arch}"
    release_dir = root / "release"
    package_dir = release_dir / package_name
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    packaged_executable = package_dir / executable_name
    shutil.copy2(built_executable, packaged_executable)
    if not args.windows:
        _chmod_executable(packaged_executable)

    shutil.copy2(root / "config.toml", package_dir / "config.toml")
    shutil.copy2(root / ".env.example", package_dir / ".env.example")
    (package_dir / "data").mkdir()

    archive_suffix = ".zip" if args.archive == "zip" else ".tar.gz"
    archive_path = release_dir / f"{package_name}{archive_suffix}"
    if archive_path.exists():
        archive_path.unlink()

    if args.archive == "zip":
        _create_zip(package_dir, archive_path)
    else:
        _create_tar_gz(package_dir, archive_path)

    _write_github_output(asset_name=archive_path.name, asset_path=archive_path)
    print(archive_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
