import argparse
import shutil
import subprocess
import sys
import zipapp
from pathlib import Path
from pyzapps import APPS


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PKG = SRC / "pepseqpred"
DIST = ROOT / "dist"
BUILD = ROOT / ".build_pyz"


def _clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
        return out.decode().strip()
    except Exception:
        return "nogit"


def _write_main(build_dir: Path, entry: str, argv0: str) -> None:
    mod, func = entry.split(":")
    (build_dir / "__main__.py").write_text(
        "import sys\n"
        f"sys.argv[0] = '{argv0}'\n"
        f"from {mod} import {func} as _main\n"
        "_main()\n"
    )


def build_one(name: str, out_dir: Path, interpreter: str, latest: bool) -> Path:
    if name.endswith(".pyz"):
        name = name[:-4]

    if name not in APPS:
        raise SystemExit(f"Unknown target '{name}'. Use --list to options.")

    if not PKG.exists():
        raise SystemExit(
            "Expected src/pepseqpred. Are you running from repo root?")

    _clean_dir(BUILD)
    BUILD.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    shutil.copytree(PKG, BUILD / "pepseqpred", dirs_exist_ok=True)
    _write_main(BUILD, APPS[name], argv0=f"{name}.pyz")

    rev = _git_rev()
    out_path = out_dir / f"{name}_{rev}.pyz"

    zipapp.create_archive(
        BUILD,
        target=out_path,
        interpreter=interpreter,
        compressed=True
    )

    if latest:
        shutil.copy2(out_path, out_dir / f"{name}_latest.pyz")

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Builds .pyz Python scripts for easy HPC usage.")
    parser.add_argument("target",
                        nargs="?",
                        default="all",
                        help="App name, app.pyz or 'all'")
    parser.add_argument("--list",
                        action="store_true",
                        help="Lists available targets to build")
    parser.add_argument("--out-dir",
                        default=str(DIST),
                        help="Output directory for target")
    parser.add_argument("--interpreter",
                        default="/usr/bin/env python3",
                        help="Path to Python interpreter")
    parser.add_argument("--no-latest",
                        action="store_true",
                        help="Do not write *_latest.pyz copies")
    args = parser.parse_args()

    if args.list:
        for k in sorted(APPS):
            print(k)
        return

    out_dir = Path(args.out_dir)

    if args.target == "all":
        for k in sorted(APPS):
            path = build_one(
                k, out_dir=out_dir, interpreter=args.interpreter, latest=(not args.no_latest))
            print(path)
        return

    path = build_one(args.target, out_dir=out_dir,
                     interpreter=args.interpreter, latest=(not args.no_latest))
    print(path)

    _clean_dir(BUILD)


if __name__ == "__main__":
    main()
