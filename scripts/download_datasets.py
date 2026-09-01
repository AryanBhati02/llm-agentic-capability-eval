"""
Download raw benchmark datasets (TaskBench and AgentBench) from GitHub.

Uses git sparse checkout to download only the necessary data directories,
avoiding full repository clones.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --taskbench-only
    python scripts/download_datasets.py --agentbench-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

TASKBENCH_REPO = "https://github.com/microsoft/JARVIS.git"
AGENTBENCH_REPO = "https://github.com/THUDM/AgentBench.git"

DATASETS_DIR = Path("datasets")


def _run_cmd(cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a shell command and return True if successful."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(cmd)}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False


def download_taskbench(dest: Path) -> bool:
    """Download TaskBench data from microsoft/JARVIS.

    TaskBench data lives in the taskbench/ subfolder of JARVIS.
    """
    if dest.exists() and any(dest.iterdir()):
        print(f"TaskBench already exists at {dest}. Skipping.")
        return True

    print(f"Downloading TaskBench from {TASKBENCH_REPO}...")
    temp_dir = dest.parent / "_tmp_jarvis"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    success = _run_cmd([
        "git", "clone", "--depth", "1", "--filter=blob:none",
        "--sparse", TASKBENCH_REPO, str(temp_dir),
    ])
    if not success:
        return False

    success = _run_cmd([
        "git", "sparse-checkout", "set", "taskbench",
    ], cwd=temp_dir)
    if not success:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    src = temp_dir / "taskbench"
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    else:
        src_data = temp_dir / "data"
        if src_data.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_data), str(dest))
        else:
            print(f"Could not find taskbench data in {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False

    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"TaskBench downloaded to {dest}")
    return True


def download_agentbench(dest: Path) -> bool:
    """Download AgentBench data from THUDM/AgentBench.

    AgentBench data lives in the data/ subfolder.
    """
    if dest.exists() and any(dest.iterdir()):
        print(f"AgentBench already exists at {dest}. Skipping.")
        return True

    print(f"Downloading AgentBench from {AGENTBENCH_REPO}...")
    temp_dir = dest.parent / "_tmp_agentbench"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    success = _run_cmd([
        "git", "clone", "--depth", "1", "--filter=blob:none",
        "--sparse", AGENTBENCH_REPO, str(temp_dir),
    ])
    if not success:
        return False

    success = _run_cmd([
        "git", "sparse-checkout", "set", "data",
    ], cwd=temp_dir)
    if not success:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    src = temp_dir / "data"
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    else:
        print(f"Could not find data/ in {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return False

    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"AgentBench downloaded to {dest}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download TaskBench and AgentBench raw datasets."
    )
    parser.add_argument(
        "--taskbench-only", action="store_true", help="Download only TaskBench"
    )
    parser.add_argument(
        "--agentbench-only", action="store_true", help="Download only AgentBench"
    )
    args = parser.parse_args()

    tb_dest = DATASETS_DIR / "taskbench"
    ab_dest = DATASETS_DIR / "agentbench"

    success = True

    if not args.agentbench_only:
        tb_ok = download_taskbench(tb_dest)
        if not tb_ok:
            print("WARNING: TaskBench download failed.")
            success = False

    if not args.taskbench_only:
        ab_ok = download_agentbench(ab_dest)
        if not ab_ok:
            print("WARNING: AgentBench download failed.")
            success = False

    if success:
        print("\nAll datasets downloaded successfully.")
        return 0
    else:
        print("\nSome downloads failed. See errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
