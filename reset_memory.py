"""
reset_memory.py: wipe the LIDA agent's long-term memory so it starts fresh.

Deletes everything the agent has learned across runs:
  vector store      LIDA_STORE_PATH         episodic, declarative and procedural memories
  scheme stats      LIDA_SCHEME_STATS_PATH  habit reliability counts
  mem0 history      LIDA_HISTORY_DB_PATH    mem0's log of memory changes
  live goals        LIDA_LIVE_GOALS_PATH    goals she revised (she goes back to goals.txt)
  goal change log   LIDA_GOALS_LOG_PATH     every revision she made
  intention         LIDA_INTENTION_PATH     what she was working toward
  mem0 files        MEM0_DIR                mem0's own state (telemetry id, migrations store)

Her workspace folder (LIDA_FILES_DIR) holds her files, not her memory, so it is
kept unless you pass --files (or say yes when asked).

Paths come from .env the same way lida-llm.py reads them; relative paths are
resolved against the current directory, so run this from where you run the agent.

Usage:
  poetry run python reset_memory.py          # lists what will go, asks first
  poetry run python reset_memory.py --yes    # no prompt (keeps her files)
  poetry run python reset_memory.py --yes --files   # also empties her workspace folder
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import portalocker
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

# keep these defaults in sync with lida-llm.py
STORE_PATH = Path(os.getenv("LIDA_STORE_PATH", "./lida_qdrant"))
SCHEME_STATS_PATH = Path(os.getenv("LIDA_SCHEME_STATS_PATH", "scheme_stats.json"))
HISTORY_DB_PATH = Path(os.getenv("LIDA_HISTORY_DB_PATH", "./lida_history.db"))
LIVE_GOALS_PATH = Path(os.getenv("LIDA_LIVE_GOALS_PATH", "./lida_goals.json"))
GOALS_LOG_PATH = Path(os.getenv("LIDA_GOALS_LOG_PATH", "./lida_goals_log.jsonl"))
INTENTION_PATH = Path(os.getenv("LIDA_INTENTION_PATH", "./lida_intention.json"))
MEM0_DIR = Path(os.getenv("MEM0_DIR", "./lida_mem0"))
FILES_DIR = Path(os.getenv("LIDA_FILES_DIR", "./workspace"))


def store_in_use(store: Path) -> bool:
    """Qdrant local mode holds an exclusive lock on <store>/.lock while open."""
    lock = store / ".lock"
    if not lock.exists():
        return False
    with open(lock, "r+") as f:
        try:
            portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.LockException:
            return True
        portalocker.unlock(f)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Clear the LIDA agent's memory.")
    ap.add_argument("-y", "--yes", action="store_true", help="don't ask for confirmation")
    ap.add_argument("--files", action="store_true",
                    help="also delete everything in her workspace folder")
    args = ap.parse_args()

    targets = [p for p in (STORE_PATH, SCHEME_STATS_PATH, HISTORY_DB_PATH,
                           LIVE_GOALS_PATH, GOALS_LOG_PATH, INTENTION_PATH, MEM0_DIR) if p.exists()]
    her_files = [p for p in FILES_DIR.rglob("*") if p.is_file()] if FILES_DIR.is_dir() else []
    if not targets and not (args.files and her_files):
        print("Nothing to clear." + (f" (Her workspace folder has {len(her_files)} file(s); "
                                     "pass --files to clear it.)" if her_files else ""))
        return 0

    if MEM0_DIR.resolve() == (Path.home() / ".mem0").resolve():
        print(f"MEM0_DIR is the shared {MEM0_DIR}; not deleting it. "
              "Point MEM0_DIR at a project folder first.", file=sys.stderr)
        return 1

    # never empty a folder that contains this project or the home folder
    files_dir = FILES_DIR.resolve()
    protected = (Path(__file__).resolve().parent, Path.cwd().resolve(), Path.home().resolve())
    if (args.files or her_files) and any(files_dir == p or files_dir in p.parents for p in protected):
        print(f"LIDA_FILES_DIR is {files_dir}, which contains this project or your home "
              "folder; not touching it. Point it at a dedicated folder first.", file=sys.stderr)
        return 1

    if STORE_PATH.exists() and store_in_use(STORE_PATH):
        print(f"{STORE_PATH} is in use. Stop the agent first.", file=sys.stderr)
        return 1

    print("This will permanently delete:")
    for p in targets:
        print(f"  {p.resolve()}")
    if args.files and her_files:
        print(f"  {files_dir}/* ({len(her_files)} file(s) of hers)")
    if not args.yes and input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Aborted.")
        return 1

    clear_files = args.files
    if her_files and not args.files and not args.yes:
        clear_files = input(f"Also empty her workspace folder {FILES_DIR.resolve()} "
                            f"({len(her_files)} file(s))? [y/N] ").strip().lower() in ("y", "yes")

    for p in targets:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    print("Memory cleared." if targets else "Memory was already empty.")
    if clear_files and FILES_DIR.is_dir():
        for p in FILES_DIR.iterdir():
            shutil.rmtree(p) if p.is_dir() and not p.is_symlink() else p.unlink()
        print(f"Workspace folder emptied ({len(her_files)} file(s)).")
    elif her_files:
        print(f"Kept her workspace folder ({len(her_files)} file(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
