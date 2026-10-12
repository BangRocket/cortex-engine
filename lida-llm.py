"""
lida_llm.py: a LIDA-style cognitive cycle with an LLM for perception and
deliberation, and mem0 as long-term memory. Embeddings are fully local.

Design rules
  * The cycle is cheap and runs constantly. The LLM is called only to perceive
    new input and to deliberate when no learned scheme fits.
  * The LLM's context is the global workspace: only what wins the attention
    competition reaches deliberation.
  * Learning is gated by the broadcast: only conscious content gets written.
  * Whether an action worked is judged by the mind itself (an appraisal LLM
    call), not by a person. With no environment to confirm outcomes, it
    judges the choice: specific, on-goal, not a repeat.
  * Deliberated actions that work get compiled into procedural schemes, so the
    LLM gets called less over time (effortful -> automatic).

Embeddings
  One local sentence-transformers model, loaded once by mem0 and shared with the
  attention codelets. Default is BAAI/bge-small-en-v1.5 (384d, ~33M params,
  ~20ms per sentence on CPU), roughly level with OpenAI text-embedding-3-small
  on MTEB averages. Switch to BAAI/bge-base-en-v1.5 (768d) for a bit more
  quality at ~2.5x the cost.

  BGE cosine scores sit in a compressed band (unrelated text often scores 0.5+),
  so every similarity goes through Calibration.relevance() onto a 0..1 scale
  before codelets, cueing or scheme confidence use it.

mem0 layout (one Memory instance, partitioned by agent_id)
  episodic     raw broadcast records, infer=False (no LLM, every broadcast)
  declarative  facts mem0 extracts from batches of episodes, infer=True
               (periodic consolidation, LIDA's transient-episodic -> declarative)
  procedural   schemes: context text -> action in metadata, infer=False.
               Reliability is tracked in a local sidecar JSON, because mem0
               updates memory text, not arbitrary counters.

Requires: poetry install   (run with: poetry run python lida-llm.py)
Needs ANTHROPIC_API_KEY (perception, deliberation, mem0 consolidation).
Config is read from .env next to this file; see .env.example.
No OpenAI key needed. Tested against mem0ai 2.2.x and transformers 5.x.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import queue
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from dotenv import load_dotenv

# before importing mem0, so its own env vars (e.g. MEM0_TELEMETRY) apply
load_dotenv(Path(__file__).with_name(".env"))
# mem0 keeps its own files (telemetry id, an internal migrations store) in
# MEM0_DIR, ~/.mem0 by default. Keep them in the project so reset_memory.py
# can clear them and nothing is shared with other mem0 users on this machine.
os.environ.setdefault("MEM0_DIR", "./lida_mem0")

import numpy as np  # noqa: E402
from mem0 import Memory  # noqa: E402

# mem0's optional extras (fastembed BM25, spaCy) are left out on purpose: with
# them, search scores blend in keyword matches and stop being the raw cosine
# that Calibration expects. Hide the "not installed" warnings about them.
for _name in ("mem0.vector_stores.qdrant", "mem0.utils.spacy_models"):
    logging.getLogger(_name).setLevel(logging.ERROR)


# --------------------------------------------------------------------------
# Config (.env / environment, with defaults)
# --------------------------------------------------------------------------
LLM_MODEL = os.getenv("LIDA_LLM_MODEL", "claude-sonnet-5-5")
MEM0_LLM_MODEL = os.getenv("LIDA_MEM0_LLM_MODEL", LLM_MODEL)
# a ceiling, not a target: thinking tokens count against it, so keep headroom
LLM_MAX_TOKENS = int(os.getenv("LIDA_LLM_MAX_TOKENS", "16000"))
# thinking depth for perception/deliberation: low|medium|high|xhigh|max (empty = model default)
LLM_EFFORT = os.getenv("LIDA_LLM_EFFORT", "low") or None
# server-side refusal fallback ("default" lets the API pick); empty disables, e.g.
# for models or gateways that don't support it
LLM_FALLBACKS = os.getenv("LIDA_LLM_FALLBACKS", "default") or None
# custom endpoint (proxy, gateway, ...); empty or unset means the default API.
# An empty value is removed outright: the SDK also reads this env var itself
# and would treat "" as the URL.
ANTHROPIC_BASE_URL = os.environ.pop("ANTHROPIC_BASE_URL", "").strip() or None
if ANTHROPIC_BASE_URL:
    os.environ["ANTHROPIC_BASE_URL"] = ANTHROPIC_BASE_URL
EMBED_MODEL = os.getenv("LIDA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
STORE_PATH = os.getenv("LIDA_STORE_PATH", "./lida_qdrant")
SCHEME_STATS_PATH = os.getenv("LIDA_SCHEME_STATS_PATH", "scheme_stats.json")
# mem0's change log; kept here instead of the shared ~/.mem0/history.db
HISTORY_DB_PATH = os.getenv("LIDA_HISTORY_DB_PATH", "./lida_history.db")
USER_ID = os.getenv("LIDA_USER_ID", "demo-user")
# who the agent is, what it's first given, and what its attention is drawn to;
# relative paths are resolved against this file's directory
PERSONALITY_FILE = os.getenv("LIDA_PERSONALITY_FILE", "personalities/clara.txt")
PROMPT_FILE = os.getenv("LIDA_PROMPT_FILE", "prompt.txt")
GOALS_FILE = os.getenv("LIDA_GOALS_FILE", "goals.txt")
# lasting traits: shown wherever goals are, but fixed (she can't revise them) and
# with no attention codelets, so goals can stay concrete things she finishes
VALUES_FILE = os.getenv("LIDA_VALUES_FILE", "values.txt")
# she can revise her own goals; the live list and a log of every change are kept
# here (goals.txt stays the original seed, used until the first change)
SELF_GOALS = os.getenv("LIDA_SELF_GOALS", "1").strip().lower() not in ("", "0", "false", "no", "off")
LIVE_GOALS_PATH = os.getenv("LIDA_LIVE_GOALS_PATH", "./lida_goals.json")
GOALS_LOG_PATH = os.getenv("LIDA_GOALS_LOG_PATH", "./lida_goals_log.jsonl")
# what she's working toward across actions; survives restarts
INTENTION_PATH = os.getenv("LIDA_INTENTION_PATH", "./lida_intention.json")
MAX_GOALS = int(os.getenv("LIDA_MAX_GOALS", "7"))
# minutes between goal checks: the next deliberation after this weighs her goals
# against what she's actually been doing (0 = only when wandering brings it up)
GOAL_REVIEW_MINUTES = float(os.getenv("LIDA_GOAL_REVIEW_MINUTES", "30"))
BROADCAST_THRESHOLD = float(os.getenv("LIDA_BROADCAST_THRESHOLD", "0.25"))
AUTOMATIC_THRESHOLD = float(os.getenv("LIDA_AUTOMATIC_THRESHOLD", "0.55"))
REFRACTORY_TAU = float(os.getenv("LIDA_REFRACTORY_TAU", "3.0"))
CONSOLIDATE_EVERY = int(os.getenv("LIDA_CONSOLIDATE_EVERY", "10"))
# raw cosine at which two workspace items count as the same thought and merge.
# Measured on bge-small: paraphrases 0.81-0.94, related-but-different 0.79-0.83
DEDUPE_SIM = float(os.getenv("LIDA_DEDUPE_SIM", "0.88"))
# memories recalled by what just reached consciousness (chains of association)
BROADCAST_CUE_K = int(os.getenv("LIDA_BROADCAST_CUE_K", "3"))
# raw cosine at which a new memory reuses an existing one instead of adding a row.
# Measured on bge-small: reworded facts 0.94-0.99, updated facts ("crashes" ->
# "no longer crashes") 0.91-0.94, different facts <= 0.79; near-identical
# episodes ~0.985, different ones ~0.77. Merges keep the newer wording.
SAME_EPISODE_SIM = float(os.getenv("LIDA_SAME_EPISODE_SIM", "0.95"))
SAME_FACT_SIM = float(os.getenv("LIDA_SAME_FACT_SIM", "0.90"))
SAME_HABIT_SIM = float(os.getenv("LIDA_SAME_HABIT_SIM", "0.90"))
# minimum seconds between actions; each action costs LLM calls, so this paces spend
ACTION_INTERVAL = float(os.getenv("LIDA_ACTION_INTERVAL", "30"))
# seconds with nothing reaching consciousness before the mind wanders to a goal
IDLE_SECONDS = float(os.getenv("LIDA_IDLE_SECONDS", "10"))
# web access for carrying out actions (Anthropic-hosted web_search + web_fetch)
WEB_ENABLED = os.getenv("LIDA_WEB", "1").strip().lower() not in ("", "0", "false", "no", "off")
WEB_MAX_USES = int(os.getenv("LIDA_WEB_MAX_USES", "3"))  # per tool, per action
WEB_MAX_CONTENT_TOKENS = int(os.getenv("LIDA_WEB_MAX_CONTENT_TOKENS", "8000"))  # per fetched page
# her own folder: the only files she can list, read, write, edit or delete
FILES_ENABLED = os.getenv("LIDA_FILES", "1").strip().lower() not in ("", "0", "false", "no", "off")
FILES_DIR = os.getenv("LIDA_FILES_DIR", "./workspace")
FILES_MAX_KB = int(os.getenv("LIDA_FILES_MAX_KB", "200"))            # per file
FILES_MAX_TOTAL_MB = int(os.getenv("LIDA_FILES_MAX_TOTAL_MB", "50"))  # whole folder
# running Python she wrote, jailed to her workspace folder (needs bubblewrap)
EXEC_ENABLED = os.getenv("LIDA_EXEC", "1").strip().lower() not in ("", "0", "false", "no", "off")
EXEC_TIMEOUT = float(os.getenv("LIDA_EXEC_TIMEOUT", "30"))        # seconds, wall clock
EXEC_MEMORY_MB = int(os.getenv("LIDA_EXEC_MEMORY_MB", "512"))
EXEC_MAX_PROCS = int(os.getenv("LIDA_EXEC_MAX_PROCS", "32"))
EXEC_NETWORK = os.getenv("LIDA_EXEC_NETWORK", "0").strip().lower() not in ("", "0", "false", "no", "off")
# she can ask the person running her a question; lines typed into the terminal
# reach her as input (answers to her question, or anything else)
ASK_ENABLED = os.getenv("LIDA_ASK", "1").strip().lower() not in ("", "0", "false", "no", "off")
USER_NAME = os.getenv("LIDA_USER_NAME", "Joshua").strip() or "Joshua"
# an unanswered question blocks new ones for this long, so she can't pile them up
ASK_EXPIRE_SECONDS = float(os.getenv("LIDA_ASK_EXPIRE_SECONDS", "1800"))


# who judges whether an action worked: "llm" (Claude, the default) or "jev"
# (TypeSafe's Jev decision model; falls back to Claude if Jev is unavailable)
DECIDER = os.getenv("LIDA_DECIDER", "llm").strip().lower()
JEV_API_KEY = os.getenv("JEV_API_KEY", "")
JEV_URL = os.getenv("JEV_URL", "https://api.typesafe.ai/v1/systemone")
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_TIMEOUT = float(os.getenv("JEV_TIMEOUT", "10"))
# Jev's probability that the action worked must reach this to count as a pass
JEV_WORKED_THRESHOLD = float(os.getenv("JEV_WORKED_THRESHOLD", "0.5"))

# print what the mind is doing step by step (0 = only actions, broadcasts, errors)
VERBOSE = os.getenv("LIDA_VERBOSE", "1").strip().lower() not in ("", "0", "false", "no", "off")
# seconds of silence before the heartbeat reports what's still running
HEARTBEAT_SECONDS = float(os.getenv("LIDA_HEARTBEAT_SECONDS", "5"))
# color log lines by what produced them: auto (only on a terminal, and not when
# NO_COLOR is set), 1 = always, 0 = never
_COLOR = os.getenv("LIDA_COLOR", "auto").strip().lower()
COLOR = (sys.stdout.isatty() and not os.getenv("NO_COLOR")) if _COLOR in ("", "auto") \
    else _COLOR not in ("0", "false", "no", "off")


# --------------------------------------------------------------------------
# Status output: timestamped phases + a heartbeat, so slow is never mistaken
# for hung
# --------------------------------------------------------------------------
# ANSI color per kind of line: who or what produced it
LOG_COLORS = {
    "llm": "94",      # bright blue: Claude calls
    "jev": "33",      # yellow: Jev calls
    "mem": "32",      # green: long-term memory, embeddings
    "lida": "36",     # cyan: the cognitive cycle (broadcasts, wandering, intention, goals)
    "act": "1;97",    # bold white: actions and what came back
    "talk": "1;95",   # bold magenta: questions to and messages from the person
    "error": "1;31",  # bold red: anything that went wrong
    "sys": "90",      # grey: startup, heartbeat, idling
}
LOG_LEGEND = [("llm", "Claude"), ("jev", "Jev"), ("mem", "memory"), ("lida", "cycle"),
              ("act", "actions"), ("talk", "conversation"), ("error", "errors"), ("sys", "system")]


class Status:
    """Phases (embedding, recall, LLM calls...) print when they start, and
    again with their duration if they took a while. A background heartbeat
    reports whatever is still running after a few quiet seconds, and says so
    outright if the main loop itself has stopped ticking. Each line has a kind
    (see LOG_COLORS) that sets its color; lines starting with "!!" are errors."""

    def __init__(self, verbose: bool = VERBOSE, heartbeat: float = HEARTBEAT_SECONDS,
                 color: bool = COLOR):
        self.verbose, self.heartbeat, self.color = verbose, heartbeat, color
        self.idle_info: Callable[[], str] | None = None
        self._stack: list[tuple[str, float, str]] = []
        self._lock = threading.Lock()
        self._last_out = self._last_tick = time.time()
        self._thread: threading.Thread | None = None

    def paint(self, text: str, kind: str) -> str:
        code = LOG_COLORS.get(kind)
        if not (self.color and code):
            return text
        # color each line separately so a wrapped or interrupted line keeps it
        return "\n".join(f"\033[{code}m{line}\033[0m" for line in text.split("\n"))

    def say(self, msg: str, kind: str = "sys") -> None:
        if msg.lstrip().startswith("!!"):
            kind = "error"
        with self._lock:
            print(f"{self.paint(time.strftime('%H:%M:%S'), 'sys')} {self.paint(msg, kind)}",
                  flush=True)
            self._last_out = time.time()

    def legend(self) -> None:
        if self.color:
            self.say("Colors: " + "  ".join(self.paint(label, k) for k, label in LOG_LEGEND))

    @contextmanager
    def phase(self, name: str, min_report: float = 1.0, kind: str = "sys"):
        t = time.time()
        if self.verbose:
            self.say(f"  · {name}…", kind)
        self._stack.append((name, t, kind))
        try:
            yield
        finally:
            self._stack.pop()
            took = time.time() - t
            if self.verbose and took >= min_report:
                self.say(f"  · {name} done ({took:.1f}s)", kind)

    def tick(self) -> None:
        self._last_tick = time.time()

    def start_heartbeat(self) -> None:
        if self.verbose and self.heartbeat > 0 and self._thread is None:
            self._thread = threading.Thread(target=self._beat, daemon=True)
            self._thread.start()

    def _beat(self) -> None:
        while True:
            time.sleep(1.0)
            now, stack = time.time(), list(self._stack)
            # report running work every `heartbeat` s, plain idling less often
            if now - self._last_out < self.heartbeat * (1 if stack else 3):
                continue
            if stack:
                path = " › ".join(name for name, _, _ in stack)
                self.say(f"  … still {path} ({now - stack[-1][1]:.0f}s)", stack[-1][2])
            elif now - self._last_tick > self.heartbeat:
                self.say(f"  !! main loop hasn't ticked for {now - self._last_tick:.0f}s; may be hung")
            else:
                info = self.idle_info() if self.idle_info else ""
                self.say(f"  … idle{', ' + info if info else ''}")


STATUS = Status()


# --------------------------------------------------------------------------
# Local embeddings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Maps raw cosine similarity onto 0..1 relevance.
    floor ~= 90th percentile of unrelated pairs, ceil ~= a strong match."""
    floor: float
    ceil: float

    def relevance(self, sim: float) -> float:
        return min(1.0, max(0.0, (sim - self.floor) / (self.ceil - self.floor)))


# dims + calibration measured on a small agent-memory retrieval set.
# Re-measure if you change models or domains.
KNOWN_MODELS: dict[str, tuple[int, Calibration]] = {
    "BAAI/bge-small-en-v1.5": (384, Calibration(floor=0.58, ceil=0.80)),
    "BAAI/bge-base-en-v1.5": (768, Calibration(floor=0.55, ceil=0.78)),
}


class LocalEmbedder:
    """Wraps the SentenceTransformer that mem0 already loaded, so the model
    lives in memory once. Caches by text since workspace content repeats."""

    def __init__(self, st_model, calib: Calibration, cache_size: int = 4096):
        self.model, self.calib, self.cache_size = st_model, calib, cache_size
        self._cache: dict[str, np.ndarray] = {}

    def many(self, texts: list[str]) -> list[np.ndarray]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            with STATUS.phase(f"embedding {len(missing)} text(s)", kind="mem"):
                vecs = self.model.encode(missing, normalize_embeddings=True,
                                         convert_to_numpy=True)
            if len(self._cache) + len(missing) > self.cache_size:
                self._cache.clear()
            self._cache.update(zip(missing, vecs))
        return [self._cache[t] for t in texts]

    def one(self, text: str) -> np.ndarray:
        return self.many([text])[0]

    def relevance(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.calib.relevance(float(np.dot(a, b)))


def _probe_dims(model_name: str) -> int:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name, device="cpu")
    fn = getattr(st, "get_embedding_dimension", None) or st.get_sentence_embedding_dimension
    return fn()


def build_memory(embed_model: str = EMBED_MODEL, store_path: str = STORE_PATH,
                 llm_model: str = MEM0_LLM_MODEL,
                 base_url: str | None = ANTHROPIC_BASE_URL) -> tuple[Memory, LocalEmbedder]:
    """mem0 with a local HF embedder, local Qdrant, and Claude for consolidation."""
    dims, calib = KNOWN_MODELS.get(embed_model, (None, Calibration(0.55, 0.80)))
    dims = dims or _probe_dims(embed_model)
    slug = re.sub(r"[^a-z0-9]+", "_", embed_model.lower()).strip("_")
    config = {
        "embedder": {"provider": "huggingface", "config": {
            "model": embed_model, "embedding_dims": dims,
            "model_kwargs": {"device": "cpu"}}},
        # collection name carries model + dims so switching models never
        # collides with an old collection of a different size
        "vector_store": {"provider": "qdrant", "config": {
            "collection_name": f"lida_{slug}_{dims}",
            "embedding_model_dims": dims, "path": store_path}},
        "history_db_path": HISTORY_DB_PATH,
        "llm": {"provider": "anthropic", "config": {
            "model": llm_model, "anthropic_base_url": base_url,
            "max_tokens": LLM_MAX_TOKENS}},
    }
    with STATUS.phase(f"loading {embed_model} and memory store", kind="mem"):
        mem = Memory.from_config(config)
    return mem, LocalEmbedder(mem.embedding_model.model, calib)


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------
class Jev:
    """Minimal client for Jev's System One endpoint: state + typed questions in,
    calibrated answers out (no text generation). Retries 429/529 with backoff,
    as the docs ask; anything else raises so the caller can fall back."""

    def __init__(self, api_key: str = JEV_API_KEY, url: str = JEV_URL,
                 model: str = JEV_MODEL, timeout: float = JEV_TIMEOUT, retries: int = 2):
        if not api_key:
            raise ValueError("JEV_API_KEY is not set")
        self.api_key, self.url, self.model = api_key.strip(), url, model
        self.timeout, self.retries = timeout, retries

    def ask(self, state, questions: dict) -> dict:
        body = json.dumps({"model": self.model, "state": state,
                           "questions": questions}).encode()
        # Jev sits behind Cloudflare, which 403s urllib's default User-Agent
        # ("error code: 1010") before the request ever reaches the API.
        req = urllib.request.Request(self.url, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
            "User-Agent": "cortex-engine/0.1"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["answers"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 529) and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                detail = e.read().decode(errors="replace")[:300]
                raise RuntimeError(f"Jev HTTP {e.code}: {detail}") from None
        raise RuntimeError("Jev: retries exhausted")


# --------------------------------------------------------------------------
# Her workspace folder: files she keeps across runs
# --------------------------------------------------------------------------
class Files:
    """Text files under one root, and nothing outside it. Every path is
    resolved (symlinks included) and must land inside the root, so "..",
    absolute paths and links pointing out are refused. No code execution."""

    def __init__(self, root: str = FILES_DIR, max_kb: int = FILES_MAX_KB,
                 max_total_mb: int = FILES_MAX_TOTAL_MB):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes, self.max_total = max_kb * 1024, max_total_mb * 1024 * 1024

    def _path(self, rel: str) -> Path:
        p = (self.root / str(rel).strip()).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"{rel!r} is outside your workspace folder")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix() or "."

    def _files(self) -> list[Path]:
        # links pointing outside the folder are invisible, not just unreadable
        return sorted(p for p in self.root.rglob("*")
                      if p.is_file() and self.root in p.resolve().parents)

    def overview(self, limit: int = 30) -> str:
        """A short listing for prompts, so she knows what she has."""
        files = self._files()
        if not files:
            return "empty"
        lines = [f"{self._rel(p)} ({p.stat().st_size / 1024:.1f} KB)" for p in files[:limit]]
        if len(files) > limit:
            lines.append(f"... and {len(files) - limit} more")
        return "\n".join(lines)

    def list_files(self, path: str = ".") -> str:
        base = self._path(path)
        if not base.is_dir():
            raise FileNotFoundError(f"no folder {path!r}")
        files = [p for p in self._files() if base == p.parent or base in p.parents]
        return "\n".join(f"{self._rel(p)}  {p.stat().st_size} bytes" for p in files[:200]) or "empty"

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 400) -> str:
        p = self._path(path)
        if not p.is_file():
            raise FileNotFoundError(f"no file {path!r}")
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            raise ValueError(f"{path!r} is not a text file") from None
        start = max(1, int(start_line))
        chunk = lines[start - 1:start - 1 + int(max_lines)]
        out = "\n".join(chunk)
        end = start - 1 + len(chunk)
        if end < len(lines):
            out += f"\n[lines {start}-{end} of {len(lines)}; read on with start_line={end + 1}]"
        return out or "(empty file)"

    def _check_size(self, p: Path, new_size: int) -> None:
        if new_size > self.max_bytes:
            raise ValueError(f"file would be {new_size / 1024:.1f} KB; the limit is "
                             f"{self.max_bytes / 1024:.0f} KB. Split it into smaller files")
        total = sum(f.stat().st_size for f in self._files() if f != p) + new_size
        if total > self.max_total:
            raise ValueError(f"workspace folder would exceed {self.max_total // (1024 * 1024)} MB")

    def write_file(self, path: str, content: str, append: bool = False) -> str:
        p = self._path(path)
        if p == self.root or p.is_dir():
            raise IsADirectoryError(f"{path!r} is a folder")
        data = (p.read_text(encoding="utf-8") if append and p.exists() else "") + content
        self._check_size(p, len(data.encode("utf-8")))
        existed = p.exists()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding="utf-8")
        verb = "appended to" if append and existed else "overwrote" if existed else "created"
        return f"{verb} {self._rel(p)} ({len(data.encode('utf-8'))} bytes)"

    def edit_file(self, path: str, old_text: str, new_text: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise FileNotFoundError(f"no file {path!r}")
        text = p.read_text(encoding="utf-8")
        n = text.count(old_text) if old_text else 0
        if n != 1:
            raise ValueError(f"old_text must appear exactly once; found it {n} times")
        data = text.replace(old_text, new_text, 1)
        self._check_size(p, len(data.encode("utf-8")))
        p.write_text(data, encoding="utf-8")
        return f"edited {self._rel(p)}"

    def delete_file(self, path: str) -> str:
        p = self._path(path)
        if p == self.root:
            raise PermissionError("can't delete the workspace folder itself")
        if p.is_dir():
            if any(p.iterdir()):
                raise OSError(f"folder {path!r} isn't empty; delete its files first")
            p.rmdir()
        elif p.is_file():
            p.unlink()
        else:
            raise FileNotFoundError(f"no file {path!r}")
        return f"deleted {self._rel(p)}"

    def handlers(self) -> dict[str, Callable[..., str]]:
        return {t["name"]: getattr(self, t["name"]) for t in FILE_TOOLS}


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"name": name, "description": description,
            "input_schema": {"type": "object", "properties": props, "required": required}}


_PATH = {"type": "string", "description": "Path inside your workspace folder, e.g. notes/gwt.md"}
FILE_TOOLS = [
    _tool("list_files", "List the files in your workspace folder (or a subfolder), with sizes.",
          {"path": {**_PATH, "description": "Subfolder to list; omit for everything."}}, []),
    _tool("read_file", "Read a text file from your workspace folder. Long files come back "
          "in chunks of lines; the result says where to read on from.",
          {"path": _PATH, "start_line": {"type": "integer", "description": "1-based; default 1."},
           "max_lines": {"type": "integer", "description": "Default 400."}}, ["path"]),
    _tool("write_file", "Create or overwrite a text file in your workspace folder (folders are "
          f"created as needed), or append to it. Max {FILES_MAX_KB} KB per file.",
          {"path": _PATH, "content": {"type": "string"},
           "append": {"type": "boolean", "description": "Add to the end instead of overwriting."}},
          ["path", "content"]),
    _tool("edit_file", "Replace one exact passage in a file. old_text must appear exactly once; "
          "include enough surrounding text to make it unique.",
          {"path": _PATH, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
          ["path", "old_text", "new_text"]),
    _tool("delete_file", "Delete a file, or an empty folder, from your workspace folder.",
          {"path": _PATH}, ["path"]),
]


class Sandbox:
    """Runs Python she wrote, in a bubblewrap jail that sees only:
      /workspace  her folder (read-write; the working directory and HOME)
      /usr        the system's Python 3 and its standard library (read-only)
      /tmp        private and empty
    Not the project, .env, your home, or any environment variables. No network
    unless LIDA_EXEC_NETWORK=1. When systemd's user manager is available the run
    also gets a cgroup with hard caps on processes and memory (so a fork bomb or
    a runaway list stays inside it); either way CPU time, file size and wall
    clock are limited. Refuses to run at all if bubblewrap is missing: there is
    no unsandboxed fallback."""

    PYTHON = "/usr/bin/python3"

    def __init__(self, files: Files, timeout: float = EXEC_TIMEOUT,
                 memory_mb: int = EXEC_MEMORY_MB, max_procs: int = EXEC_MAX_PROCS,
                 network: bool = EXEC_NETWORK, output_chars: int = 6000):
        self.bwrap = shutil.which("bwrap")
        if not self.bwrap:
            raise RuntimeError("bubblewrap (bwrap) is not installed; install it to let her run code")
        if not Path(self.PYTHON).exists():
            raise RuntimeError(f"{self.PYTHON} not found")
        self.files, self.timeout, self.network = files, timeout, network
        self.memory_mb, self.max_procs, self.output_chars = memory_mb, max_procs, output_chars
        self.cgroup = bool(shutil.which("systemd-run")) and subprocess.run(
            ["systemd-run", "--user", "--scope", "-q", "-p", "TasksMax=4", "true"],
            capture_output=True, timeout=10).returncode == 0

    def _command(self, argv: list[str]) -> list[str]:
        cmd = [self.bwrap, "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--clearenv",
               "--ro-bind", "/usr", "/usr",
               "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
               "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
               "--ro-bind-try", "/etc/python3", "/etc/python3",
               "--ro-bind-try", f"/etc/python3.{sys.version_info.minor}", f"/etc/python3.{sys.version_info.minor}",
               "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
               "--ro-bind-try", "/etc/localtime", "/etc/localtime",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
               "--bind", str(self.files.root), "/workspace", "--chdir", "/workspace",
               "--setenv", "HOME", "/workspace", "--setenv", "PATH", "/usr/bin:/bin",
               "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
               "--setenv", "PYTHONUNBUFFERED", "1"]
        if self.network:
            cmd[1:2] = ["--unshare-all", "--share-net"]
            cmd += ["--ro-bind-try", "/etc/resolv.conf", "/etc/resolv.conf",
                    "--ro-bind-try", "/etc/hosts", "/etc/hosts",
                    "--ro-bind-try", "/etc/ssl", "/etc/ssl"]
        cmd += ["--", self.PYTHON, "-I", *argv]
        if self.cgroup:
            cmd = ["systemd-run", "--user", "--scope", "-q",
                   "-p", f"TasksMax={self.max_procs}", "-p", f"MemoryMax={self.memory_mb}M",
                   "-p", "MemorySwapMax=0", "--", *cmd]
        return cmd

    def _limits(self) -> None:  # runs in the child before exec
        cpu = int(self.timeout) + 5
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        size = self.files.max_total
        resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if not self.cgroup:  # no cgroup: fall back to an address-space cap
            mem = self.memory_mb * 1024 * 1024 * 2  # headroom: AS counts more than RSS
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))

    def _clip(self, text: str, limit: int) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        half = limit // 2
        return f"{text[:half]}\n[... {len(text) - limit} characters cut ...]\n{text[-half:]}"

    def run_python(self, path: str | None = None, code: str | None = None,
                   args: list[str] | None = None, stdin: str = "") -> str:
        if bool(path) == bool(code):
            raise ValueError("give exactly one of path (a .py file in your folder) or code")
        if path:
            p = self.files._path(path)  # same containment check as the file tools
            if not p.is_file():
                raise FileNotFoundError(f"no file {path!r}")
            argv = [p.relative_to(self.files.root).as_posix()]
        else:
            argv = ["-c", code]
        argv += [str(a) for a in (args or [])]
        before = {f: f.stat().st_mtime_ns for f in self.files._files()}
        t0 = time.time()
        proc = subprocess.Popen(self._command(argv), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, errors="replace",
                                preexec_fn=self._limits, start_new_session=True)
        try:
            out, err = proc.communicate(stdin, timeout=self.timeout)
            rc = proc.returncode
            status = (f"killed by signal {-rc}: it hit the memory, CPU or process limit"
                      if rc < 0 or rc in (137, 143) else f"exit code {rc}")
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, 9)
            out, err = proc.communicate()
            status = f"killed after {self.timeout:.0f}s (time limit)"
        took = time.time() - t0
        after = {f: f.stat().st_mtime_ns for f in self.files._files()}
        changed = sorted(self.files._rel(f) for f in after if before.get(f) != after[f])
        removed = sorted(self.files._rel(f) for f in before if f not in after)
        report = [f"{status}, {took:.1f}s"]
        if changed:
            report.append("files written: " + ", ".join(changed[:20]))
        if removed:
            report.append("files removed: " + ", ".join(removed[:20]))
        total = sum(f.stat().st_size for f in after)
        if total > self.files.max_total:
            report.append(f"warning: workspace folder is now over its "
                          f"{self.files.max_total // (1024 * 1024)} MB limit")
        report.append("stdout:\n" + (self._clip(out, self.output_chars) or "(none)"))
        if err.strip():
            report.append("stderr:\n" + self._clip(err, self.output_chars // 2))
        return "\n".join(report)

    def handlers(self) -> dict[str, Callable[..., str]]:
        return {"run_python": self.run_python}


RUN_PYTHON_TOOL = _tool(
    "run_python",
    "Run Python 3 in a sandbox that sees only your workspace folder (as the current "
    "directory). Give a .py file in your folder as path, or a short snippet as code. "
    "Standard library only; " + ("network allowed; " if EXEC_NETWORK else "no network; ") +
    f"{EXEC_TIMEOUT:.0f}s and {EXEC_MEMORY_MB} MB max. Files it writes in the folder stay. "
    "Returns the exit code, stdout, stderr and which files changed.",
    {"path": _PATH, "code": {"type": "string", "description": "A snippet, instead of a file."},
     "args": {"type": "array", "items": {"type": "string"}, "description": "Command-line arguments."},
     "stdin": {"type": "string", "description": "Text fed to standard input."}}, [])


class Conversation:
    """Questions she puts to the person running her, and whatever they type
    back. Lines typed into the terminal are read on a background thread and
    handed to the mind as input; asking never waits for an answer. One question
    is open at a time (until answered or ASK_EXPIRE_SECONDS pass), so she can't
    pile them up while nobody's at the keyboard."""

    def __init__(self, name: str = USER_NAME, expire: float = ASK_EXPIRE_SECONDS,
                 stream=sys.stdin):
        self.name, self.expire = name, expire
        self.tool_name = "ask_" + (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "user")
        self.open_question: str | None = None
        self.asked_at = 0.0
        self.closed = False  # input ended (e.g. stdin isn't a terminal)
        self._lines: queue.Queue[str] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream) -> None:
        try:
            for line in stream:
                if line.strip():
                    self._lines.put(line.strip())
        except (OSError, ValueError):
            pass
        self.closed = True

    def waiting_on(self) -> str | None:
        with self._lock:
            if self.open_question and time.time() - self.asked_at < self.expire:
                return self.open_question
            return None

    def ask(self, question: str) -> str:
        question = " ".join(question.split())
        if not question:
            return "error: the question is empty"
        if self.closed:
            return f"Not asked: {self.name} can't reply right now (no input is connected)."
        open_q = self.waiting_on()
        if open_q:
            mins = (time.time() - self.asked_at) / 60
            return (f"Not asked: your earlier question to {self.name} is still unanswered "
                    f"({mins:.0f} min ago): {open_q!r}. Carry on without it for now.")
        with self._lock:
            self.open_question, self.asked_at = question, time.time()
        STATUS.say(f"  ?? to {self.name}: {question}\n     (type a reply and press Enter)", "talk")
        return (f"Asked {self.name}: {question!r}. No answer yet; if one comes it will "
                "reach you later like anything else you perceive, so don't wait on it.")

    def incoming(self) -> list[str]:
        """What was typed since the last call, phrased as input for the mind.
        The first line after a question is taken as its answer."""
        out = []
        while True:
            try:
                line = self._lines.get_nowait()
            except queue.Empty:
                return out
            with self._lock:
                q, self.open_question = self.open_question, None
            out.append(f'{self.name} answered your question ("{q}"): {line}' if q
                       else f"{self.name} says: {line}")

    def tool(self) -> dict:
        return _tool(
            self.tool_name,
            f"Ask {self.name}, the person running you, a question. Use it for what only "
            f"{self.name} can tell you, not for what you could look up or work out. It "
            "doesn't wait: the answer, if one comes, arrives later as input. One question "
            "at a time; ask another once this one is answered.",
            {"question": {"type": "string", "description": "One clear, specific question."}},
            ["question"])

    def handlers(self) -> dict[str, Callable[..., str]]:
        return {self.tool_name: self.ask}


class LLM(Protocol):
    def __call__(self, system: str, prompt: str, tools: list | None = None,
                 handlers: dict[str, Callable[..., str]] | None = None) -> str: ...


def web_tools(max_uses: int = WEB_MAX_USES,
              max_content_tokens: int = WEB_MAX_CONTENT_TOKENS) -> list[dict]:
    """Anthropic-hosted tools: they run server-side, nothing to execute here.
    web_fetch only opens URLs already in the conversation (the action text or
    search results), which is why search comes along with it.

    These are the basic variants on purpose. The newer *_20260209 ones filter
    pages through a server-side code-execution sandbox, and that sandbox kept
    failing with too_many_requests on this account while plain fetches worked."""
    return [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses},
        {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": max_uses,
         "max_content_tokens": max_content_tokens},
    ]


def _run_tool(block, handlers: dict[str, Callable[..., str]]) -> dict:
    """Run one client tool call; failures go back to the model as is_error
    results instead of raising, so it can see what went wrong."""
    out = {"type": "tool_result", "tool_use_id": block.id}
    fn = handlers.get(block.name)
    if fn is None:
        return {**out, "content": f"Unknown tool: {block.name}", "is_error": True}
    try:
        return {**out, "content": str(fn(**(block.input or {})))}
    except Exception as e:
        return {**out, "content": f"{type(e).__name__}: {e}", "is_error": True}


def anthropic_llm(model: str = LLM_MODEL,
                  base_url: str | None = ANTHROPIC_BASE_URL) -> LLM:
    import anthropic

    client = anthropic.Anthropic(base_url=base_url)

    def create(**params):
        if LLM_FALLBACKS:
            return client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks=LLM_FALLBACKS, **params)
        return client.messages.create(**params)

    def call(system: str, prompt: str, tools: list | None = None,
             handlers: dict[str, Callable[..., str]] | None = None,
             max_rounds: int = 8) -> str:
        """tools may mix server tools (web) with client tools; each client
        tool needs a handler(**input) -> str that runs here."""
        handlers = handlers or {}
        messages: list[dict] = [{"role": "user", "content": prompt}]
        params = dict(model=model, max_tokens=LLM_MAX_TOKENS, system=system)
        if LLM_EFFORT:
            params["output_config"] = {"effort": LLM_EFFORT}
        if tools:
            params["tools"] = tools
        r = create(messages=messages, **params)
        for _ in range(max_rounds):
            if r.stop_reason == "pause_turn":
                # a long server-tool turn paused; send it back as-is and the
                # server resumes (no extra "continue" message)
                messages.append({"role": "assistant", "content": r.content})
            elif r.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": r.content})
                messages.append({"role": "user", "content": [
                    _run_tool(b, handlers) for b in r.content if b.type == "tool_use"]})
            else:
                break
            r = create(messages=messages, **params)
        # the answer is the text after the last tool activity; thinking
        # blocks and any text from before a search/fetch are skipped
        parts: list[str] = []
        for b in r.content:
            if b.type == "text":
                parts.append(b.text)
            elif b.type in ("server_tool_use", "tool_use") or b.type.endswith("_tool_result"):
                parts.clear()
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError(f"LLM returned no text (stop_reason={r.stop_reason})")
        return text

    return call


def extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else []


def extract_json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


# --------------------------------------------------------------------------
# Workspace contents
# --------------------------------------------------------------------------
@dataclass
class Node:
    content: str
    activation: float = 1.0
    valence: float = 0.0   # -1..1, the "feeling" perception attaches
    urgency: float = 0.0   # 0..1
    source: str = "percept"  # percept | memory:<store> | action
    born: float = field(default_factory=time.time)
    vec: np.ndarray | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class Coalition:
    nodes: list[Node]
    codelet: "AttentionCodelet"
    activation: float

    def text(self) -> str:
        return "\n".join(f"- [{n.source}] {n.content}" for n in self.nodes)

    def key(self) -> str:
        """What this coalition is about: its seed (strongest) node. Keyed on the
        whole member set, a faint memory joining or leaving made it count as a
        new thought, and inhibition of return let the same one win every second."""
        return self.nodes[0].key


class Workspace:
    """Current situational model. Everything decays; nothing is permanent here."""

    def __init__(self, embedder: LocalEmbedder, decay: float = 0.15,
                 floor: float = 0.05, capacity: int = 40, same_sim: float = DEDUPE_SIM):
        self.embedder = embedder
        self.nodes: dict[str, Node] = {}
        self.decay, self.floor, self.capacity = decay, floor, capacity
        self.same_sim = same_sim

    def add_many(self, nodes: list[Node]) -> None:
        """Batch-embed new nodes (one model call), then add them."""
        fresh = [n for n in nodes if n.vec is None]
        for n, v in zip(fresh, self.embedder.many([n.content for n in fresh])):
            n.vec = v
        for n in nodes:
            self._add(n)

    def add(self, node: Node) -> None:
        self.add_many([node])

    def _same_as(self, node: Node) -> Node | None:
        """An existing node saying the same thing, if any: exact text, or a
        paraphrase (raw cosine >= same_sim). Action notes and outcome feelings
        are events, not claims, so they never merge: "I did: X" and "That
        worked: X" score ~0.96 but are different things to have in mind."""
        if node.key in self.nodes:
            return self.nodes[node.key]
        if node.source == "action" or node.vec is None:
            return None
        best, best_sim = None, self.same_sim
        for n in self.nodes.values():
            if n.source != "action" and n.vec is not None:
                sim = float(np.dot(n.vec, node.vec))
                if sim >= best_sim:
                    best, best_sim = n, sim
        return best

    def _add(self, node: Node) -> None:
        ex = self._same_as(node)
        if ex is not None:
            # the thought comes back: it gets louder instead of taking a second slot
            ex.activation = min(1.5, ex.activation + 0.5 * node.activation)
            ex.urgency = max(ex.urgency, node.urgency)
            if abs(node.valence) > abs(ex.valence):
                ex.valence = node.valence
            if node.source == "percept":
                # perceived again just now: fresh again, and the newer wording
                # wins ("no longer crashes" must not be kept as "crashes")
                ex.born = max(ex.born, node.born)
                if node.key != ex.key and ex.source != "action":
                    del self.nodes[ex.key]
                    ex.content, ex.vec, ex.source = node.content, node.vec, "percept"
                    self.nodes[ex.key] = ex
        else:
            self.nodes[node.key] = node
        if len(self.nodes) > self.capacity:
            weakest = sorted(self.nodes.values(), key=lambda n: n.activation)
            for n in weakest[: len(self.nodes) - self.capacity]:
                del self.nodes[n.key]

    def tick(self, dt: float) -> None:
        for k, n in list(self.nodes.items()):
            n.activation *= math.exp(-self.decay * dt)
            if n.activation < self.floor:
                del self.nodes[k]

    def all(self) -> list[Node]:
        return list(self.nodes.values())


# --------------------------------------------------------------------------
# Attention codelets: cheap, always running, each with one concern
# --------------------------------------------------------------------------
class AttentionCodelet:
    name = "base"

    def __init__(self, base_level: float = 0.5):
        self.base_level = base_level

    def relevance(self, node: Node) -> float:
        raise NotImplementedError

    def form(self, nodes: list[Node], embedder: LocalEmbedder, k: int = 4,
             coherence: float = 0.15, support: float = 0.2) -> Coalition | None:
        """Seed on the node this codelet cares about most, then recruit only
        nodes that are about the same thing as the seed. Without the coherence
        check, an urgency codelet happily bundles 'dark mode request' with
        three unrelated urgent CI percepts, and the wrong habit fires.

        Strength is the seed's, plus a little (`support`) for each member that
        backs it up. Members only ever add: averaging them in let a strong
        thought with a few faint recalled memories attached fall below the
        broadcast threshold, so wandering never reached consciousness."""
        scored = sorted(
            ((self.relevance(n) * n.activation, n) for n in nodes),
            key=lambda t: -t[0],
        )
        scored = [(s, n) for s, n in scored if s > 0.05]
        if not scored:
            return None
        seed_score, seed = scored[0]
        members = [(s, n) for s, n in scored
                   if n is seed or (n.vec is not None and seed.vec is not None
                                    and embedder.relevance(seed.vec, n.vec) >= coherence)][:k]
        act = self.base_level * (seed_score + support * sum(s for s, n in members if n is not seed))
        return Coalition([n for _, n in members], self, act)

    # attentional learning: codelets whose coalitions win get stronger
    def reinforce(self, amount: float = 0.02) -> None:
        self.base_level = min(1.0, self.base_level + amount)

    def decay(self, dt: float, floor: float, rate: float = 0.002) -> None:
        """Unused codelets weaken by `rate` per second, never below `floor`.
        The floor must sit above the broadcast threshold, or after a long
        enough run no coalition can ever win and the mind goes silent."""
        self.base_level = max(floor, self.base_level - rate * dt)


class GoalCodelet(AttentionCodelet):
    def __init__(self, goal: str, embedder: LocalEmbedder, **kw):
        super().__init__(**kw)
        self.goal, self.embedder = goal, embedder
        self.vec = embedder.one(goal)
        self.name = f"goal:{goal[:30]}"

    def relevance(self, n: Node) -> float:
        return 0.0 if n.vec is None else self.embedder.relevance(self.vec, n.vec)


class UrgencyCodelet(AttentionCodelet):
    name = "urgency"

    def relevance(self, n: Node) -> float:
        return n.urgency + 0.5 * abs(n.valence)


class NoveltyCodelet(AttentionCodelet):
    name = "novelty"

    def relevance(self, n: Node) -> float:
        age = time.time() - n.born
        return (1.0 if n.source == "percept" else 0.3) * math.exp(-age / 5.0)


# --------------------------------------------------------------------------
# Long-term memory on mem0
# --------------------------------------------------------------------------
@dataclass
class Scheme:
    id: str
    context: str
    action: str
    similarity: float   # calibrated 0..1, not raw cosine
    reliability: float

    @property
    def confidence(self) -> float:
        return self.similarity * self.reliability


class LongTermMemory:
    def __init__(self, user_id: str, mem: Memory, embedder: LocalEmbedder,
                 sidecar: str = SCHEME_STATS_PATH,
                 same_episode: float = SAME_EPISODE_SIM, same_fact: float = SAME_FACT_SIM,
                 same_habit: float = SAME_HABIT_SIM, same_action: float = DEDUPE_SIM):
        self.m, self.user_id = mem, user_id
        self.embedder, self.calib = embedder, embedder.calib
        self.same_episode, self.same_fact = same_episode, same_fact
        self.same_habit, self.same_action = same_habit, same_action
        self.sidecar = Path(sidecar)
        self.stats: dict[str, dict] = (
            json.loads(self.sidecar.read_text()) if self.sidecar.exists() else {}
        )
        self._unconsolidated: list[str] = []

    def _filters(self, store: str) -> dict:
        return {"user_id": self.user_id, "agent_id": store}

    @staticmethod
    def _results(res) -> list[dict]:
        return res.get("results", []) if isinstance(res, dict) else (res or [])

    def _search(self, query: str, store: str, top_k: int) -> list[tuple[dict, float]]:
        """mem0 search, keeping only hits above the calibration floor,
        returned with calibrated relevance instead of raw score."""
        res = self.m.search(query, filters=self._filters(store),
                            top_k=top_k, threshold=self.calib.floor)
        out = []
        for r in self._results(res):
            rel = self.calib.relevance(float(r.get("score", 0.0)))
            if rel > 0:
                out.append((r, rel))
        return out

    def _same(self, text: str, store: str, sim: float, top_k: int = 1,
              exclude: set[str] = frozenset()) -> list[tuple[dict, float]]:
        """Existing memories in `store` saying the same as `text`: raw cosine
        >= sim, best first. mem0 itself only dedups exact text."""
        res = self.m.search(text, filters=self._filters(store),
                            top_k=top_k + len(exclude), threshold=sim)
        hits = [(r, float(r.get("score", 0.0))) for r in self._results(res)
                if r.get("id") not in exclude]
        return sorted(hits, key=lambda t: -t[1])[:top_k]

    @staticmethod
    def _times(r: dict) -> int:
        return int((r.get("metadata") or {}).get("times", 1))

    # --- episodic: written on every broadcast, no LLM involved
    def write_episode(self, text: str, meta: dict) -> None:
        """A moment she has already lived through (near-identical coalition)
        reinforces that episode instead of adding a copy, and isn't queued for
        consolidation again: it holds no facts the first one didn't."""
        with STATUS.phase("storing episode (embed + write)", kind="mem"):
            same = self._same(text, "episodic", self.same_episode)
            if same:
                r, _ = same[0]
                self.m.update(r["id"], metadata={"times": self._times(r) + 1,
                                                 "last_t": meta.get("t", time.time())})
                STATUS.say(f"  · familiar moment: reinforced an existing episode "
                           f"(now {self._times(r) + 1}x)", "mem")
                return
            self.m.add([{"role": "user", "content": text}], user_id=self.user_id,
                       agent_id="episodic", metadata=meta, infer=False)
        self._unconsolidated.append(text)

    # --- declarative: periodic consolidation, mem0 extracts facts (LLM call)
    def consolidate(self) -> None:
        if not self._unconsolidated:
            return
        msgs = [{"role": "user", "content": e} for e in self._unconsolidated]
        with STATUS.phase(f"consolidating {len(msgs)} episodes into facts (LLM)", kind="llm"):
            res = self.m.add(msgs, user_id=self.user_id, agent_id="declarative", infer=True)
            merged = self._merge_new_facts(
                [r for r in self._results(res) if r.get("event") == "ADD" and r.get("id")])
        self._unconsolidated.clear()
        if merged:
            STATUS.say(f"  · {merged} new fact(s) folded into existing ones", "mem")

    def _merge_new_facts(self, new: list[dict]) -> int:
        """mem0 2.x only ever ADDs facts (exact-text dedup only), so rewordings
        and updates pile up. Fold each new fact into an existing one that says
        the same thing: the existing memory keeps its id and age, takes the
        newer wording (so "X no longer crashes" replaces "X crashes"), and the
        new row is deleted."""
        merged, gone = 0, set()
        new_ids = {r["id"] for r in new}
        for r in new:
            same = self._same(r["memory"], "declarative", self.same_fact,
                              exclude={r["id"]} | gone)
            # prefer an older fact; a fact from this same batch only if nothing older matches
            same.sort(key=lambda t: t[0]["id"] in new_ids)
            if not same:
                continue
            old, _ = same[0]
            self.m.update(old["id"], text=r["memory"],
                          metadata={"times": self._times(old) + 1})
            self.m.delete(r["id"])
            gone.add(r["id"])
            merged += 1
        return merged

    # --- cueing: local associations pulled into the workspace
    def cue(self, query: str, top_k: int = 4) -> list[Node]:
        out: list[Node] = []
        with STATUS.phase("recalling related memories", kind="mem"):
            for store, weight in (("declarative", 0.6), ("episodic", 0.4)):
                for r, rel in self._search(query, store, top_k):
                    out.append(Node(r["memory"], activation=weight * rel,
                                    source=f"memory:{store}"))
        return out

    # --- procedural
    def recruit_schemes(self, context: str, top_k: int = 3) -> list[Scheme]:
        out = []
        with STATUS.phase("checking habits", kind="mem"):
            hits = self._search(context, "procedural", top_k)
        for r, rel in hits:
            st = self.stats.get(r["id"], {"ok": 0, "n": 0})
            reliability = (st["ok"] + 1) / (st["n"] + 2)  # Laplace-smoothed
            action = (r.get("metadata") or {}).get("action", "")
            if action:
                out.append(Scheme(r["id"], r["memory"], action, rel, reliability))
        return out

    def learn_scheme(self, context: str, action: str) -> str | None:
        """Reuses an existing habit when one already covers this situation with
        the same action, so its reliability keeps accumulating in one place
        instead of being split across copies. Same situation with a different
        action is a genuinely new habit: they compete on reliability."""
        same = self._same(context, "procedural", self.same_habit, top_k=3)
        if same:
            act_vec = self.embedder.one(action)
            for r, _ in same:
                prior = (r.get("metadata") or {}).get("action", "")
                if prior and float(np.dot(self.embedder.one(prior), act_vec)) >= self.same_action:
                    STATUS.say("  · already a habit for this; strengthening it", "mem")
                    return r["id"]
        with STATUS.phase("learning a new habit", kind="mem"):
            res = self.m.add([{"role": "user", "content": context}],
                             user_id=self.user_id, agent_id="procedural",
                             metadata={"action": action}, infer=False)
        ids = [r["id"] for r in self._results(res) if "id" in r]
        return ids[0] if ids else None

    def record_outcome(self, scheme_id: str, success: bool,
                       credit: float | None = None) -> None:
        """credit (0..1) replaces the hard 1/0 when the judge gave a
        probability, so a confident failure costs more than a borderline one."""
        st = self.stats.setdefault(scheme_id, {"ok": 0, "n": 0})
        st["n"] += 1
        st["ok"] += float(success) if credit is None else credit
        self.sidecar.write_text(json.dumps(self.stats, indent=2))

    def close(self) -> None:
        """mem0's close() skips the vector store; close Qdrant explicitly so it
        isn't left to __del__ at interpreter shutdown (which fails noisily)."""
        client = getattr(getattr(self.m, "vector_store", None), "client", None)
        if client is not None:
            client.close()
        self.m.close()


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
PERCEIVE_SYS = (
    "You are the perceptual system of a cognitive agent. Turn the new input into "
    "1-5 standalone percepts, each ONE sentence under 25 words: what was noticed "
    "and how it lands. Perceive, don't plan: no options, fixes or next steps, "
    "deciding what to do is not your job. You also see the agent's goals and what is "
    "on its mind right now: use them to judge each percept, not to rewrite it. "
    "valence is how it feels to this mind (-1 bad, 1 good); urgency is how much it "
    "bears on its goals or on what it is in the middle of (0 idle, 1 drop everything). "
    "The same news can matter a lot or not at all depending on that context. "
    "Return ONLY a JSON array of objects: "
    '{"content": str, "valence": float -1..1, "urgency": float 0..1}.'
)

DELIBERATE_SYS = (
    "You are the deliberative system of a cognitive agent. You see what reached "
    "consciousness this cycle, what it has done recently and how that went, its "
    "current intention, and partial habits that might apply. Choose ONE concrete "
    "next action that moves things forward: build on what it already did and "
    "decided, don't start over or repeat what just happened. Reply with ONLY a JSON "
    'object: {"action": "one sentence", "intention": "one sentence: what you are '
    'working toward over the next several actions"}. Keep the current intention '
    "word for word unless it is done, stuck, or something matters more."
)


# appended to the deliberation prompt when a goal check is due
GOAL_CHECK = (
    "GOAL CHECK (due now{why}). Before anything else this cycle, weigh your goals "
    "against what you've actually been doing:\n{report}\n\n"
    "Goals are concrete things you're pursuing, ones you could finish or clearly "
    "make progress on, in service of your values; the values themselves are fixed "
    "and never go in this list. "
    "These goals aren't a fixed list you were handed: you can write new ones from "
    "what you've found, retire ones that are done, and drop ones that no longer "
    "pull at you. Ask of each: is it done (you did what it asked, even once: then "
    "retire it, or replace it with what comes next)? Is it still yours (one that "
    "never wins your attention may not be)? And what's missing: is there work you "
    "keep returning to, or something you've discovered you want, that no goal "
    "names yet? If anything changes, this cycle's action is to revise your goals, "
    "naming exactly what to add, drop or reword (the current intention can wait "
    "one step). If they all still fit, carry on, but say why. "
    'Add a "goal_check" field to your JSON: one honest sentence, either what you\'re '
    "changing or why each goal still earns its place.\n\n")


def _read_text(path: str) -> str:
    p = Path(path)
    return (p if p.is_absolute() else Path(__file__).parent / p).read_text(encoding="utf-8").strip()


def load_personality(path: str = PERSONALITY_FILE) -> str:
    return _read_text(path)


def load_prompt(path: str = PROMPT_FILE) -> str:
    return _read_text(path)


def load_values(path: str = VALUES_FILE) -> list[str]:
    """Her values, one per line (# comments ignored); none if the file is missing."""
    try:
        return load_goals(path)
    except FileNotFoundError:
        return []


def load_goals(path: str = GOALS_FILE) -> list[str]:
    lines = (ln.strip() for ln in _read_text(path).splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def load_live_goals(path: str = LIVE_GOALS_PATH) -> list[str] | None:
    """Her current goals if she has ever revised them, else None (use the seed)."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def numbered(items: list[str]) -> str:
    return "\n".join(f"{i}. {g}" for i, g in enumerate(items, 1)) or "none"


GOAL_REVIEW_THOUGHT = (
    "Do my goals still fit what I actually care about right now? Is one of them "
    "done? Is there something I want now that none of them names? They're mine "
    "to write: add one, retire one, reword one. Current goals: {goals}")

GOALS_TOOL = {
    "name": "revise_goals",
    "description": (
        "Change your own goals. Goals decide what your attention is drawn to and "
        "what you drift back to when nothing else is going on, so change them "
        "deliberately: add a new one when you've found something you want to "
        "pursue, retire one you've finished, drop one that no longer fits. "
        "A goal is concrete: something you could finish or clearly make progress "
        "on. Your values are fixed and aren't goals; don't add them here. "
        "Refer to existing goals by their number in the current list. Rewording "
        "and removal use the numbers from before this change. "
        f"At most {MAX_GOALS} goals, at least 1."),
    "input_schema": {
        "type": "object",
        "properties": {
            "add": {"type": "array", "items": {"type": "string"},
                    "description": "New goals, one short sentence each."},
            "remove": {"type": "array", "items": {"type": "integer"},
                       "description": "Numbers of goals to drop."},
            "reword": {"type": "array", "description": "Goals to rewrite in place.",
                       "items": {"type": "object", "properties": {
                           "number": {"type": "integer"}, "text": {"type": "string"}},
                           "required": ["number", "text"]}},
            "reason": {"type": "string", "description": "Why, in one sentence."},
        },
        "required": ["reason"],
    },
}


EXECUTE_SYS = (
    "You are the motor system of a cognitive agent: you carry out the one action "
    "it chose. Use your tools when the action needs them: the web for anything "
    "outside the mind, your workspace folder's file tools to read, write or change "
    "your own files, run_python to run code from it (check what it actually does: "
    "run it, read the output, fix and rerun when it fails), revising goals only "
    "when the action is to change them. "
    "When it needs none, carry it out in thought. Then report what you found or what happened in at most five short, "
    "factual sentences, with source URLs for anything from the web, the path "
    "of any file you touched, and for code you ran: whether it ran and what it "
    "printed or why it failed. If something "
    "failed or turned up nothing, say so plainly. The action and its result are "
    "the agent's own: report in the first person, as its own observation. The "
    "report isn't addressed to anyone, so don't ask questions or offer more in it. "
    "No preamble."
)
# appended to EXECUTE_SYS when she can ask the person running her
EXECUTE_ASK_NOTE = ("When the action is to ask {name} something, ask it with {tool} "
                    "(exactly the question the action names) and report that you asked.")

APPRAISE_SYS = (
    "You are the appraisal system of a cognitive agent. You see what was in "
    "consciousness, the agent's values and goals, the action it just chose, its recent "
    "actions, and what came back when the action was carried out. It worked if "
    "it was a sensible, specific next step given what was in consciousness and "
    "the goals, and what came back actually delivered something. It did not work "
    "if it was vague, ignored what was in consciousness, went against its values, "
    "repeated a recent action "
    "with nothing new to justify it, or came back empty or failed. When nothing "
    "came back, judge the choice alone; don't demand proof. "
    f"A question put to {USER_NAME} worked if it was specific and something only "
    f"{USER_NAME} could answer; the answer comes later, so don't count its absence. "
    "Be honest, not kind. Return ONLY a JSON object: "
    '{"worked": bool, "why": str (one sentence)}.'
)


# the same judgement as APPRAISE_SYS, phrased as a Jev yes/no question
JEV_APPRAISAL = {
    "type": "noul",
    "instructions": (
        "Judge the action this mind just chose, using everything in the state. "
        "Nothing outside the mind confirms results, so judge the choice and "
        "whatever came back from carrying it out. Be strict, not kind."),
    "criteria": {
        "true": ("It was a sensible, specific next step given the conscious "
                 "content and the goals, and what came back (if anything) "
                 "actually delivered something. A specific question put to "
                 f"{USER_NAME} that only {USER_NAME} could answer counts, even "
                 "though the answer comes later."),
        "false": ("It was vague, ignored the conscious content, went against "
                  "the values, repeated a "
                  "recent action with nothing new to justify it, or what came "
                  "back was empty or a failure."),
    },
}


def with_personality(personality: str, contract: str, note: str) -> str:
    """Personality first, then the output contract, which wins over any style
    the personality asks for (the cycle needs that exact shape back)."""
    if not personality:
        return contract
    return f"{personality}\n\n---\n\n{contract} {note}"


def deliberate_system(personality: str, abilities: str = "") -> str:
    contract = f"{DELIBERATE_SYS} {abilities}".strip()
    return with_personality(personality, contract, "Think in your own voice, "
                            "but whatever your usual style, the reply is that JSON object.")


def perceive_system(personality: str) -> str:
    return with_personality(personality, PERCEIVE_SYS, "Feel it in your own voice, "
                            "but whatever your usual style, the reply is that JSON array.")


def appraise_system(personality: str) -> str:
    return with_personality(personality, APPRAISE_SYS, "Judge in your own voice, "
                            "but whatever your usual style, the reply is that JSON object.")


class LidaAgent:
    def __init__(self, user_id: str, llm: LLM, goals: list[str],
                 act: Callable[[str], str | None], *,
                 embed_model: str = EMBED_MODEL,
                 store_path: str = STORE_PATH,
                 broadcast_threshold: float = BROADCAST_THRESHOLD,
                 automatic_threshold: float = AUTOMATIC_THRESHOLD,
                 refractory_tau: float = REFRACTORY_TAU,
                 action_interval: float = ACTION_INTERVAL,
                 idle_seconds: float = IDLE_SECONDS,
                 consolidate_every: int = CONSOLIDATE_EVERY,
                 broadcast_cue_k: int = BROADCAST_CUE_K,
                 personality: str = "",
                 abilities: str = "",
                 max_goals: int = MAX_GOALS,
                 goals_path: str | None = LIVE_GOALS_PATH,
                 goals_log_path: str | None = GOALS_LOG_PATH,
                 decider: Jev | None = None,
                 files: Files | None = None,
                 intention_path: str | None = INTENTION_PATH,
                 goal_review_minutes: float = GOAL_REVIEW_MINUTES,
                 values: list[str] | None = None):
        self.llm, self.act = llm, act
        self.decider, self.files = decider, files
        self.sandbox: Sandbox | None = None
        self.conversation: Conversation | None = None
        self.deliberate_sys = deliberate_system(personality, abilities)
        self.appraise_sys = appraise_system(personality)
        self.perceive_sys = perceive_system(personality)
        self.goals = list(goals)
        self.values = list(values or [])
        self.max_goals, self.goals_path, self.goals_log_path = max_goals, goals_path, goals_log_path
        # how each goal is actually faring: broadcasts it won this run, and when
        self.goal_wins: dict[str, int] = {g: 0 for g in self.goals}
        self.goal_last_win: dict[str, float] = {}
        self.started = time.time()
        self.goals_since = self._last_goal_change() or self.started
        self.goal_review_every = goal_review_minutes * 60
        self.last_goal_review = self.started
        self.goal_review_reason: str | None = None  # set when an intention ends
        # what she did lately and how it went: deliberation builds on it
        self.recent: deque[dict] = deque(maxlen=6)
        self.intention_path = intention_path
        self.intention: str | None = None
        if intention_path and Path(intention_path).exists():
            self.intention = json.loads(Path(intention_path).read_text(encoding="utf-8")).get("intention")
        mem, self.embedder = build_memory(embed_model, store_path)
        self.ltm = LongTermMemory(user_id, mem, self.embedder)
        self.ws = Workspace(self.embedder)
        self._goal_codelets = {g: GoalCodelet(g, self.embedder, base_level=0.6)
                               for g in self.goals}
        self._other_codelets: list[AttentionCodelet] = [UrgencyCodelet(0.6), NoveltyCodelet(0.5)]
        self.codelets: list[AttentionCodelet] = []
        self._rebuild_codelets()
        self.sensory: deque[str] = deque()
        self.refractory: dict[str, float] = {}
        self.pending: dict | None = None  # last action awaiting an outcome
        self.broadcast_threshold = broadcast_threshold
        self.automatic_threshold = automatic_threshold
        self.refractory_tau = refractory_tau
        self.action_interval, self.idle_seconds = action_interval, idle_seconds
        self.last_action = 0.0
        self.last_broadcast = time.time()
        self._wander_i = 0
        self.consolidate_every = consolidate_every
        self.broadcast_cue_k = broadcast_cue_k
        self.broadcasts = 0

    # ---- input
    def sense(self, raw: str) -> None:
        self.sensory.append(raw)

    def perceive(self, raw: str, on_mind: int = 6) -> list[Node]:
        """Percepts judged in context: valence and urgency depend on her goals
        and on what she's already thinking about, not on the input alone."""
        top = sorted(self.ws.all(), key=lambda n: -n.activation)[:on_mind]
        mind = "\n".join(f"- {n.content}" for n in top) or "nothing in particular"
        prompt = (self._values_text()
                  + f"Your current goals:\n{numbered(self.goals)}\n\n"
                  f"On your mind right now:\n{mind}\n\nNew input:\n{raw}")
        try:
            items = extract_json_array(self.llm(self.perceive_sys, prompt))
        except (json.JSONDecodeError, TypeError):
            items = []
        if not items:
            items = [{"content": raw, "valence": 0.0, "urgency": 0.3}]
        return [Node(str(i.get("content", raw)),
                     valence=max(-1.0, min(1.0, float(i.get("valence", 0.0)))),
                     urgency=max(0.0, min(1.0, float(i.get("urgency", 0.0)))))
                for i in items]

    # ---- one cognitive cycle
    # ---- goals: hers to change
    @property
    def recent_actions(self) -> list[str]:
        return [r["action"] for r in self.recent]

    def _rebuild_codelets(self) -> None:
        self.codelets = [self._goal_codelets[g] for g in self.goals] + self._other_codelets
        # the intention pulls related content into attention, like a goal does
        if self.intention:
            if getattr(self, "_intention_codelet", None) is None or \
                    self._intention_codelet.goal != self.intention:
                self._intention_codelet = GoalCodelet(self.intention, self.embedder, base_level=0.6)
                self._intention_codelet.name = f"intention:{self.intention[:30]}"
            self.codelets.append(self._intention_codelet)

    def set_intention(self, intention: str | None) -> None:
        intention = (intention or "").strip() or None
        if intention == self.intention:
            return
        if self.intention:  # the old one is done, stuck or overtaken: what now?
            self.goal_review_reason = f'you just set aside "{self.intention}"'
        self.intention = intention
        self._rebuild_codelets()
        if self.intention_path:
            Path(self.intention_path).write_text(json.dumps(
                {"intention": intention, "t": time.time()}, indent=2), encoding="utf-8")
        if intention:
            STATUS.say(f"→ intention: {intention}", "lida")

    def revise_goals(self, add: list[str] | None = None, remove: list[int] | None = None,
                     reword: list[dict] | None = None, reason: str = "") -> str:
        """Handler for the revise_goals tool. Numbers refer to the list as it
        was before this call. Returns what changed, or why nothing did."""
        def clean(text) -> str:
            return " ".join(str(text).split())[:200]

        old, n = list(self.goals), len(self.goals)
        remove, reword, add = list(remove or []), list(reword or []), list(add or [])
        numbers = remove + [r.get("number") for r in reword if isinstance(r, dict)]
        bad = [i for i in numbers if not isinstance(i, int) or not 1 <= i <= n]
        if bad or len(numbers) != len(remove) + len(reword):
            return f"Nothing changed: no goal number(s) {bad or reword}; current goals are 1-{n}."
        goals = list(old)
        for r in reword:
            goals[r["number"] - 1] = clean(r["text"])
        goals = [g for i, g in enumerate(goals, 1) if i not in set(remove)]
        goals += [clean(g) for g in add]
        goals = [g for g in dict.fromkeys(goals) if g]  # dedupe, keep order
        if not goals:
            return "Nothing changed: you need at least one goal."
        if len(goals) > self.max_goals:
            return (f"Nothing changed: that would make {len(goals)} goals, and the "
                    f"limit is {self.max_goals}. Remove or merge some first.")
        if goals == old:
            return "Nothing changed: the goals are already exactly that."
        self.set_goals(goals, reason)
        return f"Goals revised. They are now:\n{numbered(goals)}"

    def set_goals(self, goals: list[str], reason: str = "") -> None:
        """Swap in a new goal list: new goals get fresh codelets, kept goals keep
        the attentional strength they've earned. Persists and logs the change."""
        old = self.goals
        self._goal_codelets = {g: self._goal_codelets.get(g) or
                               GoalCodelet(g, self.embedder, base_level=0.6) for g in goals}
        self.goals = list(goals)
        self.goal_wins = {g: self.goal_wins.get(g, 0) for g in self.goals}
        self.goals_since = self.last_goal_review = time.time()
        self.goal_review_reason = None
        self._rebuild_codelets()
        if self.goals_path:
            Path(self.goals_path).write_text(json.dumps(self.goals, indent=2), encoding="utf-8")
        if self.goals_log_path:
            with open(self.goals_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "reason": reason,
                                    "before": old, "after": self.goals}) + "\n")
        STATUS.say(f"★ goals revised ({reason or 'no reason given'}):\n{numbered(self.goals)}", "lida")

    def _last_goal_change(self) -> float | None:
        """When the goals last changed (from the log), so 'unchanged for' spans restarts."""
        try:
            lines = Path(self.goals_log_path).read_text(encoding="utf-8").splitlines()
            changes = [e for e in map(json.loads, lines) if "after" in e]  # skip checks
            return float(changes[-1]["t"]) if changes else None
        except (OSError, TypeError, ValueError, KeyError):
            return None

    def goal_review_due(self) -> str | None:
        """Why a goal check is due now, or None: finishing or dropping an
        intention (the natural moment to ask what comes next), or the timer."""
        if self.goal_review_reason:
            return self.goal_review_reason
        if (self.goal_review_every > 0
                and time.time() - self.last_goal_review >= self.goal_review_every):
            return f"{(time.time() - self.last_goal_review) / 60:.0f} min since the last one"
        return None

    def goal_report(self) -> str:
        """Evidence for a goal check: how much attention each goal has actually
        won, against how long it has been the goal."""
        now = time.time()

        def ago(t: float) -> str:
            m = (now - t) / 60
            return f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"
        lines = []
        for i, g in enumerate(self.goals, 1):
            n, last = self.goal_wins.get(g, 0), self.goal_last_win.get(g)
            cd = self._goal_codelets.get(g)
            won = (f"won your attention {n} time{'s' * (n != 1)}"
                   + (f" (last {ago(last)} ago)" if last else "")
                   if n else "hasn't won your attention once")
            lines.append(f"{i}. {g}\n   {won}; pull strength {cd.base_level:.2f}" if cd
                         else f"{i}. {g}\n   {won}")
        total = sum(self.goal_wins.values())
        return (f"Goals unchanged for {ago(self.goals_since)}. Over the last "
                f"{ago(max(self.started, self.goals_since))} ({total} goal-driven moments):\n"
                + "\n".join(lines))

    def wander(self) -> None:
        """Nothing has reached consciousness for a while: drift somewhere and
        let it pull related memories in. No LLM call; this is what keeps an
        input-less mind going once its first input has faded. Each pass:
        what she's working on, then each goal, then whether the goals still
        fit (a standing reminder that they're hers to change)."""
        if not self.goals:
            return
        stops = (["intention"] if self.intention else []) + self.goals + ["review"]
        stop = stops[self._wander_i % len(stops)]
        self._wander_i += 1
        if stop == "intention":
            STATUS.say(f"~ mind returning to what I'm working on: {self.intention}", "lida")
            self.ws.add_many([Node(f"What I'm working on: {self.intention}", activation=1.0,
                                   urgency=0.3, source="goal")] + self.ltm.cue(self.intention))
        elif stop == "review":
            STATUS.say("~ mind wandering to: whether my goals still fit", "lida")
            self.ws.add(Node(GOAL_REVIEW_THOUGHT.format(goals="; ".join(self.goals)),
                             activation=1.0, urgency=0.8, source="goal"))  # a pressing question
        else:
            STATUS.say(f"~ mind wandering to: {stop}", "lida")
            self.ws.add_many([Node(stop, activation=1.0, source="goal")] + self.ltm.cue(stop))
        self.last_broadcast = time.time()  # give it a full idle window to win

    def step(self, dt: float = 0.1) -> Coalition | None:
        if not self.sensory and time.time() - self.last_broadcast > self.idle_seconds:
            self.wander()

        # 1. Understanding: perceive new input, cue long-term memory with it
        while self.sensory:
            raw = self.sensory.popleft()
            with STATUS.phase("perceiving new input (LLM)", kind="llm"):
                percepts = self.perceive(raw)
            self.ws.add_many(percepts + self.ltm.cue(raw))
        self.ws.tick(dt)

        # 2. Attention: codelets form coalitions, they compete
        nodes = self.ws.all()
        coalitions = [c for c in (cd.form(nodes, self.embedder) for cd in self.codelets) if c]
        for c in coalitions:
            c.activation *= self._refractory_factor(c)
        for cd in self.codelets:
            cd.decay(dt, floor=self.broadcast_threshold + 0.1)
        winner = max(coalitions, key=lambda c: c.activation, default=None)
        if not winner or winner.activation < self.broadcast_threshold:
            return None

        # 3. Broadcast -> learning + action selection
        self._broadcast(winner)
        return winner

    def _refractory_factor(self, c: Coalition) -> float:
        """Inhibition of return: recently broadcast content is suppressed."""
        t = self.refractory.get(c.key())
        return 1.0 if t is None else 1.0 - math.exp(-(time.time() - t) / self.refractory_tau)

    def _broadcast(self, w: Coalition) -> None:
        now = self.last_broadcast = time.time()
        self.refractory[w.key()] = now
        # entries this old no longer suppress anything; drop them so a long run
        # doesn't grow the dict forever
        horizon = now - 10 * self.refractory_tau
        self.refractory = {k: t for k, t in self.refractory.items() if t > horizon}
        w.codelet.reinforce()
        goal = getattr(w.codelet, "goal", None)
        if goal in self.goal_wins:  # a goal (not the intention) won attention
            self.goal_wins[goal] += 1
            self.goal_last_win[goal] = now
        for n in w.nodes:
            n.activation *= 0.5  # conscious content is "spent"

        # conscious content cues long-term memory, so one thought can call up the
        # next (LIDA's local associations). Before the episode is written, or the
        # first thing recalled would be this very moment.
        if self.broadcast_cue_k:
            self.ws.add_many(self.ltm.cue("\n".join(n.content for n in w.nodes),
                                          top_k=self.broadcast_cue_k))

        self.ltm.write_episode(w.text(), {"codelet": w.codelet.name, "t": time.time()})
        self.broadcasts += 1
        if self.broadcasts % self.consolidate_every == 0:
            self.ltm.consolidate()

        # one outstanding action at a time, and no faster than action_interval
        if self.pending is None and now - self.last_action >= self.action_interval:
            self.last_action = now
            self._select_action(w)

    # ---- action selection: habit first, LLM only when habits don't fit
    def _select_action(self, w: Coalition) -> None:
        context = w.text()
        schemes = self._not_just_done(self.ltm.recruit_schemes(context))
        best = max(schemes, key=lambda s: s.confidence, default=None)

        review = self.goal_review_due()  # a goal check needs real deliberation, not a habit
        if best and best.confidence >= self.automatic_threshold and not review:
            action, mode, sid = best.action, "automatic", best.id
        else:
            with STATUS.phase("deliberating (LLM)" + (" + goal check" if review else ""), kind="llm"):
                action, mode, sid = self._deliberate(context, schemes, review), "deliberate", None

        self.pending = {"context": context, "action": action,
                        "scheme_id": sid, "mode": mode, "result": None}
        self.ws.add(Node(f"I did: {action}", activation=0.6, source="action"))
        # whatever the action brings back re-enters through perception, like
        # any other input, and competes for attention on the next cycles
        result = self.act(action)
        if result:
            self.pending["result"] = result
            self.sense(f"What came back from \"{action}\":\n{result}")

    def _not_just_done(self, schemes: list[Scheme], last: int = 3) -> list[Scheme]:
        """A habit whose action she just did can't fire again right away
        (behavior-level inhibition of return), or a habit that worked once
        loops: same action, every time, without deliberation ever seeing it."""
        recent = self.recent_actions[-last:]
        if not schemes or not recent:
            return schemes
        done = self.embedder.many(recent)
        return [s for s in schemes
                if max(float(np.dot(self.embedder.one(s.action), d)) for d in done) < DEDUPE_SIM]

    def _recent_text(self, result_chars: int = 240) -> str:
        lines = []
        for r in self.recent:
            res = " ".join((r.get("result") or "nothing came back").split())
            if len(res) > result_chars:
                res = res[:result_chars] + "…"
            lines.append(f"- {r['action']}\n  {r['verdict']}{': ' + r['why'] if r.get('why') else ''}"
                         f"\n  came back: {res}")
        return "\n".join(lines) or "nothing yet"

    def _values_text(self) -> str:
        if not self.values:
            return ""
        return ("Your values (fixed: who you are, not things to finish):\n"
                + "\n".join(f"- {v}" for v in self.values) + "\n\n")

    def _log_goal_check(self, verdict: str, action: str) -> None:
        """A goal check happened: restart the clock and keep her verdict, so a
        'they still fit' is a decision on record, not a silent skip."""
        self.last_goal_review = time.time()
        self.goal_review_reason = None
        verdict = verdict or "(no verdict given)"
        STATUS.say(f"★ goal check: {verdict}", "lida")
        if self.goals_log_path:
            with open(self.goals_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": self.last_goal_review, "check": verdict,
                                    "action": action, "goals": self.goals,
                                    "wins": self.goal_wins}) + "\n")

    def _waiting_text(self) -> str:
        c = self.conversation
        q = c.waiting_on() if c else None
        return f"Still waiting on {c.name} to answer: {q}\n\n" if q else ""

    def _deliberate(self, context: str, schemes: list[Scheme], review: str | None = None) -> str:
        hints = "\n".join(f"- (reliability {s.reliability:.2f}) {s.action}"
                          for s in schemes) or "none"
        prompt = (f"Conscious content:\n{context}\n\n"
                  f"Current intention: {self.intention or 'none yet'}\n\n"
                  f"What you did recently (oldest first):\n{self._recent_text()}\n\n"
                  + self._waiting_text()
                  + self._values_text()
                  + f"Your current goals (concrete things you're pursuing; yours to add, "
                  f"finish and retire):\n"
                  f"{numbered(self.goals)}\n\n"
                  + (f"Your workspace folder:\n{self.files.overview()}\n\n" if self.files else "")
                  + f"Habits that partly match:\n{hints}\n\n"
                  + (GOAL_CHECK.format(why=f": {review}", report=self.goal_report())
                     if review else "")
                  + "Next action?")
        raw = self.llm(self.deliberate_sys, prompt).strip()
        try:
            choice = extract_json_object(raw)
            action = str(choice.get("action") or "").strip()
            self.set_intention(choice.get("intention") or self.intention)
            if review:  # after the intention, so changing it here doesn't re-trigger a check
                self._log_goal_check(str(choice.get("goal_check") or "").strip(), action)
        except (json.JSONDecodeError, TypeError, AttributeError):
            action = ""
        return action or raw  # unreadable JSON: treat the whole reply as the action

    # ---- appraisal: the mind judges its own action, no one else does
    def appraise(self) -> tuple[bool | None, str]:
        """(worked, why) for the pending action; worked is None when the
        judgement couldn't be read, so nothing gets learned from it."""
        p = self.pending
        if not p:
            return None, "nothing pending"
        if self.decider is not None:
            try:
                with STATUS.phase("appraising the action (Jev)", kind="jev"):
                    prob = float(self.decider.ask(
                        {"conscious_content": p["context"], "values": self.values,
                         "goals": self.goals,
                         "intention": self.intention,
                         "recent_actions_oldest_first": list(self.recent_actions),
                         "action": p["action"], "mode": p["mode"],
                         "what_came_back": p.get("result") or "nothing"},
                        {"worked": JEV_APPRAISAL})["worked"]["noul"])
                # keep the probability: habits learn from how sure the judge was
                p["p_worked"] = prob
                return prob >= JEV_WORKED_THRESHOLD, f"Jev p(worked)={prob:.2f}"
            except Exception as e:
                STATUS.say(f"  !! Jev unavailable ({e}); judging with the LLM instead")
        goals = "\n".join(f"- {g}" for g in self.goals) or "none"
        recent = "\n".join(f"- {a}" for a in self.recent_actions) or "none"
        prompt = (f"Conscious content:\n{p['context']}\n\n{self._values_text()}Goals:\n{goals}\n\n"
                  f"Current intention: {self.intention or 'none'}\n\n"
                  f"Recent actions (oldest first):\n{recent}\n\n"
                  f"Action just chosen ({p['mode']}):\n{p['action']}\n\n"
                  f"What came back:\n{p.get('result') or 'nothing'}\n\nDid it work?")
        try:
            with STATUS.phase("appraising the action (LLM)", kind="llm"):
                verdict = extract_json_object(self.llm(self.appraise_sys, prompt))
        except (json.JSONDecodeError, TypeError):
            verdict = {}
        worked = verdict.get("worked")
        if not isinstance(worked, bool):
            return None, "appraisal unreadable"
        return worked, str(verdict.get("why", "")).strip()

    def _remember_action(self, p: dict, verdict: str, why: str) -> None:
        result = " ".join((p.get("result") or "nothing came back").split())[:600]
        text = (f"I decided to: {p['action']}\n"
                + (f"Working toward: {self.intention}\n" if self.intention else "")
                + f"What came back: {result}\nIt {verdict}" + (f": {why}" if why else "."))
        try:
            self.ltm.write_episode(text, {"kind": "action", "mode": p["mode"], "t": time.time()})
        except Exception as e:  # memory trouble mustn't lose the outcome itself
            STATUS.say(f"  !! couldn't store the action episode: {e}")

    # ---- outcome: where deliberation compiles into habit
    def feedback(self, success: bool | None, why: str = "") -> None:
        """success=None drops the pending action without learning from it. Either
        way the action is remembered: recent history for the next deliberation,
        and an episode in long-term memory, so her choices outlive the
        workspace (where the action note fades long before the next choice)."""
        p = self.pending
        if not p:
            return
        verdict = {True: "worked", False: "didn't work", None: "couldn't be judged"}[success]
        self.recent.append({"action": p["action"], "mode": p["mode"], "verdict": verdict,
                            "why": why, "result": p.get("result")})
        self._remember_action(p, verdict, why)
        if success is None:
            self.pending = None
            return
        credit = p.get("p_worked")  # set when Jev judged; None for the LLM judge
        if p["scheme_id"]:
            self.ltm.record_outcome(p["scheme_id"], success, credit)
        elif success:
            sid = self.ltm.learn_scheme(p["context"], p["action"])
            if sid:
                self.ltm.record_outcome(sid, True, credit)
        # feelings: outcomes feed back into the workspace with valence. Jev's
        # p(worked) makes it graded (0.9 feels good, 0.5 barely registers);
        # the LLM judge's yes/no stays a fixed +/-0.6.
        valence = 2 * credit - 1 if credit is not None else (0.6 if success else -0.6)
        self.ws.add(Node(f"That {'worked' if success else 'failed'}: {p['action']}",
                         valence=valence, urgency=0.2,
                         activation=0.5, source="action"))
        self.pending = None


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
WEB_ABILITY = ("Actions are carried out, and can reach the web: searching for "
               "something, or reading a page at a URL. Name what to look up or the URL.")
FILES_ABILITY = ("You have a workspace folder of your own: text files you can list, "
                 "read, write, edit and delete. Notes, drafts, code, anything worth keeping: "
                 "what you put there stays between runs. Name the file and what to do with it.")
EXEC_ABILITY = ("You can also run Python you've written there (standard library, no "
                "network, sandboxed to that folder) and see what it prints, so you can "
                "test what you build instead of guessing.")
ASK_ABILITY = (f"{USER_NAME}, the person running you, is around some of the time. "
               f"An action can ask {USER_NAME} a question with the ask_"
               f"{re.sub(r'[^a-z0-9]+', '_', USER_NAME.lower()).strip('_') or 'user'} tool, "
               f"for what only {USER_NAME} can tell you; it doesn't wait, and an answer, if one comes, reaches you "
               f"later. Anything {USER_NAME} says to you reaches you the same way.")
GOALS_ABILITY = ("Your goals (concrete things you're pursuing, unlike your fixed "
                 "values) are yours to write: an action can add new ones (from "
                 "what you've found or come to want), retire ones you've finished, "
                 "drop ones that no longer pull at you, or reword them.")


def motor(llm: LLM, agent: "LidaAgent", web: bool = WEB_ENABLED,
          self_goals: bool = SELF_GOALS) -> Callable[[str], str] | None:
    """Carries out actions with whatever tools are enabled; None if none are
    (actions are then only chosen, never carried out)."""
    files, sandbox, convo = agent.files, agent.sandbox, agent.conversation
    tools = ((web_tools() if web else []) + (FILE_TOOLS if files else [])
             + ([RUN_PYTHON_TOOL] if sandbox else []) + ([convo.tool()] if convo else [])
             + ([GOALS_TOOL] if self_goals else []))
    if not tools:
        return None
    system = EXECUTE_SYS + (" " + EXECUTE_ASK_NOTE.format(name=convo.name, tool=convo.tool_name)
                            if convo else "")
    handlers = {**(files.handlers() if files else {}),
                **(sandbox.handlers() if sandbox else {}),
                **(convo.handlers() if convo else {}),
                **({"revise_goals": agent.revise_goals} if self_goals else {})}

    def carry_out(action: str) -> str:
        prompt = action
        if agent.intention:
            prompt += f"\n\nWhat this is part of: {agent.intention}"
        if files:
            prompt += f"\n\nYour workspace folder:\n{files.overview()}"
        if self_goals:
            prompt += f"\n\nCurrent goals:\n{numbered(agent.goals)}"
        return llm(system, prompt, tools=tools, handlers=handlers)

    return carry_out


if __name__ == "__main__":
    carry_out: Callable[[str], str] | None = None  # set once the agent exists

    def act(action: str) -> str | None:
        STATUS.say(f"  >> ACTION: {action}", "act")
        if carry_out is None:
            return None
        with STATUS.phase("carrying out the action (LLM + tools)", kind="llm"):
            result = carry_out(action)
        STATUS.say("  << " + result.replace("\n", "\n     "), "act")
        return result

    files = Files() if FILES_ENABLED else None
    sandbox = None
    if files and EXEC_ENABLED:
        try:
            sandbox = Sandbox(files)
            STATUS.say("Python runs sandboxed (bubblewrap"
                       + (" + cgroup limits)." if sandbox.cgroup else "; no cgroup, so rlimits only)."))
        except Exception as e:
            STATUS.say(f"  !! running Python is off: {e}")
    convo = Conversation() if ASK_ENABLED else None
    agent = LidaAgent(
        user_id=USER_ID,
        llm=anthropic_llm(),
        goals=load_live_goals() or load_goals(),
        values=load_values(),
        act=act,
        personality=load_personality(),
        abilities=" ".join(a for a, on in ((WEB_ABILITY, WEB_ENABLED),
                                           (FILES_ABILITY, FILES_ENABLED),
                                           (EXEC_ABILITY, sandbox is not None),
                                           (ASK_ABILITY, convo is not None),
                                           (GOALS_ABILITY, SELF_GOALS)) if on),
        files=files,
    )
    agent.sandbox, agent.conversation = sandbox, convo
    carry_out = motor(anthropic_llm(), agent)
    if DECIDER == "jev":
        try:
            agent.decider = Jev()
            STATUS.say("Judging actions with Jev (Claude as fallback).", "jev")
        except ValueError as e:
            STATUS.say(f"  !! LIDA_DECIDER=jev but {e}; judging with the LLM.")
    if agent.values:
        STATUS.say("Values:\n" + "\n".join(f"- {v}" for v in agent.values), "lida")
    STATUS.say(f"Goals:\n{numbered(agent.goals)}", "lida")
    agent.sense(load_prompt())
    STATUS.idle_info = lambda: (
        f"{len(agent.ws.nodes)} nodes in workspace, last broadcast "
        f"{time.time() - agent.last_broadcast:.0f}s ago, next action allowed in "
        f"{max(0.0, agent.action_interval - (time.time() - agent.last_action)):.0f}s")
    STATUS.start_heartbeat()

    STATUS.legend()
    STATUS.say("Running until Ctrl-C."
               + (f" Type a line and press Enter to talk to her as {USER_NAME}." if convo else ""))
    errors = 0
    try:
        while True:
            try:
                STATUS.tick()
                for msg in (convo.incoming() if convo else []):
                    STATUS.say(f"  ← {msg}", "talk")
                    agent.sense(msg)
                w = agent.step(dt=0.1)
                if w:
                    STATUS.say(f"[broadcast via {w.codelet.name} @ {w.activation:.2f}]\n{w.text()}", "lida")
                    if agent.pending:
                        worked, why = agent.appraise()
                        verdict = {True: "worked", False: "failed", None: "unjudged"}[worked]
                        STATUS.say(f"  ({agent.pending['mode']}) {verdict}: {why}", "act")
                        agent.feedback(worked, why)
                errors = 0
            except Exception as e:  # API/network hiccups shouldn't end the mind
                errors += 1
                agent.feedback(None)  # a stuck pending action would block all others
                backoff = min(60.0, 2.0 ** errors)
                STATUS.say(f"  !! {type(e).__name__}: {e}")
                with STATUS.phase(f"waiting {backoff:.0f}s before retrying", min_report=float("inf")):
                    time.sleep(backoff)
                STATUS.tick()
            time.sleep(0.1)
    except KeyboardInterrupt:
        STATUS.say("Stopping.")
    finally:
        try:
            agent.ltm.consolidate()
        except Exception as e:
            STATUS.say(f"  !! final consolidation failed: {e}")
        agent.ltm.close()
