"""SWE-EVO-flavoured trajectory extractor.

Why a new module?
-----------------
EvoClaw's ``OpenHandsTrajectoryExtractor`` loads a tree of ``event-*.json``
files tailed from a long-running container (``log_dir/openhands/**``).
In SWE-EVO, OpenHands runs one instance per container via
``evaluation/benchmarks/swe_bench/scripts/run_infer.sh`` and serialises the
full conversation as a **single line** of ``output.jsonl`` with a
``history`` list of raw event dicts.

The dataclasses (``TrajectoryEvent``, ``TrajectorySlice``,
``ValidationEpisode``, ``ValidationTest``, ``EditIntentContext``) and the
normalisation / episode-segmentation algorithms are copied **verbatim**
from ``EvoClaw/harness/e2e/patch_memory_trajectory.py`` so downstream
consumers (retriever, summariser, manager) work without any change.
Only ``load_conversation_events`` is replaced — it reads history lists
instead of globbing ``event-*.json`` files.

Two ingress paths are exposed:

- ``SweEvoTrajectoryExtractor(source_path=<output.jsonl>)``
    Filters ``output.jsonl`` entries by ``instance_id`` and yields the
    embedded ``history`` list.
- ``SweEvoTrajectoryExtractor.from_history(history=[...])``
    Direct use when the caller already holds a history list in memory
    (e.g. ``chain_runner`` after parsing one line).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


TEST_COMMAND_RE = re.compile(r"\b(pytest|python -m pytest|tox|nox|meson test|ctest)\b")
WRITE_COMMANDS = {"insert", "str_replace", "create", "delete"}


@dataclass
class TrajectoryEvent:
    ts: str
    kind: str
    summary: str
    tool_name: Optional[str] = None
    files: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    command: Optional[str] = None
    exit_code: Optional[int] = None
    is_error: bool = False
    output_excerpt: str = ""
    reasoning: str = ""
    raw_ref: Optional[str] = None


@dataclass
class TrajectorySlice:
    session_id: Optional[str]
    start_ts: Optional[str]
    end_ts: Optional[str]
    events: list[TrajectoryEvent] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    tests_run: list[str] = field(default_factory=list)


@dataclass
class ValidationTest:
    command: str
    exit_code: Optional[int]
    status: str
    output_excerpt: str


@dataclass
class ValidationEpisode:
    episode_id: str
    session_id: Optional[str]
    start_ts: Optional[str]
    end_ts: Optional[str]
    status: str
    files_changed: list[str] = field(default_factory=list)
    files_read: list[str] = field(default_factory=list)
    symbols_changed: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    tests: list[ValidationTest] = field(default_factory=list)
    tests_before: list[ValidationTest] = field(default_factory=list)
    reasoning_excerpts: list[str] = field(default_factory=list)


@dataclass
class EditIntentContext:
    active_files: list[str] = field(default_factory=list)
    active_features: list[str] = field(default_factory=list)
    active_symbols: list[str] = field(default_factory=list)
    recent_commands: list[str] = field(default_factory=list)
    recent_tests: list[str] = field(default_factory=list)


class SweEvoTrajectoryExtractor:
    """Extract trajectory/episodes from an OpenHands ``output.jsonl`` history list.

    Parameters
    ----------
    source_path:
        Optional path to an ``output.jsonl`` file.  If set, ``instance_id``
        must also be provided to ``load_conversation_events`` so we pick
        the right line.  When the chain runner already has the raw history
        in memory, ``source_path`` is left ``None`` and ``_history_cache``
        is populated via :meth:`set_history`.
    """

    def __init__(self, source_path: Optional[Path] = None):
        self.source_path = Path(source_path) if source_path else None
        self._history_cache: list[dict[str, Any]] = []

    @classmethod
    def from_history(cls, history: list[dict[str, Any]]) -> "SweEvoTrajectoryExtractor":
        extractor = cls()
        extractor.set_history(history)
        return extractor

    def set_history(self, history: list[dict[str, Any]]) -> None:
        """Cache an already-loaded history list."""
        self._history_cache = list(history or [])

    def _load_history_for_instance(self, instance_id: Optional[str]) -> list[dict[str, Any]]:
        if self._history_cache:
            return self._history_cache
        if not self.source_path or not self.source_path.exists():
            return []
        for line in self.source_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if instance_id and row.get("instance_id") != instance_id:
                continue
            history = row.get("history") or row.get("events") or []
            if isinstance(history, list):
                self._history_cache = history
                return history
            return []
        return []

    def load_conversation_events(
        self, session_id: Optional[str] = None
    ) -> list[TrajectoryEvent]:
        """Return a chronologically sorted list of normalised ``TrajectoryEvent``s.

        ``session_id`` is accepted for API parity with EvoClaw's extractor,
        but since ``output.jsonl`` is already scoped to one instance/session
        it doubles as the ``instance_id`` filter.  If ``set_history`` has
        been called the in-memory cache is used and ``session_id`` is a no-op.
        """
        history = self._load_history_for_instance(session_id)
        events: list[TrajectoryEvent] = []
        for idx, raw_event in enumerate(history):
            if not isinstance(raw_event, dict):
                continue
            raw_ref = f"history[{idx}]"
            event = self._normalize_event(raw_event, raw_ref)
            if event:
                events.append(event)
        return events

    def _extract_thought_text(self, raw_event: dict[str, Any]) -> str:
        thought = raw_event.get("thought")
        if not thought:
            return ""
        if isinstance(thought, str):
            return thought.strip()
        if isinstance(thought, list):
            parts: list[str] = []
            for block in thought:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text and isinstance(text, str):
                        parts.append(text.strip())
                elif isinstance(block, str):
                    parts.append(block.strip())
            return " ".join(p for p in parts if p)
        return ""

    def _normalize_event(
        self, raw_event: dict[str, Any], raw_ref: str
    ) -> Optional[TrajectoryEvent]:
        if not isinstance(raw_event, dict):
            return None

        ts = str(
            raw_event.get("timestamp")
            or raw_event.get("ts")
            or raw_event.get("created_at")
            or raw_event.get("time")
            or ""
        )
        source = str(raw_event.get("source") or "").lower()
        tool_name = str(raw_event.get("tool_name") or "").lower() or None
        content, action = self._extract_action_payload(raw_event)
        obs_command, exit_code, is_error, output_excerpt = self._extract_observation_payload(raw_event)

        reasoning = ""
        if source == "agent":
            reasoning = self._extract_thought_text(raw_event)

        kind = str(
            raw_event.get("type") or raw_event.get("event") or raw_event.get("kind") or "unknown"
        ).lower()
        summary = ""
        command = None
        files_read: list[str] = []
        files_written: list[str] = []

        if source == "agent" and tool_name == "file_editor":
            cmd = str(action.get("command") or "")
            path = str(action.get("path") or "")
            if cmd == "view":
                kind = "read"
                files_read = [path] if path else []
                summary = path
            elif cmd in WRITE_COMMANDS:
                kind = "write"
                files_written = [path] if path else []
                summary = path
            else:
                kind = cmd or "file_editor"
                summary = path or content
        elif source == "agent" and tool_name == "terminal":
            kind = "bash"
            command = str(action.get("command") or "")
            summary = command
        elif source == "environment" and tool_name == "terminal":
            kind = "terminal_result"
            command = obs_command
            summary = obs_command or output_excerpt
        elif source == "agent":
            kind = kind or "message"
            summary = content
        else:
            summary = content or output_excerpt

        files = sorted(set(files_read + files_written + self._extract_paths(raw_event)))
        if not files_read:
            files_read = [path for path in files if kind == "read"]
        if not files_written:
            files_written = [path for path in files if kind == "write"]

        summary = re.sub(r"\s+", " ", (summary or "")).strip()
        if not summary and not files and not command and not reasoning:
            return None

        return TrajectoryEvent(
            ts=ts,
            kind=kind or "unknown",
            summary=summary[:500],
            tool_name=tool_name,
            files=files,
            files_read=sorted(set(files_read)),
            files_written=sorted(set(files_written)),
            command=command,
            exit_code=exit_code,
            is_error=is_error,
            output_excerpt=output_excerpt[:2000],
            reasoning=reasoning[:1000],
            raw_ref=raw_ref,
        )

    def _extract_action_payload(self, raw_event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        action = raw_event.get("action")
        if isinstance(action, dict):
            content = str(
                raw_event.get("text")
                or raw_event.get("content")
                or action.get("path")
                or action.get("command")
                or ""
            )
            return content, action
        args = raw_event.get("args") if isinstance(raw_event.get("args"), dict) else {}
        content = str(raw_event.get("content") or raw_event.get("text") or "")
        return content, args

    def _extract_observation_payload(
        self, raw_event: dict[str, Any]
    ) -> tuple[Optional[str], Optional[int], bool, str]:
        observation = raw_event.get("observation")
        if not isinstance(observation, dict):
            return None, None, False, ""
        pieces = []
        for part in observation.get("content", []):
            if isinstance(part, dict):
                pieces.append(str(part.get("text") or ""))
        output_excerpt = "\n".join(piece for piece in pieces if piece)
        command = observation.get("command")
        exit_code = observation.get("exit_code")
        is_error = bool(observation.get("is_error")) or (exit_code not in (None, 0))
        return str(command) if command else None, exit_code, is_error, output_excerpt

    _PATH_RE = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")
    # Path prefixes that are never source code in the workdir. Any `_PATH_RE`
    # match starting with one of these is dropped before being considered as
    # a "file touched". The list is deliberately broad — it must catch the
    # apt-get install, dpkg, and system-Linux noise that bleeds into the
    # OpenHands terminal log when the agent provisions tools (e.g.
    # ``apt install r-base``).
    _NON_PATH_PREFIXES = (
        # version-control / forge URLs
        "github.com/",
        "gitlab.com/",
        "bitbucket.org/",
        # apt mirrors leaking from "Get:N http://archive.ubuntu.com/ubuntu …" lines
        "archive.ubuntu.com/",
        "security.ubuntu.com/",
        "ports.ubuntu.com/",
        "deb.debian.org/",
        # apt suite/component fragments left over after the URL is consumed
        # ("noble-updates/main", "jammy-security/universe", …)
        "noble-updates/",
        "noble-security/",
        "noble-backports/",
        "noble/",
        "jammy-updates/",
        "jammy-security/",
        "jammy-backports/",
        "jammy/",
        "focal-updates/",
        "focal-security/",
        "focal/",
        "bookworm-updates/",
        "bookworm-security/",
        "bookworm/",
        "bullseye-updates/",
        "bullseye/",
        "stable-updates/",
        "stable/",
        "testing/",
        "unstable/",
        # FHS / system trees that the agent never edits as project sources
        "names/",
        "include/",
        "usr/",
        "lib/",
        "lib64/",
        "lib32/",
        "libx32/",
        "etc/",
        "Etc/",  # Etc/UTC, Etc/GMT — timezone DB
        "var/",
        "opt/",
        "proc/",
        "sys/",
        "dev/",
        "boot/",
        "mnt/",
        "media/",
        "srv/",
        "run/",
        "root/",
        "tmp/",
        "bin/",
        "sbin/",
        # our own scratch git: we run ``git init`` to capture diffs so the
        # ``.git/...`` internals end up in stdout — never source.
        ".git/",
        # OpenHands runtime/install scratch trees. The bare ``sessions/``
        # prefix is for ``sessions/<uuid>/TASKS.md`` which OpenHands prints
        # to stdout when it persists its task list; it isn't project source.
        ".openhands/",
        ".openhands-state/",
        "sessions/",
        # truncated paths from terminal output (apt logs print "...NN-pkg.deb")
        ".../",
        "..../",
    )
    # Filenames whose final component matches one of these extensions are
    # binary / archive / compiled artefacts, never source. Apt downloads
    # populate this list heavily (``r-base_4.3.3-2build2_all.deb``).
    _NOISE_TAIL_EXT_RE = re.compile(
        r"\.(?:"
        r"deb|udeb|rpm|apk|snap|flatpak|"
        r"whl|egg|jar|war|ear|nupkg|gem|crate|"
        r"tar|tar\.gz|tgz|tar\.bz2|tbz2|tar\.xz|txz|tar\.zst|"
        r"gz|bz2|xz|zst|zip|rar|7z|lz|lzma|"
        r"so|so\.[0-9.]+|dylib|dll|"
        r"o|obj|a|lib|exe|elf|bin|"
        r"pyc|pyo|pyd|class|"
        r"jpg|jpeg|png|gif|bmp|ico|tiff|webp|svg|"
        r"mp3|mp4|mov|avi|wav|flac|ogg|webm|mkv|"
        r"woff|woff2|ttf|otf|eot|"
        r"pack|idx|pdf"
        r")$",
        re.IGNORECASE,
    )
    # Segment that looks like a hostname (foo.com, archive.ubuntu.com, …).
    # Lets us drop bare-URL tails the regex picks up, e.g.
    # ``archive.ubuntu.com/ubuntu``.
    _DOMAIN_LIKE_SEGMENT_RE = re.compile(
        r"\b[A-Za-z0-9-]+"
        r"(?:\.[A-Za-z0-9-]+)+"
        r"\.(?:com|org|net|io|dev|gov|edu|co|ai|xyz|info|us|uk|cn|de|fr|"
        r"ru|jp|tv|me|sh|ly|to|app|cloud|page|run|ubuntu|debian|fedora)"
        r"(?:/|$)"
    )
    _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b[?2004[lh]?")

    @classmethod
    def _is_noisy_path(cls, candidate: str) -> bool:
        """Return ``True`` when ``candidate`` should not be treated as a source file.

        Filters out apt download lines, system / FHS trees, hostname-shaped
        fragments, ``.../`` truncated paths, and binary/archive artefacts.
        Anything that survives is at least *plausibly* a project-source
        path and gets fed onward to the summariser.
        """
        if not candidate:
            return True
        if candidate.startswith(("http://", "https://", "ftp://", "git@")):
            return True
        # Strip a single leading slash so prefixes like "/etc/foo" match
        # the relative-path entries in `_NON_PATH_PREFIXES`.
        normalized = candidate.lstrip("/")
        if any(normalized.startswith(prefix) for prefix in cls._NON_PATH_PREFIXES):
            return True
        if cls._DOMAIN_LIKE_SEGMENT_RE.search(candidate):
            return True
        tail = normalized.rsplit("/", 1)[-1]
        if cls._NOISE_TAIL_EXT_RE.search(tail):
            return True
        # Drop pure version/numeric tails ("3.5-8-1_amd64", etc.) that the
        # `_PATH_RE` happily yields when apt prints architecture metadata.
        if re.fullmatch(r"[0-9._+-]+", tail):
            return True
        # Drop short, extension-less, shallow paths like ``kB/s`` and
        # ``files/directories`` that come from running prose in apt
        # progress lines. Real source files either have an extension or
        # live more than two segments deep (``src/foo/bar``).
        segments = [s for s in normalized.split("/") if s]
        if "." not in tail and len(segments) < 3:
            return True
        return False

    def _extract_paths(self, raw_event: dict[str, Any]) -> list[str]:
        paths: set[str] = set()

        def maybe_add(value: object) -> None:
            if not isinstance(value, str):
                return
            normalized = value.strip()
            if not normalized or normalized.startswith("http"):
                return
            if "/" not in normalized:
                return
            if self._is_noisy_path(normalized):
                return
            paths.add(normalized)

        for key in ("path", "file", "filepath"):
            maybe_add(raw_event.get(key))

        action = raw_event.get("action")
        if isinstance(action, dict):
            for key in ("path", "file", "filepath"):
                maybe_add(action.get(key))

        observation = raw_event.get("observation")
        if isinstance(observation, dict):
            for part in observation.get("content", []):
                if isinstance(part, dict):
                    text = part.get("text", "") or ""
                    clean = self._ANSI_RE.sub("", text)
                    for match in self._PATH_RE.findall(clean):
                        if not self._is_noisy_path(match):
                            paths.add(match)

        for blob_key in ("content", "text"):
            blob = raw_event.get(blob_key)
            if not isinstance(blob, str):
                continue
            for match in self._PATH_RE.findall(blob):
                if not self._is_noisy_path(match):
                    paths.add(match)

        return sorted(paths)

    def _is_test_command(self, command: str) -> bool:
        return bool(command and TEST_COMMAND_RE.search(command))

    def _classify_test_status(
        self, exit_code: Optional[int], is_error: bool, output: str
    ) -> str:
        low = output.lower()
        if is_error or exit_code not in (None, 0):
            return "failed"
        if any(token in low for token in (" failed", "failures", "assertionerror", "error:")):
            return "failed"
        return "passed"

    def _extract_symbols_from_paths(self, paths: list[str]) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        for path in paths:
            for token in re.split(r"[/_.-]+", path):
                token = token.strip().lower()
                if len(token) < 3 or token in seen:
                    continue
                seen.add(token)
                symbols.append(token)
        return symbols[:24]

    @staticmethod
    def _strip_container_prefix(path: str) -> str:
        # ``/app`` is Terminal-shift's WORKDIR; ``/testbed`` is SWE-bench's;
        # ``/workspace`` is OpenHands' default sandbox mount; ``/repo`` is
        # used by some custom Harbor agents. Stripping these makes paths
        # collide cleanly across runs of the same chain.
        for prefix in ("/testbed/", "/workspace/", "/repo/", "/app/"):
            if path.startswith(prefix):
                return path[len(prefix):]
        for prefix in ("testbed/", "app/"):
            if path.startswith(prefix):
                return path[len(prefix):]
        return path

    def get_recent_slice_for_submission(
        self,
        session_id: Optional[str],
        changed_files: list[str],
        tag_hash: Optional[str],
    ) -> TrajectorySlice:
        events = self.load_conversation_events(session_id=session_id)
        if not events:
            return TrajectorySlice(session_id=session_id, start_ts=None, end_ts=None)

        normalized_changed = {self._strip_container_prefix(f) for f in changed_files}

        def _event_paths_norm(event: TrajectoryEvent) -> set[str]:
            return {
                self._strip_container_prefix(p)
                for p in set(event.files) | set(event.files_written) | set(event.files_read)
            }

        relevant = [
            event
            for event in events
            if normalized_changed.intersection(_event_paths_norm(event))
        ]

        if len(relevant) < 8:
            relevant = events[-20:]
        else:
            relevant = relevant[-20:]

        commands = [event.command for event in relevant if event.command]
        tests_run = [command for command in commands if self._is_test_command(command)]
        files_touched = sorted(
            {self._strip_container_prefix(path) for event in relevant for path in event.files}
        )
        files_read = sorted(
            {self._strip_container_prefix(path) for event in relevant for path in event.files_read}
        )

        return TrajectorySlice(
            session_id=session_id,
            start_ts=relevant[0].ts if relevant else None,
            end_ts=relevant[-1].ts if relevant else None,
            events=relevant,
            files_touched=files_touched,
            files_read=files_read,
            commands=commands[-10:],
            tests_run=tests_run[-10:],
        )

    def extract_validation_episodes(
        self, session_id: Optional[str]
    ) -> list[ValidationEpisode]:
        events = self.load_conversation_events(session_id=session_id)
        episodes: list[ValidationEpisode] = []
        pending_writes: list[TrajectoryEvent] = []
        pending_reads: list[TrajectoryEvent] = []
        pending_commands: list[str] = []
        pending_reasoning: list[str] = []
        last_test_before: Optional[ValidationTest] = None

        for idx, event in enumerate(events):
            if event.reasoning and event.kind not in {"terminal_result"}:
                pending_reasoning.append(event.reasoning)

            if event.kind == "read":
                pending_reads.append(event)
                continue
            if event.kind == "write":
                pending_writes.append(event)
                pending_commands.append(event.summary)
                continue
            if event.kind == "bash" and event.command:
                pending_commands.append(event.command)
                if not self._is_test_command(event.command):
                    continue
                result = None
                for follow in events[idx + 1 : idx + 6]:
                    if follow.kind == "terminal_result" and follow.command == event.command:
                        result = follow
                        break
                status = self._classify_test_status(
                    result.exit_code if result else None,
                    result.is_error if result else False,
                    result.output_excerpt if result else "",
                )
                files_changed = sorted(
                    {
                        self._strip_container_prefix(path)
                        for write in pending_writes
                        for path in write.files_written or write.files
                    }
                )
                if not files_changed:
                    current_test = ValidationTest(
                        command=event.command,
                        exit_code=result.exit_code if result else None,
                        status=status,
                        output_excerpt=(result.output_excerpt if result else "")[:1200],
                    )
                    last_test_before = current_test
                    pending_writes = []
                    pending_reads = []
                    pending_commands = []
                    pending_reasoning = []
                    continue

                current_test = ValidationTest(
                    command=event.command,
                    exit_code=result.exit_code if result else None,
                    status=status,
                    output_excerpt=(result.output_excerpt if result else "")[:1200],
                )
                episode = ValidationEpisode(
                    episode_id=f"{session_id or 'nosession'}-{idx}",
                    session_id=session_id,
                    start_ts=pending_writes[0].ts if pending_writes else event.ts,
                    end_ts=(result.ts if result else event.ts),
                    status=status,
                    files_changed=files_changed,
                    files_read=sorted(
                        {
                            self._strip_container_prefix(path)
                            for read in pending_reads
                            for path in read.files_read or read.files
                        }
                    ),
                    symbols_changed=self._extract_symbols_from_paths(files_changed),
                    commands=pending_commands[-12:],
                    tests=[current_test],
                    tests_before=[last_test_before] if last_test_before else [],
                    reasoning_excerpts=[r for r in pending_reasoning[-6:] if r],
                )
                episodes.append(episode)
                last_test_before = current_test
                pending_writes = []
                pending_reads = []
                pending_commands = []
                pending_reasoning = []
                continue
            if event.kind not in {"message", "think"}:
                pending_commands = []

        return episodes

    def infer_edit_intent_context(
        self, session_id: Optional[str], runnable: list[str]
    ) -> EditIntentContext:
        events = self.load_conversation_events(session_id=session_id)
        recent = events[-40:]
        seen: set[str] = set()
        active_files: list[str] = []
        for event in reversed(recent):
            candidates = event.files_written or event.files_read or event.files
            for path in candidates:
                normalized = self._strip_container_prefix(path)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    active_files.append(normalized)
            if len(active_files) >= 15:
                break
        active_files = list(reversed(active_files[-15:]))
        recent_commands = [event.command for event in recent if event.command][-12:]
        recent_tests = [command for command in recent_commands if self._is_test_command(command)]
        active_symbols = self._extract_symbols_from_paths(active_files)
        active_features = active_symbols[:12]
        for token in runnable[:3]:
            normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", token).strip("-").lower()
            if normalized and normalized not in active_features:
                active_features.append(normalized)
        return EditIntentContext(
            active_files=active_files,
            active_features=active_features[:12],
            active_symbols=active_symbols[:20],
            recent_commands=recent_commands,
            recent_tests=recent_tests,
        )

    def infer_refine_recover_context(
        self, session_id: Optional[str], runnable: list[str]
    ) -> EditIntentContext:
        episodes = self.extract_validation_episodes(session_id=session_id)
        failed = [episode for episode in episodes if episode.status == "failed"]
        if failed:
            recent = failed[-1]
            return EditIntentContext(
                active_files=recent.files_changed,
                active_features=recent.symbols_changed[:12],
                active_symbols=recent.symbols_changed[:20],
                recent_commands=recent.commands[-12:],
                recent_tests=[test.command for test in recent.tests],
            )
        return self.infer_edit_intent_context(session_id=session_id, runnable=runnable)

    def infer_active_context_for_recover(
        self, session_id: Optional[str], runnable: list[str]
    ) -> EditIntentContext:
        return self.infer_refine_recover_context(session_id=session_id, runnable=runnable)
