"""Patch-memory manager for SWE-EVO.

Adapted from ``EvoClaw/harness/e2e/patch_memory.py`` with two minimal
mechanical changes:

1.  **No docker cp.**  EvoClaw keeps a long-running container and pushes
    ``/e2e_workspace/patch_memory`` into it after every write.  SWE-EVO
    starts/kills one container per instance (owned by OpenHands'
    CodeActAgent), so there is no persistent container to push into.
    :meth:`sync_host_memory_view` replaces :meth:`sync_container_memory_view`:
    it writes ``rendered/current_context.md`` + ``rendered/recover_context.md``
    on the host only.  Delivery to the agent is handled by
    :mod:`problem_injector` which prepends the rendered markdown to the
    next instance's ``problem_statement``.

2.  **Trajectory source swap.**  The manager holds a
    :class:`SweEvoTrajectoryExtractor` pre-loaded with the current
    instance's history list.  Callers must populate it per instance
    (``manager.load_instance_history(history)``) before invoking
    :meth:`store_submission_feature_patches`.

3.  **Generic module-path extraction.**  The original
    ``render_context_from_srs`` hard-coded ``sklearn`` regex; SWE-EVO
    spans dozens of repos.  We replace that with a generic dotted/slash
    module heuristic driven by a configurable ``repo_name``.

Record schemas, retrieval logic, summariser prompts and indexes are
preserved byte-for-byte so records produced here can in principle be
merged with records produced by the EvoClaw reference implementation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .retriever import PatchMemoryRetriever
from .summarizer import PatchMemorySummarizer
from .trajectory_extractor import (
    EditIntentContext,
    SweEvoTrajectoryExtractor,
    TrajectorySlice,
    ValidationEpisode,
)


@dataclass
class FeaturePatchRecord:
    record_id: str
    milestone_id: str
    attempt: int
    feature_id: str
    feature_title: str
    source_stage: str
    status: str
    created_at: str
    updated_at: str
    source_tag: str
    tag_hash: Optional[str]
    files_changed: list[str] = field(default_factory=list)
    symbols_changed: list[str] = field(default_factory=list)
    feature_tags: list[str] = field(default_factory=list)
    code_change: dict[str, Any] = field(default_factory=dict)
    behavior: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    trajectory: dict[str, Any] = field(default_factory=dict)
    relations: dict[str, Any] = field(default_factory=dict)


@dataclass
class MilestonePatchBundle:
    bundle_id: str
    milestone_id: str
    attempt: int
    source_tag: str
    tag_hash: Optional[str]
    feature_patch_ids: list[str]
    created_at: str


# ---------------------------------------------------------------------------
# Module-path heuristic (replaces sklearn-only regex in the EvoClaw original)
# ---------------------------------------------------------------------------


# URL / markdown link / badge / anchor / HTML-tag patterns we strip *before*
# fishing for module paths.  Without this, SWE-EVO problem_statements (which
# are basically release-notes full of ``github.com/.../pull/NNNN`` links)
# leak fragments like ``com/aws``, ``lint/pull`` into the active_files query,
# which then show up in the agent-facing "Potentially affected files" block.
_URL_RE = re.compile(
    r"""(?xi)
    \b
    (?:https?://|www\.)           # scheme or bare www.
    [^\s)>\]\"'`]+                # anything up to whitespace or closing bracket/quote
    """,
)
_MD_LINK_RE = re.compile(r"\[[^\]]{0,200}\]\([^)]{0,400}\)")
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]{0,300}>")
# Known "URL fragment" roots that the slash regex would otherwise happily
# serve up as module paths (TLDs, docs hosts, package registries, etc.).
_URL_FRAGMENT_PREFIXES = frozenset({
    "com", "org", "io", "net", "edu", "gov", "dev", "co",
    "github", "gitlab", "bitbucket",
    "docs", "api", "www",
    "pypistats", "pepy", "packagecontrol", "marketplace",
    "codecov", "discordapp",
})
# TLDs that, if they appear as *any* segment of a dotted match, reveal
# the match as a hostname rather than a real module path. Rejects e.g.
# ``archive.ubuntu.com`` from getting promoted to ``archive/ubuntu/com.py``.
_URL_FRAGMENT_TLDS = frozenset({
    "com", "org", "net", "io", "dev", "gov", "edu", "co", "ai",
    "xyz", "info", "us", "uk", "cn", "de", "fr", "ru", "jp", "tv",
    "me", "sh", "ly", "to", "app", "cloud", "page", "run",
})

# Recognised source-file extensions across the repos Harbor + SWE-EVO target.
# Used to gate slash-form matches so we only treat ``app/ars.R`` as a path
# (and not ``noble-updates/main`` from an apt log fragment), and to know
# when the caller already supplied an extension we should preserve verbatim
# instead of overwriting with ``.py`` (the historic SWE-EVO Python-only
# default that broke Terminal-shift R/text tasks).
_SOURCE_EXT_RE = re.compile(
    r"\.(?:"
    r"py|pyx|pyi|ipynb|"
    r"r|rmd|"
    r"md|rst|txt|csv|tsv|json|jsonl|ya?ml|toml|cfg|ini|env|"
    r"sh|bash|zsh|fish|ps1|"
    r"sql|"
    r"rs|go|"
    r"ts|tsx|js|jsx|mjs|cjs|"
    r"c|cc|cpp|cxx|h|hh|hpp|hxx|"
    r"java|kt|kts|scala|swift|m|mm|"
    r"rb|pl|php|lua|hs|ml|fs|fsx|nim|zig|cu|cuh|"
    r"tex|"
    r"html|htm|css|scss|sass|less|vue|svelte|astro|"
    r"tf|hcl|nix|"
    r"dockerfile|makefile|gradle|cmake|"
    r"proto|thrift|graphql|gql"
    r")$",
    re.IGNORECASE,
)
# Tail extensions that mean "binary / archive / compiled artefact" and so
# *cannot* be source even if the surrounding regex matched.
_NON_SOURCE_EXT_RE = re.compile(
    r"\.(?:"
    r"deb|udeb|rpm|apk|snap|whl|egg|jar|war|ear|nupkg|gem|crate|"
    r"tar|tgz|tbz2|txz|gz|bz2|xz|zst|zip|rar|7z|"
    r"so|so\.[0-9.]+|dylib|dll|o|obj|a|lib|exe|elf|bin|"
    r"pyc|pyo|pyd|class|"
    r"jpg|jpeg|png|gif|bmp|ico|tiff|webp|svg|"
    r"mp3|mp4|mov|avi|wav|flac|ogg|webm|mkv|"
    r"woff|woff2|ttf|otf|eot|pdf"
    r")$",
    re.IGNORECASE,
)


def _has_source_extension(path: str) -> bool:
    return bool(_SOURCE_EXT_RE.search(path))


def _has_noise_extension(path: str) -> bool:
    return bool(_NON_SOURCE_EXT_RE.search(path))


def _extract_module_paths_from_text(text: str, repo_hint: Optional[str] = None) -> list[str]:
    """Extract plausible module paths from free-text SRS / problem statement.

    Accepts both ``foo.bar.baz`` dotted forms and ``foo/bar/baz.py`` slash
    forms.  When ``repo_hint`` is provided (e.g. ``"sklearn"`` or ``"numpy"``)
    we prefer matches starting with that root, but we still emit everything
    else we see as a fallback.

    Pre-strips URLs / markdown links / HTML tags and rejects matches whose
    leading segment is a known URL/hosting-domain root — otherwise release-
    notes style text was leaking fragments like ``com/aws`` or ``lint/pull``.

    Multi-language fix
    ------------------
    The original SWE-EVO heuristic appended ``.py`` to every slash match
    and dotted match because it only ever ran against Python repos. For
    Harbor + Terminal-shift the same code is asked to scope retrieval over
    R, text, CSV, etc. — so we now:

    * accept any extension allowed by ``_SOURCE_EXT_RE`` and **preserve it
      verbatim** (no ``.py`` overwrite),
    * drop slash matches with a non-source/binary extension
      (``r-base_4.3.3.deb``, ``foo.tar.gz``, …),
    * drop slash matches that have *no* extension (``noble-updates/main``,
      ``foo/bar/baz``) — these are almost always URL fragments rather
      than real files. The previous code happily promoted them to
      ``noble-updates/main.py`` and pasted that into the agent's prompt.

    Dotted-form matches retain the ``.py`` append behaviour, since the
    only common source where ``foo.bar.baz`` denotes a code path are
    Python module references in bug reports / SRS prose.
    """
    cleaned = _URL_RE.sub(" ", text)
    cleaned = _MD_LINK_RE.sub(" ", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)

    seen: set[str] = set()
    out: list[str] = []
    # Allow dashes and dots inside non-leading segments so ``app/ars.R`` and
    # ``foo/bar-baz.tsx`` match. The leading segment stays alphanumeric/
    # underscore-only to avoid catching apt-style ``noble-updates/main``
    # by way of a ``-`` in the leading segment.
    slash_re = re.compile(
        r"[A-Za-z_][A-Za-z0-9_]*"
        r"(?:/[A-Za-z_][A-Za-z0-9_-]*)+"
        r"(?:\.[A-Za-z][A-Za-z0-9]{0,7})?"
    )
    dotted_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){2,}")

    for match in slash_re.findall(cleaned):
        head = match.split("/", 1)[0].lower()
        if head in _URL_FRAGMENT_PREFIXES:
            continue
        if _has_noise_extension(match):
            continue
        # Require an extension that we recognise as source. Anything else
        # (no extension, or extension we don't know) gets dropped — that
        # used to be the source of ``noble-updates/main.py`` style noise.
        if not _has_source_extension(match):
            continue
        path = match
        if path not in seen:
            seen.add(path)
            out.append(path)

    for match in dotted_re.findall(cleaned):
        # Skip things like "e.g." or obvious non-modules
        segments_lower = [seg.lower() for seg in match.split(".")]
        if len(segments_lower) < 3:
            continue
        if segments_lower[0] in _URL_FRAGMENT_PREFIXES:
            continue
        # If any segment looks like a TLD, this is almost certainly a
        # hostname (``archive.ubuntu.com``, ``raw.githubusercontent.com``)
        # rather than a real Python module path. Drop it.
        if any(seg in _URL_FRAGMENT_TLDS for seg in segments_lower):
            continue
        # ``foo.bar.baz`` in prose is overwhelmingly a Python module path.
        # Append ``.py`` so we hit the indexed Python files; for non-Python
        # repos this still tokenises into the right query terms because
        # the retriever runs path-segmentation, not strict equality.
        path = match.replace(".", "/") + ".py"
        if path not in seen:
            seen.add(path)
            out.append(path)

    if repo_hint:
        # Boost matches that mention the repo hint to front of list
        hint = repo_hint.lower()
        out.sort(key=lambda p: 0 if hint in p.lower() else 1)
    return out


class PatchMemoryManager:
    """Host-only patch-memory coordinator.

    Parameters
    ----------
    trial_root:
        Chain-level trial directory.  Memory state lives under
        ``<trial_root>/patch_memory/``.
    repo_name:
        Short repo identifier (e.g. ``"scikit-learn"``).  Used for logging
        and as an SRS module-extraction hint.
    model / api_key / base_url:
        Optional LLM settings forwarded to :class:`PatchMemorySummarizer`.
        If absent, the fallback summary path is used.
    log_dir:
        Unused; kept for signature parity with EvoClaw.
    """

    def __init__(
        self,
        trial_root: Path,
        repo_name: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ):
        self.trial_root = Path(trial_root)
        self.repo_name = repo_name
        self.log_dir = Path(log_dir) if log_dir else self.trial_root / "log"
        self.root = self.trial_root / "patch_memory"
        self.summarizer = PatchMemorySummarizer(model=model, api_key=api_key, base_url=base_url)
        self.trajectory_extractor = SweEvoTrajectoryExtractor()
        for directory in (self._records_dir(), self._bundles_dir(), self._indexes_dir(), self._rendered_dir()):
            directory.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # History ingress (per instance) — SWE-EVO specific entry point
    # -----------------------------------------------------------------

    def load_instance_history(self, history: list[dict[str, Any]]) -> None:
        """Point the trajectory extractor at the history of one instance."""
        self.trajectory_extractor.set_history(history or [])

    # -----------------------------------------------------------------
    # Path helpers
    # -----------------------------------------------------------------

    def _records_dir(self) -> Path:
        return self.root / "records"

    def _bundles_dir(self) -> Path:
        return self.root / "bundles"

    def _indexes_dir(self) -> Path:
        return self.root / "indexes"

    def _rendered_dir(self) -> Path:
        return self.root / "rendered"

    def _record_path(self, record_id: str) -> Path:
        return self._records_dir() / f"{record_id}.json"

    def _bundle_path(self, bundle_id: str) -> Path:
        return self._bundles_dir() / f"{bundle_id}.json"

    def _episodes_index_path(self) -> Path:
        return self._indexes_dir() / "episode_index.json"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def _load_json(self, path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write_json(self, path: Path, payload: dict | list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_record(self, record: FeaturePatchRecord) -> None:
        self._write_json(self._record_path(record.record_id), asdict(record))

    def _save_bundle(self, bundle: MilestonePatchBundle) -> None:
        self._write_json(self._bundle_path(bundle.bundle_id), asdict(bundle))

    def _load_record(self, record_id: str) -> Optional[FeaturePatchRecord]:
        payload = self._load_json(self._record_path(record_id), None)
        if not isinstance(payload, dict):
            return None
        if "intent" in payload and "behavior" not in payload:
            payload["behavior"] = payload.pop("intent")
        if "evaluation" in payload and "validation" not in payload:
            payload["validation"] = payload.pop("evaluation")
        payload.setdefault("code_change", {})
        payload.setdefault("source_stage", "submission")
        try:
            return FeaturePatchRecord(**payload)
        except TypeError:
            return None

    def _bundle_id(self, milestone_id: str, attempt: int, tag_hash: Optional[str]) -> str:
        suffix = tag_hash[:12] if tag_hash else "nohash"
        return f"{milestone_id}-attempt{attempt}-{suffix}"

    def _load_bundle_for_attempt(
        self, milestone_id: str, attempt: int
    ) -> Optional[MilestonePatchBundle]:
        prefix = f"{milestone_id}-attempt{attempt}-"
        for bundle_file in sorted(self._bundles_dir().glob(f"{prefix}*.json"), reverse=True):
            payload = self._load_json(bundle_file, None)
            if isinstance(payload, dict):
                return MilestonePatchBundle(**payload)
        return None

    def _seen_episode_ids(self) -> set[str]:
        payload = self._load_json(self._episodes_index_path(), [])
        if not isinstance(payload, list):
            return set()
        return {str(value) for value in payload}

    def _mark_episode_seen(self, episode_id: str) -> None:
        seen = self._seen_episode_ids()
        seen.add(episode_id)
        self._write_json(self._episodes_index_path(), sorted(seen))

    # -----------------------------------------------------------------
    # Normalisation
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize_path(path: str) -> str:
        for prefix in ("/testbed/", "/workspace/", "/repo/"):
            if path.startswith(prefix):
                return path[len(prefix):]
        return path

    def _normalize_paths(self, paths: list[str]) -> set[str]:
        return {self._normalize_path(p) for p in paths}

    # -----------------------------------------------------------------
    # Record building
    # -----------------------------------------------------------------

    def store_submission_feature_patches(
        self,
        milestone_id: str,
        attempt: int,
        agent_tag: str,
        tag_hash: Optional[str],
        changed_files: list[str],
        diff_text: str,
        srs_text: Optional[str],
        session_id: Optional[str] = None,
        result_dir: Optional[Path] = None,
    ) -> list[FeaturePatchRecord]:
        """Persist feature-patch records for one instance submission.

        Mirrors the signature of EvoClaw's method; ``result_dir`` is
        accepted for API parity but not required (no per-milestone on-disk
        result yet — caller may pass the chain trial dir).
        """
        trajectory_slice: Optional[TrajectorySlice] = None
        if session_id is not None:
            trajectory_slice = self.trajectory_extractor.get_recent_slice_for_submission(
                session_id=session_id,
                changed_files=changed_files,
                tag_hash=tag_hash,
            )

        all_episodes = self.trajectory_extractor.extract_validation_episodes(session_id=session_id)
        changed_set_norm = self._normalize_paths(changed_files)
        for ep in all_episodes:
            ep.files_changed = [self._normalize_path(f) for f in ep.files_changed]
            ep.files_read = [self._normalize_path(f) for f in ep.files_read]
        episodes = [
            ep for ep in all_episodes if set(ep.files_changed).intersection(changed_set_norm)
        ]

        feature_groups = self.summarizer.suggest_feature_segmentation(
            milestone_id=milestone_id,
            changed_files=changed_files,
            diff_text=diff_text,
            trajectory_slice=trajectory_slice,
            episodes=episodes or [],
        )

        # Trajectory-only fallback: if normal path / episode-based
        # segmentation produced nothing (e.g. Terminal-shift tasks where
        # the agent's outputs aren't visible to git), synthesise a single
        # milestone-wide feature group seeded from the trajectory so the
        # summariser still produces a descriptive record.
        if not feature_groups and trajectory_slice and (
            trajectory_slice.events
            or trajectory_slice.commands
            or trajectory_slice.files_touched
        ):
            traj_files = list(dict.fromkeys(
                (trajectory_slice.files_touched or [])
                + (trajectory_slice.files_read or [])
            ))
            reasoning = " ".join(
                e.reasoning for e in trajectory_slice.events
                if getattr(e, "reasoning", None)
            )[:1500]
            feature_groups = [
                {
                    "feature_id": f"{milestone_id.lower()}-feature-01",
                    "feature_title": milestone_id,
                    "files_changed": traj_files,
                    "diff_excerpt": "",
                    "trajectory_excerpt": (
                        f"commands: {trajectory_slice.commands[-12:]}\n"
                        f"tests: {trajectory_slice.tests_run[-6:]}\n"
                        f"files_touched: {traj_files[:20]}"
                    ),
                    "reasoning_excerpt": reasoning,
                    "episode_status": "unknown",
                    "tests_before": [],
                    "tests_after": [],
                }
            ]

        records: list[FeaturePatchRecord] = []

        for group in feature_groups:
            related_prior = self._find_related_prior_records(group.get("files_changed", []))
            summary = self.summarizer.summarize_feature_patch(
                milestone_id=milestone_id,
                srs_text=srs_text,
                feature_group=group,
                trajectory_slice=trajectory_slice,
                related_prior_records=related_prior,
            )
            record = self._build_record(
                milestone_id=milestone_id,
                attempt=attempt,
                agent_tag=agent_tag,
                tag_hash=tag_hash,
                feature_group=group,
                summary=summary,
                diff_text=diff_text,
                source_stage="submission",
                status="submitted",
                trajectory_slice=trajectory_slice,
                episode=None,
            )
            self._save_record(record)
            records.append(record)

        bundle = MilestonePatchBundle(
            bundle_id=self._bundle_id(milestone_id, attempt, tag_hash),
            milestone_id=milestone_id,
            attempt=attempt,
            source_tag=agent_tag,
            tag_hash=tag_hash,
            feature_patch_ids=[record.record_id for record in records],
            created_at=self._now_iso(),
        )
        self._save_bundle(bundle)
        self._update_indexes(records)
        self.render_memory_readme()
        return records

    def store_new_validation_episodes(
        self,
        session_id: Optional[str],
        milestone_id: str = "in_session",
        attempt: int = 0,
        diff_text: str = "",
    ) -> list[FeaturePatchRecord]:
        """Harvest new validation episodes from the current history.

        Differs from the EvoClaw version only in that we take ``diff_text``
        from the caller rather than reading a live git worktree inside a
        container.  Pass an empty string when the caller does not yet have
        a diff (records will still carry trajectory/test evidence, just no
        code hunks).
        """
        seen = self._seen_episode_ids()
        stored: list[FeaturePatchRecord] = []
        for episode in self.trajectory_extractor.extract_validation_episodes(session_id=session_id):
            if episode.episode_id in seen:
                continue
            episode.files_changed = [self._normalize_path(f) for f in episode.files_changed]
            episode.files_read = [self._normalize_path(f) for f in episode.files_read]
            if episode.status == "passed":
                stored.extend(
                    self._store_episode_records(
                        milestone_id=milestone_id,
                        attempt=attempt,
                        episode=episode,
                        agent_tag="",
                        tag_hash=None,
                        diff_text=diff_text,
                        source_stage="local_validation",
                        status="working",
                    )
                )
            else:
                stored.extend(
                    self._store_episode_records(
                        milestone_id=milestone_id,
                        attempt=attempt,
                        episode=episode,
                        agent_tag="",
                        tag_hash=None,
                        diff_text=diff_text,
                        source_stage="local_failure",
                        status="failing",
                    )
                )
            self._mark_episode_seen(episode.episode_id)
        return stored

    def _store_episode_records(
        self,
        milestone_id: str,
        attempt: int,
        episode: ValidationEpisode,
        agent_tag: str,
        tag_hash: Optional[str],
        diff_text: str,
        source_stage: str,
        status: str,
    ) -> list[FeaturePatchRecord]:
        feature_groups = self.summarizer.suggest_feature_segmentation(
            milestone_id=milestone_id,
            changed_files=episode.files_changed,
            diff_text=diff_text,
            trajectory_slice=None,
            episodes=[episode],
        )
        records: list[FeaturePatchRecord] = []
        for group in feature_groups:
            related_prior = self._find_related_prior_records(group.get("files_changed", []))
            summary = self.summarizer.summarize_feature_patch(
                milestone_id=milestone_id,
                srs_text=None,
                feature_group=group,
                trajectory_slice=None,
                related_prior_records=related_prior,
            )
            record = self._build_record(
                milestone_id=milestone_id,
                attempt=attempt,
                agent_tag=agent_tag,
                tag_hash=tag_hash,
                feature_group=group,
                summary=summary,
                diff_text=diff_text,
                source_stage=source_stage,
                status=status,
                trajectory_slice=None,
                episode=episode,
            )
            self._save_record(record)
            records.append(record)
        self._update_indexes(records)
        self.render_memory_readme()
        return records

    def _build_record(
        self,
        milestone_id: str,
        attempt: int,
        agent_tag: str,
        tag_hash: Optional[str],
        feature_group: dict[str, Any],
        summary: dict[str, Any],
        diff_text: str,
        source_stage: str,
        status: str,
        trajectory_slice: Optional[TrajectorySlice],
        episode: Optional[ValidationEpisode],
    ) -> FeaturePatchRecord:
        now = self._now_iso()
        record_id = (
            f"{feature_group['feature_id']}-{source_stage}-{(tag_hash or 'nohash')[:12]}"
        )
        return FeaturePatchRecord(
            record_id=record_id,
            milestone_id=milestone_id,
            attempt=attempt,
            feature_id=feature_group["feature_id"],
            feature_title=summary.get("feature_title")
            or feature_group.get("feature_title")
            or feature_group["feature_id"],
            source_stage=source_stage,
            status=status,
            created_at=now,
            updated_at=now,
            source_tag=agent_tag,
            tag_hash=tag_hash,
            files_changed=feature_group.get("files_changed", []),
            symbols_changed=summary.get("symbols_changed", []),
            feature_tags=summary.get("feature_tags", []),
            code_change=self._build_code_change_payload(diff_text, feature_group.get("files_changed", [])),
            behavior=self._build_behavior_payload(summary),
            validation=self._build_validation_payload(episode, feature_group),
            trajectory=self._build_trajectory_payload(trajectory_slice, episode, feature_group),
            relations={"related_records": []},
        )

    def _build_code_change_payload(
        self, diff_text: str, files_changed: list[str]
    ) -> dict[str, Any]:
        return {
            "patch_text": self._bounded_patch_text(diff_text, files_changed),
            "diff_hunks": self._extract_patch_hunks(diff_text, files_changed),
        }

    @staticmethod
    def _coerce_str(value: Any) -> str:
        """Normalise LLM-returned values into a single plain string.

        The JSON schema handed to the summariser asks for strings in the
        ``what_changed`` / ``why_changed`` / ``before_behavior`` /
        ``after_behavior`` fields, but OpenAI models occasionally wrap
        the value in a single-element list (or return a numeric/bool).
        We defensively flatten so downstream consumers (BM25 tokeniser,
        markdown renderer) never hit a non-string where they expect a str.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return " ".join(
                PatchMemoryManager._coerce_str(item) for item in value if item is not None
            )
        return str(value)

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        """Normalise LLM-returned values into a list of plain strings."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [
                PatchMemoryManager._coerce_str(item) for item in value if item is not None
            ]
        return [PatchMemoryManager._coerce_str(value)]

    def _build_behavior_payload(self, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "what_changed": self._coerce_str(summary.get("what_changed", "")),
            "why_changed": self._coerce_str(summary.get("why_changed", "")),
            "before_summary": self._coerce_str(summary.get("before_behavior", "")),
            "after_summary": self._coerce_str(summary.get("after_behavior", "")),
            "constraints_to_preserve": self._coerce_str_list(
                summary.get("constraints_to_preserve", [])
            ),
            "known_risks": self._coerce_str_list(summary.get("known_risks", [])),
        }

    def _build_validation_payload(
        self,
        episode: Optional[ValidationEpisode],
        feature_group: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not episode:
            fg = feature_group or {}
            tests_before = fg.get("tests_before", [])
            tests_after = fg.get("tests_after", [])
            if tests_before or tests_after:
                return {
                    "validation_before": tests_before,
                    "validation_after": tests_after,
                    "tests": tests_after,
                    "last_status": tests_after[-1].get("status", "unknown")
                    if tests_after
                    else "unknown",
                }
            return {
                "validation_before": [],
                "validation_after": [],
                "tests": [],
                "last_status": "unknown",
            }

        tests_after = [
            {
                "command": test.command,
                "exit_code": test.exit_code,
                "status": test.status,
                "output_excerpt": test.output_excerpt,
            }
            for test in episode.tests
        ]
        tests_before = [
            {
                "command": test.command,
                "exit_code": test.exit_code,
                "status": test.status,
                "output_excerpt": test.output_excerpt,
            }
            for test in episode.tests_before
        ]
        return {
            "validation_before": tests_before,
            "validation_after": tests_after,
            "tests": tests_after,
            "last_status": episode.status,
        }

    def _build_trajectory_payload(
        self,
        trajectory_slice: Optional[TrajectorySlice],
        episode: Optional[ValidationEpisode],
        feature_group: dict[str, Any],
    ) -> dict[str, Any]:
        reasoning_excerpts = feature_group.get("reasoning_excerpt", "")
        if episode:
            return {
                "session_id": episode.session_id,
                "files_touched": episode.files_changed,
                "files_read": episode.files_read,
                "commands": episode.commands[-8:],
                "tests_run": [test.command for test in episode.tests[-8:]],
                "event_count": 0,
                "excerpt": feature_group.get("trajectory_excerpt", ""),
                "reasoning": reasoning_excerpts,
            }
        return {
            "session_id": trajectory_slice.session_id if trajectory_slice else None,
            "files_touched": trajectory_slice.files_touched if trajectory_slice else [],
            "files_read": trajectory_slice.files_read if trajectory_slice else [],
            "commands": trajectory_slice.commands[-8:] if trajectory_slice else [],
            "tests_run": trajectory_slice.tests_run[-8:] if trajectory_slice else [],
            "event_count": len(trajectory_slice.events) if trajectory_slice else 0,
            "excerpt": feature_group.get("trajectory_excerpt", ""),
            "reasoning": reasoning_excerpts,
        }

    def _bounded_patch_text(
        self, diff_text: str, files_changed: list[str], max_chars: int = 6000
    ) -> str:
        if not diff_text:
            return ""
        if not files_changed:
            return self._dedupe_patch_blocks(diff_text)[:max_chars]
        lines = diff_text.splitlines()
        collected: list[str] = []
        keep = False
        for line in lines:
            if line.startswith("diff --git "):
                keep = any(f"a/{path}" in line or f"b/{path}" in line for path in files_changed)
            if keep:
                collected.append(line)
        patch_text = "\n".join(collected)
        patch_text = self._dedupe_patch_blocks(patch_text)
        return patch_text[:max_chars]

    def _dedupe_patch_blocks(self, patch_text: str) -> str:
        """Drop exact duplicate `diff --git ...` blocks while preserving order."""
        if not patch_text:
            return ""
        lines = patch_text.splitlines()
        blocks: list[str] = []
        current: list[str] = []
        for line in lines:
            if line.startswith("diff --git "):
                if current:
                    blocks.append("\n".join(current))
                current = [line]
            else:
                if not current:
                    current = [line]
                else:
                    current.append(line)
        if current:
            blocks.append("\n".join(current))

        seen: set[str] = set()
        unique: list[str] = []
        for block in blocks:
            key = block.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(block)
        return "\n".join(unique)

    def _extract_patch_hunks(
        self, diff_text: str, files_changed: list[str]
    ) -> list[dict[str, Any]]:
        patch_text = self._bounded_patch_text(diff_text, files_changed, max_chars=12000)
        if not patch_text:
            return []
        hunks: list[dict[str, Any]] = []
        current_file = None
        current_header = None
        current_lines: list[str] = []
        for line in patch_text.splitlines():
            if line.startswith("diff --git "):
                if current_file and current_header and current_lines:
                    hunks.append(self._hunk_payload(current_file, current_header, current_lines))
                parts = line.split()
                current_file = (
                    parts[2][2:] if len(parts) >= 3 and parts[2].startswith("a/") else None
                )
                current_header = None
                current_lines = []
            elif line.startswith("@@ "):
                if current_file and current_header and current_lines:
                    hunks.append(self._hunk_payload(current_file, current_header, current_lines))
                current_header = line
                current_lines = []
            elif current_header is not None:
                current_lines.append(line)
        if current_file and current_header and current_lines:
            hunks.append(self._hunk_payload(current_file, current_header, current_lines))
        deduped: list[dict[str, Any]] = []
        seen_hunks: set[tuple[str, str, str, str]] = set()
        for hunk in hunks:
            key = (
                str(hunk.get("file", "")),
                str(hunk.get("hunk_header", "")),
                str(hunk.get("before_snippet", "")),
                str(hunk.get("after_snippet", "")),
            )
            if key in seen_hunks:
                continue
            seen_hunks.add(key)
            deduped.append(hunk)
        return deduped[:12]

    def _hunk_payload(
        self, file_path: str, header: str, lines: list[str]
    ) -> dict[str, Any]:
        before = [line[1:] for line in lines if line.startswith("-")][:20]
        after = [line[1:] for line in lines if line.startswith("+")][:20]
        return {
            "file": file_path,
            "hunk_header": header,
            "before_snippet": "\n".join(before),
            "after_snippet": "\n".join(after),
        }

    # -----------------------------------------------------------------
    # Post-evaluation finalisation
    # -----------------------------------------------------------------

    def finalize_feature_patches_with_eval(
        self,
        milestone_id: str,
        attempt: int,
        eval_status: str,
        eval_res: Any = None,
        error_msg: Optional[str] = None,
    ) -> None:
        bundle = self._load_bundle_for_attempt(milestone_id, attempt)
        if not bundle:
            return

        updated_records: list[FeaturePatchRecord] = []
        for record_id in bundle.feature_patch_ids:
            record = self._load_record(record_id)
            if not record:
                continue
            record.updated_at = self._now_iso()
            record.validation["eval_status"] = eval_status
            if error_msg:
                record.validation["eval_error"] = error_msg
            if eval_status == "passed":
                record.status = "validated"
            elif eval_status == "failed":
                record.status = "regressive"
            else:
                record.status = "error"
            self._save_record(record)
            updated_records.append(record)

        self._update_indexes(updated_records)

    # -----------------------------------------------------------------
    # Retrieval / rendering
    # -----------------------------------------------------------------

    def _find_related_prior_records(
        self, files_changed: list[str]
    ) -> list[dict[str, Any]]:
        related: list[dict[str, Any]] = []
        file_index = self._load_json(self._indexes_dir() / "file_index.json", {})
        seen: set[str] = set()
        for path in files_changed:
            for record_id in file_index.get(path, []):
                if record_id in seen:
                    continue
                seen.add(record_id)
                record = self._load_record(record_id)
                if not record:
                    continue
                related.append(
                    {
                        "record_id": record.record_id,
                        "feature_title": record.feature_title,
                        "status": record.status,
                        "files_changed": record.files_changed[:5],
                        "constraints_to_preserve": record.behavior.get(
                            "constraints_to_preserve", []
                        ),
                    }
                )
        return related[:5]

    def _update_indexes(self, records: list[FeaturePatchRecord]) -> None:
        milestone_index = self._load_json(self._indexes_dir() / "milestone_index.json", {})
        file_index = self._load_json(self._indexes_dir() / "file_index.json", {})
        feature_index = self._load_json(self._indexes_dir() / "feature_index.json", {})
        symbol_index = self._load_json(self._indexes_dir() / "symbol_index.json", {})

        for record in records:
            milestone_index.setdefault(record.milestone_id, [])
            if record.record_id not in milestone_index[record.milestone_id]:
                milestone_index[record.milestone_id].append(record.record_id)
            for path in record.files_changed:
                file_index.setdefault(path, [])
                if record.record_id not in file_index[path]:
                    file_index[path].append(record.record_id)
            for tag in record.feature_tags:
                feature_index.setdefault(tag, [])
                if record.record_id not in feature_index[tag]:
                    feature_index[tag].append(record.record_id)
            for symbol in record.symbols_changed:
                symbol_index.setdefault(symbol, [])
                if record.record_id not in symbol_index[symbol]:
                    symbol_index[symbol].append(record.record_id)

        self._write_json(self._indexes_dir() / "milestone_index.json", milestone_index)
        self._write_json(self._indexes_dir() / "file_index.json", file_index)
        self._write_json(self._indexes_dir() / "feature_index.json", feature_index)
        self._write_json(self._indexes_dir() / "symbol_index.json", symbol_index)

    def retrieve_for_edit_intent(
        self,
        active_files: list[str],
        active_features: list[str],
        active_symbols: list[str],
    ) -> list[FeaturePatchRecord]:
        retriever = PatchMemoryRetriever(self._all_records())
        return [
            r
            for r, _ in retriever.retrieve(
                active_files, active_features, active_symbols, mode="edit_intent"
            )
        ]

    def retrieve_for_refine_recover(
        self,
        active_files: list[str],
        active_features: list[str],
        active_symbols: list[str],
    ) -> list[FeaturePatchRecord]:
        retriever = PatchMemoryRetriever(self._all_records())
        return [
            r
            for r, _ in retriever.retrieve(
                active_files, active_features, active_symbols, mode="recover"
            )
        ]

    def _all_records(self) -> list[FeaturePatchRecord]:
        records: list[FeaturePatchRecord] = []
        for record_file in sorted(self._records_dir().glob("*.json")):
            record = self._load_record(record_file.stem)
            if record:
                records.append(record)
        return records

    def render_edit_guardrail_context(
        self,
        active_files: list[str],
        active_features: list[str],
        active_symbols: list[str],
        char_budget: int = 3500,
    ) -> str:
        records = self.retrieve_for_edit_intent(active_files, active_features, active_symbols)
        records = [r for r in records if r.status in ("validated", "working", "submitted")]
        lines = ["# Retrieved Prior Patches — Check Before Editing", ""]
        lines.append(
            "The following patches were retrieved as potentially relevant to files you may be editing. "
            "They were previously validated (tests passed). "
            "If any of these files overlap with your current changes, preserve the behavior described — "
            "reverting it will cause regressions in previously passing tests."
        )
        lines.append("")
        if active_files:
            lines.append(f"Overlapping files: {', '.join(active_files[:8])}")
            lines.append("")
        if not records:
            lines.append("No overlapping validated patches found.")
            return "\n".join(lines).strip() + "\n"
        lines.extend(
            self._render_records_within_budget(records, include_tests=True, char_budget=char_budget)
        )
        return "\n".join(lines).strip() + "\n"

    def render_refine_recover_context(
        self,
        active_files: list[str],
        active_features: list[str],
        active_symbols: list[str],
        char_budget: int = 3500,
    ) -> str:
        records = self.retrieve_for_refine_recover(active_files, active_features, active_symbols)
        records = [r for r in records if r.status in ("validated", "working", "submitted")]
        lines = ["# Retrieved Prior Patches (May Be Relevant)", ""]
        lines.append(
            "The following patches were retrieved as potentially relevant to your current task. "
            "They may overlap with the files or features you are working on — but this is based on "
            "similarity scoring, not a guarantee. Review them and, if relevant, use this information to:"
        )
        lines.append("1. Avoid reverting or conflicting with prior validated changes")
        lines.append("2. Understand the intent behind earlier modifications to overlapping files")
        lines.append("")
        if active_files:
            lines.append(f"Overlapping files: {', '.join(active_files[:8])}")
        if active_features:
            lines.append(f"Feature hints: {', '.join(active_features[:8])}")
        lines.append("")
        if not records:
            lines.append("No relevant prior validated patches found.")
            return "\n".join(lines).strip() + "\n"
        lines.extend(
            self._render_records_within_budget(records, include_tests=True, char_budget=char_budget)
        )
        return "\n".join(lines).strip() + "\n"

    _DETAIL_LEVELS = ("full", "medium", "compact")

    def _render_records_within_budget(
        self,
        records: list[FeaturePatchRecord],
        include_tests: bool,
        char_budget: int,
    ) -> list[str]:
        actionable = [
            r
            for r in records
            if r.code_change.get("patch_text") or r.status in ("validated", "working")
        ]
        if not actionable:
            actionable = records[:1]

        all_lines: list[str] = []
        used = 0
        for rank, record in enumerate(actionable):
            detail = self._DETAIL_LEVELS[rank] if rank < len(self._DETAIL_LEVELS) else "compact"
            block = self._render_record_block(record, include_tests=include_tests, detail=detail)
            block_text = "\n".join(block)
            if rank > 0 and used + len(block_text) > char_budget:
                break
            all_lines.extend(block)
            used += len(block_text)
        return all_lines

    def _render_record_block(
        self,
        record: FeaturePatchRecord,
        include_tests: bool,
        detail: str = "full",
    ) -> list[str]:
        lines = [f"- `{record.record_id}` [{record.status}] {record.feature_title}"]
        lines.append(f"  Files: {', '.join(record.files_changed[:4])}")
        why = record.behavior.get("why_changed", "")
        if why:
            max_why = 200 if detail != "compact" else 150
            lines.append(f"  Change intent: {why[:max_why]}")
        constraints = record.behavior.get("constraints_to_preserve", [])
        if constraints:
            lines.append(f"  MUST preserve: {constraints[0]}")

        if detail != "compact":
            patch_text = record.code_change.get("patch_text", "")
            if patch_text:
                if detail == "full":
                    max_chars, max_lines = 1200, 25
                else:
                    max_chars, max_lines = 600, 15
                lines.append("  Patch:")
                lines.append("  ```diff")
                for line in patch_text[:max_chars].splitlines()[:max_lines]:
                    lines.append(f"  {line}")
                lines.append("  ```")
            else:
                lines.append("  (No code diff available — in-session change not yet committed)")

        if include_tests:
            tests = record.validation.get("validation_after") or record.validation.get("tests", [])
            for test in tests[:2]:
                lines.append(
                    f"  Passed local test: {test.get('command', '')} [{test.get('status', '')}]"
                )
            tests_before = record.validation.get("validation_before", [])
            for test in tests_before[:1]:
                lines.append(
                    f"  State before change: {test.get('command', '')} [{test.get('status', '')}]"
                )
        return lines

    def render_memory_readme(self) -> str:
        content = (
            "# Patch Memory\n\n"
            "This directory stores feature-level patch records with actual code hunks.\n"
            "`rendered/current_context.md` is the markdown that gets prepended to the "
            "next instance's problem_statement (guardrail mode). "
            "`rendered/recover_context.md` is the refine/recover variant.\n"
        )
        (self._rendered_dir() / "README.md").write_text(content, encoding="utf-8")
        return content

    def render_context_from_problem_statement(
        self,
        problem_statements: list[str],
        milestone_ids: list[str],
        char_budget: int = 3500,
    ) -> str:
        """SWE-EVO analog of ``render_context_from_srs``.

        The "upcoming task" signal comes from the next instance's
        ``problem_statement`` (itself derived from SWE-EVO SRS docs /
        generated diffs).  We mine plausible module paths + function
        symbols from the raw text and use them as the retrieval query.
        """
        active_files: list[str] = []
        seen_files: set[str] = set()
        for text in problem_statements:
            for path in _extract_module_paths_from_text(text, repo_hint=self.repo_name):
                if path not in seen_files:
                    seen_files.add(path)
                    active_files.append(path)

        active_features: list[str] = list(milestone_ids)
        for text in problem_statements:
            for match in re.finditer(r"\*\*FR\d+\*\*:\s*(.+)", text):
                active_features.append(match.group(1).strip()[:60])
            for match in re.finditer(r"\*\*REQ-[A-Z0-9_-]+\*\*", text):
                active_features.append(match.group(0))

        active_symbols = self._extract_symbols_from_text(problem_statements)

        records = self.retrieve_for_edit_intent(active_files, active_features, active_symbols)
        records = [r for r in records if r.status in ("validated", "working", "submitted")]

        lines = ["# Retrieved Prior Patches — Check Before Editing", ""]
        lines.append(
            "The following patches were retrieved from prior instances in this SWE-EVO chain. "
            "They were previously submitted (and optionally validated by the SWE-bench harness). "
            "If any of these files overlap with your current changes, preserve the behavior described — "
            "reverting it will cause regressions in previously passing tests."
        )
        lines.append("")
        if active_files:
            lines.append(f"Potentially affected files: {', '.join(active_files[:8])}")
            lines.append("")
        if not records:
            lines.append("No overlapping validated patches found.")
            return "\n".join(lines).strip() + "\n"
        lines.extend(
            self._render_records_within_budget(records, include_tests=True, char_budget=char_budget)
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _extract_symbols_from_text(texts: list[str]) -> list[str]:
        symbols: list[str] = []
        seen: set[str] = set()
        for text in texts:
            for match in re.finditer(r"`(\w{4,})`", text):
                name = match.group(1)
                if name not in seen:
                    seen.add(name)
                    symbols.append(name)
        return symbols[:20]

    # -----------------------------------------------------------------
    # Host-only memory view (replaces docker cp path)
    # -----------------------------------------------------------------

    def sync_host_memory_view(
        self,
        task_context_md: str,
        recover_context_md: Optional[str] = None,
    ) -> None:
        """Persist rendered markdown under ``<trial_root>/patch_memory/rendered/``.

        Unlike EvoClaw's ``sync_container_memory_view`` this **does not**
        ``docker cp`` into a running container.  In SWE-EVO the container
        starts fresh for each instance and the patch-memory markdown
        reaches the agent via ``problem_statement`` (see
        ``problem_injector.py``).  Writing the rendered files on the host
        is still valuable for post-hoc inspection / debugging.
        """
        rendered_dir = self._rendered_dir()
        rendered_dir.mkdir(parents=True, exist_ok=True)
        (rendered_dir / "current_context.md").write_text(task_context_md, encoding="utf-8")
        if recover_context_md is not None:
            (rendered_dir / "recover_context.md").write_text(recover_context_md, encoding="utf-8")
        self.render_memory_readme()

    def sync_dynamic_memory_from_context(
        self, context: EditIntentContext, recover: bool = False
    ) -> None:
        current = self.render_edit_guardrail_context(
            active_files=context.active_files,
            active_features=context.active_features,
            active_symbols=context.active_symbols,
        )
        recover_md: Optional[str] = None
        if recover:
            recover_md = self.render_refine_recover_context(
                active_files=context.active_files,
                active_features=context.active_features,
                active_symbols=context.active_symbols,
            )
        self.sync_host_memory_view(current, recover_context_md=recover_md)
