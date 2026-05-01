# harbor_patch_memory

Harbor agents that wrap **OpenHands** with EvoClaw-style **Patch Memory**, used to evaluate the [Terminal-shift-main](../Terminal-shift-main) chain dataset.

This package is a sibling of `~/qingchuan/harbor` and `~/qingchuan/SWE-EVO`. It does **not** modify either of those upstream projects. It plugs into Harbor purely via `--agent-import-path`.

```
harbor_patch_memory/
├── pyproject.toml
├── README.md
├── harbor_patch_memory/                # python package
│   ├── __init__.py
│   ├── chain_id.py                     # auto-detect chain id from task name
│   ├── patch_capture.py                # git baseline + diff in the container
│   ├── memory_bridge.py                # host-side glue (LLM cfg, manager build)
│   ├── memory/                         # vendored from swe_evo_patch_memory
│   │   ├── manager.py
│   │   ├── retriever.py
│   │   ├── summarizer.py
│   │   └── trajectory_extractor.py
│   └── agents/
│       ├── openhands_baseline.py       # OpenHands + git capture, no PM
│       └── openhands_pm.py             # OpenHands + Patch Memory
├── scripts/
│   ├── run_chain.py                    # chain runner (sequential within / parallel across)
│   ├── list_chains.py                  # enumerate chains discovered in the dataset
│   ├── _dispatch.sh                    # internal: tmux-side parallel dispatcher
│   ├── launch_runs.sh                  # generic launcher (--variant pm|baseline)
│   ├── launch_pm.sh                    # one-click launch of OpenHands+PM, all chains
│   ├── launch_baseline.sh              # one-click launch of OpenHands baseline
│   ├── kill_runs.sh                    # tmux + pgid + docker teardown (no zombies)
│   └── status.py                       # liveness + per-task pass/fail + accuracy
└── tests/
```

## How Patch Memory works here

Conceptually, *Patch Memory captures **how the OpenHands agent's working memory has updated** during a task* — i.e., what the agent has learned, decided, edited, run and concluded — and replays a curated subset of those updates as guardrails on the next sibling task. It is **not** "we paste git diffs into the prompt".

Per Harbor trial (one task) the agent does the following:

1. **Chain id** — derived from the task name (default: strip `-EVO-N` / `-v-<token>`; override with `--ae HARBOR_PM_CHAIN_ID=<chain>`). All five `bn-fit-modify*` tasks share `chain_id = "bn-fit-modify"` and therefore the same memory store.
2. **Chain-scoped memory store** at `~/.harbor_patch_memory/<chain_id>/patch_memory/{records,bundles,indexes,rendered}` — outlives any single Harbor trial; that's how the prototype's records reach the four variants.
3. **Render → inject.** The retriever scores existing records against the *new* task's instruction (active files / active symbols / BM25 over intent text). The top-K records are rendered as a markdown block:
   - `feature_title` and `behavior.what_changed / why_changed` (the descriptive patch);
   - `behavior.constraints_to_preserve` and `behavior.known_risks`;
   - `before_summary` / `after_summary` (≈ what behaviour the agent thinks it changed);
   - **at most ~2 KB of trimmed `code_change` hunks** per record, only when relevant.
   The block is fenced with `<!-- BEGIN PATCH MEMORY -->` and prepended to OpenHands' `--task=` instruction.
4. **Run OpenHands** the same way Harbor's stock `openhands.py` does (same `install`, `RUNTIME=local`, `SAVE_TRAJECTORY_PATH`). The full conversation / events / file-edits go into `logs/agent/openhands.trajectory.json` — that's the **primary signal** for the summariser.
5. **Ingest.** After OpenHands exits we feed three things to `PatchMemoryManager.store_submission_feature_patches`:
   - The OpenHands **trajectory** (extracted into `TrajectorySlice` + `ValidationEpisode`s by `SweEvoTrajectoryExtractor`) — what the agent did and why.
   - **`changed_files`** + a **bounded `diff_text`** captured from a one-shot scratch `git` repo at `/app` (see "About the git capture" below).
   - The original problem statement (`srs_text`).
   The summariser LLM produces structured `FeaturePatchRecord`s (`behavior` + `validation` + small bounded `code_change`). Records are indexed by file / symbol / feature so the next variant's run can find them.

### About the git capture

Terminal-shift containers don't ship with git, so we transparently `git init` `/app` before the run, commit a baseline, and read `git diff` afterwards. **This is purely an internal scoping signal for the summariser** ("which files did the agent touch, and roughly how"). It is never pasted into the prompt verbatim — only the LLM-summarised, descriptively-rephrased fields end up in memory.

If a task produces no diffable changes (e.g. the agent only emits CSV blobs, or git wasn't installable), `diff_text` is empty: the summariser falls back to **trajectory-only** records and `code_change.patch_text` simply stays `""`. The descriptive patch (intent / constraints / behaviour) is the same either way.

### Path-noise filter

When the agent provisions tooling (`apt-get install r-base`, etc.) the package manager prints lines like

```
Get:178 http://archive.ubuntu.com/ubuntu noble-updates/main amd64 r-base 4.3.3-2build2_all.deb
```

The naive path-extraction regex (`(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+`) happily picks `archive.ubuntu.com/ubuntu`, `noble-updates/main`, `r-base 4.3.3-2build2_all.deb` and our own `git init` `.git/` internals out of those lines. To prevent those bogus paths from polluting the descriptive patches, **every path candidate is run through `SweEvoTrajectoryExtractor._is_noisy_path`** and dropped if it matches:

- a system / FHS prefix (`etc/`, `usr/`, `lib/`, `proc/`, `var/`, `Etc/`, `.git/`, `.openhands/`, `sessions/`, …),
- an apt mirror prefix (`archive.ubuntu.com/`, `noble-updates/`, `jammy-security/`, `bookworm/`, …),
- a `.../`-truncated tail from an apt progress line,
- a hostname-shaped segment (anything ending in `.com / .org / .ubuntu / .debian / …` followed by `/`),
- a binary / archive extension on the final segment (`.deb / .rpm / .so / .so.X / .tar.gz / .zip / .pyc / .ttf / …`),
- a purely numeric-looking final segment (apt's `3.5-8-1_amd64` arch tags), or
- a 1- or 2-segment path with no extension (catches `kB/s`, `files/directories` from prose lines).

The same filter runs once at extraction time (`_extract_paths`) and once again as defence-in-depth in `memory_bridge._filter_changed_files`, which also re-runs over `git diff --name-only` results in case the scratch repo accidentally tracked something it shouldn't.

The retriever's free-text path heuristic in `memory.manager._extract_module_paths_from_text` is multi-language: it accepts source extensions (`.py .R .ipynb .ts .rs .go .md .csv .txt .yaml .toml …`) verbatim instead of force-rewriting everything to `.py`. Only the dotted-form (`numpy.core.fromnumeric`) still gets `.py` appended, since dotted module paths almost always denote Python in bug reports / SRS prose.

## Two agent variants

| Variant   | Import path                                                              | What it does                                                          |
|-----------|--------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `pm`      | `harbor_patch_memory.agents.openhands_pm:OpenHandsPatchMemory`           | OpenHands + chain-scoped Patch Memory (render → inject → ingest).     |
| `baseline`| `harbor_patch_memory.agents.openhands_baseline:OpenHandsBaseline`        | OpenHands + the same internal git capture, but no memory render / ingest. |

The baseline performs the same internal `git init` / post-run `git diff` capture and dumps `/logs/agent/harbor_pm/{diff.patch,changed_files.txt,meta.json}` for parity. The only behavioural difference vs. PM is that the baseline never injects retrieved memory and never writes records — so any chain-level effect you observe is attributable to Patch Memory.

## Install

```bash
cd ~/qingchuan/harbor_patch_memory
pip install -e .
# Make sure Harbor's CLI is on PATH (already true on this server)
which harbor
```

The package depends on Harbor + OpenHands at runtime; both are already installed in this environment. The vendored memory modules don't pull anything new — the summariser uses the same OpenAI-compatible client that the rest of `~/qingchuan` already has.

## LLM configuration

Three sources, in priority order:

1. Explicit kwargs on the agent (`--agent-kwarg model=...`).
2. `--ae LLM_API_KEY=...`, `--ae LLM_MODEL=...`, `--ae LLM_BASE_URL=...`.
3. Fallback to `[llm.swe_evo_pm]` in `~/qingchuan/OpenHands_upstream/config.toml`.

Both the OpenHands run *and* the Patch Memory summariser use the resolved triple (so you only ever set it once).

To swap the fallback block, pass `--agent-kwarg oh_llm_block=<name>` or `--agent-kwarg oh_config_path=/path/to/config.toml`.

## Smoke test (one chain, sequentially)

The `bn-fit-modify` chain is the recommended first run — it's Python-only and the Docker image is light.

```bash
cd ~/qingchuan/harbor_patch_memory

# PM variant
python scripts/run_chain.py \
  --dataset ~/qingchuan/Terminal-shift-main \
  --chain bn-fit-modify \
  --variant pm \
  --trials-dir ~/qingchuan/harbor_patch_memory/runs/bn-fit-modify-pm

# Baseline variant (separate trials dir so artefacts don't collide)
python scripts/run_chain.py \
  --dataset ~/qingchuan/Terminal-shift-main \
  --chain bn-fit-modify \
  --variant baseline \
  --trials-dir ~/qingchuan/harbor_patch_memory/runs/bn-fit-modify-baseline
```

The runner discovers `bn-fit-modify` plus all `bn-fit-modify-EVO-{1,2,3,4}` and runs them in order. Per-task Harbor stdout is captured to `<trials_dir>/_logs/<task_name>.harbor.log`. A summary JSON lands at `<trials_dir>/_logs/chain_runner_summary.json`.

## Running multiple chains in parallel

Sequential within each chain (Patch Memory needs that), but parallel across chains:

```bash
python scripts/run_chain.py \
  --dataset ~/qingchuan/Terminal-shift-main \
  --chain bn-fit-modify \
  --chain adaptive-rejection-sampler \
  --variant pm \
  --max-parallel 2
```

## Full-dataset deployment via tmux

For a hands-off, reboot-safe run over **every** chain in the dataset, use
the tmux-based launchers. They're designed so you can ssh in, fire and
forget, log out, and come back later to inspect status / accuracy.

Layout:

```
scripts/launch_pm.sh        # OpenHands + Patch Memory  (tmux session: harbor-pm)
scripts/launch_baseline.sh  # OpenHands baseline        (tmux session: harbor-baseline)
scripts/kill_runs.sh        # clean teardown
scripts/status.py           # progress + accuracy report
```

Within one chain the dispatcher runs prototype → variants serially (so
Patch Memory accumulates). Across chains it runs **3 chains
concurrently** by default (override with `--parallel N`).

### Deploy

```bash
cd ~/qingchuan/harbor_patch_memory

# PM run — full dataset, 3 chains in parallel
scripts/launch_pm.sh

# Baseline run — same shape, separate tmux session, separate trials dir
scripts/launch_baseline.sh
```

After each command you'll see the tmux session name and the log paths.
You can detach and log out; the runs continue.

Common variations:

```bash
# Subset of chains
scripts/launch_pm.sh --chains 'bn-fit-modify adaptive-rejection-sampler'

# Skip singleton chains (no variant — Patch Memory has no chain effect)
scripts/launch_pm.sh --no-singletons

# Different concurrency
scripts/launch_pm.sh --parallel 4

# Resume / re-run into a fresh trials dir
scripts/launch_pm.sh --trials-dir ~/runs-pm-attempt2
```

### Monitor

```bash
# One-shot snapshot
scripts/status.py

# Refresh every 30s
scripts/status.py --watch

# Per-task pass/fail listing
scripts/status.py --detail

# Just one variant
scripts/status.py --variant pm

# Machine-readable
scripts/status.py --json
```

The states `status.py` reports per variant:

* **RUNNING** — tmux session live + dispatcher process alive
* **FINISHED** — dispatcher exited; tmux pane held open by `remain-on-exit` so you can attach and read the final log. Run `kill_runs.sh` to free it.
* **STOPPED** — clean teardown; everything reaped
* **NOT-STARTED** — no `run_meta.json` for this variant
* **INCONSISTENT** — tmux died but the dispatcher process didn't (rare; usually means someone SIGKILL'd tmux). `kill_runs.sh` will reap the dispatcher pgid.

### Tear down

```bash
# One variant
scripts/kill_runs.sh --variant pm

# Both
scripts/kill_runs.sh --all

# Be aggressive with leftover docker containers (only kills containers
# created **after** this run started — won't touch unrelated workloads
# on the same host)
scripts/kill_runs.sh --variant pm --reap-docker
```

The killer first SIGTERMs (gives Harbor + OpenHands' atexit cleanup a
chance to remove their docker container), waits `--grace SECONDS`
(default 30), then SIGKILLs anything that survived. Per-chain pgids are
read from `<trials_dir>/_logs/runner/chain.<id>.pgid` so we kill *every*
descendant — no zombie processes.

### File layout produced

```text
runs/full-pm/
├── _logs/
│   └── runner/
│       ├── run_meta.json           # tmux session, dispatcher pid, start time
│       ├── chains.txt              # one chain id per line
│       ├── dispatcher.{pid,log}    # the xargs orchestrator
│       └── chain.<id>.{pgid,log,exit}   # per-chain pgid for kill, log, final exit code
├── _logs/chain_runner_summary.json # aggregated summary written by run_chain.py
├── <task_name>__<rand>/            # one Harbor trial per task
│   ├── result.json                 # ← status.py reads this for pass/fail
│   ├── verifier/reward.txt
│   ├── trial.log
│   ├── agent/                      # OpenHands artefacts incl. trajectory
│   └── ...
└── ...
```

## Inspecting Patch Memory state

After a chain run, the chain memory lives at `~/.harbor_patch_memory/<chain_id>/`:

```text
~/.harbor_patch_memory/bn-fit-modify/
└── patch_memory/
    ├── records/     # one JSON per FeaturePatchRecord
    ├── bundles/     # one JSON per milestone (task)
    ├── indexes/     # file_index / symbol_index / feature_index / milestone_index
    └── rendered/
        ├── current_context.md
        └── README.md
```

Per-trial artefacts (handy when debugging which memory was prepended):

```text
<trials_dir>/<trial_name>/logs/agent/harbor_pm/
├── diff.patch                   # raw diff captured against the baseline commit
├── changed_files.txt
├── meta.json                    # baseline_sha / head_sha / sizes
├── pre_run_meta.json            # chain_id, n memory chars, etc
├── original_instruction.md      # task prompt as Harbor handed it to us
├── injected_instruction.md      # prompt actually fed to OpenHands (memory + original)
└── rendered_memory.md           # the markdown block that was prepended
```

## Skipped / out of scope

- The Harbor verifier reward (`/logs/verifier/reward.txt`) is **not** wired into Patch Memory records — every record is stored with `status="submitted"` per the user's choice. To turn this on later, post-process `verifier/reward.txt` and call `PatchMemoryManager.finalize_feature_patches_with_eval(...)`.
- The chain id detector is regex-based. It collapses `<base>-EVO-N` and `<base>-v-<token>` (`<token>` may include hyphens) onto `<base>`. A small handful of dataset chains use idiosyncratic naming — e.g. `install-windows-3.11` is the prototype but its variants are named `install-windows-v-*`. The detector treats these as 5 singleton chains rather than 1 chain of 5; Patch Memory simply doesn't kick in for that group, which is reflected in the dataset's "WARN" coverage.
- **Singletons are skipped by default in both launchers** so the pm and baseline runs evaluate the same task set (88 chains / 440 tasks). Use `launch_pm.sh --include-singletons` (and the same for baseline) if you want to evaluate the full 93 chains / 445 tasks.
