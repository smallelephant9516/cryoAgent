# CryoAgent

## 1. Introduction

CryoAgent is an agent-driven pipeline for single-particle cryo-EM (SPA). It uses an LLM-backed ReAct loop to drive **CryoSPARC** jobs through a multi-stage workflow from preprocessing through reconstruction, optional polish, heterogeneity analysis, and post-run improvement.

The primary and fully validated processing backend is **CryoSPARC**. RELION-related interfaces are retained in the codebase as an **experimental** integration for further development; they are not presented as a validated end-to-end pipeline.

The principal design is **progressive autonomy**: start with a reliable, expert-defined workflow, then allow specialized agents to optimize parameters, and finally run hypothesis-driven exploration once enough processing history has accumulated.

## 2. Progressive autonomy architecture

CryoAgent increases agent freedom in three levels:

| Level | CLI | Description |
| --- | --- | --- |
| Rigid / guided workflow | `--mode guided` (default) | Fixed stage order from `session.json`. Workflow example prompts improve reliability across LLM providers. |
| Dynamic planning | `--mode dynamic` | LLM chooses each next stage from prior stage JSON outputs. |
| De novo exploration (DE) | `--mode full_dynamic` | Single agent with the full tool set; minimal predefined workflow. Higher run-to-run variance; success depends strongly on LLM capability (see paper Table S2). |
| Hypothesis-driven exploration (HE) | `--improve` | Improvement agent reads prior run outputs in `--outputs-dir`, diagnoses limiting factors, and iteratively tests hypotheses. |

Structured workflow prompts reduce stochastic failures, especially with less capable models. Unconstrained DE mode is feasible with stronger models but is generally less reliable than the guided workflow.

## 3. Features

- **CryoSPARC-first end-to-end workflow** — Preprocessing, picking, 2D/3D optimization, reconstruction, polish, and heterogeneity stages orchestrated through specialized agents (enable stages in `session.json`).
- **Failure recovery** — Distinguishes execution failures (job error → CryoSPARC forum search → parameter remapping → retry) from processing-state failures (job completes but output is unsuitable → threshold adjustment → repeat). Upstream failures block downstream cascade.
- **Automated heterogeneity analysis** — Optional stages (`heterogeneity`, `heterogeneity_depth`) combine deterministic density comparison with an agentic decision layer that decides when to subdivide, which branches to keep, and when to stop.
- **Iterative optimization** — 2D optimization (classification and CryoSift-assisted refinement) and 3D box-size optimization. Box size respects a CTF-delocalization **minimum** from acquisition parameters; FSC and cFAR compare candidates at or above that bound rather than selecting undersized boxes for higher FSC alone.
- **Monitoring** — Structured logs, per-stage JSON artifacts, and an optional GUI visualizer (see section 8).

## 4. Requirements and compatibility

| Requirement | Notes |
| --- | --- |
| CryoSPARC | **Version ≤ 4.7.1** (tested). Newer releases may change APIs; downgrade to a compatible version if needed ([CryoSPARC install docs](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure)). |
| LLM API | DeepSeek, OpenAI, or Panshi (see section 5). |
| Conda | `cryoagent` environment created by `install.sh`. |
| Optional | Helicon, CryoSift, CryoAlign2, ChimeraX — required only when corresponding stages are enabled. |

**RELION (experimental):** connection notes in [note/relion_agent.txt](note/relion_agent.txt). Not part of the validated end-to-end path.

### Glossary

- **FSC** — Fourier shell correlation; measures reproducibility between independent half-maps.
- **cFAR** — Conical FSC area ratio; orientation diagnostic (higher is generally better).

## 5. Installation and configuration

Step-by-step prerequisites, Zenodo tarball for CryoAlign2, `install.sh`, and JSON edits are documented in **[note/installation.md](note/installation.md)**.

For LLM API keys and CryoSPARC credentials, see the subsections below. Example configuration templates: [note/connection_configs/gibh_feilab/](note/connection_configs/gibh_feilab/) and [note/connection_configs/ibp_118/](note/connection_configs/ibp_118/).

### Install

```bash
bash install.sh
```

Step 4 of `install.sh` interactively prompts for API keys and license values and can write them to `~/.bashrc`.

### LLM and environment variables

Secrets are referenced in `configs/master_config.json` as `${VAR_NAME}` and resolved at runtime (including from a `.env` file in the repo root).

| Variable | Used for |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek LLM |
| `OPENAI_API_KEY` | OpenAI LLM |
| `PANSHI_API_KEY` | Panshi LLM |
| `LICENSE_ID` | CryoSPARC license (in config) |
| `CRYOSPARC_LICENSE_ID` | CryoSPARC license (alternate env name) |

**Setup steps:**

1. Run `bash install.sh` step 4, or export the variables manually / add them to `.env`.
2. Edit `configs/master_config.json`:
   - `agent.provider` — active LLM provider (`deepseek`, `openai`, or `panshi`).
   - `agent.models.<provider>.model_name` — model to use.
   - `cryosparc.*` — host, port, username, password, license.
3. Verify connections (also run automatically by `scripts/run_cryoagent.sh`):

```bash
python check_LLM_connection.py
python check_cryosparc_connection.py
python check_cryosift_connection.py
```

### CryoSPARC connection

Edit `configs/master_config.json` (`cryosparc` section). Ensure CryoSPARC is reachable from the machine where you run CryoAgent.

### RELION (experimental)

Edit `relion` in `configs/master_config.json` or `configs/session.json` if testing RELION integration. See [note/relion_agent.txt](note/relion_agent.txt). RELION tool-call parsing is more sensitive to LLM output format; check logs for `TOOL EXECUTION:` lines to confirm actions were actually invoked.

## 6. Preparing each dataset

Per dataset you maintain two JSON files. Shared templates live under `configs/`; for batch runs, each dataset folder holds its own copies (see section 7).

### `configs/microscope_config.json` — acquisition and data paths

Set **dataset-specific** acquisition and input paths under `microscope_parameters`:

| Field | Role |
| --- | --- |
| `pixel_size`, `voltage`, `cs_mm`, `dose` | Microscope / exposure parameters |
| `particle_diameter`, `symmetry` | For picking and reconstruction |
| `movies_path` | Movies to import (wildcards allowed, e.g. `*.mrc`) |
| `micrographs_path` | Optional: pre-corrected micrographs (skips movie import / motion correction when used) |
| `gain_ref_path`, `gain_rot`, `gain_flip` | Gain reference and orientation |

Human-readable explanations for these keys are in `parameter_descriptions` in the same file.

### `configs/session.json` — CryoSPARC session and pipeline stages

- **`master_workflow.stages`** — List of stages with `enabled` flags (e.g. preprocessing, particle picking, `optimization_2d`, reconstruction, `optimization`, `polish`, heterogeneity stages). Turn stages on or off without editing `master_config.json`.
- **`workflow.project_uid`** and **`workflow.workspace_uid`** — CryoSPARC project and workspace for this dataset. These are the main per-dataset fields most users change.

`session.json` in the same directory as `master_config.json` is **merged on top of** `master_config.json` (session wins on conflicts).

## 7. How to run

### Recommended entry point

Use the wrapper script so you do not need to cd into the source tree or invoke Python directly. Before launching the workflow it runs four preflight checks:

1. **cryoagent** conda environment  
2. **LLM** API connection (`check_LLM_connection.py`)  
3. **CryoSPARC** connection (`check_cryosparc_connection.py`)  
4. **CryoSift** env + weights + evaluator (`check_cryosift_connection.py`)  

```bash
bash scripts/run_cryoagent.sh --workflow test    # preflight + verify setup
bash scripts/run_cryoagent.sh                    # preflight + full guided pipeline
bash scripts/run_cryoagent.sh --skip-checks ...  # skip preflight (advanced)
bash scripts/run_cryoagent.sh --help
```

The script activates the `cryoagent` conda environment and runs `cryoagent_workflow.py` from the repository root.

### Alternative: direct Python invocation

```bash
conda activate cryoagent
python cryoagent_workflow.py --workflow test
python cryoagent_workflow.py
```

Default master config: `configs/master_config.json`. Per dataset, set up `configs/microscope_config.json` and `configs/session.json`.

### Workflow modes

```bash
# Guided workflow (default) — fixed stage order from session.json
bash scripts/run_cryoagent.sh --mode guided

# Dynamic — LLM picks each next stage from prior JSON outputs
bash scripts/run_cryoagent.sh --mode dynamic --goal "Reach best resolution supported by data"

# De novo exploration — single agent, full tool set, no predefined workflow
bash scripts/run_cryoagent.sh --mode full_dynamic --goal "Process movies to 3D density"

# Hypothesis-driven exploration — improve a prior run (does not re-run main workflow)
bash scripts/run_cryoagent.sh --improve --outputs-dir outputs
bash scripts/run_cryoagent.sh --mode full_dynamic --improve --outputs-dir outputs
```

### Custom stages and other flags

```bash
# Only selected stages (comma-separated, no spaces)
bash scripts/run_cryoagent.sh --workflow custom --stages preprocessing,particle_picking
```

Other useful flags: `--config`, `--outputs-dir`, `--conversation-id`, `--verbose`, `--dry-run`, `--mode`, `--improve`, `--goal`. Run `bash scripts/run_cryoagent.sh --help` for the full list.

### Batch runs — `run_batch_datasets.py`

Runs the same workflow over **many datasets** in sequence. Each dataset is a folder under `datasets/unfinished_datasets/` (default) containing:

- `{dataset_name}/configs/session.json`
- `{dataset_name}/configs/microscope_config.json`

Example:

- `datasets/unfinished_datasets/10240/configs/session.json`
- `datasets/unfinished_datasets/10240/configs/microscope_config.json`

The runner copies the repository `configs/master_config.json` into a temporary config for each dataset, overlays that dataset's `session.json`, points the workflow at the dataset's `microscope_config.json`, then calls `cryoagent_workflow.py`. Finished datasets can be moved to `datasets/finished_datasets/`.

```bash
python run_batch_datasets.py
python run_batch_datasets.py --datasets my_dataset_a,my_dataset_b --workflow complete
```

## 8. Monitoring and troubleshooting

All artifacts are written under `--outputs-dir` (default: `outputs/`).

### Output files

| File | Purpose | When to check |
| --- | --- | --- |
| `llm_conversation_<stage>_<timestamp>.log` | LLM reasoning, `TOOL EXECUTION:` lines, parameters, status | **First** on any failure |
| `llm_conversation_full_dynamic_*.log` | De novo exploration run trace | After `--mode full_dynamic` or `--improve` on a DE run |
| `workflow_state.json` | Structured stage history and metrics | Guided/dynamic runs; input for `--improve` |
| `*_results_cryosparc_*.json` | Per-stage metrics and job UIDs | Stage-level debugging |
| Final report JSON/MD | End-of-run summary | After a successful run |

Only lines containing `TOOL EXECUTION:` indicate a tool was actually invoked and parsed. Natural-language text such as `Action: Execute blob_picker...` without a matching `TOOL EXECUTION:` line means the action was not executed.

### Troubleshooting checklist

1. Open the most recent `llm_conversation_*.log` for the failing stage.
2. Find the last `TOOL EXECUTION:` line — was a tool actually called?
3. Read the error message or job log output in the same file.
4. Confirm upstream stages produced valid outputs (avoid cascade failures after an early stage failed).
5. For stochastic failures, rerun with `--mode guided` and review workflow prompts in `cryoagent/prompts/`.

### GUI visualizer

The optional workflow visualizer reads `workflow_state.json` and stage result files from `outputs/`:

```bash
bash cryoagent_viz/run.sh
```

Open `http://localhost:3000` in a browser. Backend runs on port 8000.

## 9. Prompts and customization

LLM prompts are separated from Python execution logic. Templates live under [cryoagent/prompts/](cryoagent/prompts/); see [cryoagent/prompts/README.md](cryoagent/prompts/README.md) for layout and editing conventions.

- Workflow one-shot examples are in prompt templates, not embedded in source code.
- Modify, version, and reuse prompts without changing agent or tool-execution code.
- Regenerate Claude Code / Cursor skill files after prompt edits: `python scripts/sync_claude_openclaw_prompts.py`.

## License

This project is licensed under the Apache 2.0 License; see the `LICENSE` file.
