# CryoAgent

## 1. Introduction

CryoAgent is an agent-driven pipeline for single-particle cryo-EM. It drives **RELION** and **CryoSPARC** through a shared configuration model, uses an LLM-backed ReAct loop to choose actions, and runs a multi-stage workflow from preprocessing through reconstruction and optional polish and heterogeneity steps.

## 2. Features

- **RELION and CryoSPARC integration** — One workflow can use both stacks where each stage expects them (CryoSPARC jobs, RELION directories and executables, and Helicon-style transitions when configured).
- **Workflow monitoring and failure handling** — Stages are tracked; the orchestrator can retry, apply fallback strategies, and surface errors so runs do not fail silently.
- **Automated heterogeneity analysis** — Optional stages (`heterogeneity`, `heterogeneity_depth`) explore multiple structural states using ab initio, heterogeneous refinement, and density comparison (enable them in `session.json` when you need this).
- **Iterative optimization** — Built-in 2D optimization (e.g. classification and CryoSift-assisted refinement), box-size / diameter optimization, and repeated refinement loops toward better maps.

## 3. Installation

Step-by-step prerequisites, Zenodo tarball for CryoAlign2, `install.sh`, and JSON edits are documented in **[note/installation.md](note/installation.md)**.

You will still need any external tools (RELION, CryoSPARC client access, Helicon, CryoSift) configured separately to match `configs/master_config.json` and `configs/session.json`.

### Configuration after install

Edit `**configs/master_config.json`** for CryoSPARC host, credentials, LLM provider, and shared options. Prefer environment variables for secrets (for example `${DEEPSEEK_API_KEY}` in the config files). Step **4** of `install_all_envs.sh` can help set API keys and license values. In the frist round of installation, the script will guide you to input these information accordingly. 

Ensure CryoSPARC is reachable from the machine where you run CryoAgent, and that RELION paths and conda env names in config match your cluster or workstation.

Here are several ways to check the connections

```bash
# Verify setup (CryoSPARC / config sanity)
python check_cryosparc_connection.py

# Verify LLM connection
python check_LLM_connection.py
```

## 4. Preparing each dataset

Per dataset you mainly maintain two JSON files. Shared templates live under `configs/` in the repo; for batch runs, each dataset folder holds its own copies (see section 5).

### `configs/microscope_config.json` — acquisition and data paths

Set **dataset-specific** acquisition and input paths under `microscope_parameters`:


| Field                                    | Role                                                                                   |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| `pixel_size`, `voltage`, `cs_mm`, `dose` | Microscope / exposure parameters                                                       |
| `particle_diameter`, `symmetry`          | For picking and reconstruction                                                |
| `movies_path`                            | Movies to import (wildcards allowed, e.g. `*.mrc`)                                     |
| `micrographs_path`                       | Optional: pre-corrected micrographs (skips movie import / motion correction when used) |
| `gain_ref_path`, `gain_rot`, `gain_flip` | Gain reference and orientation                                                         |


Human-readable explanations for these keys are in `parameter_descriptions` in the same file.

### `configs/session.json` — modular pipeline and RELION/CryoSPARC session

- `**master_workflow.stages`** — List of stages with `enabled` flags (e.g. preprocessing, particle picking, `optimization_2d`, reconstruction, `optimization`, `polish`, heterogeneity stages). Turn stages on or off without editing `master_config.json`.
- `**relion**` — RELION executable, working directory (`relion_dir`), and backend options (timeouts, concurrency, `conda_env`).
- `**workflow**` — CryoSPARC `**project_uid**` and `**workspace_uid**` for this dataset.

For many datasets you only change `**relion.relion_dir**` (and related RELION paths if needed) and `**workflow.project_uid**` (and workspace if it differs); keep the rest aligned with your standard pipeline.

`session.json` in the same directory as `master_config.json` is **merged on top of** `master_config.json` (session wins on conflicts).

## 5. How to run

Activate the **cryoagent** conda environment by

```bash
conda activate cryoagent
```

### Single run — `cryoagent_workflow.py`

From the repository root, the default master config is `configs/master_config.json`. 
For each dataset specifically, one need to setup the `configs/microscope_config.json` and `configs/session.json`

```bash
# Verify setup (CryoSPARC / config sanity) (Can be skipped)
python cryoagent_workflow.py --workflow test

# Full enabled pipeline
python cryoagent_workflow.py

# Only selected stages (comma-separated, no spaces; useful for debugging)
python cryoagent_workflow.py --workflow custom --stages preprocessing,particle_picking
```

Other useful flags: `--config`, `--outputs-dir`, `--conversation-id`, `--verbose`, `--dry-run`. Run `python cryoagent_workflow.py --help` for the full list.

### Batch runs — `run_batch_datasets.py`

Runs the same workflow over **many datasets** in sequence. Each dataset is a folder under `datasets/unfinished_datasets/` (default) containing:

- `{dataset_name}/configs/session.json`
- `{dataset_name}/configs/microscope_config.json`

For example, a dataset named

- `datasets/unfinished_datasets/10240/configs/session.json`
- `datasets/unfinished_datasets/10240/configs/microscope_config.json`

The runner copies the repository `configs/master_config.json` into a temporary config for each dataset, overlays that dataset’s `session.json`, points the workflow at the dataset’s `microscope_config.json`, then calls `cryoagent_workflow.py`. Finished datasets can be moved to `datasets/finished_datasets/` 

```bash
# All dataset folders under the default unfinished directory
python run_batch_datasets.py

# Only named datasets
python run_batch_datasets.py --datasets my_dataset_a,my_dataset_b --workflow complete

```

## License

This project is licensed under the Apache 2.0 License; see the `LICENSE` file.