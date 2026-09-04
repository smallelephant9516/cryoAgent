# CryoAgent

## Introduction

CryoAgent is an agent-driven pipeline for single-particle cryo-EM (SPA). It uses an LLM-backed ReAct loop to drive **CryoSPARC** jobs through a multi-stage workflow: preprocessing, particle picking, 2D/3D optimization, reconstruction, optional polish, heterogeneity analysis, and post-run improvement.

The primary and fully validated backend is **CryoSPARC**. RELION interfaces are retained as an **experimental** integration and are not part of the validated end-to-end path.

CryoAgent uses **progressive autonomy**: a reliable guided workflow first, then parameter optimization, then hypothesis-driven exploration (`--improve`) once enough processing history has accumulated. The default run script executes guided workflow followed by exploration automatically.

## Requirements


| Requirement | Notes                                                                      |
| ----------- | -------------------------------------------------------------------------- |
| CryoSPARC   | **Version ≤ 4.7.1** (tested). Newer releases may change APIs.              |
| LLM API     | DeepSeek, OpenAI, or Panshi                                                |
| Conda       | `cryoagent` environment (created by `install.sh`)                          |
| CryoSift    | Required when `optimization_2d` or CryoSift-based picking is enabled       |
| Optional    | Helicon, CryoAlign2, ChimeraX — only when corresponding stages are enabled |


Example configuration templates: [note/connection_configs/gibh_feilab/](note/connection_configs/gibh_feilab/) and [note/connection_configs/ibp_118/](note/connection_configs/ibp_118/).

Further install notes (CryoAlign2, Docker): [note/installation.md](note/installation.md).

## Installation

```bash
git clone https://gitee.com/fei_sun_lab/cryoagent
cd cryoagent

# Download CryoAlign2 (optional; needed for map alignment in heterogeneity analysis)
curl -L https://zenodo.org/records/19552663/files/cryoalign_env.tar.gz?download=1 -o ./cryoalign_env.tar.gz

# Install conda environments (cryoagent, helicon, magellon2DAssess, CryoAlign2)
bash install.sh
```

Step 4 of `install.sh` interactively prompts for API keys and CryoSPARC license and can write them to `~/.bashrc`.

## Configuration files

CryoAgent is driven by three JSON files under `configs/`. Edit these before your first run.

For batch runs, each dataset folder under `datasets/unfinished_datasets/{name}/configs/` holds its own `session.json` and `microscope_config.json` while sharing `master_config.json`.

`session.json` is **merged on top of** `master_config.json` at runtime (session values win on conflicts).

### `configs/master_config.json`

Machine-wide settings: software connections, LLM provider, and shared tool paths. Edit once per installation.


| Section          | Key fields                                                                | Purpose                                                                |
| ---------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `cryosparc`      | `host`, `base_port`, `username`, `password`, `license_id`                 | CryoSPARC server connection. Use `${LICENSE_ID}` for the license.      |
| `agent`          | `provider`, `models.<provider>.model_name`, `models.<provider>.api_key`   | Active LLM provider and model. API keys as `${DEEPSEEK_API_KEY}`, etc. |
| `cryosift`       | `cryosift_env`, `cryosift_weights_path`, `cryosift_evaluator_script_path` | CryoSift conda env and file paths for 2D class evaluation.             |
| `job_management` | `default_timeout`, `max_retries`                                          | Job wait and retry defaults.                                           |
| `error_handling` | `max_consecutive_failures`, `fallback_strategies`                         | Workflow-level failure handling.                                       |
| `relion`         | `relion_exe`, `relion_dir`, `backend_execution`                           | **Experimental.** RELION paths if testing RELION integration.          |
| `transition`     | `helicon` conda env                                                       | Helicon-based format conversion between CryoSPARC and RELION.          |


**Typical first-time edits:**

```json
"cryosparc": {
  "host": "localhost",
  "base_port": 39000,
  "username": "your_email@example.com",
  "password": "your_password",
  "license_id": "${LICENSE_ID}"
},
"agent": {
  "provider": "deepseek",
  "models": {
    "deepseek": {
      "api_key": "${DEEPSEEK_API_KEY}",
      "model_name": "deepseek-v4-flash"
    }
  }
}
```



### `configs/session.json`

Per-dataset pipeline and CryoSPARC session. Change this file most often when switching datasets.


| Section                  | Key fields                                      | Purpose                                           |
| ------------------------ | ----------------------------------------------- | ------------------------------------------------- |
| `master_workflow.stages` | `name`, `enabled`, `agent_group`, `agent_class` | Which pipeline stages run and in what order.      |
| `workflow`               | `project_uid`, `workspace_uid`                  | CryoSPARC project and workspace for this dataset. |


**Available stages** (set `enabled: true/false`):


| Stage                 | Description                                                            |
| --------------------- | ---------------------------------------------------------------------- |
| `preprocessing`       | Import movies, motion correction, CTF estimation, micrograph selection |
| `particle_picking`    | Blob/template picking and particle extraction                          |
| `optimization_2d`     | 2D classification optimization with CryoSift                           |
| `reconstruction`      | Ab initio and heterogeneous/homogeneous refinement                     |
| `heterogeneity`       | Structural-state discovery via density comparison                      |
| `heterogeneity_depth` | Recursive heterogeneity refinement                                     |
| `optimization`        | 3D box-size / diameter optimization                                    |
| `polish`              | Final CTF refinement and polish                                        |


**Typical per-dataset edits:**

```json
"workflow": {
  "project_uid": "P1",
  "workspace_uid": "W1"
}
```



### `configs/microscope_config.json`

Per-dataset acquisition parameters and input data paths. All values under `microscope_parameters`:


| Field               | Role                                                                          |
| ------------------- | ----------------------------------------------------------------------------- |
| `pixel_size`        | Physical pixel size (Å)                                                       |
| `voltage`           | Accelerating voltage (kV)                                                     |
| `cs_mm`             | Spherical aberration coefficient (mm)                                         |
| `dose`              | Electron dose per frame (e⁻/Å²)                                               |
| `particle_diameter` | Estimated particle diameter (Å); used for picking and reconstruction defaults |
| `symmetry`          | Point-group symmetry (e.g. `C1`, `D7`)                                        |
| `movies_path`       | Path to movie files (wildcards allowed, e.g. `*.mrc`)                         |
| `micrographs_path`  | Optional pre-corrected micrographs (skips movie import and motion correction) |
| `gain_ref_path`     | Gain reference file (.mrc)                                                    |
| `gain_rot`          | Clockwise rotation for gain reference (0–3, in 90° steps)                     |
| `gain_flip`         | Gain flip code (0 = none, 1 = flip Y, 2 = flip X)                             |


Human-readable descriptions for every key are in `parameter_descriptions` in the same file.

**Example:**

```json
"microscope_parameters": {
  "pixel_size": 0.88,
  "voltage": 200,
  "cs_mm": 1.4,
  "dose": 61.4,
  "particle_diameter": 180,
  "symmetry": "D2",
  "movies_path": "/path/to/Movies/*.tiff",
  "gain_ref_path": "/path/to/gain.mrc",
  "gain_rot": 0,
  "gain_flip": 0
}
```



## Environment and credentials

Secrets are referenced in `configs/master_config.json` as `${VAR_NAME}` and resolved at runtime (including from a `.env` file in the repository root).


| Variable           | Used for                      |
| ------------------ | ----------------------------- |
| `DEEPSEEK_API_KEY` | DeepSeek LLM                  |
| `OPENAI_API_KEY`   | OpenAI LLM                    |
| `PANSHI_API_KEY`   | Panshi LLM (not fully tested) |
| `LICENSE_ID`       | CryoSPARC license             |


Set these via `install.sh` step 4, manual `export`, or a `.env` file. Then set `agent.provider` and `agent.models.<provider>.model_name` in `master_config.json`.

Verify setup (also run automatically by `scripts/run_cryoagent.sh`):

```bash
conda activate cryoagent
python check_LLM_connection.py
python check_cryosparc_connection.py
python check_cryosift_connection.py
```



## Quick start

**Using Python script:**

```bash
conda activate cryoagent
python cryoagent_workflow.py
# Then running the improvement mode if necessary
python cryoagent_workflow.py --improve
```

**Alternative (direct using bash script to include everything together)**:

```bash
# Default: guided workflow, then hypothesis-driven exploration
bash scripts/run_cryoagent.sh
```


| `--pipeline`     | What it runs                                         |
| ---------------- | ---------------------------------------------------- |
| `full` (default) | Guided workflow, then `--improve` if guided succeeds |
| `guided`         | `--mode guided` complete workflow only               |
| `exploration`    | `--improve` on outputs in `--outputs-dir` only       |


The wrapper activates the `cryoagent` conda environment, runs preflight checks (conda, LLM, CryoSPARC, CryoSift), then launches the selected pipeline. Run `bash scripts/run_cryoagent.sh --help` for all options.

**Batch processing:**

```bash
python run_batch_datasets.py
python run_batch_datasets.py --datasets my_dataset_a,my_dataset_b --workflow complete
```



## Advanced workflow modes

The run script `--pipeline` flag controls guided vs exploration at a high level. For finer control, pass flags through to `cryoagent_workflow.py`:


| Mode                | Flag                  | Description                                         |
| ------------------- | --------------------- | --------------------------------------------------- |
| Guided workflow     | `--mode guided`       | Fixed stage order from `session.json`               |
| Dynamic planning    | `--mode dynamic`      | LLM picks each next stage from prior JSON outputs   |
| De novo exploration | `--mode full_dynamic` | Single agent, full tool set, no predefined workflow |


```bash
bash scripts/run_cryoagent.sh --mode full_dynamic
# the goal mode is under test
bash scripts/run_cryoagent.sh --mode full_dynamic --goal "Process movies to 3D density"
```



## Monitoring and troubleshooting

Artifacts are written under `--outputs-dir` (default: `outputs/`).


| File                                       | When to check                                               |
| ------------------------------------------ | ----------------------------------------------------------- |
| `llm_conversation_<stage>_<timestamp>.log` | **First** on any failure — contains `TOOL EXECUTION:` lines |
| `workflow_state.json`                      | Structured stage history; input for `--improve`             |
| `*_results_cryosparc_*.json`               | Per-stage metrics and job UIDs                              |
| Final report JSON/MD                       | End-of-run summary                                          |


Only lines with `TOOL EXECUTION:` mean a tool was actually invoked. Natural-language "Action: ..." text without a matching `TOOL EXECUTION:` line was not executed.

**GUI visualizer:**

```bash
bash cryoagent_viz/run.sh   # open http://localhost:3000
```



## Prompts and customization

LLM prompts are in [cryoagent/prompts/](cryoagent/prompts/) (see [cryoagent/prompts/README.md](cryoagent/prompts/README.md)). Edit prompt templates without changing Python code.

## License

This project is licensed under the Apache 2.0 License; see the `LICENSE` file.

## References

If you use CryoAgent in your research, please cite:

- **Bioarxiv Paper:** https://www.biorxiv.org/content/10.64898/2026.04.16.718662v1
- **Code:** [https://gitee.com/fei_sun_lab/cryoagent](https://gitee.com/fei_sun_lab/cryoagent)

