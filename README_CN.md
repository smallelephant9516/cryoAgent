# CryoAgent

## 1. 简介

CryoAgent 是基于智能体的单颗粒冷冻电镜（cryo-EM）流程工具。它通过统一的配置模型驱动 **RELION** 与 **CryoSPARC**，采用由大语言模型（LLM）支撑的 ReAct 循环来选择动作，并执行从预处理到重构、以及异质性分析等多阶段工作流。

## 2. 功能特性

- **RELION 与 CryoSPARC 集成** — 同一工作流可在各阶段按需使用两套软件栈（CryoSPARC 作业、RELION 目录与可执行文件，以及在配置启用时的 Helicon 风格衔接）。
- **工作流监控与失败处理** — 各阶段可被追踪；编排器可重试、应用回退策略并暴露错误，避免运行静默失败。
- **自动化异质性分析** — 可选阶段（`heterogeneity`、`heterogeneity_depth`）通过从头算、异质性精修与密度比较探索多种结构状态（需要时在 `session.json` 中启用）。
- **迭代优化** — 内置二维优化（如分类与 CryoSift 辅助精修）、盒尺寸 / 直径优化，以及面向更好电子密度图的重复精修循环。

## 3. 安装

分步前置条件、CryoAlign2 的 Zenodo 压缩包、`install.sh` 以及 JSON 修改说明见 **[note/installation.md](note/installation.md)**。

你仍需单独配置外部工具（RELION、CryoSPARC 客户端访问、Helicon、CryoSift），并与 `configs/master_config.json`、`configs/session.json` 保持一致。

### 安装后的配置

编辑 **`configs/master_config.json`**，设置 CryoSPARC 主机、凭据、LLM 提供商及共享选项。建议对密钥使用环境变量（例如配置中的 `${DEEPSEEK_API_KEY}`）。`install_all_envs.sh` 的步骤 **4** 可帮助设置 API 密钥与许可证相关值。首次安装时，脚本会引导你按步骤输入这些信息。

确保从运行 CryoAgent 的机器可以访问 CryoSPARC，且配置中的 RELION 路径与 conda 环境名称与你的集群或工作站一致。

可通过以下方式检查连接：

```bash
# 检查环境与 CryoSPARC / 配置是否正常
python check_cryosparc_connection.py

# 检查 LLM 连接
python check_LLM_connection.py
```

## 4. 为每个数据集做准备

每个数据集主要维护两个 JSON 文件。仓库中的共享模板位于 `configs/`；批量运行时，每个数据集目录下会有自己的副本（见第 5 节）。

### `configs/microscope_config.json` — 采集参数与数据路径

在 `microscope_parameters` 下设置**数据集专用**的采集与输入路径：


| 字段 | 作用 |
| ---------------------------------------- | -------------------------------------------------------------------------------------- |
| `pixel_size`, `voltage`, `cs_mm`, `dose` | 显微镜 / 曝光参数 |
| `particle_diameter`, `symmetry` | 颗粒挑选与重构的默认值 |
| `movies_path` | 要导入的原始电影文件（可使用通配符，例如 `*.mrc`） |
| `micrographs_path` | 可选：已校正的显微照片（使用时跳过电影导入 / 运动校正） |
| `gain_ref_path`, `gain_rot`, `gain_flip` | 增益参考及其方向 |


同一文件中的 `parameter_descriptions` 对这些键有更易读的说明。

### `configs/session.json` — 模块化流程与 RELION / CryoSPARC 会话

- **`master_workflow.stages`** — 带 `enabled` 标志的阶段列表（如预处理、颗粒挑选、`optimization_2d`、重构、`optimization`、`polish`、异质性相关阶段）。无需编辑 `master_config.json` 即可开关各阶段。
- **`relion`** — RELION 可执行文件、工作目录（`relion_dir`）及后端选项（超时、并发、`conda_env`）。
- **`workflow`** — 本数据集在 CryoSPARC 中的 **`project_uid`** 与 **`workspace_uid`**。

对许多数据集，通常只需修改 **`relion.relion_dir`**（及必要的其他 RELION 路径）与 **`workflow.project_uid`**（若 workspace 不同则一并修改）；其余项与标准流程保持一致即可。

与 `master_config.json` 同目录下的 `session.json` 会**叠加合并**到 `master_config.json` 之上（发生冲突时以 session 为准）。

## 5. 如何运行

先激活 **cryoagent** conda 环境：

```bash
conda activate cryoagent
```

### 单次运行 — `cryoagent_workflow.py`

在仓库根目录下，默认主配置为 `configs/master_config.json`。  
针对每个具体数据集，需要设置好该数据集的 `configs/microscope_config.json` 与 `configs/session.json`。

```bash
# 检查环境与 CryoSPARC / 配置（可跳过）
python cryoagent_workflow.py --workflow test

# 运行当前已启用的完整流程
python cryoagent_workflow.py

# 仅运行指定阶段（逗号分隔，勿加空格；便于调试）
python cryoagent_workflow.py --workflow custom --stages preprocessing,particle_picking
```

其他常用参数：`--config`、`--outputs-dir`、`--conversation-id`、`--verbose`、`--dry-run`。完整列表请执行 `python cryoagent_workflow.py --help`。

### 批量运行 — `run_batch_datasets.py`

按顺序对**多个数据集**运行同一工作流。每个数据集是 `datasets/unfinished_datasets/`（默认）下的一个文件夹，其中包含：

- `{dataset_name}/configs/session.json`
- `{dataset_name}/configs/microscope_config.json`

例如名为 `10240` 的数据集：

- `datasets/unfinished_datasets/10240/configs/session.json`
- `datasets/unfinished_datasets/10240/configs/microscope_config.json`

运行器会为每个数据集复制仓库中的 `configs/master_config.json` 到临时配置，叠加该数据集的 `session.json`，将工作流指向该数据集的 `microscope_config.json`，再调用 `cryoagent_workflow.py`。完成的数据集可移至 `datasets/finished_datasets/`。

```bash
# 默认未完成目录下的所有数据集文件夹
python run_batch_datasets.py

# 仅运行指定名称的数据集
python run_batch_datasets.py --datasets my_dataset_a,my_dataset_b --workflow complete

```
