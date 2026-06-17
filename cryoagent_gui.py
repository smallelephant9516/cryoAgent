#!/usr/bin/env python3
"""CryoAgent interactive GUI (Streamlit).

A second way to drive the pipeline alongside the existing CLI:

- **Quick mode** mirrors the current CLI: set the few key acquisition parameters
  and launch the workflow with the configured stages.
- **Interactive mode** is agentic. The app builds an editable *plan* (which
  stages run, in what order, and each step's parameters) from a non-destructive
  working copy of ``configs/``. You chat in natural language ("change the binning
  factor of motion correction to 2", "skip the polish stage"); the agent
  interprets your intent into concrete plan edits, you confirm them, and only
  then does it run the existing orchestrator against the edited working copy.

Run with:

    streamlit run cryoagent_gui.py

The canonical ``configs/`` directory is never modified by this app; each session
works on a copy under ``outputs/interactive/<run_id>/configs/``.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st

from cryoagent.interactive import (
    PlanApplier,
    PlanIntentParser,
    build_plan,
    create_working_configs,
)
from cryoagent.interactive.plan_intent_parser import IntentResult, PlanEdit

PROJECT_ROOT = Path(__file__).resolve().parent
CANONICAL_CONFIGS = PROJECT_ROOT / "configs"
INTERACTIVE_ROOT = PROJECT_ROOT / "outputs" / "interactive"

KEY_MICROSCOPE_PARAMS = [
    "movies_path",
    "micrographs_path",
    "pixel_size",
    "voltage",
    "cs_mm",
    "dose",
    "particle_diameter",
    "symmetry",
]


# ----------------------------------------------------------------------
# Session state helpers
# ----------------------------------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("session_started", False)
    ss.setdefault("run_id", None)
    ss.setdefault("working_configs", None)  # Path
    ss.setdefault("run_dir", None)  # Path
    ss.setdefault("outputs_dir", None)  # Path
    ss.setdefault("chat", [])  # list[{role, content}]
    ss.setdefault("pending", None)  # IntentResult awaiting confirmation
    ss.setdefault("proc_pid", None)
    ss.setdefault("log_path", None)  # Path
    ss.setdefault("running", False)


def _start_session() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = INTERACTIVE_ROOT / run_id
    working_configs = create_working_configs(CANONICAL_CONFIGS, run_dir)
    st.session_state.update(
        session_started=True,
        run_id=run_id,
        run_dir=run_dir,
        working_configs=working_configs,
        outputs_dir=run_dir / "outputs",
        chat=[],
        pending=None,
        proc_pid=None,
        log_path=None,
        running=False,
    )


def _master_config_path() -> Path:
    return Path(st.session_state.working_configs) / "master_config.json"


# ----------------------------------------------------------------------
# Pipeline launching (subprocess -> log file the UI tails)
# ----------------------------------------------------------------------
def _launch_pipeline(mode: str, workflow: str, goal: Optional[str]) -> None:
    run_dir: Path = st.session_state.run_dir
    outputs_dir: Path = st.session_state.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "cryoagent_workflow.py"),
        "--config", str(_master_config_path()),
        "--workflow", workflow,
        "--outputs-dir", str(outputs_dir),
        "--mode", mode,
        "--conversation-id", f"interactive_{st.session_state.run_id}",
    ]
    if goal:
        cmd += ["--goal", goal]

    log_file = open(log_path, "w", encoding="utf-8")
    log_file.write(f"$ {' '.join(cmd)}\n\n")
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    st.session_state.proc_pid = proc.pid
    st.session_state.log_path = log_path
    st.session_state.running = True
    # Keep a reference so the process isn't garbage-collected.
    st.session_state["_proc"] = proc


def _poll_pipeline() -> Optional[int]:
    """Return the process return code if finished, else None."""
    proc = st.session_state.get("_proc")
    if proc is None:
        return None
    code = proc.poll()
    if code is not None:
        st.session_state.running = False
    return code


def _read_log(tail_lines: int = 400) -> str:
    log_path = st.session_state.get("log_path")
    if not log_path or not Path(log_path).exists():
        return ""
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-tail_lines:])


# ----------------------------------------------------------------------
# Rendering: plan view
# ----------------------------------------------------------------------
def _render_plan() -> None:
    plan = build_plan(st.session_state.working_configs)

    st.subheader("Current plan")
    st.caption(
        "Stages run top-to-bottom in this order. Toggle a stage to enable/disable "
        "it; use the chat to change step parameters or reorder stages."
    )

    for stage in plan.stages:
        label = f"{stage.order + 1}. {stage.name}  ({stage.agent_group})"
        new_enabled = st.checkbox(
            label,
            value=stage.enabled,
            key=f"enable_{stage.name}",
            help=stage.description,
        )
        if new_enabled != stage.enabled:
            applier = PlanApplier(st.session_state.working_configs)
            op = "enable_stage" if new_enabled else "disable_stage"
            applier.apply([PlanEdit(op=op, stage=stage.name, summary=f"{op} {stage.name}")])
            st.rerun()

        if stage.steps:
            with st.expander(f"{stage.name} parameters", expanded=False):
                for step in stage.steps:
                    if not step.params:
                        continue
                    st.markdown(f"**{step.name}**" + (f" — {step.description}" if step.description else ""))
                    rows = {p.name: p.value for p in step.params}
                    st.table({"parameter": list(rows.keys()), "value": [str(v) for v in rows.values()]})

    if plan.microscope_params:
        with st.expander("Microscope parameters", expanded=False):
            rows = {p.name: p.value for p in plan.microscope_params}
            st.table({"parameter": list(rows.keys()), "value": [str(v) for v in rows.values()]})


# ----------------------------------------------------------------------
# Rendering: chat + intent confirmation
# ----------------------------------------------------------------------
def _render_edit_proposal(result: IntentResult) -> None:
    valid = result.valid_edits
    invalid = result.invalid_edits

    if valid:
        st.markdown("**Proposed changes** (confirm to apply):")
        for edit in valid:
            st.markdown(f"- {edit.summary or _describe_edit(edit)}")
    if invalid:
        st.markdown("**Could not apply:**")
        for edit in invalid:
            st.markdown(f"- {edit.summary or _describe_edit(edit)} — _{edit.error}_")

    col_a, col_b = st.columns(2)
    with col_a:
        if valid and st.button("Confirm & apply", type="primary", key="confirm_edits"):
            applier = PlanApplier(st.session_state.working_configs)
            messages = applier.apply(valid)
            st.session_state.chat.append(
                {"role": "assistant", "content": "Applied:\n" + "\n".join(f"- {m}" for m in messages)}
            )
            st.session_state.pending = None
            st.rerun()
    with col_b:
        if st.button("Discard", key="discard_edits"):
            st.session_state.pending = None
            st.session_state.chat.append({"role": "assistant", "content": "Discarded the proposed changes."})
            st.rerun()


def _describe_edit(edit: PlanEdit) -> str:
    if edit.op == "set_param":
        return f"Set {edit.stage}.{edit.step}.{edit.param} = {edit.value}"
    if edit.op == "enable_stage":
        return f"Enable stage {edit.stage}"
    if edit.op == "disable_stage":
        return f"Disable stage {edit.stage}"
    if edit.op == "reorder":
        return f"Reorder stages: {' -> '.join(edit.order or [])}"
    return edit.op


def _render_chat() -> None:
    st.subheader("Talk to the agent")
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pending: Optional[IntentResult] = st.session_state.pending
    if pending is not None:
        with st.chat_message("assistant"):
            if pending.reply:
                st.markdown(pending.reply)
            _render_edit_proposal(pending)

    prompt = st.chat_input("e.g. change the binning factor of motion correction to 2")
    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        plan = build_plan(st.session_state.working_configs)
        parser = PlanIntentParser(str(_master_config_path()))
        with st.spinner("Interpreting your request..."):
            result = parser.parse(prompt, plan)
        st.session_state.pending = result
        st.rerun()


# ----------------------------------------------------------------------
# Rendering: run controls + logs
# ----------------------------------------------------------------------
def _render_run_controls() -> None:
    st.subheader("Run")
    code = _poll_pipeline()

    col1, col2, col3 = st.columns(3)
    with col1:
        run_mode = st.selectbox(
            "Execution mode",
            ["guided", "dynamic"],
            help="guided: run enabled stages in order, re-plan on failure. "
            "dynamic: the LLM planner chooses each next stage.",
        )
    with col2:
        workflow = st.selectbox("Workflow", ["complete", "preprocessing"], index=0)
    with col3:
        goal = st.text_input("Goal (optional)", value="")

    disabled = st.session_state.running
    if st.button("Run pipeline", type="primary", disabled=disabled):
        _launch_pipeline(run_mode, workflow, goal.strip() or None)
        st.rerun()

    if st.session_state.running:
        st.info(f"Pipeline running (PID {st.session_state.proc_pid})...")
    elif code is not None:
        if code == 0:
            st.success("Pipeline finished successfully.")
        else:
            st.error(f"Pipeline exited with code {code}.")

    if st.session_state.log_path:
        auto = st.checkbox("Auto-refresh logs", value=st.session_state.running)
        st.code(_read_log() or "(no output yet)", language="text")
        if auto and st.session_state.running:
            time.sleep(2)
            st.rerun()


# ----------------------------------------------------------------------
# Quick mode (mirrors current CLI)
# ----------------------------------------------------------------------
def _render_quick_mode() -> None:
    import json

    st.header("Quick mode")
    st.caption(
        "The current CLI behavior: set the key acquisition parameters, then run "
        "the workflow with the stages configured in configs/session.json."
    )

    micro_path = CANONICAL_CONFIGS / "microscope_config.json"
    try:
        micro = json.loads(micro_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read {micro_path}: {exc}")
        return

    params = micro.get("microscope_parameters", {})
    st.warning(
        "Quick mode edits the canonical configs/microscope_config.json in place "
        "(matching the CLI). Use Interactive mode for a non-destructive session."
    )

    edited = {}
    with st.form("quick_params"):
        for key in KEY_MICROSCOPE_PARAMS:
            if key not in params:
                continue
            current = params.get(key)
            if isinstance(current, bool):
                edited[key] = st.checkbox(key, value=current)
            elif isinstance(current, (int, float)) and not isinstance(current, bool):
                edited[key] = st.number_input(key, value=float(current))
            else:
                edited[key] = st.text_input(key, value="" if current is None else str(current))
        workflow = st.selectbox("Workflow", ["complete", "preprocessing"], index=0)
        run_mode = st.selectbox("Execution mode", ["guided", "dynamic"], index=0)
        submitted = st.form_submit_button("Save parameters & show run command")

    if submitted:
        for key, value in edited.items():
            if isinstance(params.get(key), int) and not isinstance(params.get(key), bool):
                value = int(value)
            if value == "":
                value = None
            params[key] = value
        micro["microscope_parameters"] = params
        micro_path.write_text(json.dumps(micro, indent=2, ensure_ascii=False), encoding="utf-8")
        st.success(f"Saved parameters to {micro_path}.")
        st.markdown("Run from a terminal:")
        st.code(
            f"python cryoagent_workflow.py --config configs/master_config.json "
            f"--workflow {workflow} --mode {run_mode}",
            language="bash",
        )


# ----------------------------------------------------------------------
# Interactive mode
# ----------------------------------------------------------------------
def _render_interactive_mode() -> None:
    st.header("Interactive mode")

    if not st.session_state.session_started:
        st.caption(
            "Start a session to create a non-destructive working copy of your "
            "configs. You can then chat to shape the plan before running."
        )
        if st.button("Start interactive session", type="primary"):
            _start_session()
            st.rerun()
        return

    st.caption(f"Session `{st.session_state.run_id}` — working configs at `{st.session_state.working_configs}`")
    if st.button("Reset session (new working copy)"):
        _start_session()
        st.rerun()

    left, right = st.columns([1, 1])
    with left:
        _render_plan()
    with right:
        _render_chat()

    st.divider()
    _render_run_controls()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="CryoAgent", layout="wide")
    _init_state()

    st.title("CryoAgent")
    mode = st.sidebar.radio(
        "Mode",
        ["Interactive (agentic)", "Quick (current CLI)"],
        help="Interactive: chat to shape and run the plan. Quick: set a few "
        "parameters and run, like the existing CLI.",
    )

    if mode.startswith("Quick"):
        _render_quick_mode()
    else:
        _render_interactive_mode()


if __name__ == "__main__":
    main()
