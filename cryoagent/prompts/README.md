# CryoAgent Prompt Layout

Prompts are separated from Python so the same text works in:

- **CryoAgent runtime** (LangChain ReAct via `load_prompt()`)
- **Claude Code** subagents (`.claude/agents/*.md`)
- **Cursor / OpenClaw** skills (`.cursor/skills/*/SKILL.md`)

## Directory layout

```
cryoagent/prompts/
├── prompt_loader.py          # {{placeholder}} rendering
├── shared/
│   └── react-wrapper.md      # ReAct user-message wrapper
└── cryosparc/
    ├── preprocessing/
    │   ├── system.md         # Agent system prompt (subagent body)
    │   └── task.md           # Per-run task (minimal)
    ├── optimization_2d/
    │   ├── system.md
    │   ├── task_both.md
    │   ├── task_f1_only.md
    │   ├── task_f2_only.md
    │   └── task_none.md
    └── ...                   # one folder per stage

.claude/agents/               # generated Claude Code subagents
.cursor/skills/               # generated Cursor skills
```

## Two prompt roles

| File | Role | Analogue |
|------|------|----------|
| `system.md` | Stage manual: tools, rules, edge cases | Claude Code **subagent body** |
| `task*.md` | This run: job UIDs, enabled branch | Synthetic **user message** |

Python only builds a **context dict** and selects which template to load.

## Template syntax

Use `{{variable_name}}` placeholders. Python fills them at runtime:

```python
from cryoagent.prompts.prompt_loader import load_prompt

prompt = load_prompt(
    "cryosparc/optimization_2d/system.md",
    {"project_uid": "P30", "enable_f1_status": "ENABLED", ...},
)
```

## Claude Code / OpenClaw compatibility

### Subagents (Claude Code)

Generated files live in `.claude/agents/cryosparc-<stage>.md`:

```markdown
---
name: cryosparc-preprocessing
description: When to delegate to this agent...
model: inherit
---

[contents of system.md]
```

Regenerate after editing prompts:

```bash
python scripts/sync_claude_openclaw_prompts.py
```

### Skills (Cursor / OpenClaw)

Generated files live in `.cursor/skills/cryosparc-<stage>/SKILL.md` with standard frontmatter:

```markdown
---
name: cryosparc-preprocessing
description: ...
disable-model-invocation: true
---
```

OpenClaw uses the same `SKILL.md` + YAML frontmatter pattern. Point your OpenClaw workspace at `.cursor/skills/` or copy skills into your OpenClaw skills directory.

## Migration checklist (per stage)

1. **Extract** embedded f-strings:
   ```bash
   python scripts/extract_cryosparc_prompts.py
   ```
2. **Convert** remaining `{python_expr}` in `system.md` / `task.md` to `{{name}}`.
3. **Add** `_get_system_prompt_context()` on the agent; return `load_prompt(...)`.
4. **Replace** workflow inline prompts with `load_prompt("cryosparc/<stage>/task....md", ctx)`.
5. **Sync** Claude/Cursor definitions:
   ```bash
   python scripts/sync_claude_openclaw_prompts.py
   ```

## Migration status

| Stage | system.md | task.md | Python wired |
|-------|-----------|---------|--------------|
| preprocessing | done | done | **done** |
| particle_picking | done | done | **done** |
| optimization_2d | done | done | **done** |
| reconstruction | done | done | **done** |
| heterogeneity | done | done | **done** |
| heterogeneity_depth | done | done | **done** |
| optimization | done | done (+ sections/) | **done** |
| polish | done | done | **done** |

All CryoSPARC agents load prompts via `load_prompt()` — no inline prompt strings remain in `cryosparc_*` Python modules.

## Reference implementation

See `cryoagent/core/cryosparc_2Doptimize/optimizer_2d_agent.py` and `optimizer_2d_workflow.py`.
