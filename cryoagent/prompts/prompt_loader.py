"""Load CryoSPARC agent prompts from markdown files (Claude Code / OpenClaw / Cursor compatible)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

PROMPTS_ROOT = Path(__file__).resolve().parent


class PromptNotFoundError(FileNotFoundError):
    """Raised when a prompt markdown file cannot be resolved."""


def prompt_path(*parts: str) -> Path:
    """Resolve a path under cryoagent/prompts/."""
    return PROMPTS_ROOT.joinpath(*parts)


def load_prompt(
    relative_path: str,
    variables: Optional[Mapping[str, Any]] = None,
    *,
    strict: bool = False,
) -> str:
    """
    Load a markdown prompt and substitute ``{{key}}`` placeholders.

    Args:
        relative_path: Path relative to cryoagent/prompts/ (e.g. "cryosparc/preprocessing/system.md").
        variables: Template variables. Missing keys leave the placeholder unchanged unless strict=True.
        strict: If True, raise KeyError when a placeholder has no matching variable.

    Returns:
        Rendered prompt text without YAML frontmatter (frontmatter is stripped when present).
    """
    path = prompt_path(relative_path)
    if not path.is_file():
        raise PromptNotFoundError(f"Prompt file not found: {path}")

    text = path.read_text(encoding="utf-8")
    text = _strip_frontmatter(text)

    if not variables:
        return text

    return render_template(text, variables, strict=strict)


def render_template(
    template: str,
    variables: Mapping[str, Any],
    *,
    strict: bool = False,
) -> str:
    """Replace ``{{key}}`` placeholders in template text."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        if key not in variables:
            if strict:
                raise KeyError(f"Missing template variable: {key}")
            return match.group(0)
        value = variables[key]
        return "" if value is None else str(value)

    return re.sub(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}", _replace, template)


def _strip_frontmatter(text: str) -> str:
    """Remove optional YAML frontmatter block used by SKILL.md / subagent.md files."""
    if not text.startswith("---\n"):
        return text.lstrip("\n")
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + 5 :].lstrip("\n")


def load_subagent_prompt(agent_name: str) -> str:
    """
    Load a Claude Code / OpenClaw subagent body from cryoagent/prompts/subagents/<name>.md.

    The file may include YAML frontmatter; only the markdown body is returned for LangChain.
    """
    return load_prompt(f"subagents/{agent_name}.md")
