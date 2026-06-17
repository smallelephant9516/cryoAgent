"""LangChain callback handler that logs the LLM's intermediate reasoning.

The AgentExecutor only returns its final ``output`` text; the per-step reasoning
the model emits before each tool call (LangChain exposes this as ``action.log``,
shown in the terminal as the "responded: ..." lines) is otherwise lost. This
handler taps the agent's callbacks and forwards that intermediate reasoning — and
the final answer — to the realtime conversation logger so each stage's .log file
is a faithful transcript of what the LLM actually said, not just which tools ran.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:  # pragma: no cover - fallback for older langchain layouts
    from langchain.callbacks.base import BaseCallbackHandler


def _clean_reasoning(text: str) -> str:
    """Trim LangChain's action log to the human-readable reasoning portion.

    ``action.log`` often contains the model's narration followed by an
    ``Invoking: `tool` with `args``` line. We keep the narration (the part the
    user sees after "responded:") and drop trailing tool-invocation boilerplate,
    since the tool call itself is logged separately by the tool executor.
    """
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Invoking:") or stripped.startswith("Action:") \
                or stripped.startswith("Action Input:"):
            break
        lines.append(line)
    return "\n".join(lines).strip()


class ReasoningLogCallbackHandler(BaseCallbackHandler):
    """Forward intermediate LLM reasoning + final output to a realtime logger."""

    def __init__(self, realtime_logger):
        super().__init__()
        self.realtime_logger = realtime_logger
        self._last_logged_text = None

    def _log(self, text: str, metadata: dict) -> None:
        """Write reasoning to the log, skipping an immediate duplicate."""
        if not text or text == self._last_logged_text:
            return
        self._last_logged_text = text
        try:
            self.realtime_logger.log_assistant_response(text, metadata=metadata)
        except Exception:
            # Logging must never break the workflow.
            pass

    def _enabled(self) -> bool:
        return (
            self.realtime_logger is not None
            and getattr(self.realtime_logger, "current_log_file", None)
        )

    @staticmethod
    def _content_to_text(content) -> str:
        """Flatten an AIMessage.content (str OR list of content blocks) to text."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    # text blocks look like {"type": "text", "text": "..."}
                    if block.get("type") == "text" and block.get("text"):
                        parts.append(block["text"])
                    elif block.get("text"):
                        parts.append(block["text"])
            return "\n".join(p for p in parts if p).strip()
        return str(content).strip()

    def on_llm_end(self, response, **kwargs: Any) -> None:
        """Capture the model's reasoning text emitted alongside each tool-call turn.

        With create_tool_calling_agent the model returns an AIMessage whose
        ``content`` holds the narration ("responded: ..." text) and whose
        ``tool_calls`` hold the action. on_agent_action does NOT fire for these
        agents, so this is where the intermediate reasoning is captured.
        """
        if not self._enabled():
            return
        try:
            generations = getattr(response, "generations", None) or []
            for gen_list in generations:
                for gen in gen_list:
                    message = getattr(gen, "message", None)
                    # Reasoning text: prefer the message content, fall back to gen.text.
                    text = ""
                    if message is not None:
                        text = self._content_to_text(getattr(message, "content", None))
                    if not text:
                        text = (getattr(gen, "text", "") or "").strip()
                    if not text:
                        continue
                    # Note the tool(s) this turn is about to invoke, if any.
                    metadata = {"phase": "reasoning"}
                    tool_calls = getattr(message, "tool_calls", None) if message is not None else None
                    if tool_calls:
                        names = [tc.get("name") for tc in tool_calls if isinstance(tc, dict) and tc.get("name")]
                        if names:
                            metadata["next_tool"] = ", ".join(names)
                    self._log(text, metadata)
        except Exception:
            # Logging must never break the workflow.
            pass

    def on_agent_action(self, action, **kwargs: Any) -> None:
        """Fallback for ReAct-style agents that DO emit AgentAction.log."""
        if not self._enabled():
            return
        reasoning = _clean_reasoning(getattr(action, "log", "") or "")
        if not reasoning:
            return
        tool = getattr(action, "tool", None)
        metadata = {"phase": "reasoning"}
        if tool:
            metadata["next_tool"] = tool
        self._log(reasoning, metadata)

    def on_agent_finish(self, finish, **kwargs: Any) -> None:
        """Capture the model's final answer text."""
        if not self._enabled():
            return
        text = ""
        ret = getattr(finish, "return_values", None)
        if isinstance(ret, dict):
            text = ret.get("output") or ret.get("text") or ""
        if not text:
            text = _clean_reasoning(getattr(finish, "log", "") or "")
        if not text:
            return
        # The final answer often already arrived via on_llm_end (same text); _log
        # dedups it. Use 'final' metadata only when it's genuinely new text.
        if text == self._last_logged_text:
            return
        self._log(text.strip(), {"phase": "final"})
