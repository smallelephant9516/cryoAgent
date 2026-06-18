"""Flexible LangChain tool factory.

All CryoSPARC tool adapters in this codebase take a SINGLE string argument and
parse it internally (via ``BaseReActAgent._parse_tool_input``). Wrapping them in
a plain single-input ``langchain.tools.Tool`` breaks under ``create_tool_calling_agent``:
when the model emits a structured multi-argument call (e.g.
``{"particles_job_uid": "J1", "volume_job_uid": "J2"}`` — which it does because
the descriptions enumerate named params), LangChain raises
``Too many arguments to single-input tool`` BEFORE the adapter runs.

``make_flexible_tool`` wraps any single-string adapter in a permissive
``StructuredTool`` (pydantic ``extra="allow"``) that accepts every input shape the
model might produce and routes it to the adapter unchanged:
named multi-arg, a single ``tool_input`` string or dict, a bare positional
string, or no args at all.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict


class FlexibleToolInput(BaseModel):
    """Permissive args schema: any named params the model emits are accepted
    (``extra="allow"``), or a single ``tool_input`` JSON string / value."""
    model_config = ConfigDict(extra="allow")
    tool_input: Optional[Any] = None


def make_flexible_tool(name: str, description: str,
                       func: Callable[[str], str]) -> StructuredTool:
    """Wrap a single-string adapter ``func(input_str)`` as a StructuredTool that
    tolerates structured multi-argument tool calls.

    The wrapper normalizes whatever the model sends into the single string the
    adapter already knows how to parse.
    """
    def wrapper(tool_input: Any = None, **kwargs: Any) -> str:
        if kwargs:
            # Model passed named params directly -> hand the adapter a JSON object.
            return func(json.dumps(kwargs))
        if tool_input is None:
            return func("{}")
        if isinstance(tool_input, str):
            return func(tool_input)
        return func(json.dumps(tool_input))

    return StructuredTool.from_function(
        func=wrapper, name=name, description=description,
        args_schema=FlexibleToolInput,
    )
