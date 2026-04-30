"""DeepSeek-specific chat model adapter for multi-turn thinking mode."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr


class DeepSeekChatAdapter(ChatOpenAI):
    """
    DeepSeek adapter that replays reasoning_content for thinking-mode continuity.

    DeepSeek V4 thinking mode requires the previous assistant reasoning payload to
    be passed back on every continuation turn.  This adapter:

      - Overrides ``_generate`` (and ``_agenerate``) to use a raw urllib HTTP call
        instead of the openai Python client, giving precise control over the payload.
      - Before each request: injects the cached ``reasoning_content`` into the last
        assistant message that contains ``tool_calls``.
      - After each response: captures the new ``reasoning_content`` for the next turn.

    ``disable_streaming = True`` forces LangChain to always use ``_generate`` rather
    than ``_stream``, even when verbose/streaming callbacks are present (e.g. when
    AgentExecutor is configured with verbose=True).

    Private state uses Pydantic ``PrivateAttr`` so it persists across model copies.
    """

    # Force LangChain to always call _generate (not _stream), regardless of callbacks.
    disable_streaming: bool = True

    # Maps the first tool_call id of each assistant response → its reasoning_content.
    # This lets every assistant message in a multi-turn history get the right reasoning
    # replayed, not just the most-recent one.
    _reasoning_by_tool_call_id: Dict[str, Any] = PrivateAttr(default_factory=dict)
    # Tool call ids from the most recent assistant response (used during capture).
    _pending_tool_call_ids: List[str] = PrivateAttr(default_factory=list)

    # ------------------------------------------------------------------
    # Pydantic lifecycle
    # ------------------------------------------------------------------

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)

    # ------------------------------------------------------------------
    # API parameter resolution
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        key_obj = getattr(self, "openai_api_key", None)
        if key_obj is None:
            return ""
        if hasattr(key_obj, "get_secret_value"):
            return str(key_obj.get_secret_value() or "")
        return str(key_obj or "")

    def _resolve_base_url(self) -> str:
        base = (
            getattr(self, "openai_api_base", None)
            or getattr(self, "base_url", None)
            or ""
        )
        return str(base).strip().rstrip("/")

    def _resolve_timeout(self) -> float:
        timeout = getattr(self, "request_timeout", None) or getattr(self, "timeout", None)
        try:
            return float(timeout if timeout is not None else 60.0)
        except (TypeError, ValueError):
            return 60.0

    # ------------------------------------------------------------------
    # Raw HTTP
    # ------------------------------------------------------------------

    def _post_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        base_url = self._resolve_base_url()
        if not base_url:
            raise ValueError("[DeepSeekAdapter] missing base URL")
        api_key = self._resolve_api_key()
        if not api_key:
            raise ValueError("[DeepSeekAdapter] missing API key")

        url = f"{base_url}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            url=url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        timeout = self._resolve_timeout()
        try:
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"[DeepSeekAdapter] HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"[DeepSeekAdapter] request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Reasoning-content capture
    # ------------------------------------------------------------------

    def _capture_reasoning(self, response_json: Dict[str, Any]) -> None:
        """Store reasoning_content keyed by each tool_call_id the model just produced."""
        reasoning = self._find_reasoning(response_json)

        # Extract tool_call ids from the response
        choices = response_json.get("choices") or []
        tool_call_ids: List[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") or {}
            for tc in message.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    tool_call_ids.append(tc["id"])

        if reasoning and tool_call_ids:
            for tc_id in tool_call_ids:
                self._reasoning_by_tool_call_id[tc_id] = reasoning
        # Always keep a fallback for non-tool-call turns (e.g. final answer after last tool)
        if reasoning:
            self._pending_tool_call_ids = tool_call_ids

    def _find_reasoning(self, value: Any) -> Any:
        if isinstance(value, str):
            return None
        if isinstance(value, Mapping):
            for key in ("reasoning_content", "reasoning"):
                candidate = value.get(key)
                if candidate:
                    return candidate
            for nested in value.values():
                found = self._find_reasoning(nested)
                if found:
                    return found
        elif isinstance(value, Sequence):
            for item in value:
                found = self._find_reasoning(item)
                if found:
                    return found
        return None

    # ------------------------------------------------------------------
    # Reasoning-content injection
    # ------------------------------------------------------------------

    def _inject_reasoning(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inject reasoning_content into every assistant message that has tool_calls."""
        if not self._reasoning_by_tool_call_id:
            return messages

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue
            if msg.get("reasoning_content"):
                continue  # already present

            # Find matching reasoning by looking up any of the message's tool_call ids
            matched_reasoning = None
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if tc_id and tc_id in self._reasoning_by_tool_call_id:
                    matched_reasoning = self._reasoning_by_tool_call_id[tc_id]
                    break

            if matched_reasoning:
                msg["reasoning_content"] = matched_reasoning

        return messages

    # ------------------------------------------------------------------
    # Payload builder
    # ------------------------------------------------------------------

    def _build_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            return super()._get_request_payload(messages, stop=stop, **kwargs)
        except Exception as exc:
            print(f"[DeepSeekAdapter] _get_request_payload failed ({exc}), falling back", flush=True)
            return self._build_payload_manually(messages, stop, **kwargs)

    def _build_payload_manually(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
            "function": "function",
        }
        serialized: List[Dict[str, Any]] = []
        for msg in messages:
            role = role_map.get(getattr(msg, "type", ""), "user")
            content = getattr(msg, "content", "") or ""
            entry: Dict[str, Any] = {"role": role, "content": content}

            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id:
                entry["tool_call_id"] = tool_call_id

            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("args", {})),
                        },
                    }
                    for tc in tool_calls
                    if isinstance(tc, dict)
                ]

            additional_kwargs = dict(getattr(msg, "additional_kwargs", {}) or {})
            if additional_kwargs.get("reasoning_content"):
                entry["reasoning_content"] = additional_kwargs["reasoning_content"]

            serialized.append(entry)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": serialized,
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)
        return payload

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response_json: Dict[str, Any]) -> AIMessage:
        choices = response_json.get("choices") or []
        if not choices:
            raise RuntimeError(f"[DeepSeekAdapter] response missing choices: {response_json}")

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content: str = message.get("content") or ""

        tool_calls_raw: List[Dict[str, Any]] = message.get("tool_calls") or []
        parsed_tool_calls: List[Dict[str, Any]] = []
        for tc in tool_calls_raw:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name")
            args_raw = fn.get("arguments", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
            except json.JSONDecodeError:
                args = {}
            if name:
                parsed_tool_calls.append(
                    {"id": tc.get("id"), "type": "tool_call", "name": name, "args": args}
                )

        additional_kwargs: Dict[str, Any] = {}
        if tool_calls_raw:
            additional_kwargs["tool_calls"] = tool_calls_raw

        reasoning_content = (
            message.get("reasoning_content")
            or choice.get("reasoning_content")
            or response_json.get("reasoning_content")
        )
        if reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content

        return AIMessage(
            content=content,
            tool_calls=parsed_tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata={
                "finish_reason": choice.get("finish_reason"),
                "model": response_json.get("model"),
                "usage": response_json.get("usage"),
                "reasoning_content": reasoning_content,
            },
        )

    # ------------------------------------------------------------------
    # Core generation override
    # ------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(deepcopy(messages), stop=stop, **kwargs)
        payload["messages"] = self._inject_reasoning(payload["messages"])

        response_json = self._post_chat_completions(payload)
        self._capture_reasoning(response_json)

        ai_message = self._parse_response(response_json)
        generation = ChatGeneration(
            message=ai_message,
            generation_info={"raw_response": response_json},
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "token_usage": response_json.get("usage"),
                "model": response_json.get("model"),
            },
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        import asyncio
        return await asyncio.to_thread(
            self._generate, messages, stop, run_manager, **kwargs
        )
