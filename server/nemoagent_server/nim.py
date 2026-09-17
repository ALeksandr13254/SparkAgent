"""Model client: OpenCode Go ($10/mo) for text roles, NVIDIA NIM for embeddings.

SparkAgent keeps NemoAgent's internal chat shape (messages + tools) so agent.py,
traces and UI logs are untouched; translation happens here per model transport:
- Responses (`/responses`): Muse Spark contributors, Grok 4.6, GPT 5.6 Luna, DeepSeek flashes.
  System messages -> `instructions`, rest -> `input` items, reasoning effort pinned per role.
- Chat (`/chat/completions`): all other Go models, OpenAI SSE shape with tools.
Requests carry the Go fingerprint (own UA + stable x-opencode-session per conversation).

Embeddings for the client's long-term memory still go to NVIDIA NIM.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

import httpx

from .config import settings

log = logging.getLogger("nim")

# The free pool is flaky rather than slow: retry with short waits first
# (this is a voice assistant — every second counts).
RETRY_DELAYS = (0.8, 1.5, 3.0, 6.0, 10.0)
_OVERLOAD_RE = re.compile(r"request limit reached|overloaded|ResourceExhausted|temporarily unavailable|capacity|"
                          r"Agent failed|API failed|timed out after|high demand|rate limit", re.I)
_GEO_RE = re.compile(r"geo|location|region|country|not available in|blocked|forbidden|proxy", re.I)
# A repetition loop ("ellsellsellsells…" until max_tokens): a short piece repeated ten or more times at the end
_DEGENERATE_RE = re.compile(r"(.{2,16}?)\1{9,}\s*$", re.S)


class UpstreamError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass
class Completion:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    first_token_at: float = 0.0


def _error_detail(text: str) -> str:
    try:
        j = json.loads(text)
        d = j.get("detail") or (j.get("error") or {}).get("message") or j.get("error") or j.get("message") or j.get("title") or text
        return d if isinstance(d, str) else json.dumps(d)
    except Exception:
        return text


def _classify(err: Exception) -> Optional[str]:
    msg = str(err)
    if isinstance(err, UpstreamError):
        if err.status == 403 or _GEO_RE.search(msg):
            return "geo"
        if _OVERLOAD_RE.search(msg):
            return "overloaded"
        if err.status in (429, 500, 502, 503, 504, 529):
            return "overloaded"
        return "fatal"
    if isinstance(err, (httpx.TransportError, httpx.RemoteProtocolError)):
        return "dropped"
    return None


def _plain_text(content: Any) -> str:
    """Plain text of a chat content (media parts dropped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


def _part_to_inputs(part: dict) -> list[dict]:
    t = part.get("type")
    if t == "text":
        text = part.get("text") or ""
        return [{"type": "input_text", "text": text}] if text else []
    if t == "image_url":
        url = (part.get("image_url") or {}).get("url") or ""
        return [{"type": "input_image", "image_url": url}] if url else []
    if t in ("audio_url", "video_url"):
        kind = "аудио" if t == "audio_url" else "видео"
        # The Responses endpoint takes no audio/video parts: keep a stub so the turn
        # does not fail (file names are already listed in the message text by media.py).
        return [{"type": "input_text",
                 "text": f"[{kind}-вложение: бинарное содержимое не передано модели, "
                         f"она принимает текст, картинки и документы. Опиши его словами или приложи кадром.]"}]
    return []


def _message_to_inputs(msg: dict) -> list[dict]:
    """One chat message -> Responses input items (system messages are handled separately)."""
    role = msg.get("role")
    if role == "system":
        return []
    if role == "tool":
        return [{"type": "function_call_output",
                 "call_id": msg.get("tool_call_id") or f"call_{uuid.uuid4().hex[:8]}",
                 "output": str(msg.get("content") or "")[:8000]}]
    parts: list[dict] = []
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            parts.append({"type": "input_text", "text": content})
    elif isinstance(content, list):
        for p in content:
            if isinstance(p, dict):
                parts.extend(_part_to_inputs(p))
    items: list[dict] = []
    if parts and role == "user":
        items.append({"type": "message", "role": role, "content": parts})
    elif role == "assistant":
        # Console Go requires output_text (not input_text) on assistant history messages.
        texts = [p.get("text", "") for p in parts if p.get("type") == "input_text"]
        if any(t.strip() for t in texts):
            items.append({"type": "message", "role": "assistant",
                          "content": [{"type": "output_text", "text": "\n".join(texts)}]})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        items.append({"type": "function_call",
                      "call_id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                      "name": fn.get("name") or "",
                      "arguments": fn.get("arguments") or "{}"})
    return items


def _convert_tools(tools: Optional[list[dict]]) -> list[dict]:
    """OpenAI chat tools -> Responses function tools (only `type: function` is accepted)."""
    out: list[dict] = []
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            fn = t["function"]
            item: dict[str, Any] = {"type": "function", "name": fn.get("name") or "",
                                    "description": fn.get("description") or ""}
            if fn.get("parameters"):
                item["parameters"] = fn["parameters"]
            out.append(item)
        elif t.get("type") == "function" and t.get("name"):
            out.append(t)  # already Responses-shaped
    return out


def _map_usage(u: Optional[dict]) -> Optional[dict]:
    if not u:
        return None
    return {"prompt_tokens": u.get("input_tokens", u.get("prompt_tokens", 0)),
            "completion_tokens": u.get("output_tokens", u.get("completion_tokens", 0)),
            "total_tokens": u.get("total_tokens", 0)}


# Models served on the Responses transport (verified Sep 2026); everything else on
# the Go endpoint speaks OpenAI chat/completions (also verified per model, except
# minimax-m2.7 / union-alpha which flaked with 500s — kept in the lists anyway).
RESPONSES_MODELS = frozenset({
    "muse-spark-1.3-contributor",
    "muse-spark-1.2-contributor",
    "grok-4.6",
    "gpt-5.6-luna",
    "deepseek-v4-flash",
    "deepseek-v4-flash-vision-exp",
})

# Only Muse Spark models accept the Responses `reasoning.effort` parameter.
REASONING_MODELS = frozenset({
    "muse-spark-1.3-contributor",
    "muse-spark-1.2-contributor",
})


def _model_key(model: str) -> str:
    return (model or "").split("/")[-1].strip()


def _transport(model: str) -> str:
    return "responses" if _model_key(model) in RESPONSES_MODELS else "chat"


class _ThinkFilter:
    """Splits streamed text into speech vs <think> reasoning: some chat models put
    thinking inside content, which must never reach TTS. Holds at most a tag tail."""

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, text: str) -> tuple[str, str]:
        self._buf += text or ""
        speech, reasoning = [], []
        while True:
            if self._in_think:
                i = self._buf.find("</think>")
                if i < 0:
                    break
                reasoning.append(self._buf[:i])
                self._buf = self._buf[i + len("</think>"):]
                self._in_think = False
            else:
                i = self._buf.find("<think>")
                if i < 0:
                    keep = min(len(self._buf), len("<think>") - 1)
                    speech.append(self._buf[:len(self._buf) - keep])
                    self._buf = self._buf[len(self._buf) - keep:]
                    break
                speech.append(self._buf[:i])
                self._buf = self._buf[i + len("<think>"):]
                self._in_think = True
        return "".join(speech), "".join(reasoning)

    def finish(self) -> tuple[str, str]:
        if self._in_think:  # unclosed think at the end counts as reasoning, not speech
            reasoning, self._buf = self._buf, ""
            self._in_think = False
            return "", reasoning
        speech, self._buf = self._buf, ""
        return speech, ""


class NIMClient:
    """Same public shape as NemoAgent's client; text goes to OpenCode Go, embeddings to NIM."""

    def __init__(self) -> None:
        go_base = settings.GO_BASE_URL.rstrip("/") + "/"
        self._zen = httpx.AsyncClient(
            base_url=go_base,
            headers={"Authorization": f"Bearer {settings.GO_API_KEY}", "Accept": "application/json",
                     # Go docs: identify with our own UA (not a generic SDK name) + stable session id.
                     "User-Agent": "SparkAgent/1.0",
                     "x-opencode-client": "sparkagent",
                     "x-opencode-project": "global"},
            timeout=httpx.Timeout(connect=20.0, read=float(settings.UPSTREAM_TIMEOUT), write=60.0, pool=60.0),
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32, keepalive_expiry=60),
            proxy=settings.LLM_PROXY or None,
        )
        self._nim = httpx.AsyncClient(
            base_url=settings.NIM_BASE_URL,
            headers={"Authorization": f"Bearer {settings.NVIDIA_API_KEY}", "Accept": "application/json"},
            timeout=httpx.Timeout(connect=20.0, read=120.0, write=60.0, pool=60.0),
            limits=httpx.Limits(max_keepalive_connections=16, max_connections=32, keepalive_expiry=60),
        )

    async def aclose(self) -> None:
        await self._zen.aclose()
        await self._nim.aclose()

    # ------------------------------------------------------------------ chat
    async def chat_stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        *,
        model: Optional[str] = None,
        thinking: Optional[bool] = None,  # accepted for compatibility; Go reasons internally
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Any = "auto",
        on_event: Optional[Callable[[str, dict], Awaitable[None]]] = None,
        session_id: Optional[str] = None,  # stable per conversation -> x-opencode-session (routing/cache)
        reasoning_effort: Optional[str] = None,  # minimal|low|medium|high|xhigh; default: medium w/ tools, else low
    ) -> Completion:
        """Stream one assistant turn. on_event(kind, data) gets 'delta' / 'reasoning' / 'wait' events."""
        model = model or settings.LLM_MODEL
        if _transport(model) == "chat":
            return await self._chat_turn(
                messages, tools, model=model, temperature=temperature, max_tokens=max_tokens,
                tool_choice=tool_choice, on_event=on_event, session_id=session_id)
        instructions: list[str] = []
        inputs: list[dict] = []
        for m in messages or []:
            if m.get("role") == "system":
                t = _plain_text(m.get("content"))
                if t.strip():
                    instructions.append(t)
            else:
                inputs.extend(_message_to_inputs(m))
        body: dict[str, Any] = {
            "model": model,
            "max_output_tokens": settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        }
        if instructions:
            body["instructions"] = "\n\n".join(instructions)
        body["input"] = inputs
        # NOTE: the Go gateway pins temperature server-side and rejects the parameter,
        # so it is never sent (the `temperature` argument is accepted for compatibility only).
        rtools = _convert_tools(tools)
        if _model_key(model) in REASONING_MODELS:
            body["reasoning"] = {"effort": reasoning_effort or ("medium" if rtools else "low")}
        if rtools:
            body["tools"] = rtools
            # Console Go accepts only tool_choice "auto": emulate "required" (and named
            # choices) with a directive, which the executor relies on as a fallback.
            requested = tool_choice or "auto"
            if requested != "auto":
                extra = ("[System directive: you MUST call at least one of the provided functions "
                         "in this turn; a plain-text answer is not accepted.]")
                body["instructions"] = ((body.get("instructions") or "") + "\n\n" + extra).strip()
            body["tool_choice"] = "auto"
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Go request: model=%s tool_choice=%s tools=%s",
                      model, body.get("tool_choice"), [t.get("name") for t in rtools])
        # Per-request fingerprint: stable session id + fresh request id (Go docs).
        extra_headers = self._fp_headers(session_id)

        attempt = 0
        temp_dropped = False
        while True:
            try:
                if rtools:
                    return await self._once(body, on_event, extra_headers)
                return await self._stream(body, on_event, extra_headers)
            except _Emitted as e:
                raise e.inner
            except Exception as err:  # noqa: BLE001
                kind = _classify(err)
                text = str(err)
                if kind == "geo":
                    raise UpstreamError(403, f"OpenCode Go отклонил запрос (403): {text[:250]} "
                                             "Если причина — регион, задайте в server/.env "
                                             "LLM_PROXY=http(s)/socks5://… и повторите.") from err
                if not temp_dropped and "temperature" in body and "temperature" in text.lower() and attempt < 2:
                    temp_dropped = True
                    attempt += 1
                    log.warning("backend rejected temperature — retrying without it")
                    del body["temperature"]
                    continue
                if kind in ("overloaded", "dropped") and attempt < len(RETRY_DELAYS):
                    delay = 1.5 if kind == "dropped" else RETRY_DELAYS[attempt]
                    attempt += 1
                    log.warning("Go %s (%s) — retry %d in %.1fs", kind, text[:120], attempt, delay)
                    if on_event:
                        await on_event("wait", {"stage": "retry", "reason": kind, "attempt": attempt, "delay": delay})
                    await asyncio.sleep(delay)
                    continue
                raise

    async def _once(self, body: dict, on_event, extra_headers: dict) -> Completion:
        """One non-streamed Responses request (executor rounds with tools)."""
        r = await self._zen.post("responses", json={**body, "stream": False}, headers=extra_headers)
        if r.status_code != 200:
            raise UpstreamError(r.status_code, _error_detail(r.text))
        data = r.json()
        acc = Completion()
        for item in data.get("output") or []:
            itype = item.get("type")
            if itype == "message":
                for c in item.get("content") or []:
                    if c.get("type") in ("output_text", "text") and c.get("text"):
                        acc.content += c["text"]
            elif itype == "function_call":
                acc.tool_calls.append({"id": item.get("call_id") or f"call_{uuid.uuid4().hex[:8]}",
                                       "type": "function",
                                       "function": {"name": item.get("name") or "",
                                                    "arguments": item.get("arguments") or "{}"}})
        acc.usage = _map_usage(data.get("usage"))
        acc.finish_reason = "tool_calls" if acc.tool_calls else "stop"
        if acc.content:
            loop = _DEGENERATE_RE.search(acc.content[-400:])
            if loop:
                acc.content = acc.content[:-len(loop.group(0))]
                acc.finish_reason = "degenerate"
                log.warning("degenerate output after %d chars", len(acc.content))
        if not acc.content.strip() and not acc.tool_calls:
            raise UpstreamError(502, "empty response without a result")
        if on_event and acc.content:
            await on_event("delta", {"content": acc.content})
        return acc

    async def _stream(self, body: dict, on_event, extra_headers: dict) -> Completion:
        req = {**body, "stream": True}
        async with self._zen.stream("POST", "responses", json=req,
                                    headers={"Accept": "text/event-stream", **extra_headers}) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "ignore")
                raise UpstreamError(resp.status_code, _error_detail(text))
            return await self._consume(resp, on_event)

    async def _consume(self, resp: httpx.Response, on_event) -> Completion:
        acc = Completion()
        emitted = False

        async def deliver(text: str) -> None:
            nonlocal emitted
            if not text:
                return
            emitted = True
            if not acc.first_token_at:
                acc.first_token_at = time.time()
            acc.content += text
            if on_event:
                await on_event("delta", {"content": text})

        try:
            async for line in resp.aiter_lines():
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    evt = json.loads(p)
                except json.JSONDecodeError:
                    continue
                if evt.get("error"):
                    e = evt["error"]
                    raise UpstreamError(int(e.get("code") or 0), e.get("message") or json.dumps(e))
                t = evt.get("type") or ""
                if t == "response.output_text.delta":
                    await deliver(evt.get("delta") or "")
                    if len(acc.content) >= 60:
                        loop = _DEGENERATE_RE.search(acc.content[-400:])
                        if loop:
                            # the model fell into a repetition loop: cut it here
                            acc.content = acc.content[:-len(loop.group(0))]
                            acc.finish_reason = "degenerate"
                            log.warning("degenerate output after %d chars — stream aborted", len(acc.content))
                            break
                elif t == "response.completed":
                    u = ((evt.get("response") or {}).get("usage")) or evt.get("usage")
                    acc.usage = _map_usage(u)
                elif t in ("response.failed", "response.incomplete"):
                    if t == "response.incomplete" and acc.content.strip():
                        # cut by max_output_tokens (reasoning eats the budget too): keep what we got
                        u = ((evt.get("response") or {}).get("usage")) or evt.get("usage")
                        if u:
                            acc.usage = _map_usage(u)
                        acc.finish_reason = "length"
                        break
                    detail = json.dumps(evt)[:300]
                    raise UpstreamError(502, f"response did not complete: {detail}")
        except Exception as e:
            if emitted:
                raise _Emitted(e)
            raise
        if acc.finish_reason is None:
            acc.finish_reason = "stop"
        if not acc.content.strip():
            raise UpstreamError(502, "stream ended without a result")
        return acc

    @staticmethod
    def _fp_headers(session_id: Optional[str]) -> dict:
        return {"x-opencode-session": session_id or f"ses_{uuid.uuid4().hex}",
                "x-opencode-request": f"msg_{uuid.uuid4().hex}"}

    # ------------------------------------------------- chat/completions transport
    @staticmethod
    def _force_tools(messages: list[dict]) -> list[dict]:
        """Copy of messages with a must-call-tools directive (the gateway takes only
        tool_choice=auto, so `required` is emulated). Never mutates the input."""
        directive = ("[System directive: you MUST call at least one of the provided functions "
                     "in this turn; a plain-text answer is not accepted.]")
        out = [dict(m) for m in messages]
        if out and out[0].get("role") == "system":
            first = dict(out[0])
            c = first.get("content")
            if isinstance(c, str):
                first["content"] = (c + "\n\n" + directive).strip()
            elif isinstance(c, list):
                first["content"] = [*c, {"type": "text", "text": directive}]
            else:
                first["content"] = directive
            out[0] = first
        else:
            out.insert(0, {"role": "system", "content": directive})
        return out

    async def _chat_turn(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        *,
        model: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        tool_choice: Any,
        on_event,
        session_id: Optional[str],
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if settings.LLM_TOP_P:
            payload["top_p"] = float(settings.LLM_TOP_P)
        if tools:
            if (tool_choice or "auto") != "auto":
                payload["messages"] = self._force_tools(messages)
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Accept": "text/event-stream", **self._fp_headers(session_id)}
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Go chat request: model=%s tools=%s",
                      model, [t["function"]["name"] for t in (tools or [])])
        attempt = 0
        while True:
            try:
                return await self._chat_consume(payload, headers, on_event)
            except _Emitted as e:
                raise e.inner
            except Exception as err:  # noqa: BLE001
                kind = _classify(err)
                text = str(err)
                if kind == "geo":
                    raise UpstreamError(403, f"OpenCode Go отклонил запрос (403): {text[:250]}") from err
                dropped = [k for k in ("temperature", "top_p") if k in payload and k in text.lower()]
                if dropped and attempt < 2:
                    for k in dropped:
                        del payload[k]
                    attempt += 1
                    log.warning("backend rejected %s — retrying without it", ",".join(dropped))
                    continue
                if kind in ("overloaded", "dropped") and attempt < len(RETRY_DELAYS):
                    delay = 1.5 if kind == "dropped" else RETRY_DELAYS[attempt]
                    attempt += 1
                    log.warning("Go chat %s (%s) — retry %d in %.1fs", kind, text[:120], attempt, delay)
                    if on_event:
                        await on_event("wait", {"stage": "retry", "reason": kind, "attempt": attempt, "delay": delay})
                    await asyncio.sleep(delay)
                    continue
                raise

    async def _chat_consume(self, payload: dict, headers: dict, on_event) -> Completion:
        acc = Completion()
        tool_slots: dict[int, dict] = {}
        think = _ThinkFilter()
        emitted = False

        async def deliver_speech(text: str) -> None:
            nonlocal emitted
            speech, reasoning = think.feed(text)
            if reasoning:
                emitted = True
                if on_event:
                    await on_event("reasoning", {"content": reasoning})
            if speech:
                if not acc.first_token_at:
                    acc.first_token_at = time.time()
                acc.content += speech
                emitted = True
                if on_event:
                    await on_event("delta", {"content": speech})

        try:
            async with self._zen.stream("POST", "chat/completions", json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "ignore")
                    raise UpstreamError(resp.status_code, _error_detail(text))
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    p = line[5:].strip()
                    if p == "[DONE]":
                        break
                    try:
                        chunk = json.loads(p)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("error"):
                        e = chunk["error"]
                        raise UpstreamError(int(e.get("code") or 0), e.get("message") or json.dumps(e))
                    choice = (chunk.get("choices") or [None])[0]
                    if not choice:
                        continue
                    if choice.get("finish_reason"):
                        acc.finish_reason = choice["finish_reason"]
                    d = choice.get("delta") or {}
                    reasoning = d.get("reasoning_content") or d.get("reasoning")
                    if reasoning:
                        emitted = True
                        if on_event:
                            await on_event("reasoning", {"content": reasoning})
                    if d.get("content"):
                        if not acc.first_token_at:
                            acc.first_token_at = time.time()
                        await deliver_speech(d["content"])
                        if len(acc.content) >= 60:
                            loop = _DEGENERATE_RE.search(acc.content[-400:])
                            if loop:
                                acc.content = acc.content[:-len(loop.group(0))]
                                acc.finish_reason = "degenerate"
                                log.warning("degenerate output after %d chars — stream aborted", len(acc.content))
                                break
                    for tc in d.get("tool_calls") or []:
                        if not acc.first_token_at:
                            acc.first_token_at = time.time()
                        i = tc.get("index", 0)
                        slot = tool_slots.setdefault(i, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
        except Exception as e:
            if emitted:
                raise _Emitted(e)
            raise
        tail_speech, tail_reasoning = think.finish()
        if tail_reasoning and on_event:
            await on_event("reasoning", {"content": tail_reasoning})
        if tail_speech:
            if not acc.first_token_at:
                acc.first_token_at = time.time()
            acc.content += tail_speech
            emitted = True
            if on_event:
                await on_event("delta", {"content": tail_speech})
        acc.tool_calls = [tool_slots[i] for i in sorted(tool_slots)]
        if acc.finish_reason is None:
            acc.finish_reason = "tool_calls" if acc.tool_calls else "stop"
        if not acc.content.strip() and not acc.tool_calls and not acc.reasoning:
            raise UpstreamError(502, "stream ended without a result")
        return acc

    # ------------------------------------------------------------ embeddings (NVIDIA NIM, unchanged)
    async def embed(self, model: str, inputs: list[str], input_type: str = "passage") -> list[list[float]]:
        """Embed texts (or data:image/... URIs for the VL model). Batches of 64."""
        if not settings.NVIDIA_API_KEY:
            raise UpstreamError(401, "NVIDIA_API_KEY is not set — memory embeddings need it (server/.env)")
        out: list[list[float]] = []
        for i in range(0, len(inputs), 64):
            batch = inputs[i:i + 64]
            for attempt in range(4):
                try:
                    r = await self._nim.post("embeddings", json={
                        "model": model, "input": batch, "input_type": input_type, "encoding_format": "float",
                    }, timeout=120)
                    if r.status_code in (429, 502, 503, 504, 529):
                        raise UpstreamError(r.status_code, _error_detail(r.text))
                    if r.status_code != 200:
                        raise UpstreamError(r.status_code, _error_detail(r.text))
                    data = sorted(r.json().get("data", []), key=lambda d: d.get("index", 0))
                    out.extend(d["embedding"] for d in data)
                    break
                except Exception as err:  # noqa: BLE001
                    if _classify(err) in ("overloaded", "dropped") and attempt < 3:
                        await asyncio.sleep(RETRY_DELAYS[attempt])
                        continue
                    raise
        return out


class _Emitted(Exception):
    """Wraps an error raised after tokens were already streamed — must not be retried silently."""

    def __init__(self, inner: Exception):
        super().__init__(str(inner))
        self.inner = inner
