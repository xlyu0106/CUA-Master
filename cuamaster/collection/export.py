"""Merge agent transcripts and actuator logs into canonical CUA trajectories."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import shutil
import struct
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from cuamaster.action_space import CUAAction
from .episode import build_episode
from .ledger import EpisodeLedger
from .profiles import VMProfile
from .quality import validate_episode
from .schemas import EpisodeStatus
from .trace_store import TraceStore, atomic_write_json, atomic_write_text


EVENT_SCHEMA = "cua_trajectory.event.v1"
MANIFEST_SCHEMA = "cua_trajectory.v1"

_SHELL_TOOL_NAMES = {
    "bash",
    "cli_command",
    "exec",
    "exec_command",
    "run_shell_command",
    "shell",
    "shell.exec",
    "shell_exec",
    "terminal",
}
_CANONICAL_GUI_ACTIONS = {
    "computer.click",
    "computer.double_click",
    "computer.triple_click",
    "computer.move",
    "computer.drag",
    "computer.scroll",
    "computer.type",
    "computer.press",
    "computer.key_down",
    "computer.key_up",
    "computer.mouse_down",
    "computer.mouse_up",
    "computer.wait",
}
_SHELL_EXIT_CODE_PATTERNS = (
    re.compile(r"^\s*Exit code\s+(-?\d+)\b", re.I),
    re.compile(r"^\s*Process exited with code\s+(-?\d+)\b", re.I),
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def _digest(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _message(record: dict[str, Any]) -> dict[str, Any]:
    message = record.get("message")
    if isinstance(message, dict):
        return message
    return record if record.get("role") else {}


def _parse_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {"_value": value} if value is not None else {}


def _assistant_text_blocks(message: dict[str, Any], chat_index: int) -> list[dict[str, Any]]:
    content = message.get("content")
    values = content if isinstance(content, list) else [content]
    blocks: list[dict[str, Any]] = []
    for value in values:
        text: Any = None
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict) and value.get("type") in {"text", "output_text"}:
            text = value.get("text")
            if isinstance(text, dict):
                text = text.get("value")
        if isinstance(text, str) and text.strip():
            blocks.append({"chat_index": chat_index, "text": text.strip()})
    return blocks


def _decision_context(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks:
        return {}
    texts = [str(block["text"]) for block in blocks]
    return {
        "source": "chat.jsonl",
        "visibility": "observable_assistant_output",
        "intent": texts[-1],
        "text": "\n\n".join(texts),
        "chat_indices": list(dict.fromkeys(block["chat_index"] for block in blocks)),
        "blocks": [dict(block) for block in blocks],
    }


def _chat_tool_calls(task_dir: Path) -> list[dict[str, Any]]:
    """Read OpenClaw-v3 or native OpenAI chat logs without losing tool order."""
    calls: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    pending_results: dict[str, dict[str, Any]] = {}
    pending_assistant_text: list[dict[str, Any]] = []

    def add_call(
        call_id: Any,
        name: Any,
        tool_input: Any,
        chat_index: int,
        decision_context: dict[str, Any],
    ) -> None:
        identifier = str(call_id or "")
        tool_name = str(name or "")
        if not (_is_computer_tool(tool_name) or _is_shell_tool(tool_name)):
            return
        call = {
            "tool_call_id": identifier or None,
            "tool_name": tool_name,
            "request": _parse_arguments(tool_input),
            "chat_index": chat_index,
            "result": pending_results.pop(identifier, None) if identifier else None,
            "decision_context": decision_context,
        }
        calls.append(call)
        if identifier:
            by_id[identifier] = call

    def add_result(call_id: Any, content: Any, block: dict[str, Any]) -> None:
        identifier = str(call_id or "")
        result = {
            "content": content,
            "is_error": block.get("is_error", block.get("isError")),
            "details": block.get("details"),
        }
        if identifier in by_id:
            by_id[identifier]["result"] = result
        elif identifier:
            pending_results[identifier] = result

    for chat_index, record in enumerate(_jsonl(task_dir / "chat.jsonl"), 1):
        message = _message(record)
        role = str(message.get("role") or "")
        content = message.get("content")
        blocks = content if isinstance(content, list) else []

        assistant_has_tool_call = False
        if role == "assistant":
            pending_assistant_text.extend(_assistant_text_blocks(message, chat_index))
            context = _decision_context(pending_assistant_text)
            for block in blocks:
                if isinstance(block, dict) and block.get("type") in {
                    "tool_use",
                    "toolCall",
                }:
                    assistant_has_tool_call = True
                    add_call(
                        block.get("id"),
                        block.get("name", block.get("toolName")),
                        block.get("input", block.get("arguments")),
                        chat_index,
                        context,
                    )
            # Also accept an unconverted OpenAI chat-completions transcript.
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function") or {}
                if isinstance(function, dict):
                    assistant_has_tool_call = True
                    add_call(
                        tool_call.get("id"),
                        function.get("name") or tool_call.get("name"),
                        function.get("arguments", tool_call.get("arguments")),
                        chat_index,
                        context,
                    )
            if assistant_has_tool_call:
                pending_assistant_text.clear()

        has_tool_result = False
        for block in blocks:
            if isinstance(block, dict) and block.get("type") in {
                "tool_result",
                "toolResult",
            }:
                add_result(
                    block.get("tool_use_id", block.get("toolCallId")),
                    block.get("content"),
                    block,
                )
                has_tool_result = True
        if role in {"tool", "toolResult"}:
            add_result(
                message.get("tool_call_id", message.get("toolCallId")),
                content,
                message,
            )
            has_tool_result = True
        if has_tool_result or role in {"user", "tool", "toolResult"}:
            pending_assistant_text.clear()
    return calls


def _is_computer_tool(name: str) -> bool:
    return _tool_basename(name) == "computer"


def _tool_basename(name: str) -> str:
    lower = name.strip().lower()
    if lower in _SHELL_TOOL_NAMES:
        return lower
    parts = [part for part in re.split(r"[/:.]+|__", lower) if part]
    return parts[-1] if parts else lower


def _is_shell_tool(name: str) -> bool:
    return _tool_basename(name) in _SHELL_TOOL_NAMES


def _valid_environment_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, dict):
        return None
    try:
        canonical = CUAAction.from_dict(action).to_dict()
    except (TypeError, ValueError):
        return None
    action_type = str(canonical.get("type") or "")
    if not (
        action_type.startswith("computer.")
        or action_type in {"shell.exec", "finish"}
    ):
        # tool.call runs host-side; the GUI+CLI environment timeline keeps
        # actuator-grounded actions only.
        return None
    if action_type == "computer.type" and not str(
        (canonical.get("args") or {}).get("text") or ""
    ):
        return None
    return canonical


def _shell_timeout_seconds(
    tool_input: dict[str, Any],
    tool_name: str,
) -> int | None:
    timeout = tool_input.get("timeout_sec")
    if timeout is None:
        timeout = tool_input.get("timeout")
        if (
            isinstance(timeout, (int, float))
            and not isinstance(timeout, bool)
            and _tool_basename(tool_name) == "bash"
        ):
            timeout = timeout / 1000
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        return None
    return math.ceil(timeout)


def _shell_action(
    tool_input: dict[str, Any],
    tool_name: str = "",
) -> dict[str, Any] | None:
    command = next(
        (
            tool_input[key]
            for key in ("command", "cmd", "script")
            if isinstance(tool_input.get(key), str) and tool_input[key].strip()
        ),
        None,
    )
    if command is None and isinstance(tool_input.get("_raw"), str):
        command = tool_input["_raw"]
    if command is None:
        return None
    args: dict[str, Any] = {"command": command}
    cwd = tool_input.get("cwd", tool_input.get("workdir"))
    if isinstance(cwd, str) and cwd:
        args["cwd"] = cwd
    timeout = _shell_timeout_seconds(tool_input, tool_name)
    if timeout is not None:
        args["timeout_sec"] = timeout
    return {"type": "shell.exec", "args": args}


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        values: list[str] = []
        for item in content:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    values.append(text)
        return "\n".join(values)
    return ""


def _shell_result_exit_code(content: Any, details: Any = None) -> int | None:
    if isinstance(details, dict):
        value = details.get("exitCode", details.get("exit_code"))
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    text = _result_text(content)
    for pattern in _SHELL_EXIT_CODE_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def _raw_gui_actions(tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    value = tool_input.get("actions")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    value = tool_input.get("action", tool_input)
    if isinstance(value, str):
        return [tool_input]
    return [value] if isinstance(value, dict) and value else []


def _png_dimensions(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    try:
        header = path.read_bytes()[:24]
    except OSError:
        return None
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return (width, height) if width and height else None


def _number(value: Any, default: float = 0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coordinate(value: Any, extent: int | None) -> int:
    number = _number(value)
    if extent:
        number = number * 1000 / extent
    return round(max(0, min(1000, number)))


def _point(raw: dict[str, Any], dimensions: tuple[int, int] | None) -> tuple[int, int]:
    width, height = dimensions or (None, None)
    return _coordinate(raw.get("x"), width), _coordinate(raw.get("y"), height)


def _canonical_gui_action(
    raw: dict[str, Any], dimensions: tuple[int, int] | None
) -> list[dict[str, Any]]:
    """Map one MCP action to one or more canonical atomic actions."""
    action_type = str(raw.get("type") or raw.get("action") or "").strip()
    lowered = action_type.lower()
    if lowered in _CANONICAL_GUI_ACTIONS and isinstance(raw.get("args"), dict):
        return [{"type": lowered, "args": dict(raw["args"])}]
    if lowered.startswith("computer."):
        lowered = lowered.removeprefix("computer.")
    if lowered in {"screenshot", "cursor_position", ""}:
        return []
    if lowered in {"click", "left_click", "right_click", "middle_click"}:
        x, y = _point(raw, dimensions)
        default_button = "right" if lowered == "right_click" else "middle" if lowered == "middle_click" else "left"
        button = raw.get("button", default_button)
        if isinstance(button, (int, float)) and not isinstance(button, bool):
            button = {1: "left", 2: "middle", 3: "right"}.get(int(button), "left")
        if button not in {"left", "right", "middle"}:
            button = "left"
        return [{"type": "computer.click", "args": {"x": x, "y": y, "button": button}}]
    if lowered in {"double_click", "doubleclick"}:
        x, y = _point(raw, dimensions)
        return [{"type": "computer.double_click", "args": {"x": x, "y": y, "button": "left"}}]
    if lowered == "triple_click":
        x, y = _point(raw, dimensions)
        return [{"type": "computer.triple_click", "args": {"x": x, "y": y, "button": "left"}}]
    if lowered in {"move", "mouse_move", "mousemove"}:
        x, y = _point(raw, dimensions)
        return [{"type": "computer.move", "args": {"x": x, "y": y}}]
    if lowered in {"type", "keyboard"}:
        text = str(raw.get("text") or "")
        return [{"type": "computer.type", "args": {"text": text}}] if text else []
    if lowered in {"keypress", "key"}:
        keys = raw.get("keys", raw.get("text", raw.get("key")))
        if isinstance(keys, str):
            keys = [keys]
        keys = [str(key).strip().lower() for key in keys or [] if str(key).strip()]
        return [{"type": "computer.press", "args": {"keys": keys}}] if keys else []
    if lowered in {"key_down", "keydown", "key_up", "keyup"}:
        key = raw.get("key", raw.get("text"))
        if key is None and isinstance(raw.get("keys"), list) and raw["keys"]:
            key = raw["keys"][0]
        key = str(key or "").strip().lower()
        if not key:
            return []
        action = "computer.key_down" if lowered in {"key_down", "keydown"} else "computer.key_up"
        return [{"type": action, "args": {"key": key}}]
    if lowered in {
        "left_mouse_down",
        "mousedown",
        "mouse_down",
        "left_mouse_up",
        "mouseup",
        "mouse_up",
    }:
        action = (
            "computer.mouse_down"
            if lowered in {"left_mouse_down", "mousedown", "mouse_down"}
            else "computer.mouse_up"
        )
        args = {"button": "left"}
        if raw.get("x") is not None and raw.get("y") is not None:
            x, y = _point(raw, dimensions)
            args.update({"x": x, "y": y})
        return [{"type": action, "args": args}]
    if lowered == "scroll":
        x, y = _point(raw, dimensions)
        # Canonical dx/dy are the pyautogui delta that was actually executed
        # (positive dy scrolls up). MCP tool input uses OpenAI-style
        # scroll_x/scroll_y, which the guest actuator negates before execution,
        # so apply the same negation when recording. Logs that already carry
        # canonical dx/dy are passed through unchanged.
        dx = _number(raw["dx"]) if "dx" in raw else -_number(raw.get("scroll_x"))
        dy = _number(raw["dy"]) if "dy" in raw else -_number(raw.get("scroll_y"))
        args: dict[str, Any] = {"dx": dx, "dy": dy, "x": x, "y": y}
        return [{"type": "computer.scroll", "args": args}] if args["dx"] or args["dy"] else []
    if lowered == "drag":
        path = raw.get("path") if isinstance(raw.get("path"), list) else []
        if len(path) >= 2:
            first, last = path[0], path[-1]
            if isinstance(first, list):
                first = {"x": first[0], "y": first[1]}
            if isinstance(last, list):
                last = {"x": last[0], "y": last[1]}
        else:
            first = {"x": raw.get("x1"), "y": raw.get("y1")}
            last = {"x": raw.get("x2"), "y": raw.get("y2")}
        if isinstance(first, dict) and isinstance(last, dict):
            x1, y1 = _point(first, dimensions)
            x2, y2 = _point(last, dimensions)
            args = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "button": "left"}
            return [{"type": "computer.drag", "args": args}]
    if lowered == "wait":
        if "seconds" in raw:
            seconds = _number(raw["seconds"], 1)
        else:
            seconds = _number(raw.get("ms", raw.get("duration", 1000)), 1000) / 1000
        return [{"type": "computer.wait", "args": {"seconds": max(0.001, seconds)}}]
    return []


def _canonical_gui_actions(
    tool_input: dict[str, Any], dimensions: tuple[int, int] | None
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for raw in _raw_gui_actions(tool_input):
        actions.extend(_canonical_gui_action(raw, dimensions))
    return actions


def _category(actions: list[dict[str, Any]]) -> str:
    categories: set[str] = set()
    for action in actions:
        action_type = str(action.get("type") or "")
        if action_type.startswith("computer."):
            categories.add("gui")
        elif action_type == "shell.exec":
            categories.add("cli")
        elif action_type == "finish":
            categories.add("control")
        else:
            raise ValueError(f"unsupported GUI+CLI action type: {action_type}")
    if not categories:
        return "observation"
    return next(iter(categories)) if len(categories) == 1 else "mixed"


def _screenshot_path(task_dir: Path, remote: Any) -> tuple[str | None, Path | None]:
    name = Path(str(remote)).name if remote else None
    local = task_dir / "screenshots" / name if name else None
    if local is not None and local.is_file():
        return f"screenshots/{name}", local
    return None, None


def _embedded_result_screenshot(
    task_dir: Path,
    call: dict[str, Any],
    sequence: int,
) -> Path | None:
    result = call.get("result")
    if not isinstance(result, dict):
        return None
    content = result.get("content")
    blocks = content if isinstance(content, list) else []
    for block in reversed(blocks):
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        encoded = block.get("data")
        if not isinstance(encoded, str) or not encoded:
            continue
        mime_type = str(block.get("mimeType") or block.get("media_type") or "")
        if encoded.startswith("data:"):
            header, separator, encoded = encoded.partition(",")
            if not separator or ";base64" not in header:
                continue
            if not mime_type:
                mime_type = header[5:].split(";", 1)[0]
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            continue
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            extension = ".png"
        elif payload.startswith(b"\xff\xd8\xff"):
            extension = ".jpg"
        elif mime_type == "image/png":
            extension = ".png"
        elif mime_type in {"image/jpeg", "image/jpg"}:
            extension = ".jpg"
        else:
            continue
        screenshot_dir = task_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        destination = screenshot_dir / f"screenshot_{sequence:04d}_tool_result{extension}"
        if not destination.is_file() or destination.read_bytes() != payload:
            destination.write_bytes(payload)
        return destination
    return None


def _gui_event(
    task_dir: Path,
    event: dict[str, Any],
    *,
    call: dict[str, Any] | None = None,
    fallback_screenshot: Path | None = None,
) -> dict[str, Any]:
    remote = event.get("screenshot_path")
    screenshot, local = _screenshot_path(task_dir, remote)
    if local is None and fallback_screenshot is not None and fallback_screenshot.is_file():
        local = fallback_screenshot
        screenshot = f"screenshots/{fallback_screenshot.name}"
    request = (
        call.get("request") if call and isinstance(call.get("request"), dict)
        else event.get("request") if isinstance(event.get("request"), dict)
        else {"actions": event.get("actions") or []}
    )
    actions = _canonical_gui_actions(request, _png_dimensions(local))
    pre_screenshot, _ = _screenshot_path(task_dir, event.get("pre_screenshot_path"))
    normalized_executions = []
    for execution in event.get("executions") or []:
        if not isinstance(execution, dict):
            normalized_executions.append(execution)
            continue
        normalized = dict(execution)
        execution_screenshot, _ = _screenshot_path(
            task_dir, normalized.pop("screenshot_path", None)
        )
        if execution_screenshot:
            normalized["screenshot"] = execution_screenshot
        normalized_executions.append(normalized)
    item = dict(event)
    item.update({
        "schema_version": EVENT_SCHEMA,
        "source": (
            "chat+mcp_event_log" if call and event else
            "chat" if call else "mcp_event_log"
        ),
        "category": _category(actions),
        "tool_call_id": call.get("tool_call_id") if call else item.get("tool_call_id"),
        "tool_name": call.get("tool_name") if call else item.get("tool_name", "computer"),
        "request": request,
        "actions": actions,
        "executions": normalized_executions or item.get("executions") or [],
        "pre_screenshot": pre_screenshot,
        "screenshot": screenshot,
    })
    item.pop("screenshot_path", None)
    item.pop("pre_screenshot_path", None)
    if call:
        item["chat_index"] = call.get("chat_index")
        if call.get("decision_context"):
            item["decision_context"] = call["decision_context"]
    return item


def _shell_event(
    call: dict[str, Any],
    shell_captures: dict[str, dict[str, Any]] | None = None,
    fallback_shell_capture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(call.get("tool_name") or "unknown")
    tool_input = call.get("request") if isinstance(call.get("request"), dict) else {}
    action = _shell_action(tool_input, name)
    if action is None:
        raise ValueError(f"invalid shell tool input for {name}")
    capture = (shell_captures or {}).get(str(call.get("tool_call_id") or ""), {})
    if not capture and fallback_shell_capture:
        capture = fallback_shell_capture
    result = call.get("result") if isinstance(call.get("result"), dict) else None
    is_error = result.get("is_error") if result else None
    ok = None if result is None else not bool(is_error)
    shell_exit_code = (
        _shell_result_exit_code(result.get("content"), result.get("details"))
        if result is not None
        else None
    )
    if shell_exit_code is not None:
        ok = shell_exit_code == 0 and not bool(is_error)
    execution: dict[str, Any] = {
        "action": action,
        "ok": ok,
        "status_source": "chat_tool_result" if result is not None else "missing",
    }
    if result is not None:
        execution["output"] = result.get("content")
    if shell_exit_code is not None:
        execution["exit_code"] = shell_exit_code
    captured_execution = capture.get("execution")
    if isinstance(captured_execution, dict):
        execution.update(captured_execution)
        ok = captured_execution.get("ok", ok)
    event = {
        "schema_version": EVENT_SCHEMA,
        "source": "chat+shell_capture" if capture else "chat",
        "category": _category([action]),
        "chat_index": call.get("chat_index"),
        "tool_call_id": call.get("tool_call_id"),
        "tool_name": name,
        "request": tool_input,
        "actions": [action],
        "executions": [execution],
        "ok": ok,
        "errors": (
            []
            if not is_error and shell_exit_code in {None, 0}
            else [str((result or {}).get("content") or "tool call failed")]
        ),
        "pre_screenshot": capture.get("pre"),
        "screenshot": capture.get("post"),
    }
    if call.get("decision_context"):
        event["decision_context"] = call["decision_context"]
    return event


def _shell_capture_index(
    task_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    captures: dict[str, dict[str, Any]] = {}
    ordered: dict[str, dict[str, Any]] = {}
    for row in _jsonl(task_dir / "shell_events.jsonl"):
        tool_call_id = str(row.get("tool_call_id") or "")
        sequence = str(row.get("sequence") or "")
        phase = str(row.get("phase") or "").strip().lower()
        if phase not in {"pre", "post"}:
            continue
        screenshot, _ = _screenshot_path(task_dir, row.get("screenshot_path"))
        target = captures.setdefault(tool_call_id, {}) if tool_call_id else (
            ordered.setdefault(sequence, {}) if sequence else None
        )
        if target is None:
            continue
        if screenshot:
            target[phase] = screenshot
        if phase == "post" and isinstance(row.get("execution"), dict):
            target["execution"] = dict(row["execution"])
    return captures, list(ordered.values())


def _ensure_initial_screenshot(task_dir: Path) -> str | None:
    destination = task_dir / "screenshots" / "init_screenshot.png"
    if destination.is_file():
        return "screenshots/init_screenshot.png"
    source = task_dir / "init_screenshot.png"
    if not source.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    return "screenshots/init_screenshot.png"


def _canonical_runtime_events(task_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_screenshot = _canonical_runtime_screenshot(
        task_dir, "screenshots/step_0000.png"
    ) or _ensure_initial_screenshot(task_dir)
    for row in _jsonl(task_dir / "canonical_trajectory.jsonl"):
        action = _valid_environment_action(row.get("action"))
        if action is None:
            continue
        execution = row.get("execution_result")
        if not isinstance(execution, dict):
            execution = {"ok": None, "observation": {}}
        pre_screenshot = _canonical_runtime_screenshot(
            task_dir, row.get("pre_screenshot_path")
        ) or previous_screenshot
        screenshot = _canonical_runtime_screenshot(
            task_dir,
            row.get("post_screenshot_path") or row.get("screenshot_path"),
        )
        event = {
                "schema_version": EVENT_SCHEMA,
                "source": "canonical_runtime",
                "step": row.get("step", len(events) + 1),
                "category": _category([action]),
                "request": action.get("args") or {},
                "actions": [action],
                "executions": [{"action": action, **execution}],
                "ok": execution.get("ok"),
                "errors": [],
                "pre_screenshot": pre_screenshot,
                "screenshot": screenshot,
                "raw_model_text": row.get("raw_model_text"),
                "end_reason": row.get("end_reason"),
            }
        raw_model_text = row.get("raw_model_text")
        if isinstance(raw_model_text, str) and raw_model_text.strip():
            text = raw_model_text.strip()
            event["decision_context"] = {
                "source": "canonical_trajectory.jsonl",
                "visibility": "observable_model_output",
                "intent": text,
                "text": text,
                "chat_indices": [],
                "blocks": [],
            }
        events.append(event)
        if screenshot:
            previous_screenshot = screenshot
    return events


def _canonical_runtime_screenshot(task_dir: Path, value: Any) -> str | None:
    if not value:
        return None
    screenshot_path = Path(str(value))
    if not screenshot_path.is_absolute():
        screenshot_path = task_dir / screenshot_path
    try:
        relative = screenshot_path.resolve().relative_to(task_dir.resolve())
    except (OSError, ValueError):
        return None
    return str(relative) if screenshot_path.is_file() else None


def _merged_events(task_dir: Path) -> tuple[list[dict[str, Any]], str]:
    canonical_events = _canonical_runtime_events(task_dir)
    if canonical_events:
        return canonical_events, "canonical_runtime"
    event_log = task_dir / "computer_events.jsonl"
    mcp_events = list(_jsonl(event_log)) if event_log.is_file() else []
    calls = _chat_tool_calls(task_dir)
    shell_captures, ordered_shell_captures = _shell_capture_index(task_dir)
    has_shell_captures = bool(shell_captures or ordered_shell_captures)
    screenshots = sorted((task_dir / "screenshots").glob("screenshot_*.png"))
    merged: list[dict[str, Any]] = []
    computer_index = 0
    shell_index = 0

    for call in calls:
        if _is_computer_tool(str(call.get("tool_name") or "")):
            event = mcp_events[computer_index] if computer_index < len(mcp_events) else {}
            fallback = _embedded_result_screenshot(
                task_dir, call, computer_index + 1
            )
            if fallback is None and computer_index < len(screenshots):
                fallback = screenshots[computer_index]
            merged.append(_gui_event(task_dir, event, call=call, fallback_screenshot=fallback))
            computer_index += 1
        elif _is_shell_tool(str(call.get("tool_name") or "")):
            fallback_shell_capture = None
            if shell_index < len(ordered_shell_captures):
                fallback_shell_capture = ordered_shell_captures[shell_index]
            shell_index += 1
            merged.append(
                _shell_event(
                    call,
                    shell_captures,
                    fallback_shell_capture=fallback_shell_capture,
                )
            )

    # A truncated/missing transcript must not discard actuator-grounded events.
    for event in mcp_events[computer_index:]:
        merged.append(_gui_event(task_dir, event))

    initial_screenshot = _ensure_initial_screenshot(task_dir)
    if merged and initial_screenshot and not merged[0].get("pre_screenshot"):
        merged[0]["pre_screenshot"] = initial_screenshot

    for index, item in enumerate(merged, 1):
        item["step"] = index
    if calls and mcp_events and has_shell_captures:
        source = "chat+mcp_event_log+shell_capture"
    elif calls and mcp_events:
        source = "chat+mcp_event_log"
    elif calls and has_shell_captures:
        source = "chat+shell_capture"
    elif calls:
        source = "chat"
    else:
        source = "mcp_event_log"
    return merged, source


def export_task(
    task_dir: Path,
    profile: VMProfile,
    *,
    run_id: str = "standalone",
    ledger: EpisodeLedger | None = None,
    trace_store: TraceStore | None = None,
) -> bool:
    normalized, source = _merged_events(task_dir)
    if not normalized:
        return False
    trajectory = task_dir / "trajectory.jsonl"
    atomic_write_text(
        trajectory,
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in normalized),
    )
    score = _read_json(task_dir / "score.json")
    status = _read_json(task_dir / "execution_status.json")
    fingerprint = task_dir / "environment_fingerprint.json"
    event_categories = Counter(item.get("category") or "unknown" for item in normalized)
    action_types = Counter(
        str(action.get("type") or "unknown")
        for item in normalized
        for action in item.get("actions") or []
        if isinstance(action, dict)
    )
    event_log = task_dir / "computer_events.jsonl"
    canonical_log = task_dir / "canonical_trajectory.jsonl"
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.as_metadata(),
        "task": {
            "id": score.get("task_id", task_dir.name),
            "category": score.get("category", task_dir.parent.name),
            "mode": score.get("mode", "gui"),
        },
        "agent": {
            "harness": score.get("harness") or status.get("harness") or "unknown",
            "model": score.get("model") or status.get("model"),
            "status": score.get("agent_status"),
            "done": status.get("agent_done", score.get("agent_done")),
        },
        "source": source,
        "collection_backend": profile.backend,
        "step_count": len(normalized),
        "action_count": sum(len(item.get("actions") or []) for item in normalized),
        "event_categories": dict(sorted(event_categories.items())),
        "action_types": dict(sorted(action_types.items())),
        "score": score.get("score"),
        "files": {
            "trajectory": trajectory.name,
            "chat": "chat.jsonl" if (task_dir / "chat.jsonl").is_file() else None,
            "computer_events": event_log.name if event_log.is_file() else None,
            "shell_events": (
                "shell_events.jsonl" if (task_dir / "shell_events.jsonl").is_file() else None
            ),
            "canonical_trajectory": canonical_log.name if canonical_log.is_file() else None,
            "environment_fingerprint": fingerprint.name if fingerprint.is_file() else None,
            "screenshots": sum(1 for item in normalized if item.get("screenshot")),
        },
        "source_sha256": {
            "chat": _digest(task_dir / "chat.jsonl"),
            "computer_events": _digest(event_log),
            "canonical_trajectory": _digest(canonical_log),
            "environment_fingerprint": _digest(fingerprint),
        },
    }
    atomic_write_json(task_dir / "trajectory_manifest.json", manifest)
    episode_manifest = build_episode(task_dir, profile, normalized, run_id=run_id)
    quality = validate_episode(task_dir)
    identifier = str(episode_manifest["episode_id"])
    if ledger is not None:
        existing = ledger.episode(identifier)
        ledger.register_episode(
            identifier,
            run_id=run_id,
            profile_id=profile.id,
            task_id=str(episode_manifest["task"]["id"]),
            agent_id=str(episode_manifest["agent"]["id"]),
            output_dir=str(task_dir),
            metadata={"quality": quality["training_tier"]},
        )
        if existing is None or existing.get("status") not in {
            status.value for status in EpisodeStatus
            if status not in {EpisodeStatus.PENDING, EpisodeStatus.RUNNING}
        }:
            ledger.mark_running(identifier, worker_id="exporter")
        final_status = (
            EpisodeStatus(episode_manifest["status"])
            if quality["valid"] else EpisodeStatus.CORRUPT
        )
        ledger.finish_episode(identifier, final_status, metadata={
            "quality": quality,
            "manifest": str(task_dir / "episode_manifest.json"),
        })
    if trace_store is not None:
        indexes = trace_store.index_episode(
            task_dir,
            episode_id=identifier,
            task_id=str(episode_manifest["task"]["id"]),
            agent_id=str(episode_manifest["agent"]["id"]),
            run_id=run_id,
            created_at=str(episode_manifest["created_at"]),
        )
        episode_manifest["indexes"] = indexes
        atomic_write_json(task_dir / "episode_manifest.json", episode_manifest)
    return True


def export_result_tree(
    result_root: str | Path,
    profile: VMProfile,
    *,
    run_id: str = "standalone",
    ledger: EpisodeLedger | None = None,
    trace_store: TraceStore | None = None,
) -> int:
    root = Path(result_root)
    candidates = {path.parent for path in root.rglob("computer_events.jsonl")}
    candidates.update(path.parent for path in root.rglob("chat.jsonl"))
    candidates.update(path.parent for path in root.rglob("canonical_trajectory.jsonl"))
    return sum(
        1
        for task_dir in sorted(candidates)
        if export_task(
            task_dir,
            profile,
            run_id=run_id,
            ledger=ledger,
            trace_store=trace_store,
        )
    )
