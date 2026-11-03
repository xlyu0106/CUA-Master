"""Canonical atomic action space for CUA trajectories.

This module is deliberately dependency-free so labeling, execution, and
evaluation code can all reuse the same schema. The schema is intentionally small: GUI actions model
primitive human input, shell actions model one sandboxed command execution, and
tool actions model one registered structured call.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal


ActionCategory = Literal["gui", "cli", "tool", "control"]
ArgValidator = Callable[[dict[str, Any], str], None]

MOUSE_BUTTONS = {"left", "right", "middle"}
FINISH_STATUSES = {"success", "failure"}
MAX_SCROLL_DELTA = 50.0


@dataclass(frozen=True)
class ActionSpec:
    """Static schema metadata for one action type."""

    type: str
    category: ActionCategory
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    validate_args: ArgValidator | None = None

    @property
    def allowed(self) -> frozenset[str]:
        return self.required | self.optional


def _spec(
    action_type: str,
    category: ActionCategory,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
    validate_args: ArgValidator | None = None,
) -> ActionSpec:
    return ActionSpec(
        type=action_type,
        category=category,
        required=frozenset(required),
        optional=frozenset(optional),
        validate_args=validate_args,
    )


@dataclass(frozen=True)
class CUAAction:
    """One canonical CUA action.

    Coordinates use normalized 0-1000 screen space. Adapters are responsible for
    mapping them to the current screen resolution.
    """

    type: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CUAAction":
        if not isinstance(data, dict):
            raise TypeError("action must be a dict")
        if "type" not in data:
            raise ValueError("action type is required")
        action = cls(type=data["type"], args=data.get("args") or {})
        action.validate()
        return action

    def validate(self) -> None:
        validate_action(self)


@dataclass(frozen=True)
class CUAResponse:
    """Structured execution response for one action."""

    ok: bool
    observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolDefinition:
    """Minimal contract for a registered structured tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    read_only: bool = True
    destructive: bool = False
    open_world: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_action(action: CUAAction | dict[str, Any]) -> None:
    """Raise ValueError/TypeError if an action does not match the schema."""

    if isinstance(action, dict):
        action = CUAAction.from_dict(action)
        return
    if not isinstance(action, CUAAction):
        raise TypeError("action must be CUAAction or dict")
    if action.type not in ACTION_SPECS:
        raise ValueError(f"unsupported action type: {action.type}")
    if not isinstance(action.args, dict):
        raise TypeError("action args must be a dict")

    spec = ACTION_SPECS[action.type]
    _validate_shape(action.args, spec)
    if spec.validate_args is not None:
        spec.validate_args(action.args, action.type)


def from_legacy_action(name: str, *params: Any, **kwargs: Any) -> CUAAction:
    """Convert common OpenCUA/Qwen-style action names to canonical actions.

    This helper is for importing historical logs. New annotations should be
    emitted directly in the canonical schema.
    """

    key = name.strip().lower()
    converter = _LEGACY_CONVERTERS.get(key)
    if converter is None:
        raise ValueError(f"unsupported legacy action: {name}")
    return converter(params, kwargs)


def _validate_shape(args: dict[str, Any], spec: ActionSpec) -> None:
    missing = sorted(spec.required - set(args))
    if missing:
        raise ValueError(f"missing required args: {', '.join(missing)}")
    unknown = sorted(set(args) - spec.allowed)
    if unknown:
        raise ValueError(f"unknown args: {', '.join(unknown)}")


def _validated(action_type: str, args: dict[str, Any]) -> CUAAction:
    action = CUAAction(type=action_type, args=args)
    action.validate()
    return action


def _xy(params: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
    return kwargs.get("x", params[0] if len(params) > 0 else None), kwargs.get(
        "y", params[1] if len(params) > 1 else None
    )


def _require_number(value: Any, name: str, *, positive: bool = False, allow_negative: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0")
    if not allow_negative and value < 0:
        raise ValueError(f"{name} must be >= 0")


def _require_coord(value: Any, name: str) -> None:
    _require_number(value, name)
    if value > 1000:
        raise ValueError(f"{name} must be in normalized 0-1000 coordinates")


def _require_button(args: dict[str, Any]) -> None:
    button = args.get("button", "left")
    if button not in MOUSE_BUTTONS:
        raise ValueError(f"button must be one of {sorted(MOUSE_BUTTONS)}")


def _validate_point(args: dict[str, Any], _: str) -> None:
    _require_coord(args["x"], "x")
    _require_coord(args["y"], "y")


def _validate_point_click(args: dict[str, Any], action_type: str) -> None:
    _validate_point(args, action_type)
    _require_button(args)


def _validate_drag(args: dict[str, Any], _: str) -> None:
    for key in ("x1", "y1", "x2", "y2"):
        _require_coord(args[key], key)
    _require_button(args)
    if "duration_ms" in args:
        _require_number(args["duration_ms"], "duration_ms", positive=True)


def _validate_scroll(args: dict[str, Any], _: str) -> None:
    for axis in ("dx", "dy"):
        value = args[axis]
        _require_number(value, axis, allow_negative=True)
        if not math.isfinite(value):
            raise ValueError(f"{axis} must be finite")
        if abs(value) > MAX_SCROLL_DELTA:
            raise ValueError(
                f"{axis} must be between {-MAX_SCROLL_DELTA:g} and {MAX_SCROLL_DELTA:g}"
            )
    if "x" in args:
        _require_coord(args["x"], "x")
    if "y" in args:
        _require_coord(args["y"], "y")


def _validate_type_text(args: dict[str, Any], _: str) -> None:
    if not isinstance(args["text"], str):
        raise TypeError("text must be a string")
    if not args["text"]:
        raise ValueError("text must not be empty")


def _validate_press_keys(args: dict[str, Any], _: str) -> None:
    keys = args["keys"]
    if not isinstance(keys, list) or not keys:
        raise TypeError("keys must be a non-empty list")
    if not all(isinstance(key, str) and key for key in keys):
        raise TypeError("all keys must be non-empty strings")


def _validate_single_key(args: dict[str, Any], _: str) -> None:
    key = args["key"]
    if not isinstance(key, str) or not key:
        raise TypeError("key must be a non-empty string")


def _validate_mouse_button_state(args: dict[str, Any], _: str) -> None:
    _require_button(args)
    has_x = "x" in args
    has_y = "y" in args
    if has_x != has_y:
        raise ValueError("x and y must be provided together")
    if has_x:
        _validate_point(args, "")


def _validate_wait(args: dict[str, Any], _: str) -> None:
    _require_number(args["seconds"], "seconds", positive=True)


def _validate_shell_exec(args: dict[str, Any], _: str) -> None:
    if not isinstance(args["command"], str) or not args["command"].strip():
        raise TypeError("command must be a non-empty string")
    if "cwd" in args and not isinstance(args["cwd"], str):
        raise TypeError("cwd must be a string")
    if "timeout_sec" in args:
        _require_number(args["timeout_sec"], "timeout_sec", positive=True)


def _validate_tool_call(args: dict[str, Any], _: str) -> None:
    if not isinstance(args["name"], str) or not args["name"].strip():
        raise TypeError("name must be a non-empty string")
    if not isinstance(args["args"], dict):
        raise TypeError("args must be a dict")


def _validate_finish(args: dict[str, Any], _: str) -> None:
    if args["status"] not in FINISH_STATUSES:
        raise ValueError(f"status must be one of {sorted(FINISH_STATUSES)}")
    if "answer" in args and not isinstance(args["answer"], str):
        raise TypeError("answer must be a string")


ACTION_SPECS: dict[str, ActionSpec] = {
    "computer.click": _spec("computer.click", "gui", ("x", "y"), ("button",), _validate_point_click),
    "computer.double_click": _spec(
        "computer.double_click", "gui", ("x", "y"), ("button",), _validate_point_click
    ),
    "computer.triple_click": _spec(
        "computer.triple_click", "gui", ("x", "y"), ("button",), _validate_point_click
    ),
    "computer.move": _spec("computer.move", "gui", ("x", "y"), validate_args=_validate_point),
    "computer.drag": _spec(
        "computer.drag", "gui", ("x1", "y1", "x2", "y2"), ("button", "duration_ms"), _validate_drag
    ),
    "computer.scroll": _spec("computer.scroll", "gui", ("dx", "dy"), ("x", "y"), _validate_scroll),
    "computer.type": _spec("computer.type", "gui", ("text",), validate_args=_validate_type_text),
    "computer.press": _spec("computer.press", "gui", ("keys",), validate_args=_validate_press_keys),
    "computer.key_down": _spec(
        "computer.key_down", "gui", ("key",), validate_args=_validate_single_key
    ),
    "computer.key_up": _spec(
        "computer.key_up", "gui", ("key",), validate_args=_validate_single_key
    ),
    "computer.mouse_down": _spec(
        "computer.mouse_down",
        "gui",
        (),
        ("x", "y", "button"),
        _validate_mouse_button_state,
    ),
    "computer.mouse_up": _spec(
        "computer.mouse_up",
        "gui",
        (),
        ("x", "y", "button"),
        _validate_mouse_button_state,
    ),
    "computer.wait": _spec("computer.wait", "gui", ("seconds",), validate_args=_validate_wait),
    "shell.exec": _spec("shell.exec", "cli", ("command",), ("cwd", "timeout_sec"), _validate_shell_exec),
    "tool.call": _spec("tool.call", "tool", ("name", "args"), validate_args=_validate_tool_call),
    "finish": _spec("finish", "control", ("status",), ("answer",), _validate_finish),
}

ACTION_TYPES = frozenset(ACTION_SPECS)
GUI_ACTIONS = frozenset(action for action, spec in ACTION_SPECS.items() if spec.category == "gui")
CLI_ACTIONS = frozenset(action for action, spec in ACTION_SPECS.items() if spec.category == "cli")
TOOL_ACTIONS = frozenset(action for action, spec in ACTION_SPECS.items() if spec.category == "tool")
CONTROL_ACTIONS = frozenset(action for action, spec in ACTION_SPECS.items() if spec.category == "control")


def _legacy_click(params: tuple[Any, ...], kwargs: dict[str, Any], *, button: str = "left") -> CUAAction:
    x, y = _xy(params, kwargs)
    return _validated("computer.click", {"x": x, "y": y, "button": kwargs.get("button", button)})


def _legacy_double_click(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    x, y = _xy(params, kwargs)
    return _validated("computer.double_click", {"x": x, "y": y, "button": kwargs.get("button", "left")})


def _legacy_triple_click(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    x, y = _xy(params, kwargs)
    return _validated("computer.triple_click", {"x": x, "y": y, "button": kwargs.get("button", "left")})


def _legacy_move(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    x, y = _xy(params, kwargs)
    return _validated("computer.move", {"x": x, "y": y})


def _legacy_drag(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    args = {
        "x1": kwargs.get("x1", params[0] if len(params) > 0 else None),
        "y1": kwargs.get("y1", params[1] if len(params) > 1 else None),
        "x2": kwargs.get("x2", params[2] if len(params) > 2 else kwargs.get("x")),
        "y2": kwargs.get("y2", params[3] if len(params) > 3 else kwargs.get("y")),
        "button": kwargs.get("button", "left"),
    }
    if "duration_ms" in kwargs:
        args["duration_ms"] = kwargs["duration_ms"]
    return _validated("computer.drag", args)


def _legacy_scroll(params: tuple[Any, ...], kwargs: dict[str, Any], *, horizontal: bool = False) -> CUAAction:
    dx = kwargs.get("dx", params[0] if len(params) > 0 else 0)
    dy = kwargs.get("dy", params[1] if len(params) > 1 else 0)
    if horizontal and "dx" not in kwargs and len(params) == 1:
        dx, dy = params[0], 0
    args = {"dx": dx, "dy": dy}
    for key in ("x", "y"):
        if key in kwargs:
            args[key] = kwargs[key]
    return _validated("computer.scroll", args)


def _legacy_type(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    return _validated("computer.type", {"text": kwargs.get("text", params[0] if params else None)})


def _legacy_press(params: tuple[Any, ...], kwargs: dict[str, Any], *, hotkey: bool = False) -> CUAAction:
    keys = kwargs.get("keys")
    if keys is None:
        keys = list(params) if hotkey else [kwargs.get("key", params[0] if params else None)]
    return _validated("computer.press", {"keys": keys})


def _legacy_key_state(
    params: tuple[Any, ...], kwargs: dict[str, Any], *, action_type: str
) -> CUAAction:
    key = kwargs.get("key", params[0] if params else None)
    return _validated(action_type, {"key": key})


def _legacy_mouse_state(
    params: tuple[Any, ...], kwargs: dict[str, Any], *, action_type: str
) -> CUAAction:
    args: dict[str, Any] = {"button": kwargs.get("button", "left")}
    if len(params) >= 2:
        args.update({"x": params[0], "y": params[1]})
    elif "x" in kwargs or "y" in kwargs:
        args.update({"x": kwargs.get("x"), "y": kwargs.get("y")})
    return _validated(action_type, args)


def _legacy_wait(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    return _validated("computer.wait", {"seconds": kwargs.get("seconds", params[0] if params else 1)})


def _legacy_shell_exec(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    args = {"command": kwargs.get("command", params[0] if params else None)}
    for key in ("cwd", "timeout_sec"):
        if key in kwargs:
            args[key] = kwargs[key]
    return _validated("shell.exec", args)


def _legacy_tool_call(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    return _validated(
        "tool.call",
        {
            "name": kwargs.get("name", params[0] if params else None),
            "args": kwargs.get("args", params[1] if len(params) > 1 else {}),
        },
    )


def _legacy_finish(params: tuple[Any, ...], kwargs: dict[str, Any]) -> CUAAction:
    status = kwargs.get("status", params[0] if params else "success")
    if status == "failed":
        status = "failure"
    args = {"status": status}
    answer = kwargs.get("answer", params[1] if len(params) > 1 else None)
    if answer is not None:
        args["answer"] = answer
    return _validated("finish", args)


_LEGACY_CONVERTERS: dict[str, Callable[[tuple[Any, ...], dict[str, Any]], CUAAction]] = {
    "click": _legacy_click,
    "left_click": _legacy_click,
    "rightclick": lambda params, kwargs: _legacy_click(params, kwargs, button="right"),
    "right_click": lambda params, kwargs: _legacy_click(params, kwargs, button="right"),
    "middleclick": lambda params, kwargs: _legacy_click(params, kwargs, button="middle"),
    "middle_click": lambda params, kwargs: _legacy_click(params, kwargs, button="middle"),
    "doubleclick": _legacy_double_click,
    "double_click": _legacy_double_click,
    "tripleclick": _legacy_triple_click,
    "triple_click": _legacy_triple_click,
    "moveto": _legacy_move,
    "move": _legacy_move,
    "mouse_move": _legacy_move,
    "dragto": _legacy_drag,
    "drag": _legacy_drag,
    "left_click_drag": _legacy_drag,
    "scroll": _legacy_scroll,
    "hscroll": lambda params, kwargs: _legacy_scroll(params, kwargs, horizontal=True),
    "write": _legacy_type,
    "type": _legacy_type,
    "press": _legacy_press,
    "key": _legacy_press,
    "hotkey": lambda params, kwargs: _legacy_press(params, kwargs, hotkey=True),
    "keydown": lambda params, kwargs: _legacy_key_state(
        params, kwargs, action_type="computer.key_down"
    ),
    "key_down": lambda params, kwargs: _legacy_key_state(
        params, kwargs, action_type="computer.key_down"
    ),
    "keyup": lambda params, kwargs: _legacy_key_state(
        params, kwargs, action_type="computer.key_up"
    ),
    "key_up": lambda params, kwargs: _legacy_key_state(
        params, kwargs, action_type="computer.key_up"
    ),
    "mousedown": lambda params, kwargs: _legacy_mouse_state(
        params, kwargs, action_type="computer.mouse_down"
    ),
    "mouse_down": lambda params, kwargs: _legacy_mouse_state(
        params, kwargs, action_type="computer.mouse_down"
    ),
    "mouseup": lambda params, kwargs: _legacy_mouse_state(
        params, kwargs, action_type="computer.mouse_up"
    ),
    "mouse_up": lambda params, kwargs: _legacy_mouse_state(
        params, kwargs, action_type="computer.mouse_up"
    ),
    "wait": _legacy_wait,
    "cli_command": _legacy_shell_exec,
    "shell.exec": _legacy_shell_exec,
    "shell_exec": _legacy_shell_exec,
    "api_call": _legacy_tool_call,
    "tool.call": _legacy_tool_call,
    "tool_call": _legacy_tool_call,
    "terminate": _legacy_finish,
    "finish": _legacy_finish,
}
