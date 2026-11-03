from __future__ import annotations

import pytest

from cuamaster.action_space import ACTION_SPECS, ACTION_TYPES, CUAAction, from_legacy_action, validate_action


def test_action_space_has_no_screenshot_action() -> None:
    assert "computer.screenshot" not in ACTION_TYPES
    with pytest.raises(ValueError):
        validate_action({"type": "computer.screenshot", "args": {}})


def test_long_press_is_not_in_action_space() -> None:
    assert "computer.long_press" not in ACTION_TYPES
    with pytest.raises(ValueError):
        validate_action({"type": "computer.long_press", "args": {"x": 500, "y": 600}})
    with pytest.raises(ValueError):
        from_legacy_action("longPress", 500, 600)


def test_action_types_are_derived_from_specs() -> None:
    assert ACTION_TYPES == frozenset(ACTION_SPECS)
    assert ACTION_SPECS["computer.click"].required == frozenset({"x", "y"})
    assert ACTION_SPECS["computer.click"].optional == frozenset({"button"})


def test_qwen_input_primitives_are_canonical_actions() -> None:
    validate_action(
        {
            "type": "computer.triple_click",
            "args": {"x": 100, "y": 200, "button": "left"},
        }
    )
    validate_action({"type": "computer.key_down", "args": {"key": "shift"}})
    validate_action({"type": "computer.key_up", "args": {"key": "shift"}})
    validate_action(
        {"type": "computer.mouse_down", "args": {"button": "left"}}
    )
    validate_action(
        {
            "type": "computer.mouse_up",
            "args": {"x": 100, "y": 200, "button": "left"},
        }
    )


def test_mouse_button_state_requires_complete_optional_coordinate() -> None:
    with pytest.raises(ValueError, match="x and y must be provided together"):
        validate_action(
            {"type": "computer.mouse_down", "args": {"x": 100, "button": "left"}}
        )


def test_press_represents_single_key_and_hotkey() -> None:
    validate_action({"type": "computer.press", "args": {"keys": ["enter"]}})
    validate_action({"type": "computer.press", "args": {"keys": ["ctrl", "shift", "t"]}})


def test_legacy_press_and_hotkey_map_to_press() -> None:
    assert from_legacy_action("press", "enter").to_dict() == {
        "type": "computer.press",
        "args": {"keys": ["enter"]},
    }
    assert from_legacy_action("hotkey", "ctrl", "c").to_dict() == {
        "type": "computer.press",
        "args": {"keys": ["ctrl", "c"]},
    }
    assert from_legacy_action("keyDown", "shift").to_dict() == {
        "type": "computer.key_down",
        "args": {"key": "shift"},
    }
    assert from_legacy_action("keyUp", "shift").to_dict() == {
        "type": "computer.key_up",
        "args": {"key": "shift"},
    }


def test_legacy_mouse_actions_map_to_canonical_actions() -> None:
    assert from_legacy_action("click", 100, 200).to_dict() == {
        "type": "computer.click",
        "args": {"x": 100, "y": 200, "button": "left"},
    }
    assert from_legacy_action("rightClick", 100, 200).to_dict() == {
        "type": "computer.click",
        "args": {"x": 100, "y": 200, "button": "right"},
    }
    assert from_legacy_action("doubleClick", 100, 200).to_dict() == {
        "type": "computer.double_click",
        "args": {"x": 100, "y": 200, "button": "left"},
    }
    assert from_legacy_action("tripleClick", 100, 200).to_dict() == {
        "type": "computer.triple_click",
        "args": {"x": 100, "y": 200, "button": "left"},
    }
    assert from_legacy_action("dragTo", 100, 200, 300, 400).to_dict() == {
        "type": "computer.drag",
        "args": {"x1": 100, "y1": 200, "x2": 300, "y2": 400, "button": "left"},
    }


def test_scroll_allows_negative_direction() -> None:
    validate_action({"type": "computer.scroll", "args": {"dx": 0, "dy": -5}})
    assert from_legacy_action("hscroll", -3).to_dict() == {
        "type": "computer.scroll",
        "args": {"dx": -3, "dy": 0},
    }


@pytest.mark.parametrize("value", [51, -51, float("inf"), float("nan")])
def test_scroll_rejects_non_finite_or_excessive_values(value: float) -> None:
    with pytest.raises(ValueError):
        validate_action({"type": "computer.scroll", "args": {"dx": 0, "dy": value}})


def test_coordinate_validation_uses_normalized_range() -> None:
    validate_action({"type": "computer.click", "args": {"x": 0, "y": 1000}})
    with pytest.raises(ValueError):
        validate_action({"type": "computer.click", "args": {"x": 1001, "y": 1}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_coordinates_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        validate_action({"type": "computer.click", "args": {"x": value, "y": 0}})
    with pytest.raises(ValueError):
        validate_action(
            {"type": "computer.drag", "args": {"x1": 0, "y1": 0, "x2": value, "y2": 0}}
        )
    with pytest.raises(ValueError):
        validate_action({"type": "computer.wait", "args": {"seconds": value}})


def test_validation_rejects_unknown_args() -> None:
    with pytest.raises(ValueError):
        validate_action({"type": "computer.type", "args": {"text": "hello", "delay": 1}})


def test_from_dict_requires_type() -> None:
    with pytest.raises(ValueError):
        CUAAction.from_dict({"args": {}})


def test_shell_tool_and_finish_shapes() -> None:
    validate_action({"type": "shell.exec", "args": {"command": "pwd", "timeout_sec": 5}})
    validate_action({"type": "tool.call", "args": {"name": "ask_user", "args": {"prompt": "Continue?"}}})
    validate_action({"type": "finish", "args": {"status": "success", "answer": "done"}})


def test_legacy_terminate_failed_normalizes_to_failure() -> None:
    assert from_legacy_action("terminate", "failed", "blocked").to_dict() == {
        "type": "finish",
        "args": {"status": "failure", "answer": "blocked"},
    }
