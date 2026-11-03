from __future__ import annotations

import base64
import io
import importlib.util
import json
import subprocess
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuamaster.collection.episode import canonical_steps
from cuamaster.collection.export import (
    _canonical_runtime_events,
    _canonical_gui_action,
    _chat_tool_calls,
    _shell_action,
    export_result_tree,
)
from cuamaster.collection.assets import claudecode_shell_screenshot_hook
from cuamaster.collection.image_store import ImageStore
from cuamaster.collection.profiles import expand_platform_profiles, load_matrix_config
from cuamaster.collection.task_specs import task_supports_profile
from cuamaster.contracts import TaskSpec

def _fake_weavebench_root(tmp_path):
    root = tmp_path / "WeaveBench"
    (root / "weavebench/eval/_runners").mkdir(parents=True, exist_ok=True)
    (root / "weavebench/eval/_runners/messages_gui.py").write_text("", encoding="utf-8")
    return str(root)



def _config(tmp_path: Path, profiles: list[dict]) -> Path:
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "defaults": {"provider": "docker", "num_envs": 2},
        "profiles": profiles,
    }), encoding="utf-8")
    return path


def _profile(tmp_path: Path):
    return load_matrix_config(_config(tmp_path, [{
        "id": "ubuntu",
        "os": {"type": "Ubuntu"},
        "image": {},
    }])).profiles[0]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _png_header(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", width, height)


def test_collection_preserves_qwen_gui_primitives_as_canonical_actions():
    assert _canonical_gui_action(
        {"type": "triple_click", "x": 100, "y": 200}, (1000, 1000)
    ) == [
        {
            "type": "computer.triple_click",
            "args": {"x": 100, "y": 200, "button": "left"},
        }
    ]
    assert _canonical_gui_action(
        {"type": "key_down", "keys": ["SHIFT"]}, None
    ) == [{"type": "computer.key_down", "args": {"key": "shift"}}]
    assert _canonical_gui_action({"type": "left_mouse_up"}, None) == [
        {"type": "computer.mouse_up", "args": {"button": "left"}}
    ]
    assert _canonical_gui_action({"type": "type", "text": ""}, None) == []
    assert _canonical_gui_action({"type": "scroll", "dx": 0, "dy": 0}, None) == []
    assert _canonical_gui_action({"type": "unsupported_gui_primitive"}, None) == []


def test_canonical_scroll_records_the_delta_the_guest_actuator_executes():
    from cuamaster.action_space import CUAAction
    from cuamaster.runtime.core import _gui_command

    canonical = _canonical_gui_action(
        {"type": "scroll", "x": 100, "y": 200, "scroll_x": 0, "scroll_y": 3},
        (1000, 1000),
    )

    assert canonical == [
        {"type": "computer.scroll", "args": {"dx": 0.0, "dy": -3.0, "x": 100, "y": 200}}
    ]
    # Replaying the recorded action must reproduce the negated delta the guest
    # actuator executed instead of scrolling the opposite way.
    assert _gui_command(CUAAction.from_dict(canonical[0]), 1000, 1000).endswith(
        "pyautogui.scroll(-3.0)"
    )


def test_canonical_scroll_passes_through_canonical_deltas():
    assert _canonical_gui_action(
        {"type": "scroll", "dx": 2, "dy": -4, "x": 0, "y": 0}, None
    ) == [
        {"type": "computer.scroll", "args": {"dx": 2.0, "dy": -4.0, "x": 0, "y": 0}}
    ]


def test_guest_actuator_scrolls_each_axis_on_its_own_axis():
    from cuamaster.collection.assets.weavebench_computer_mcp.server import action_to_python

    horizontal = action_to_python(
        {"type": "scroll", "x": 10, "y": 20, "scroll_x": 4, "scroll_y": 0}
    )
    assert "pyautogui.hscroll(-4)" in horizontal
    assert "pyautogui.scroll(" not in horizontal

    both = action_to_python(
        {"type": "scroll", "x": 10, "y": 20, "scroll_x": 4, "scroll_y": 3}
    )
    assert "pyautogui.hscroll(-4)" in both
    assert "pyautogui.scroll(-3)" in both


def test_chat_parser_ignores_all_non_gui_cli_tools(tmp_path):
    _write_jsonl(tmp_path / "chat.jsonl", [
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use",
            "id": "todo-1",
            "name": "TodoWrite",
            "input": {"todos": []},
        }]}},
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use",
            "id": "read-1",
            "name": "Read",
            "input": {"file_path": "/tmp/demo.txt"},
        }]}},
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use",
            "id": "shell-1",
            "name": "Bash",
            "input": {"command": "pwd"},
        }]}},
    ])

    calls = _chat_tool_calls(tmp_path)

    assert [call["tool_name"] for call in calls] == ["Bash"]


def test_canonical_steps_propagate_initial_screenshot_across_shell_actions():
    steps = canonical_steps([
        {
            "pre_screenshot": "screenshots/init.png",
            "actions": [{"type": "shell.exec", "args": {"command": "pwd"}}],
            "executions": [{"ok": True}],
        },
        {
            "actions": [{"type": "shell.exec", "args": {"command": "ls"}}],
            "executions": [{"ok": True}],
        },
    ], success=True)

    assert steps[0]["pre_observation"]["screenshot"] == "screenshots/init.png"
    assert steps[1]["pre_observation"]["screenshot"] == "screenshots/init.png"


def test_canonical_runtime_filters_invalid_and_internal_actions(tmp_path):
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    (tmp_path / "init_screenshot.png").write_bytes(_png_header(800, 600))
    _write_jsonl(tmp_path / "canonical_trajectory.jsonl", [
        {
            "step": 1,
            "action": {"type": "computer.type", "args": {"text": ""}},
            "execution_result": {"ok": True},
        },
        {
            "step": 2,
            "action": {
                "type": "tool.call",
                "args": {"name": "TaskOutput", "args": {"task_id": "1"}},
            },
            "execution_result": {"ok": True},
        },
        {
            "step": 3,
            "action": {"type": "computer.wait", "args": {"seconds": 1}},
            "execution_result": {"ok": True},
        },
    ])

    events = _canonical_runtime_events(tmp_path)

    assert [event["actions"][0]["type"] for event in events] == ["computer.wait"]
    assert events[0]["pre_screenshot"] == "screenshots/init_screenshot.png"


def test_profile_loading_and_relative_image(tmp_path):
    config = load_matrix_config(_config(tmp_path, [{
        "id": "ubuntu-22.04",
        "os": {"type": "Ubuntu", "version": "22.04"},
        "image": {"path": "images/base.qcow2"},
        "software": {"libreoffice": "7.3.7.2"},
    }]))
    profile = config.profiles[0]
    assert profile.num_envs == 2
    assert profile.image_path == str((tmp_path / "images/base.qcow2").resolve())


def test_duplicate_profile_rejected(tmp_path):
    row = {"id": "same", "os": {"type": "Ubuntu"}, "image": {}}
    with pytest.raises(ValueError, match="duplicate profile"):
        load_matrix_config(_config(tmp_path, [row, row]))


def test_all_platform_mode_expands_each_logical_profile(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "defaults": {
            "provider": "sandbox",
            "platform_mode": "all",
            "platform_worker_selection": "random_balanced",
            "platform_profiles": {
                "linux": {
                    "os": {"type": "Ubuntu", "version": "22.04"},
                    "image": {"id": "linux-image"},
                    "backend_options": {
                        "sandbox_vm_backend": "desktop_env"
                    },
                },
                "windows": {
                    "os": {"type": "Windows", "version": "10"},
                    "image": {"id": "windows-image"},
                    "backend_options": {
                        "sandbox_vm_backend": "desktop_env"
                    },
                },
                "macos": {
                    "os": {"type": "Darwin", "version": "15.2"},
                    "image": {"id": "macos-image"},
                    "backend_options": {
                        "sandbox_vm_backend": "macos_kvm"
                    },
                },
            },
        },
        "profiles": [{
            "id": "claude-hybrid",
            "backend": "weavebench",
            "os": {"type": "Ubuntu"},
            "image": {},
            "backend_options": {
                "mode": "hybrid",
                "task_filter": "task-a,task-b",
            },
        }],
    }), encoding="utf-8")
    config = load_matrix_config(path)

    profiles = expand_platform_profiles(config, config.enabled_profiles())

    assert [profile.id for profile in profiles] == [
        "claude-hybrid-linux",
        "claude-hybrid-windows",
        "claude-hybrid-macos",
    ]
    assert [profile.os_type for profile in profiles] == [
        "Ubuntu",
        "Windows",
        "Darwin",
    ]
    assert [profile.image_id for profile in profiles] == [
        "linux-image",
        "windows-image",
        "macos-image",
    ]
    assert {
        profile.backend_options["task_filter"] for profile in profiles
    } == {"task-a,task-b"}
    assert all(
        profile.backend_options["force_all_tasks_on_platform"] is True
        for profile in profiles
    )
    assert all(
        profile.backend_options["sandbox_worker_selection"]
        == "random_balanced"
        for profile in profiles
    )
    assert all(
        profile.backend_options["sandbox_worker_selection_seed"]
        == "{resource_id}"
        for profile in profiles
    )
    assert profiles[-1].backend_options["sandbox_vm_backend"] == "macos_kvm"


def test_configured_platform_mode_does_not_expand_profiles(tmp_path):
    config = load_matrix_config(_config(tmp_path, [{
        "id": "ubuntu",
        "os": {"type": "Ubuntu"},
        "image": {},
    }]))

    assert expand_platform_profiles(config, list(config.profiles), "configured") == [
        config.profiles[0]
    ]


def test_task_platform_constraints_match_profile(tmp_path):
    profile = load_matrix_config(_config(tmp_path, [{
        "id": "windows-11",
        "os": {"type": "Windows", "version": "11", "architecture": "x86_64"},
        "image": {},
        "software": {"notepad": "11"},
    }])).profiles[0]
    task = TaskSpec.from_dict({
        "id": "windows-task",
        "instruction": "Open Notepad.",
        "required_capabilities": ["computer", "app:notepad"],
        "target_environments": [{
            "os_type": "Windows",
            "os_versions": ["11"],
            "architectures": ["x86_64"],
        }],
    })
    assert task_supports_profile(task, profile)
    assert task.to_dict()["target_environments"][0]["os_type"] == "Windows"


def test_task_platform_constraints_reject_other_os(tmp_path):
    profile = _profile(tmp_path)
    task = TaskSpec.from_dict({
        "id": "windows-task",
        "instruction": "Open Notepad.",
        "target_environments": [{"os_type": "Windows"}],
    })
    assert not task_supports_profile(task, profile)


def test_event_export(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "gui" / "model" / "DOC" / "task-1"
    (task / "screenshots").mkdir(parents=True)
    (task / "screenshots" / "screenshot_0001_click.png").write_bytes(b"png")
    event = {
        "step": 1,
        "actions": [{"type": "click", "x": 1, "y": 2}],
        "executions": [{"ok": True}],
        "ok": True,
        "screenshot_path": "/tmp_workspace/_screenshots/screenshot_0001_click.png",
    }
    (task / "computer_events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    assert export_result_tree(tmp_path / "results", profile) == 1
    exported = json.loads((task / "trajectory.jsonl").read_text(encoding="utf-8"))
    assert exported["screenshot"] == "screenshots/screenshot_0001_click.png"
    assert exported["actions"] == [{
        "type": "computer.click",
        "args": {"x": 1, "y": 2, "button": "left"},
    }]


def test_export_merges_cli_and_gui_in_chat_order(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "gui" / "model" / "DOC" / "mixed-task"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True)
    shot = screenshots / "screenshot_0001_click.png"
    shot.write_bytes(_png_header(2000, 1000))

    _write_jsonl(task / "chat.jsonl", [
        {"type": "message", "message": {"role": "assistant", "content": [{
            "type": "text", "text": "Create the workspace before using the GUI.",
        }]}},
        {"type": "message", "message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "shell-1", "name": "Bash",
            "input": {"command": "mkdir -p /tmp/demo", "timeout": 5000},
        }]}},
        {"type": "message", "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "shell-1", "content": "created",
            "is_error": False,
        }]}},
        {"type": "message", "message": {"role": "assistant", "content": [{
            "type": "text", "text": "Click the center control next.",
        }, {
            "type": "tool_use", "id": "gui-1",
            "name": "mcp__weavebench_computer__computer",
            "input": {"actions": [{"action": "click", "x": 1000, "y": 500, "button": 1}]},
        }]}},
        {"type": "message", "message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "shell-2", "name": "functions.exec_command",
            "input": {"cmd": "pwd", "workdir": "/tmp/demo"},
        }]}},
        {"type": "message", "message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "shell-2", "content": "/tmp/demo\n",
        }]}},
    ])
    _write_jsonl(task / "computer_events.jsonl", [{
        "step": 1,
        "request": {"actions": [{"type": "click", "x": 1000, "y": 500, "button": "left"}]},
        "executions": [{"ok": True}],
        "ok": True,
        "screenshot_path": f"/tmp_workspace/_screenshots/{shot.name}",
    }])

    assert export_result_tree(tmp_path / "results", profile) == 1
    rows = [json.loads(line) for line in (task / "trajectory.jsonl").read_text().splitlines()]
    assert [row["category"] for row in rows] == ["cli", "gui", "cli"]
    assert [row["actions"][0]["type"] for row in rows] == [
        "shell.exec", "computer.click", "shell.exec",
    ]
    assert rows[0]["actions"][0]["args"] == {
        "command": "mkdir -p /tmp/demo", "timeout_sec": 5.0,
    }
    assert rows[0]["executions"][0]["output"] == "created"
    assert rows[1]["actions"][0]["args"] == {"x": 500, "y": 500, "button": "left"}
    assert rows[1]["source"] == "chat+mcp_event_log"
    assert rows[0]["decision_context"]["intent"] == (
        "Create the workspace before using the GUI."
    )
    assert rows[0]["decision_context"]["visibility"] == (
        "observable_assistant_output"
    )
    assert rows[1]["decision_context"]["intent"] == "Click the center control next."
    assert rows[2]["actions"][0]["args"] == {"command": "pwd", "cwd": "/tmp/demo"}

    canonical_steps = [
        json.loads(line)
        for line in (task / "canonical" / "steps.jsonl").read_text().splitlines()
    ]
    assert canonical_steps[0]["decision_context"]["intent"] == (
        "Create the workspace before using the GUI."
    )
    assert canonical_steps[0]["provenance"]["assistant_intent"] == (
        "Create the workspace before using the GUI."
    )

    manifest = json.loads((task / "trajectory_manifest.json").read_text())
    assert manifest["schema_version"] == "cua_trajectory.v1"
    assert manifest["source"] == "chat+mcp_event_log"
    assert manifest["event_categories"] == {"cli": 2, "gui": 1}
    assert manifest["action_types"] == {"computer.click": 1, "shell.exec": 2}


def test_shell_capture_hook_records_post_screenshot(monkeypatch, tmp_path):
    trajectory_dir = tmp_path / "_trajectory"
    screenshot_dir = tmp_path / "_shell_screenshots"
    event_log = trajectory_dir / "shell_events.jsonl"
    monkeypatch.setattr(claudecode_shell_screenshot_hook, "TRAJECTORY_DIR", trajectory_dir)
    monkeypatch.setattr(claudecode_shell_screenshot_hook, "SCREENSHOT_DIR", screenshot_dir)
    monkeypatch.setattr(claudecode_shell_screenshot_hook, "EVENT_LOG", event_log)
    monkeypatch.setenv("GUI_TRAJECTORY_CAPTURE_SHELL_SCREENSHOTS", "true")
    monkeypatch.setenv("GUI_TRAJECTORY_SHELL_POST_SCREENSHOT_DELAY_S", "0")

    def capture(path):
        path.write_bytes(_png_header(1280, 720) + b"capture")
        return None

    monkeypatch.setattr(claudecode_shell_screenshot_hook, "_capture", capture)
    payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "session-1",
        "tool_use_id": "Bash:7",
        "tool_name": "Bash",
        "tool_input": {"command": "touch /tmp/demo"},
        "tool_response": {"returncode": 0, "stdout": "created"},
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    claudecode_shell_screenshot_hook.capture_hook()

    events = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert [event["phase"] for event in events] == ["post"]
    assert {event["tool_call_id"] for event in events} == {"Bash:7"}
    assert events[0]["execution"]["ok"] is True
    assert events[0]["execution"]["returncode"] == 0
    assert events[0]["execution"]["stdout"] == "created"
    assert (screenshot_dir / "shell_Bash_7.png").is_file()


def test_shell_capture_hook_records_failure_without_screenshot(monkeypatch, tmp_path):
    trajectory_dir = tmp_path / "trajectory"
    event_log = trajectory_dir / "shell_events.jsonl"
    monkeypatch.setattr(claudecode_shell_screenshot_hook, "TRAJECTORY_DIR", trajectory_dir)
    monkeypatch.setattr(claudecode_shell_screenshot_hook, "EVENT_LOG", event_log)
    payload = {
        "hook_event_name": "PostToolUseFailure",
        "session_id": "session-1",
        "tool_use_id": "Bash:8",
        "tool_name": "Bash",
        "tool_input": {"command": "false"},
        "tool_response": {"returncode": 1, "stderr": "failed"},
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    claudecode_shell_screenshot_hook.capture_hook()

    event = json.loads(event_log.read_text())
    assert event["screenshot_path"] is None
    assert event["execution"] == {
        "ok": False,
        "status_source": "claude_code_hook",
        "hook_event_name": "PostToolUseFailure",
        "returncode": 1,
        "stderr": "failed",
        "tool_response": {"returncode": 1, "stderr": "failed"},
    }


def test_export_builds_shell_state_chain_from_initial_and_post_screenshot(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "gui" / "model" / "DOC" / "shell-capture-task"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True)
    (task / "init_screenshot.png").write_bytes(_png_header(1280, 720))
    (screenshots / "shell_Bash_0.png").write_bytes(_png_header(1280, 720))
    _write_jsonl(task / "chat.jsonl", [
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use",
            "id": "Bash:0",
            "name": "Bash",
            "input": {"command": "touch /tmp/demo"},
        }]}},
        {"message": {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": "Bash:0",
            "content": "",
        }]}},
    ])
    _write_jsonl(task / "shell_events.jsonl", [
        {
            "phase": "post",
            "tool_call_id": "Bash:0",
            "screenshot_path": "/tmp_workspace/_shell_screenshots/shell_Bash_0.png",
        },
    ])

    assert export_result_tree(tmp_path / "results", profile) == 1
    event = json.loads((task / "trajectory.jsonl").read_text())
    assert event["source"] == "chat+shell_capture"
    assert event["pre_screenshot"] == "screenshots/init_screenshot.png"
    assert event["screenshot"] == "screenshots/shell_Bash_0.png"
    steps = [
        json.loads(line)
        for line in (task / "canonical" / "steps.jsonl").read_text().splitlines()
    ]
    assert steps[0]["action"]["type"] == "shell.exec"
    assert steps[0]["pre_observation"] == {
        "screenshot": "screenshots/init_screenshot.png"
    }
    assert steps[0]["post_observation"] == {
        "screenshot": "screenshots/shell_Bash_0.png"
    }
    assert steps[0]["observation_quality"] == "complete"


def test_consecutive_shell_actions_reuse_previous_post_as_next_pre(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "gui" / "model" / "DOC" / "shell-chain-task"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True)
    (task / "init_screenshot.png").write_bytes(_png_header(1280, 720))
    (screenshots / "shell_Bash_0.png").write_bytes(_png_header(1280, 720))
    (screenshots / "shell_Bash_1.png").write_bytes(_png_header(1280, 720))
    _write_jsonl(task / "chat.jsonl", [
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "Bash:0", "name": "Bash",
            "input": {"command": "touch /tmp/one"},
        }]}},
        {"message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "Bash:0", "content": "",
        }]}},
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "Bash:1", "name": "Bash",
            "input": {"command": "touch /tmp/two"},
        }]}},
        {"message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "Bash:1", "content": "",
        }]}},
    ])
    _write_jsonl(task / "shell_events.jsonl", [
        {
            "phase": "post", "tool_call_id": "Bash:0",
            "screenshot_path": "/tmp_workspace/_shell_screenshots/shell_Bash_0.png",
        },
        {
            "phase": "post", "tool_call_id": "Bash:1",
            "screenshot_path": "/tmp_workspace/_shell_screenshots/shell_Bash_1.png",
        },
    ])

    assert export_result_tree(tmp_path / "results", profile) == 1
    steps = [
        json.loads(line)
        for line in (task / "canonical" / "steps.jsonl").read_text().splitlines()
    ]
    assert steps[0]["pre_observation"]["screenshot"] == "screenshots/init_screenshot.png"
    assert steps[0]["post_observation"]["screenshot"] == "screenshots/shell_Bash_0.png"
    assert steps[1]["pre_observation"] == steps[0]["post_observation"]
    assert steps[1]["post_observation"]["screenshot"] == "screenshots/shell_Bash_1.png"


def test_host_guest_shell_captures_are_matched_in_call_order(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "hybrid" / "model" / "DOC" / "host-shell-task"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True)
    for name in ("pre-1.png", "post-1.png", "pre-2.png", "post-2.png"):
        (screenshots / name).write_bytes(_png_header(1280, 720))
    _write_jsonl(task / "chat.jsonl", [
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "tool-a",
            "name": "mcp__weavebench_computer__shell",
            "input": {"command": "Get-Date"},
        }]}},
        {"message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "tool-a", "content": "ok",
        }]}},
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "tool-b",
            "name": "mcp__weavebench_computer__shell",
            "input": {"command": "pwd"},
        }]}},
        {"message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "tool-b", "content": "ok",
        }]}},
    ])
    _write_jsonl(task / "shell_events.jsonl", [
        {"sequence": 1, "phase": "pre", "screenshot_path": str(screenshots / "pre-1.png")},
        {
            "sequence": 1,
            "phase": "post",
            "screenshot_path": str(screenshots / "post-1.png"),
            "execution": {"ok": True, "returncode": 0, "stdout": "date"},
        },
        {"sequence": 2, "phase": "pre", "screenshot_path": str(screenshots / "pre-2.png")},
        {
            "sequence": 2,
            "phase": "post",
            "screenshot_path": str(screenshots / "post-2.png"),
            "execution": {"ok": True, "returncode": 0, "stdout": "/Users/oai"},
        },
    ])

    assert export_result_tree(tmp_path / "results", profile) == 1
    rows = [json.loads(line) for line in (task / "trajectory.jsonl").read_text().splitlines()]
    assert [row["actions"][0]["type"] for row in rows] == ["shell.exec", "shell.exec"]
    assert rows[0]["pre_screenshot"] == "screenshots/pre-1.png"
    assert rows[0]["screenshot"] == "screenshots/post-1.png"
    assert rows[0]["executions"][0]["stdout"] == "date"
    assert rows[1]["pre_screenshot"] == "screenshots/pre-2.png"
    assert rows[1]["screenshot"] == "screenshots/post-2.png"


@pytest.mark.parametrize("tool_name", ["exec", "Bash", "terminal", "exec_command"])
def test_cli_only_chat_is_exported_for_every_harness(tmp_path, tool_name):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / tool_name
    task.mkdir(parents=True)
    _write_jsonl(task / "chat.jsonl", [{
        "role": "assistant",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps({"command": "pwd"})},
        }],
    }])

    assert export_result_tree(tmp_path / "results", profile) == 1
    row = json.loads((task / "trajectory.jsonl").read_text())
    assert row["category"] == "cli"
    assert row["actions"] == [{"type": "shell.exec", "args": {"command": "pwd"}}]


def test_shell_timeout_units_follow_harness_contract():
    assert _shell_action({"command": "pwd", "timeout": 1200}, "exec") == {
        "type": "shell.exec",
        "args": {"command": "pwd", "timeout_sec": 1200},
    }
    assert _shell_action({"command": "pwd", "timeout": 1200}, "Bash") == {
        "type": "shell.exec",
        "args": {"command": "pwd", "timeout_sec": 2},
    }


def test_chat_tool_result_marks_shell_execution_as_verified(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "shell-result"
    task.mkdir(parents=True)
    _write_jsonl(task / "chat.jsonl", [
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "Bash:1", "name": "Bash",
            "input": {"command": "pwd"},
        }]}},
        {"message": {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "Bash:1", "content": "/tmp",
        }]}},
    ])

    assert export_result_tree(tmp_path / "results", profile) == 1
    row = json.loads((task / "trajectory.jsonl").read_text())
    assert row["executions"][0]["ok"] is True
    assert row["executions"][0]["status_source"] == "chat_tool_result"


def test_chat_tool_result_uses_explicit_nonzero_shell_exit_code(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "shell-failure"
    task.mkdir(parents=True)
    _write_jsonl(task / "chat.jsonl", [
        {"message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "Bash:1", "name": "Bash",
            "input": {"command": "ls /missing"},
        }]}},
        {"message": {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": "Bash:1",
            "content": "Exit code 2\nls: cannot access '/missing'",
        }]}},
    ])

    assert export_result_tree(tmp_path / "results", profile) == 1
    row = json.loads((task / "trajectory.jsonl").read_text())
    assert row["ok"] is False
    assert row["executions"][0]["ok"] is False
    assert row["executions"][0]["exit_code"] == 2
    assert row["errors"] == ["Exit code 2\nls: cannot access '/missing'"]


def test_openclaw_camel_case_chat_exports_actions_and_embedded_screenshot(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "openclaw-chat"
    task.mkdir(parents=True)
    screenshot = _png_header(1920, 1080)
    _write_jsonl(task / "chat.jsonl", [
        {"type": "message", "message": {"role": "assistant", "content": [{
            "type": "toolCall",
            "id": "gui-1",
            "name": "__computer__",
            "arguments": {"action": {"type": "click", "x": 960, "y": 540}},
        }]}},
        {"type": "message", "message": {
            "role": "toolResult",
            "toolCallId": "gui-1",
            "toolName": "__computer__",
            "content": [{
                "type": "image",
                "mimeType": "image/png",
                "data": base64.b64encode(screenshot).decode("ascii"),
            }],
            "isError": False,
        }},
        {"type": "message", "message": {"role": "assistant", "content": [{
            "type": "toolCall",
            "id": "shell-1",
            "name": "exec",
            "arguments": {"command": "false"},
        }]}},
        {"type": "message", "message": {
            "role": "toolResult",
            "toolCallId": "shell-1",
            "toolName": "exec",
            "content": [{"type": "text", "text": "command failed"}],
            "details": {"exitCode": 1},
            "isError": False,
        }},
    ])

    assert export_result_tree(tmp_path / "results", profile) == 1
    rows = [json.loads(line) for line in (task / "trajectory.jsonl").read_text().splitlines()]
    assert [row["category"] for row in rows] == ["gui", "cli"]
    assert rows[0]["actions"] == [{
        "type": "computer.click",
        "args": {"x": 500, "y": 500, "button": "left"},
    }]
    assert rows[0]["screenshot"] == "screenshots/screenshot_0001_tool_result.png"
    assert (task / rows[0]["screenshot"]).read_bytes() == screenshot
    assert rows[1]["executions"][0]["exit_code"] == 1
    assert rows[1]["executions"][0]["ok"] is False


def test_non_shell_tools_are_ignored_at_export_source(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "tools"
    task.mkdir(parents=True)
    _write_jsonl(task / "chat.jsonl", [
        {
            "type": "message", "message": {"role": "assistant", "content": [{
                "type": "tool_use", "id": "read-1", "name": "Read",
                "input": {"file_path": "/tmp/demo.txt"},
            }]},
        },
        {
            "type": "message", "message": {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "read-1", "content": "demo",
            }]},
        },
    ])

    assert export_result_tree(tmp_path / "results", profile) == 0
    assert not (task / "trajectory.jsonl").exists()


def test_canonical_runtime_tool_observation_is_rejected(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "canonical-tool"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True)
    screenshot = screenshots / "step_0001.png"
    screenshot.write_bytes(b"png")
    _write_jsonl(task / "canonical_trajectory.jsonl", [{
        "step": 1,
        "raw_model_text": "Code: tool.call(name='ui.inspect', args={'query': 'Save'})",
        "action": {
            "type": "tool.call",
            "args": {"name": "ui.inspect", "args": {"query": "Save"}},
        },
        "execution_result": {
            "ok": True,
            "observation": {"category": "tool", "result": {"nodes": [{"name": "Save"}]}},
        },
        "screenshot_path": str(screenshot),
        "end_reason": None,
    }])

    assert export_result_tree(tmp_path / "results", profile) == 0
    assert not (task / "trajectory.jsonl").exists()
    assert not (task / "trajectory_manifest.json").exists()


def test_canonical_gui_sidecar_actions_remain_canonical(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "canonical-sidecar"
    task.mkdir(parents=True)
    _write_jsonl(task / "computer_events.jsonl", [{
        "actions": [{
            "type": "computer.drag",
            "args": {"x1": 10, "y1": 20, "x2": 30, "y2": 40, "button": "left"},
        }],
        "ok": True,
    }])

    assert export_result_tree(tmp_path / "results", profile) == 1
    row = json.loads((task / "trajectory.jsonl").read_text())
    assert row["actions"] == [{
        "type": "computer.drag",
        "args": {"x1": 10, "y1": 20, "x2": 30, "y2": 40, "button": "left"},
    }]


def test_computer_chat_fallback_works_without_mcp_sidecar(tmp_path):
    profile = _profile(tmp_path)
    task = tmp_path / "results" / "chat-gui-fallback"
    screenshots = task / "screenshots"
    screenshots.mkdir(parents=True)
    (screenshots / "screenshot_0001_click.png").write_bytes(_png_header(100, 100))
    _write_jsonl(task / "chat.jsonl", [{
        "type": "message", "message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "gui-1", "name": "computer",
            "input": {"action": {"type": "click", "x": 50, "y": 25}},
        }]},
    }])

    assert export_result_tree(tmp_path / "results", profile) == 1
    row = json.loads((task / "trajectory.jsonl").read_text())
    assert row["source"] == "chat"
    assert row["actions"] == [{
        "type": "computer.click",
        "args": {"x": 500, "y": 250, "button": "left"},
    }]
    assert row["screenshot"] == "screenshots/screenshot_0001_click.png"


def test_export_only_rebuilds_existing_mixed_trajectory(tmp_path):
    from cuamaster.collection.collector import main

    config_path = _config(tmp_path, [{
        "id": "ubuntu",
        "os": {"type": "Ubuntu"},
        "image": {},
    }])
    output_root = tmp_path / "existing-results"
    task = output_root / "ubuntu" / "task-1"
    task.mkdir(parents=True)
    _write_jsonl(task / "chat.jsonl", [{
        "type": "message", "message": {"role": "assistant", "content": [{
            "type": "tool_use", "id": "shell-1", "name": "terminal",
            "input": {"command": "pwd"},
        }]},
    }])

    assert main([
        "--config", str(config_path),
        "--output-root", str(output_root),
        "--export-only",
    ]) == 0
    row = json.loads((task / "trajectory.jsonl").read_text())
    assert row["actions"] == [{"type": "shell.exec", "args": {"command": "pwd"}}]


def test_image_store_resolves_managed_path(tmp_path):
    store_root = tmp_path / "images"
    managed = store_root / "linux" / "test" / "disk.qcow2"
    managed.parent.mkdir(parents=True)
    managed.write_bytes(b"image")
    descriptor = managed.parent / "descriptor.json"
    descriptor.write_text(json.dumps({
        "schema_version": 1,
        "image_id": "test",
        "os": {"type": "Fedora", "version": "42", "architecture": "x86_64"},
        "apps": [{"name": "Test App", "version": "1.0"}],
    }), encoding="utf-8")
    (store_root / "catalog.json").write_text(json.dumps({
        "schema_version": 1,
        "images": {
            "test": {
                "path": "linux/test/disk.qcow2",
                "descriptor": "linux/test/descriptor.json",
                "format": "qcow2",
                "os": {"type": "Fedora", "version": "42", "architecture": "x86_64"},
                "size_bytes": 5,
            }
        },
    }), encoding="utf-8")
    record = ImageStore(store_root).resolve("test")
    assert record.path == managed
    assert record.descriptor_path == descriptor
    assert record.os_type == "Fedora"

    descriptor.write_text(json.dumps({
        "schema_version": 1,
        "image_id": "test",
        "os": {"type": "Fedora", "version": "41", "architecture": "x86_64"},
        "apps": [{"name": "Test App"}],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor OS version mismatch"):
        ImageStore(store_root).resolve("test")

    descriptor.write_text(json.dumps({
        "schema_version": 1,
        "image_id": "wrong-image",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor image_id mismatch"):
        ImageStore(store_root).resolve("test")


def test_image_store_adopts_verified_managed_image(tmp_path):
    store_root = tmp_path / "images"
    store_root.mkdir()
    (store_root / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "images": {}}), encoding="utf-8"
    )
    disk = store_root / "linux" / "baked" / "disk.qcow2"
    disk.parent.mkdir(parents=True)
    disk.write_bytes(b"image")
    (disk.parent / "descriptor.json").write_text(json.dumps({
        "schema_version": 1,
        "image_id": "baked",
        "os": {"type": "Ubuntu", "version": "22.04", "architecture": "x86_64"},
        "apps": [{"name": "Test App"}],
    }), encoding="utf-8")

    store = ImageStore(store_root)
    record = store.register(
        image_id="baked",
        source=disk,
        os_type="Ubuntu",
        os_version="22.04",
        architecture="x86_64",
        snapshot="clean-v1",
        software={"codex": "1.2.3"},
        mode="adopt",
        derived_from="base",
        build={"verified": True, "harness": "codex"},
    )
    assert record.path == disk
    assert record.derived_from == "base"
    assert record.build["harness"] == "codex"
    store.verify_record(record)
    assert not list(store_root.glob(".catalog.json.*.tmp"))


def test_image_store_accepts_verified_artifact_with_pending_inventory(tmp_path):
    store_root = tmp_path / "images"
    store_root.mkdir()
    (store_root / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "images": {}}), encoding="utf-8"
    )
    disk = store_root / "windows" / "base" / "disk.qcow2"
    disk.parent.mkdir(parents=True)
    disk.write_bytes(b"image")
    (disk.parent / "descriptor.json").write_text(
        json.dumps({
            "schema_version": 1,
            "image_id": "base",
            "os": {
                "type": "Windows",
                "version": "10",
                "architecture": "x86_64",
            },
            "inventory_status": "pending",
            "apps": [],
        }),
        encoding="utf-8",
    )

    store = ImageStore(store_root)
    record = store.register(
        image_id="base",
        source=disk,
        os_type="Windows",
        os_version="10",
        architecture="x86_64",
        snapshot="upstream-base",
        software={},
        mode="adopt",
        build={"artifact_verified": True, "boot_verified": False},
    )
    store.verify_record(record)


def test_image_store_adopt_rejects_noncanonical_source(tmp_path):
    store_root = tmp_path / "images"
    store_root.mkdir()
    (store_root / "catalog.json").write_text(
        json.dumps({"schema_version": 1, "images": {}}), encoding="utf-8"
    )
    disk = tmp_path / "outside.qcow2"
    disk.write_bytes(b"image")
    with pytest.raises(ValueError, match="managed path"):
        ImageStore(store_root).register(
            image_id="baked",
            source=disk,
            os_type="Ubuntu",
            os_version="22.04",
            architecture="x86_64",
            snapshot="clean-v1",
            software={},
            mode="adopt",
        )


def test_command_backend_is_not_weavebench_bound(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = load_matrix_config(_config(tmp_path, [{
        "id": "fedora-generic",
        "backend": "command",
        "provider": "custom-hypervisor",
        "os": {"type": "Fedora", "version": "42"},
        "image": {"path": "disk.qcow2"},
        "backend_options": {
            "command": ["/bin/echo", "{profile_id}", "{image_path}", "{output_dir}"]
        },
    }]))
    profile = config.profiles[0]
    spec = create_backend("command").prepare(
        config=config,
        profile=profile,
        output_dir=tmp_path / "output",
        cli=Namespace(),
        base_env={},
    )
    assert spec.env["CUA_ACTION_PROTOCOL"] == "qwen_xml"
    assert spec.env["GUI_TRAJECTORY_ACTION_PROTOCOL"] == "qwen_xml"
    assert spec.metadata["action_protocol"] == "qwen_xml"
    assert spec.command[0] == "/bin/echo"
    assert "fedora-generic" in spec.command
    assert spec.metadata["backend"] == "command"


def test_weavebench_backend_passes_profile_harness(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    image = tmp_path / "disk.qcow2"
    image.write_bytes(b"image")
    config = load_matrix_config(_config(tmp_path, [{
        "id": "codex-baked",
        "backend": "weavebench",
        "provider": "docker",
        "os": {"type": "Ubuntu", "version": "22.04"},
        "image": {"path": str(image)},
        "backend_options": {
            "root": _fake_weavebench_root(tmp_path),
            "tasks_root": str(tasks),
            "model": "openai/gpt-5.5",
            "harness": "codex",
            "image_strategy": "runtime-overlay",
        },
    }]))
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )
    assert spec.env["WEAVEBENCH_HARNESS"] == "codex"
    assert spec.env["WEAVEBENCH_IMAGE_STRATEGY"] == "runtime-overlay"
    assert spec.metadata["harness"] == "codex"
    assert spec.metadata["image_strategy"] == "runtime-overlay"


def test_weavebench_backend_passes_openclaw_chat_endpoint(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {
        "harness": "openclaw",
        "transport": "chat",
        "runtime_location": "guest",
        "model": "/models/moonshotai--Kimi-K3",
        "litellm_base_url": "http://10.0.0.1:30000/v1",
        "litellm_api_key_env": "KIMI_K3_API_KEY",
    })
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={"KIMI_K3_API_KEY": "test-key"},
    )

    endpoint_index = spec.command.index("--litellm_base_url")
    assert spec.command[endpoint_index + 1] == "http://10.0.0.1:30000/v1"
    assert spec.env["WEAVEBENCH_HARNESS"] == "openclaw"
    assert spec.env["WEAVEBENCH_TRANSPORT"] == "chat"
    assert spec.env["GUI_TRAJECTORY_RESULT_DIR"] == str(tmp_path / "output")
    assert spec.env["GUI_TRAJECTORY_REQUESTED_MODE"] == "gui"
    assert spec.env["WEAVEBENCH_LITELLM_KEY"] == "test-key"
    assert spec.env["OPENROUTER_API_KEY"] == "test-key"
    assert spec.env["OPENAI_API_KEY"] == "test-key"
    assert spec.env["OPENAI_BASE_URL"] == "http://10.0.0.1:30000/v1"
    assert spec.env["WEAVEBENCH_LITELLM_BASE_URL"] == (
        "http://10.0.0.1:30000/v1"
    )
    assert spec.metadata["transport"] == "chat"


def test_weavebench_backend_allows_unauthenticated_local_endpoint(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {
        "harness": "openclaw",
        "transport": "chat",
        "runtime_location": "guest",
        "litellm_base_url": "http://10.0.0.1:30000/v1",
        "allow_unauthenticated_model_endpoint": True,
    })
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )

    assert spec.env["WEAVEBENCH_LITELLM_KEY"] == "EMPTY"
    assert "GUI_TRAJECTORY_MISSING_API_KEY" not in spec.env


def test_task_trajectory_pipeline_defaults_to_end_to_end_run(tmp_path):
    pipeline = _load_task_trajectory_pipeline_module()

    args = pipeline.parse_args([
        "--run-dir",
        str(tmp_path / "run"),
        "--generation-config",
        str(tmp_path / "generation.yaml"),
        "--matrix-config",
        str(tmp_path / "matrix.json"),
        "--output-root",
        str(tmp_path / "output"),
    ])

    assert args.stage == "run"
    assert args.endpoint is None
    assert args.model is None
    assert args.taskset == args.output_root / "taskset"
    assert args.collection == args.output_root / "collection"
    assert args.dataset == args.output_root / "dataset"
    assert args.retry_failed_attempts == 1
    assert args.action_protocol == "qwen_xml"


def test_task_trajectory_pipeline_loads_yaml_config(tmp_path):
    pipeline = _load_task_trajectory_pipeline_module()
    config_dir = tmp_path / "config" / "trajectory_collection"
    config_dir.mkdir(parents=True)
    pipeline_config = config_dir / "pipeline.yaml"
    pipeline_config.write_text(
        """\
schema_version: 1
generation:
  config: ../task_generation/example.yaml
  pool: accepted
  format: weavebench
collection:
  concurrency: 8
  max_parallel_profiles: 3
  retry_failed_attempts: 2
api:
  model: test-model
  base_urls:
    - http://model-a.example/v1
    - http://model-b.example/v1
  timeout_seconds: 90
scoring:
  enabled: true
  concurrency: 3
  threshold: 0.7
export:
  copy_images: false
  visual_window: 6
vm_matrix:
  defaults:
    image_store: ../../images
  profiles:
    - id: test-profile
      os:
        type: Ubuntu
      image:
        id: test-image
"""
    )

    args = pipeline.parse_args([
        "--config",
        str(pipeline_config),
        "--run-dir",
        str(tmp_path / "generated"),
        "--output-root",
        str(tmp_path / "output"),
    ])

    assert args.generation_config == (
        tmp_path / "config" / "task_generation" / "example.yaml"
    ).resolve()
    assert args.matrix_config is None
    assert args.embedded_matrix["defaults"]["image_store"] == str(
        (tmp_path / "images").resolve()
    )
    assert args.pool == "accepted"
    assert args.concurrency == 8
    assert args.max_parallel_profiles == 3
    assert args.api_timeout == 90
    assert args.model == "test-model"
    assert args.base_urls == [
        "http://model-a.example/v1",
        "http://model-b.example/v1",
    ]
    assert args.api_key_env is None
    assert args.allow_unauthenticated is True
    assert args.copy_images is False
    assert args.visual_window == 6
    assert args.retry_failed_attempts == 2
    assert args.score_with_model is True
    assert args.scoring_concurrency == 3
    assert args.scoring_threshold == 0.7
    assert args.action_protocol == "qwen_xml"

    matrix_path = pipeline._ensure_matrix_config(args)
    assert matrix_path == args.output_root / ".pipeline" / "vm_matrix.json"
    matrix = json.loads(matrix_path.read_text())
    assert matrix["schema_version"] == 1
    assert matrix["profiles"][0]["id"] == "test-profile"
    backend_options = matrix["defaults"]["backend_options"]
    assert backend_options["model"] == "test-model"
    assert backend_options["action_protocol"] == "qwen_xml"
    assert "litellm_api_key_env" not in backend_options
    assert backend_options["allow_unauthenticated_model_endpoint"] is True
    assert matrix["profiles"][0]["backend_options"]["litellm_base_url"] == (
        "http://model-a.example/v1"
    )


def test_task_trajectory_pipeline_loads_endpoint_pool(tmp_path):
    pipeline = _load_task_trajectory_pipeline_module()
    config_dir = tmp_path / "config" / "trajectory_collection"
    model_dir = tmp_path / "config" / "models"
    config_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    pool_path = model_dir / "kimi.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: kimi-k3\nbase_urls:\n"
        "  - http://model-a.example/v1\n  - http://model-b.example/v1\n",
        encoding="utf-8",
    )
    pipeline_config = config_dir / "pipeline.yaml"
    pipeline_config.write_text(
        """\
schema_version: 1
generation:
  config: ../task_generation/example.yaml
api:
  endpoint_pool: ../models/kimi.yaml
vm_matrix:
  defaults:
    image_store: ../../images
  profiles:
    - id: test-profile
      os:
        type: Ubuntu
      image:
        id: test-image
""",
        encoding="utf-8",
    )

    args = pipeline.parse_args([
        "--config",
        str(pipeline_config),
        "--run-dir",
        str(tmp_path / "generated"),
        "--output-root",
        str(tmp_path / "output"),
    ])
    matrix_path = pipeline._ensure_matrix_config(args)
    matrix = json.loads(matrix_path.read_text())

    assert args.model == "kimi-k3"
    assert args.endpoint_pool == pool_path.resolve()
    assert len(args.base_urls) == 2
    assert matrix["defaults"]["backend_options"]["model_endpoint_pool"] == str(
        pool_path.resolve()
    )


def test_task_trajectory_pipeline_inherits_endpoint_pool_api_key_env(tmp_path):
    pipeline = _load_task_trajectory_pipeline_module()
    config_dir = tmp_path / "config" / "trajectory_collection"
    model_dir = tmp_path / "config" / "models"
    config_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    pool_path = model_dir / "gpt.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: GPT-5.6-sol\nbase_urls:\n"
        "  - https://llm-gateway.example.com/v1\n"
        "api:\n  wire_api: responses\n  api_key_env: LLM_GATEWAY_API_KEY\n",
        encoding="utf-8",
    )
    pipeline_config = config_dir / "pipeline.yaml"
    pipeline_config.write_text(
        """\
schema_version: 1
generation:
  config: ../task_generation/example.yaml
api:
  endpoint_pool: ../models/gpt.yaml
vm_matrix:
  defaults:
    image_store: ../../images
    backend_options:
      harness: codex
  profiles:
    - id: test-profile
      os:
        type: Ubuntu
      image:
        id: test-image
""",
        encoding="utf-8",
    )

    args = pipeline.parse_args([
        "--config",
        str(pipeline_config),
        "--run-dir",
        str(tmp_path / "generated"),
        "--output-root",
        str(tmp_path / "output"),
    ])
    matrix_path = pipeline._ensure_matrix_config(args)
    matrix = json.loads(matrix_path.read_text())

    assert args.api_key_env == "LLM_GATEWAY_API_KEY"
    assert args.allow_unauthenticated is False
    assert matrix["defaults"]["backend_options"]["litellm_api_key_env"] == (
        "LLM_GATEWAY_API_KEY"
    )


def test_task_trajectory_pipeline_allows_collection_model_override(tmp_path):
    pipeline = _load_task_trajectory_pipeline_module()
    config_dir = tmp_path / "config" / "trajectory_collection"
    model_dir = tmp_path / "config" / "models"
    config_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    pool_path = model_dir / "kimi.yaml"
    pool_path.write_text(
        "schema_version: 1\nmodel: kimi-k3-hq\nbase_urls:\n"
        "  - https://llm-gateway-b.example.com/v1\n"
        "api:\n  wire_api: chat_completions\n  api_key_env: LLM_GATEWAY_API_KEY\n",
        encoding="utf-8",
    )
    pipeline_config = config_dir / "pipeline.yaml"
    pipeline_config.write_text(
        """\
schema_version: 1
generation:
  config: ../task_generation/example.yaml
api:
  endpoint_pool: ../models/kimi.yaml
  model: kimi-k3-hq-anthropic
vm_matrix:
  defaults:
    image_store: ../../images
    backend_options:
      harness: claudecode
  profiles:
    - id: test-profile
      os:
        type: Ubuntu
      image:
        id: test-image
""",
        encoding="utf-8",
    )

    args = pipeline.parse_args([
        "--config",
        str(pipeline_config),
        "--run-dir",
        str(tmp_path / "generated"),
        "--output-root",
        str(tmp_path / "output"),
    ])
    matrix_path = pipeline._ensure_matrix_config(args)
    matrix = json.loads(matrix_path.read_text())

    assert args.model == "kimi-k3-hq-anthropic"
    assert matrix["defaults"]["backend_options"]["model"] == (
        "kimi-k3-hq-anthropic"
    )


def test_task_trajectory_pipeline_checks_claudecode_messages_api(monkeypatch):
    pipeline = _load_task_trajectory_pipeline_module()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({
                "content": [{"type": "text", "text": "OK"}],
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(pipeline.urllib.request, "urlopen", fake_urlopen)

    pipeline._check_api_target({
        "endpoint": "http://model.example/v1",
        "model": "/models/kimi-k3",
        "api_key_env": "",
        "protocol": "messages",
    }, 12)

    assert captured["url"] == "http://model.example/v1/messages"
    assert captured["headers"]["X-api-key"] == "EMPTY"
    assert captured["headers"]["Anthropic-version"] == "2023-06-01"
    assert captured["payload"]["model"] == "/models/kimi-k3"
    assert captured["timeout"] == 12


def test_task_trajectory_pipeline_run_executes_all_stages(
    monkeypatch, capsys, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()

    calls = []
    args = SimpleNamespace(
        skip_api_check=False,
        collection=tmp_path / "collection",
        taskset=None,
        retry_failed_attempts=0,
    )
    monkeypatch.setattr(pipeline, "prepare", lambda value: calls.append("prepare"))
    monkeypatch.setattr(pipeline, "check_api", lambda value: calls.append("check-api"))
    monkeypatch.setattr(
        pipeline,
        "check_collection_dependencies",
        lambda value: calls.append("check-dependencies"),
    )
    monkeypatch.setattr(
        pipeline,
        "collect",
        lambda value, dry_run: calls.append(("collect", dry_run)),
    )
    monkeypatch.setattr(pipeline, "score", lambda value: calls.append("score"))
    monkeypatch.setattr(pipeline, "export", lambda value: calls.append("export"))

    pipeline.run_pipeline(args)

    assert calls == [
        "prepare",
        "check-api",
        "check-dependencies",
        ("collect", False),
        "score",
        "export",
    ]
    captured = capsys.readouterr()
    assert "[task-trajectory] [1/5] start stage=prepare" in captured.err
    assert "[task-trajectory] [5/5] done stage=export" in captured.err
    assert "[task-trajectory] complete stages=5" in captured.err


def test_task_trajectory_pipeline_dependency_check_reports_missing_libgl(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    taskset = tmp_path / "taskset"
    taskset.mkdir()
    (taskset / "taskset_manifest.json").write_text(
        json.dumps({
            "tasks": [{"execution_adapter": "legacy_osworld"}],
        }),
        encoding="utf-8",
    )
    evaluator_root = tmp_path / "osworld"
    matrix_config = tmp_path / "matrix.json"
    matrix_config.write_text(
        json.dumps({
            "schema_version": 1,
            "defaults": {
                "backend": "weavebench",
                "backend_options": {
                    "osworld_evaluator_root": str(evaluator_root),
                },
            },
            "profiles": [{"id": "ubuntu", "enabled": True}],
        }),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "ImportError: libGL.so.1: cannot open shared object file: "
                "No such file or directory"
            ),
        )

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    args = SimpleNamespace(
        taskset=taskset,
        matrix_config=matrix_config,
        embedded_matrix=None,
    )

    with pytest.raises(RuntimeError, match="opencv-python-headless") as exc_info:
        pipeline.check_collection_dependencies(args)

    assert "profile 'ubuntu'" in str(exc_info.value)
    assert captured["command"][:2] == [sys.executable, "-c"]
    assert captured["env"]["GUI_TRAJECTORY_OSWORLD_EVALUATOR_ROOT"] == str(
        evaluator_root
    )


def test_task_trajectory_pipeline_dependency_check_skips_nonlegacy_tasksets(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    taskset = tmp_path / "taskset"
    taskset.mkdir()
    (taskset / "taskset_manifest.json").write_text(
        json.dumps({"tasks": [{"execution_adapter": "task_class"}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        pipeline.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("dependency probe should not run"),
    )

    pipeline.check_collection_dependencies(SimpleNamespace(taskset=taskset))


def test_task_trajectory_pipeline_detects_retryable_collection_outputs(tmp_path):
    pipeline = _load_task_trajectory_pipeline_module()
    result_root = tmp_path / "collection" / "profile-a" / "hybrid" / "model" / "os"

    def write_result(
        task_id,
        *,
        score=1.0,
        error=None,
        status=None,
        trajectory=True,
        agent_status="ok",
    ):
        task_dir = result_root / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text(
            json.dumps({"task_id": task_id}),
            encoding="utf-8",
        )
        if score is not None:
            (task_dir / "score.json").write_text(
                json.dumps({
                    "score": score,
                    "error": error,
                    "agent_done": True,
                    "agent_status": agent_status,
                }),
                encoding="utf-8",
            )
        if trajectory:
            (task_dir / "trajectory.jsonl").write_text(
                json.dumps({"step": 1, "actions": [{"type": "shell.exec"}]})
                + "\n",
                encoding="utf-8",
            )
            canonical = task_dir / "canonical"
            canonical.mkdir()
            (canonical / "task.json").write_text(
                json.dumps({"id": task_id}),
                encoding="utf-8",
            )
        if status is not None:
            (task_dir / "execution_status.json").write_text(
                json.dumps(status),
                encoding="utf-8",
            )
        return task_dir

    write_result("successful", score=1.0)
    successful_raw_gui = write_result(
        "successful-raw-gui",
        score=1.0,
        trajectory=False,
    )
    (successful_raw_gui / "computer_events.jsonl").write_text(
        json.dumps({"step": 1, "action": {"type": "click"}}) + "\n",
        encoding="utf-8",
    )
    write_result("zero-score", score=0.0, error="judge_failed: unavailable")
    write_result("errored", score=1.0, error="upload failed")
    vm_upload_with_trajectory = write_result(
        "vm-upload-failed",
        score=0.0,
        error=(
            "500 Server Error: Internal Server Error for url: "
            "http://10.0.0.1:21004/setup/upload"
        ),
        trajectory=False,
    )
    (vm_upload_with_trajectory / "computer_events.jsonl").write_text(
        json.dumps({"step": 1, "action": {"type": "click"}}) + "\n",
        encoding="utf-8",
    )
    write_result(
        "vm-upload-no-trajectory",
        score=0.0,
        error=(
            "500 Server Error: Internal Server Error for url: "
            "http://10.0.0.1:21004/setup/upload"
        ),
        trajectory=False,
    )
    write_result(
        "empty-run",
        score=0.0,
        error="agent_empty_run: clean exit but no model output",
        trajectory=False,
        agent_status="empty_run",
    )
    write_result(
        "timed-out",
        score=None,
        status={"agent_done": False, "timed_out": True, "exit_code": None},
        trajectory=False,
    )
    write_result(
        "transport-failed",
        score=None,
        status={
            "agent_done": False,
            "timed_out": False,
            "exit_code": None,
            "transport_failure": True,
        },
        trajectory=False,
    )
    invalid = write_result("invalid-score", score=1.0)
    (invalid / "score.json").write_text("{", encoding="utf-8")

    failures = pipeline._failed_collection_tasks(tmp_path / "collection")

    assert {failure["task_id"] for failure in failures} == {
        "empty-run",
        "timed-out",
        "transport-failed",
        "vm-upload-failed",
        "vm-upload-no-trajectory",
    }
    reasons = {failure["task_id"]: failure["reason"] for failure in failures}
    assert "agent_empty_run" in reasons["empty-run"]
    assert "agent_timed_out" in reasons["timed-out"]
    assert "missing_or_empty_trajectory" in reasons["timed-out"]
    assert "vm_transport_unavailable" in reasons["transport-failed"]
    assert "missing_or_empty_trajectory" in reasons["transport-failed"]
    assert reasons["vm-upload-failed"] == "vm_transport_unavailable"
    assert reasons["vm-upload-no-trajectory"] == (
        "missing_or_empty_trajectory | vm_transport_unavailable"
    )


def test_task_trajectory_pipeline_archives_and_retries_failed_tasks(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    collection = tmp_path / "collection"
    result_root = collection / "profile-a" / "hybrid" / "model" / "os"
    for task_id in ("successful", "retry-b", "retry-a", "vm-retry"):
        task_dir = result_root / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text(
            json.dumps({"task_id": task_id}),
            encoding="utf-8",
        )
        (task_dir / "score.json").write_text(
            json.dumps({
                "score": 1.0 if task_id == "successful" else 0.0,
                "agent_done": True,
                "agent_status": (
                    "ok" if task_id in {"successful", "vm-retry"} else "empty_run"
                ),
                "error": (
                    None
                    if task_id == "successful"
                    else (
                        "500 Server Error for url: http://10.0.0.2:21004/setup/upload"
                        if task_id == "vm-retry"
                        else "agent_empty_run: clean exit but no model output"
                    )
                ),
            }),
            encoding="utf-8",
        )
        if task_id in {"successful", "vm-retry"}:
            (task_dir / "trajectory.jsonl").write_text(
                '{"step":1,"actions":[{"type":"shell.exec"}]}\n',
                encoding="utf-8",
            )

    calls = []
    monkeypatch.setattr(
        pipeline,
        "collect",
        lambda args, **kwargs: calls.append(kwargs),
    )
    args = SimpleNamespace(collection=collection, retry_failed_attempts=1)

    pipeline._retry_failed_collection(args)

    assert calls == [{
        "dry_run": False,
        "task_filter": ["retry-a", "retry-b", "vm-retry"],
        "profile_filter": ["profile-a"],
        "force_rerun": True,
        "allow_config_change": True,
        "cap_profile_concurrency": True,
    }]
    assert (result_root / "successful" / "score.json").is_file()
    assert not (result_root / "retry-a").exists()
    archive_root = collection / ".collection_state" / "retry_attempts" / "round-01"
    assert (
        archive_root
        / "profile-a"
        / "hybrid"
        / "model"
        / "os"
        / "retry-a"
        / "score.json"
    ).is_file()
    manifest = json.loads((archive_root / "retry_manifest.json").read_text())
    assert {task["task_id"] for task in manifest["tasks"]} == {
        "retry-a",
        "retry-b",
        "vm-retry",
    }
    archived_vm = next(
        task for task in manifest["tasks"] if task["task_id"] == "vm-retry"
    )
    assert Path(archived_vm["archive_dir"], "trajectory.jsonl").is_file()


def test_task_trajectory_pipeline_retries_after_collector_nonzero(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    collection = tmp_path / "collection"
    result_root = collection / "profile-a" / "hybrid" / "model" / "os"
    failed_dir = result_root / "retry-me"
    successful_dir = result_root / "keep-me"
    failed_dir.mkdir(parents=True)
    successful_dir.mkdir(parents=True)
    (failed_dir / "task.json").write_text(
        json.dumps({"task_id": "retry-me"}),
        encoding="utf-8",
    )
    (failed_dir / "score.json").write_text(
        json.dumps({
            "score": 0.0,
            "agent_done": True,
            "agent_status": "empty_run",
            "error": "agent_empty_run: no model output",
        }),
        encoding="utf-8",
    )
    (successful_dir / "task.json").write_text(
        json.dumps({"task_id": "keep-me"}),
        encoding="utf-8",
    )
    (successful_dir / "trajectory.jsonl").write_text(
        '{"step":1,"actions":[{"type":"shell.exec"}]}\n',
        encoding="utf-8",
    )
    successful_marker = successful_dir / "preserve.txt"
    successful_marker.write_text("unchanged", encoding="utf-8")

    calls = []

    def fake_collect(args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, ["collector"])
        failed_dir.mkdir(parents=True)
        (failed_dir / "task.json").write_text(
            json.dumps({"task_id": "retry-me"}),
            encoding="utf-8",
        )
        (failed_dir / "score.json").write_text(
            json.dumps({"score": 1.0, "agent_done": True, "agent_status": "ok"}),
            encoding="utf-8",
        )
        (failed_dir / "trajectory.jsonl").write_text(
            '{"step":1,"actions":[{"type":"shell.exec"}]}\n',
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(1, ["collector"])

    monkeypatch.setattr(pipeline, "collect", fake_collect)
    monkeypatch.setattr(pipeline, "check_collection_dependencies", lambda args: None)
    args = SimpleNamespace(
        collection=collection,
        taskset=None,
        retry_failed_attempts=1,
    )

    pipeline._collect_with_retries(args)

    assert calls == [
        {"dry_run": False},
        {
            "dry_run": False,
            "task_filter": ["retry-me"],
            "profile_filter": ["profile-a"],
            "force_rerun": True,
            "allow_config_change": True,
            "cap_profile_concurrency": True,
        },
    ]
    assert successful_marker.read_text(encoding="utf-8") == "unchanged"
    assert (failed_dir / "trajectory.jsonl").is_file()


def test_task_trajectory_pipeline_reports_missing_retry_output(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    collection = tmp_path / "collection"
    failed_dir = collection / "profile-a" / "hybrid" / "model" / "os" / "retry-me"
    failed_dir.mkdir(parents=True)
    (failed_dir / "task.json").write_text(
        json.dumps({"task_id": "retry-me"}),
        encoding="utf-8",
    )
    (failed_dir / "score.json").write_text(
        json.dumps({
            "score": 0.0,
            "agent_done": True,
            "agent_status": "empty_run",
            "error": "agent_empty_run: no model output",
        }),
        encoding="utf-8",
    )

    def failed_collect(args, **kwargs):
        raise subprocess.CalledProcessError(1, ["collector"])

    monkeypatch.setattr(pipeline, "collect", failed_collect)
    monkeypatch.setattr(pipeline, "check_collection_dependencies", lambda args: None)
    args = SimpleNamespace(
        collection=collection,
        taskset=None,
        retry_failed_attempts=1,
    )

    with pytest.raises(RuntimeError, match="missing_retry_output"):
        pipeline._collect_with_retries(args)

    resumed_failures = pipeline._failed_collection_tasks(collection)
    assert resumed_failures == [{
        "profile_id": "profile-a",
        "task_id": "retry-me",
        "output_dir": str(failed_dir),
        "reason": "missing_retry_output",
    }]


def test_task_trajectory_pipeline_continues_retry_round_numbers(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    collection = tmp_path / "collection"
    failed_dir = collection / "profile-a" / "hybrid" / "model" / "os" / "retry-me"
    failed_dir.mkdir(parents=True)
    (failed_dir / "task.json").write_text(
        json.dumps({"task_id": "retry-me"}),
        encoding="utf-8",
    )
    (failed_dir / "score.json").write_text(
        json.dumps({
            "score": 0.0,
            "agent_done": True,
            "agent_status": "empty_run",
            "error": "agent_empty_run: no model output",
        }),
        encoding="utf-8",
    )
    retry_root = collection / ".collection_state" / "retry_attempts"
    (retry_root / "round-02").mkdir(parents=True)

    def successful_collect(args, **kwargs):
        failed_dir.mkdir(parents=True)
        (failed_dir / "task.json").write_text(
            json.dumps({"task_id": "retry-me"}),
            encoding="utf-8",
        )
        (failed_dir / "trajectory.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "collect", successful_collect)
    args = SimpleNamespace(
        collection=collection,
        taskset=None,
        retry_failed_attempts=1,
    )

    assert pipeline._retry_failed_collection(args) == []
    assert (retry_root / "round-03" / "retry_manifest.json").is_file()


def test_task_trajectory_pipeline_export_uses_dataset_cli_arguments(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    commands = []
    monkeypatch.setattr(pipeline, "_run", commands.append)
    args = SimpleNamespace(
        collection=tmp_path / "collection",
        dataset=tmp_path / "dataset",
        bundle_format="canonical",
        visual_window=10,
        max_history_actions=3,
        copy_images=True,
        require_pre_screenshot=True,
    )

    pipeline.export(args)

    command = commands[0]
    assert command.count(str(args.collection)) == 1
    assert command[command.index("--collection-root") + 1] == str(args.collection)
    assert command[command.index("--output-dir") + 1] == str(args.dataset)
    assert command[command.index("--action-protocol") + 1] == "qwen_xml"


def test_task_trajectory_pipeline_collect_pins_qwen_xml_protocol(
    monkeypatch, tmp_path
):
    pipeline = _load_task_trajectory_pipeline_module()
    taskset = tmp_path / "taskset"
    taskset.mkdir()
    (taskset / "taskset_manifest.json").write_text("{}", encoding="utf-8")
    matrix = tmp_path / "matrix.json"
    matrix.write_text('{"schema_version":1,"profiles":[]}', encoding="utf-8")
    commands = []
    monkeypatch.setattr(pipeline, "_run", commands.append)
    args = SimpleNamespace(
        taskset=taskset,
        matrix_config=matrix,
        embedded_matrix=None,
        collection=tmp_path / "collection",
        concurrency=None,
        max_parallel_profiles=None,
        progress_interval_seconds=30,
        model=None,
        endpoint=None,
        action_protocol="qwen_xml",
        force_rerun=False,
    )

    pipeline.collect(
        args,
        dry_run=True,
        profile_filter=["profile-a"],
        allow_config_change=True,
        cap_profile_concurrency=True,
    )

    command = commands[0]
    protocol_index = command.index("action_protocol=qwen_xml")
    assert command[protocol_index - 1] == "--backend-option"
    assert command[command.index("--profiles") + 1] == "profile-a"
    assert "--allow-config-change" in command
    assert "--cap-profile-concurrency" in command


def test_collector_streams_backend_output_to_terminal_and_log(tmp_path, capsys):
    from cuamaster.collection import collector

    log_path = tmp_path / "collector.log"
    returncode = collector._run_streaming_process(
        [sys.executable, "-c", "print('task 1/2'); print('task 2/2')"],
        cwd=tmp_path,
        env={},
        log_path=log_path,
        prefix="ubuntu-profile",
    )

    assert returncode == 0
    assert log_path.read_text() == "task 1/2\ntask 2/2\n"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "[ubuntu-profile] task 1/2" in captured.err
    assert "[ubuntu-profile] task 2/2" in captured.err


def test_collector_reports_backend_task_progress(tmp_path, capsys):
    from cuamaster.collection import collector

    log_path = tmp_path / "collector.log"
    script = (
        "print('modes=[gui] total_runs=2'); "
        "print('▶ START  [env-1][gui] task-a'); "
        "print('✓ DONE   [env-1][gui] task-a score=1.000'); "
        "print('⊘ SKIP   [env-2][gui] task-b (already done)')"
    )

    returncode = collector._run_streaming_process(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={},
        log_path=log_path,
        prefix="ubuntu-profile",
        progress_interval_seconds=0,
    )

    assert returncode == 0
    captured = capsys.readouterr()
    assert "profile=ubuntu-profile progress=1/2" in captured.err
    assert "profile=ubuntu-profile progress=2/2" in captured.err


def test_collector_reports_heartbeat_while_backend_is_silent(tmp_path, capsys):
    from cuamaster.collection import collector

    log_path = tmp_path / "collector.log"
    returncode = collector._run_streaming_process(
        [sys.executable, "-c", "import time; time.sleep(0.08)"],
        cwd=tmp_path,
        env={},
        log_path=log_path,
        prefix="ubuntu-profile",
        progress_interval_seconds=0.02,
    )

    assert returncode == 0
    captured = capsys.readouterr()
    assert "profile=ubuntu-profile progress=0/?" in captured.err
    assert "silent_for=" in captured.err


def test_task_trajectory_pipeline_accepts_arbitrary_generated_tasks(tmp_path):
    pipeline = _load_task_trajectory_pipeline_module()
    run_dir = tmp_path / "generated"
    for task_id, evaluator_type, environment_id in (
        ("task-a", "custom_evaluator", "ubuntu-image"),
        ("task-b", "another_evaluator", "windows-image"),
    ):
        task_dir = run_dir / "promoted" / task_id
        task_dir.mkdir(parents=True)
        (task_dir / "generated_task.json").write_text(json.dumps({
            "task": {
                "id": task_id,
                "evaluator": {"type": evaluator_type},
                "metadata": {"environment_id": environment_id},
            },
            "recipe": {},
        }))

    stats = pipeline._validate_promoted(run_dir, "promoted")

    assert stats == {
        "source_tasks": 2,
        "environment_counts": {"ubuntu-image": 1, "windows-image": 1},
    }


def _load_task_trajectory_pipeline_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "task_trajectory_pipeline.py"
    spec = importlib.util.spec_from_file_location("task_trajectory_pipeline_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weavebench_mode_config(tmp_path, backend_options):
    tasks = tmp_path / "tasks"
    tasks.mkdir(exist_ok=True)
    image = tmp_path / "disk.qcow2"
    image.write_bytes(b"image")
    return load_matrix_config(_config(tmp_path, [{
        "id": "mode-test",
        "backend": "weavebench",
        "provider": "docker",
        "os": {"type": "Ubuntu", "version": "22.04"},
        "image": {"path": str(image)},
        "backend_options": {
            "root": _fake_weavebench_root(tmp_path),
            "tasks_root": str(tasks),
            "model": "openai/gpt-5.5",
            **backend_options,
        },
    }]))


def test_weavebench_backend_defaults_to_gui_mode(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {})
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )
    mode_index = spec.command.index("--mode")
    assert spec.command[mode_index + 1] == "gui"
    assert spec.metadata["mode"] == "gui"
    assert spec.metadata["interaction_harness"]["name"] == "gui"
    assert spec.env["CUA_INTERACTION_HARNESS"] == "gui"
    assert spec.metadata["action_protocol"] == "qwen_xml"
    assert spec.env["CUA_ACTION_PROTOCOL"] == "qwen_xml"
    assert spec.env["GUI_TRAJECTORY_ACTION_PROTOCOL"] == "qwen_xml"


def test_weavebench_backend_passes_interaction_harness(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(
        tmp_path, {"interaction_harness": "hybrid"}
    )
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )
    mode_index = spec.command.index("--mode")
    assert spec.command[mode_index + 1] == "hybrid"
    assert spec.metadata["mode"] == "hybrid"
    assert spec.metadata["interaction_harness"]["capabilities"] == ["cli", "gui"]
    assert spec.env["CUA_INTERACTION_HARNESS"] == "hybrid"


def test_weavebench_backend_keeps_legacy_mode_compatibility(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {"mode": "cli"})
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )

    assert spec.metadata["interaction_harness"]["name"] == "cli"
    assert spec.env["CUA_INTERACTION_HARNESS"] == "cli"


def test_weavebench_backend_passes_timeout_controls(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {
        "task_timeout_seconds": 3600,
        "monitor_max_seconds": 172800,
    })
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )

    monitor_index = spec.command.index("--monitor_max_seconds")
    assert spec.command[monitor_index + 1] == "172800"
    assert spec.env["GUI_TRAJECTORY_TASK_TIMEOUT_SECONDS"] == "3600"


def test_weavebench_backend_passes_legacy_proxy_from_named_environment(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {
        "legacy_osworld_enable_proxy": True,
        "legacy_osworld_proxy_url_env": "OSWORLD_PROXY_URL",
    })
    proxy_url = "http://proxy-user:proxy-pass@10.0.0.8:3128"
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={"OSWORLD_PROXY_URL": proxy_url},
    )

    assert spec.env["GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_ENABLED"] == "true"
    assert spec.env["GUI_TRAJECTORY_LEGACY_OSWORLD_PROXY_URL"] == proxy_url
    assert spec.metadata["legacy_osworld_proxy_enabled"] is True
    assert spec.metadata["legacy_osworld_proxy_url_env"] == "OSWORLD_PROXY_URL"
    assert proxy_url not in json.dumps(spec.metadata)


def test_weavebench_backend_requires_configured_legacy_proxy_environment(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {
        "legacy_osworld_enable_proxy": True,
        "legacy_osworld_proxy_url_env": "OSWORLD_PROXY_URL",
    })
    with pytest.raises(ValueError, match="OSWORLD_PROXY_URL"):
        create_backend("weavebench").prepare(
            config=config,
            profile=config.profiles[0],
            output_dir=tmp_path / "output",
            cli=Namespace(backend_options={}),
            base_env={},
        )


def test_weavebench_backend_rejects_invalid_mode(tmp_path):
    from argparse import Namespace

    import pytest

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(tmp_path, {"mode": "desktop"})
    with pytest.raises(ValueError, match="unknown interaction harness"):
        create_backend("weavebench").prepare(
            config=config,
            profile=config.profiles[0],
            output_dir=tmp_path / "output",
            cli=Namespace(backend_options={}),
            base_env={},
        )


def test_weavebench_backend_rejects_conflicting_interaction_harness(tmp_path):
    from argparse import Namespace

    import pytest

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(
        tmp_path,
        {"interaction_harness": "hybrid", "mode": "gui"},
    )
    with pytest.raises(ValueError, match="conflicts with legacy mode"):
        create_backend("weavebench").prepare(
            config=config,
            profile=config.profiles[0],
            output_dir=tmp_path / "output",
            cli=Namespace(backend_options={}),
            base_env={},
        )


def test_weavebench_backend_enables_guest_shell_for_host_hybrid(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    config = _weavebench_mode_config(
        tmp_path, {"mode": "hybrid", "runtime_location": "host"}
    )
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )

    assert spec.env["GUI_TRAJECTORY_ENABLE_GUEST_SHELL"] == "true"
    assert spec.metadata["runtime_location"] == "host"
    assert spec.metadata["shell_runtime_location"] == "guest"


def test_weavebench_backend_loads_model_and_endpoint_from_pool(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    pool = tmp_path / "model_pool.yaml"
    pool.write_text(
        "\n".join([
            "schema_version: 1",
            "model: /models/kimi-k3",
            "base_urls:",
            "  - http://model-a:30000/v1",
        ])
        + "\n",
        encoding="utf-8",
    )
    config = _weavebench_mode_config(tmp_path, {
        "mode": "hybrid",
        "runtime_location": "host",
        "model": None,
        "model_endpoint_pool": str(pool),
        "osworld_evaluator_root": "legacy-eval",
        "allow_unauthenticated_model_endpoint": True,
    })
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )

    model_index = spec.command.index("--model")
    endpoint_index = spec.command.index("--litellm_base_url")
    assert spec.command[model_index + 1] == "/models/kimi-k3"
    assert spec.command[endpoint_index + 1] == "http://model-a:30000/v1"
    assert spec.env["GUI_TRAJECTORY_ENDPOINT_POOL"] == str(pool.resolve())
    assert spec.env["GUI_TRAJECTORY_OSWORLD_EVALUATOR_ROOT"] == str(
        (tmp_path / "legacy-eval").resolve()
    )
    assert spec.env["WEAVEBENCH_LITELLM_KEY"] == "EMPTY"
    assert spec.env["OPENROUTER_API_KEY"] == "EMPTY"


def test_weavebench_backend_uses_host_runtime_for_windows(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    image = tmp_path / "windows.qcow2"
    image.write_bytes(b"image")
    config = load_matrix_config(_config(tmp_path, [{
        "id": "windows-host",
        "backend": "weavebench",
        "provider": "docker",
        "os": {"type": "Windows", "version": "11"},
        "image": {"path": str(image)},
        "backend_options": {
            "root": _fake_weavebench_root(tmp_path),
            "tasks_root": str(tasks),
            "model": "openai/gpt-5.5",
            "harness": "claudecode",
            "image_strategy": "runtime-overlay",
        },
    }]))
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )
    assert spec.env["GUI_TRAJECTORY_RUNTIME_LOCATION"] == "host"
    assert spec.env["GUI_TRAJECTORY_REQUIRE_EXPLICIT_PLATFORM"] == "true"
    assert spec.metadata["runtime_location"] == "host"
    assert spec.metadata["controller_transport"] == "osworld-http"


def test_weavebench_backend_keeps_explicit_windows_guest_runtime(tmp_path):
    """Explicit guest runtime remains available for compatible custom images."""
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    image = tmp_path / "windows.qcow2"
    image.write_bytes(b"image")
    config = load_matrix_config(_config(tmp_path, [{
        "id": "windows-guest",
        "backend": "weavebench",
        "provider": "docker",
        "os": {"type": "Windows", "version": "11"},
        "image": {"path": str(image)},
        "backend_options": {
            "root": _fake_weavebench_root(tmp_path),
            "tasks_root": str(tasks),
            "model": "openai/gpt-5.5",
            "runtime_location": "guest",
            "mode": "hybrid",
        },
    }]))
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={},
    )
    assert spec.env["GUI_TRAJECTORY_RUNTIME_LOCATION"] == "guest"
    assert spec.metadata["runtime_location"] == "guest"


def test_weavebench_backend_rejects_legacy_manager_provider(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    config = load_matrix_config(_config(tmp_path, [{
        "id": "macos-manager",
        "backend": "weavebench",
        "provider": "manager",
        "os": {"type": "Darwin", "version": "14.6", "architecture": "arm64"},
        "image": {"id": "macos14-pages"},
        "backend_options": {
            "root": _fake_weavebench_root(tmp_path),
            "tasks_root": str(tasks),
            "model": "openai/gpt-5.5",
            "harness": "claudecode",
            "runtime_location": "host",
            "image_strategy": "baked",
        },
    }]))
    with pytest.raises(ValueError, match="provider=manager is disabled"):
        create_backend("weavebench").prepare(
            config=config,
            profile=config.profiles[0],
            output_dir=tmp_path / "output",
            cli=Namespace(backend_options={}),
            base_env={},
        )


def test_weavebench_backend_maps_remote_ssh_without_exposing_password(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    image = tmp_path / "windows.qcow2"
    image.write_bytes(b"image")
    config = load_matrix_config(_config(tmp_path, [{
        "id": "windows-remote",
        "backend": "weavebench",
        "provider": "remote_ssh",
        "os": {"type": "Windows", "version": "11"},
        "image": {"path": str(image)},
        "backend_options": {
            "root": _fake_weavebench_root(tmp_path),
            "tasks_root": str(tasks),
            "model": "openai/gpt-5.5",
            "ssh_host": "example.invalid",
            "ssh_port": 13022,
            "ssh_user": "root",
            "ssh_password_env": "TEST_REMOTE_VM_PASSWORD",
            "client_password": "osworld-public-evaluation",
        },
    }]))
    spec = create_backend("weavebench").prepare(
        config=config,
        profile=config.profiles[0],
        output_dir=tmp_path / "output",
        cli=Namespace(backend_options={}),
        base_env={"TEST_REMOTE_VM_PASSWORD": "test-only-secret"},
    )

    assert spec.command[spec.command.index("--provider_name") + 1] == "docker"
    assert spec.env["GUI_TRAJECTORY_REMOTE_SSH_PROVIDER"] == "1"
    assert spec.env["GUI_TRAJECTORY_SSH_HOST"] == "example.invalid"
    assert spec.env["GUI_TRAJECTORY_SSH_PORT"] == "13022"
    assert spec.env["GUI_TRAJECTORY_SSH_PASSWORD"] == "test-only-secret"
    assert spec.env["GUI_TRAJECTORY_SSH_REMOTE_IMAGE_PATH"] == str(image)
    assert spec.metadata["vm_transport"] == "ssh"
    assert spec.command[spec.command.index("--client_password") + 1] == (
        "osworld-public-evaluation"
    )
    assert "test-only-secret" not in " ".join(spec.command)
    assert "test-only-secret" not in json.dumps(spec.metadata)


def test_weavebench_backend_rejects_plaintext_ssh_password(tmp_path):
    from argparse import Namespace

    from cuamaster.collection.backends import create_backend

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    config = load_matrix_config(_config(tmp_path, [{
        "id": "remote-with-password",
        "backend": "weavebench",
        "provider": "remote_ssh",
        "os": {"type": "Windows"},
        "image": {},
        "backend_options": {
            "root": _fake_weavebench_root(tmp_path),
            "tasks_root": str(tasks),
            "model": "openai/gpt-5.5",
            "ssh_host": "example.invalid",
            "ssh_password": "must-not-be-stored",
            "remote_image_path": "/srv/vms/windows.qcow2",
        },
    }]))

    with pytest.raises(ValueError, match="do not store ssh_password in config"):
        create_backend("weavebench").prepare(
            config=config,
            profile=config.profiles[0],
            output_dir=tmp_path / "output",
            cli=Namespace(backend_options={}),
            base_env={},
        )


def test_runtime_harness_matrix_reuses_one_base_image():
    matrix_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "collection"
        / "vm_harness_matrix.json"
    )
    config = load_matrix_config(matrix_path)

    assert {profile.image_id for profile in config.profiles} == {
        "osworld-ubuntu-22.04-20260701-controller-v2"
    }
    assert {profile.provider for profile in config.profiles} == {"sandbox"}
    assert {
        profile.backend_options["harness"] for profile in config.profiles
    } == {"openclaw", "codex", "claudecode", "hermes"}
    assert config.defaults["backend_options"]["image_strategy"] == "runtime-overlay"
