from types import SimpleNamespace

import numpy as np
import pytest

from tools.stellarium import extract_star as collector


def test_capture_cli_default_url_and_camera_hold():
    args = collector.build_parser().parse_args(["--output-dir", "new-session", "--hold-for-camera"])
    assert args.api_url == "http://localhost:8090/api"
    assert args.hold_for_camera


def test_capture_output_refuses_to_overwrite(tmp_path):
    image = tmp_path / "stellarium_view.png"
    image.write_bytes(b"original fixture")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        collector.prepare_output(tmp_path)
    assert image.read_bytes() == b"original fixture"


def test_capture_restores_state_when_prepare_scene_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.sys, "platform", "win32")
    monkeypatch.setattr(collector, "load_capture_dependencies", lambda: None)
    monkeypatch.setattr(collector, "api_request", lambda *a, **k: None)
    restored = []

    def failed_setup(saved):
        saved["time"] = {"timerate": 1}
        raise RuntimeError("unsupported projection property")

    monkeypatch.setattr(collector, "freeze_and_clean_stellarium", failed_setup)
    monkeypatch.setattr(collector, "restore_stellarium_gui", lambda state: restored.append(state.copy()))
    with pytest.raises(RuntimeError, match="unsupported projection"):
        collector.main(["--output-dir", str(tmp_path / "session")])
    assert restored == [{"time": {"timerate": 1}}]


def test_api_uses_basic_auth_with_empty_username(monkeypatch):
    calls = []
    response = SimpleNamespace(text="ok", raise_for_status=lambda: None)

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(collector, "requests", SimpleNamespace(get=request, post=request), raising=False)
    monkeypatch.setattr(collector, "API_AUTH", ("", "dummy-test-password"))
    collector.api_request("main/time", {"timerate": 0})
    assert calls[0][0].endswith("/api/main/time")
    assert calls[0][1]["auth"] == ("", "dummy-test-password")
    assert calls[0][1]["data"] == {"timerate": 0}


def test_collector_tangent_plane_matches_vertical_fov():
    result = collector.compute_unrotated_offsets([
        {"ra": 0, "dec": 0}, {"ra": 0, "dec": 5},
    ], 0, 0, 10, 1000)
    assert np.allclose(result[0][:2], [0, 0])
    assert np.allclose(result[1][:2], [0, -500])


def test_linux_capture_rejects_wayland(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("DISPLAY", ":0")
    with pytest.raises(RuntimeError, match="X11 desktop"):
        collector.linux_capture_region()


def test_linux_capture_checks_process_and_client_geometry(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    calls = []

    def xdo(command, **kwargs):
        calls.append(command)
        args = command[1:]
        return {
            ("search", "--onlyvisible", "--class", "^stellarium$"): "42\n",
            ("getwindowpid", "42"): "123\n",
            ("windowactivate", "--sync", "42"): "",
            ("getactivewindow",): "42\n",
            ("getwindowgeometry", "--shell", "42"): "WINDOW=42\nX=-100\nY=30\nWIDTH=1920\nHEIGHT=1080\nSCREEN=0\n",
        }[tuple(args)]

    monkeypatch.setattr(collector.subprocess, "check_output", xdo)
    monkeypatch.setattr(collector.Path, "resolve", lambda *a, **k: collector.Path("/usr/bin/stellarium"))
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    assert collector.linux_capture_region() == {"left": -100, "top": 30, "width": 1920, "height": 1080}
    assert ["xdotool", "windowactivate", "--sync", "42"] in calls


def test_linux_capture_refuses_wrong_foreground(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    responses = iter(["42", "123", "", "99"])
    monkeypatch.setattr(collector.subprocess, "check_output", lambda *a, **k: next(responses))
    monkeypatch.setattr(collector.Path, "resolve", lambda *a, **k: collector.Path("/usr/bin/stellarium"))
    monkeypatch.setattr(collector.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="foreground"):
        collector.linux_capture_region()


def test_linux_capture_refuses_unverified_process(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    responses = iter(["42", "123"])
    monkeypatch.setattr(collector.subprocess, "check_output", lambda *a, **k: next(responses))
    monkeypatch.setattr(collector.Path, "resolve", lambda *a, **k: collector.Path("/usr/bin/other-app"))
    with pytest.raises(RuntimeError, match="exactly one"):
        collector.linux_capture_region()


def test_linux_capture_missing_xdotool_is_actionable(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")

    def missing(*a, **k):
        raise FileNotFoundError("xdotool")

    monkeypatch.setattr(collector.subprocess, "check_output", missing)
    with pytest.raises(RuntimeError, match="sudo apt install xdotool"):
        collector.linux_capture_region()
