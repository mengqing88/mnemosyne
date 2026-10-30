import json
import shutil
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

from mnemosyne import cli, diagnose


def _check(payload, name):
    entries = payload.get("checks") or payload.get("entries", [])
    return next(entry for entry in entries if entry["check"] == name)


def test_runtime_checks_run_in_the_selected_venv(tmp_path, monkeypatch):
    environment = tmp_path / "provider-env"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    site_packages = next(environment.rglob("site-packages"))
    package = site_packages / "mnemosyne"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "mnemosyne" / "runtime_diagnostics.py",
        package / "runtime_diagnostics.py",
    )
    (site_packages / "fastembed.py").write_text("__version__ = 'provider-only'\n", encoding="utf-8")

    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    payload = diagnose.run_diagnostics(python_executable=str(python))

    assert payload["runtime_scope"] == "target_interpreter"
    assert payload["runtime_executable"] == python.name
    assert _check(payload, "fastembed")["status"] == "OK"
    assert _check(payload, "fastembed")["detail"] == "version=provider-only"
    assert _check(payload, "fastembed")["scope"] == "target_interpreter"
    assert str(environment) not in json.dumps(payload)


def test_default_runtime_checks_are_labeled_as_cli_interpreter(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

    payload = diagnose.run_diagnostics()

    assert payload["runtime_scope"] == "cli_interpreter"
    assert payload["runtime_executable"] == Path(sys.executable).name
    assert _check(payload, "fastembed")["scope"] == "cli_interpreter"


def test_unavailable_target_does_not_fall_back_to_cli_interpreter(tmp_path):
    missing_python = tmp_path / "missing-python"

    payload = diagnose.collect_runtime_diagnostics(str(missing_python))

    assert payload["unavailable"] is True
    assert payload["scope"] == "target_interpreter"
    assert [entry["check"] for entry in payload["checks"]] == ["target_interpreter"]
    assert payload["checks"][0]["status"] == "ERROR"
    assert str(tmp_path) not in json.dumps(payload)


def test_target_payload_with_invalid_check_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "mnemosyne.runtime_diagnostics.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='MNEMOSYNE_DIAGNOSTICS={"checks":[null]}\n',
        ),
    )

    result = diagnose.collect_runtime_diagnostics("./python")

    assert result["unavailable"] is True
    assert result["checks"][0]["check"] == "target_interpreter"


def test_target_runtime_replaces_invalid_child_output_bytes(monkeypatch):
    captured = {}

    def run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=1, stdout="", stderr="\ufffd")

    monkeypatch.setattr("mnemosyne.runtime_diagnostics.subprocess.run", run)

    result = diagnose.collect_runtime_diagnostics("./python")

    assert captured["errors"] == "replace"
    assert result["unavailable"] is True


def test_unavailable_target_is_unknown_and_never_auto_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    calls = []
    monkeypatch.setattr(diagnose.subprocess, "run", lambda *args, **kwargs: calls.append(args))
    target_python = str(tmp_path / "missing-python")

    result = diagnose.run_diagnostics(python_executable=target_python)

    assert result["runtime_scope"] == "target_interpreter"
    assert any("dependency status is unknown" in finding for finding in result["key_findings"])
    assert not any("fastembed not available" in finding for finding in result["key_findings"])
    calls.clear()
    fix_result = diagnose.auto_fix(result["entries"], python_executable=target_python)
    assert fix_result["failed"] == [
        {"label": "target Python", "error": "diagnostics could not run in the selected interpreter"}
    ]
    assert calls == []


def test_auto_fix_uses_selected_python_module_pip(monkeypatch, tmp_path):
    target_python = tmp_path / "provider env" / "python.exe"
    calls = []
    monkeypatch.setattr(
        diagnose.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(stderr=""),
    )
    monkeypatch.setattr(
        diagnose,
        "collect_runtime_diagnostics",
        lambda python=None: {"checks": [{"check": "fastembed", "status": "OK"}]},
    )

    result = diagnose.auto_fix(
        [{"check": "fastembed", "status": "MISSING"}],
        python_executable=str(target_python),
    )

    assert result["fixed"] == ["fastembed (embeddings engine)"]
    assert calls[0][0] == [
        str(target_python.resolve()),
        "-m",
        "pip",
        "install",
        "mnemosyne-memory[embeddings]",
    ]


def test_auto_fix_does_not_claim_success_when_isolated_check_still_fails(
    monkeypatch, tmp_path, capsys
):
    target_python = tmp_path / "provider env" / "python.exe"
    calls = []
    monkeypatch.setattr(
        diagnose.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or SimpleNamespace(stderr=""),
    )
    monkeypatch.setattr(
        diagnose,
        "collect_runtime_diagnostics",
        lambda python=None: {"checks": [{"check": "sqlite_vec", "status": "MISSING"}]},
    )

    result = diagnose.auto_fix(
        [{"check": "sqlite_vec", "status": "MISSING"}],
        python_executable=str(target_python),
    )

    assert result["fixed"] == []
    assert result["failed"] == [
        {
            "label": "sqlite-vec (vector search)",
            "error": "pip exited successfully, but the dependency was not verified in the selected interpreter",
        }
    ]
    assert calls[0][0][0] == str(target_python.resolve())
    assert "Could not verify sqlite-vec" in capsys.readouterr().out


def test_target_python_bare_name_resolves_through_path(monkeypatch):
    resolved = "C:/venvs/provider/Scripts/python.exe"
    monkeypatch.setattr(
        "mnemosyne.runtime_diagnostics.shutil.which",
        lambda executable: resolved if executable == "python" else None,
    )
    commands = []
    monkeypatch.setattr(
        "mnemosyne.runtime_diagnostics.subprocess.run",
        lambda command, **kwargs: commands.append(command)
        or SimpleNamespace(returncode=1, stdout="", stderr=""),
    )

    diagnose.collect_runtime_diagnostics("python")

    assert commands[0][0] == resolved


def test_auto_fix_dry_run_uses_selected_python_basename(monkeypatch, tmp_path):
    target_python = tmp_path / "provider env" / "python.exe"
    calls = []
    monkeypatch.setattr(diagnose.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    result = diagnose.auto_fix(
        [{"check": "fastembed", "status": "MISSING"}],
        dry_run=True,
        python_executable=str(target_python),
    )

    assert result["fixed"] == [
        "WOULD install: fastembed (embeddings engine) (python.exe -m pip install mnemosyne-memory[embeddings])"
    ]
    assert calls == []


def test_cli_forwards_python_to_diagnostics_and_auto_fix(monkeypatch, capsys):
    calls = []

    def run_diagnostics(**kwargs):
        calls.append(("diagnose", kwargs))
        return {
            "runtime_scope": "target_interpreter",
            "runtime_executable": "python.exe",
            "checks_passed": 1,
            "checks_total": 1,
            "entries": [],
            "key_findings": [],
        }

    def auto_fix(entries, **kwargs):
        calls.append(("fix", kwargs))
        return {"fixed": [], "failed": [], "skipped": [], "ran": True}

    monkeypatch.setattr(diagnose, "run_diagnostics", run_diagnostics)
    monkeypatch.setattr(diagnose, "auto_fix", auto_fix)

    cli.cmd_diagnose(["--python", "provider/python", "--fix"])

    assert calls == [
        (
            "diagnose",
            {"repair_vec_working": False, "dry_run": False, "python_executable": "provider/python"},
        ),
        ("fix", {"dry_run": False, "python_executable": "provider/python"}),
    ]
    assert "Runtime checks: target_interpreter (python.exe)" in capsys.readouterr().out


def test_cli_accepts_equals_form_for_python_option(monkeypatch):
    calls = []
    monkeypatch.setattr(diagnose, "run_diagnostics", lambda **kwargs: calls.append(kwargs) or {
        "runtime_scope": "target_interpreter",
        "runtime_executable": "python",
        "checks_passed": 0,
        "checks_total": 0,
        "entries": [],
        "key_findings": [],
    })

    cli.cmd_diagnose(["--python=provider/python"])

    assert calls[0]["python_executable"] == "provider/python"
