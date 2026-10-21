"""Code-release checks use fabricated files and never execute clinical training."""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_script("run_pipeline")
release = load_script("check_release")


def runner_config(tmp_path, monkeypatch, source="synthetic"):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    folder = tmp_path / "configs"
    folder.mkdir()
    config = folder / "fixture.yaml"
    config.write_text(yaml.safe_dump({"data": {"source": source, "cache_dir": "data/fresh"},
                                     "results_root": "results_fixture", "seeds": [0, 1]}),
                      encoding="utf-8")
    return config


def test_runner_is_dry_and_uses_current_interpreter(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    _, commands = runner.build_plan(config)
    assert len(commands) == 7
    assert all(command[0] == sys.executable for command in commands)
    assert not (tmp_path / "data").exists()
    def no_execution(*args, **kwargs):
        raise AssertionError("Dry-run must not execute subprocesses")
    monkeypatch.setattr(runner.subprocess, "run", no_execution)
    assert runner.main(["--config", str(config), "--dry-run"]) == 0


def test_runner_requires_restricted_data_acknowledgement(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch, source="mimic_dir")
    with pytest.raises(ValueError, match="allow-restricted-data"):
        runner.build_plan(config)
    assert runner.build_plan(config, allow_restricted_data=True)[1]


def test_runner_refuses_existing_outputs(tmp_path, monkeypatch):
    config = runner_config(tmp_path, monkeypatch)
    (tmp_path / "results_fixture").mkdir()
    with pytest.raises(FileExistsError, match="existing"):
        runner.build_plan(config)


def test_release_rejects_patient_file_types(tmp_path):
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "fixture.parquet").write_bytes(b"fabricated-not-a-real-parquet")
    assert any("Non-release file" in problem for problem in release.inspect_tree(tmp_path)[1])


def test_release_detects_credential_without_echoing_value(tmp_path):
    credential = "gh" + "p_" + "a" * 30
    (tmp_path / "README.md").write_text(credential, encoding="utf-8")
    _, problems, _ = release.inspect_tree(tmp_path)
    assert any("credential pattern" in problem for problem in problems)
    assert all(credential not in problem for problem in problems)


def test_explicit_hashing_does_not_contact_a_model_provider(monkeypatch):
    from wjem_ed_audit.data import text_embed
    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit hashing must not load a transformer")
    monkeypatch.setattr(text_embed, "_bert_embed", forbidden)
    embedded, encoder = text_embed.embed_texts(["fabricated complaint", ""],
                                              {"text": {"encoder": "hashing"}})
    assert embedded.shape == (2, 768)
    assert not embedded[1].any()
    assert encoder == "hashing_explicit"
