"""Lifecycle and process-ownership regression checks; no weights or GPU needed."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / 'tools' / 'model.py'
spec = importlib.util.spec_from_file_location('local_model_switch', PATH)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def arguments(**kw):
    return SimpleNamespace(memory_fraction=None, dry_run=False, restart=False, offline=True,
                           stop_timeout=1, timeout=1, **kw)


@pytest.fixture
def envdir(tmp_path, monkeypatch):
    env = tmp_path / 'runtime'
    (env / 'bin').mkdir(parents=True)
    for name in ('vllm', 'python'):
        p = env / 'bin' / name
        p.write_text('#!/bin/sh\n')
        p.chmod(0o755)
    monkeypatch.setattr(model, 'STATE', tmp_path / 'state')
    return env


@pytest.mark.parametrize('profile,precision,quantized', [('3b', 'BF16', False), ('8b', 'BF16', False),
                                                      ('8b-quantized', '4-bit', True), ('3b-quantized', '4-bit', True)])
def test_dry_run_is_non_mutating_and_preserves_precision(profile, precision, quantized, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(model, 'run', lambda *a, **k: pytest.fail('dry-run invoked a subprocess'))
    monkeypatch.setattr(model, 'STATE', tmp_path / 'state')
    model.main([profile, '--dry-run'])
    data = json.loads(capsys.readouterr().out)
    assert data['precision'] == precision
    assert data['checkpoint'].endswith('-4bit') == quantized
    assert '--served-model-name askiisc' in data['command']
    assert not (tmp_path / 'state').exists()


def test_model_flag_and_alias_work_without_positional_argument(capsys):
    model.main(['--model', '8b-q4', '--dry-run'])
    assert json.loads(capsys.readouterr().out)['profile'] == '8b-quantized'


@pytest.mark.parametrize('args', [['8b', '--model', '3b'], ['invalid'], ['8b', '--memory-fraction', 'nan'],
                                 ['8b', '--memory-fraction', '1.1'], ['3b', '--timeout', '0']])
def test_invalid_arguments_fail_before_mutation(args):
    with pytest.raises(SystemExit) as error:
        model.main(args)
    assert error.value.code == 2


def test_unmanaged_port_is_never_interrupted(monkeypatch):
    monkeypatch.setattr(model, 'run', lambda *a, **k: pytest.fail('sent a process signal'))
    p = {'pid': 100, 'dead': False, 'id': '%1'}
    with pytest.raises(model.SwitchError, match='unmanaged'):
        model.assert_owned([('askiisc-model:vllm', p)], table={100: (1, 'vllm'), 300: (1, 'other')}, port_pids={300})


def test_lookalike_alias_is_not_accepted_as_our_server():
    assert not model.is_model_process('vllm serve model --served-model-name askiisc-other')
    assert model.is_model_process('/env/bin/python /env/bin/vllm serve model --served-model-name askiisc --port 8000')


def test_non_model_process_in_known_pane_is_not_interrupted():
    with pytest.raises(model.SwitchError, match='something other'):
        model.assert_owned([('askiisc-model:vllm', {'pid': 100, 'dead': False})],
                           table={100: (1, 'python webserver.py')}, port_pids={100})


def test_download_failure_leaves_running_model_untouched(envdir, monkeypatch):
    monkeypatch.setattr(model, 'controlled_panes', lambda: [])
    monkeypatch.setattr(model, 'assert_owned', lambda *a: None)
    monkeypatch.setattr(model, 'models', lambda: [])
    monkeypatch.setattr(model, 'run', lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr(model, 'stop', lambda *a: pytest.fail('stopped working server on download failure'))
    with pytest.raises(model.SwitchError, match='left running'):
        model.start(arguments(), '3b', envdir)


def test_same_ready_checkpoint_is_idempotent(envdir, monkeypatch):
    monkeypatch.setattr(model, 'controlled_panes', lambda: [])
    monkeypatch.setattr(model, 'assert_owned', lambda *a: None)
    monkeypatch.setattr(model, 'models', lambda: [{'id': 'askiisc', 'root': model.PROFILES['3b-quantized'].repo}])
    monkeypatch.setattr(model, 'cache_model', lambda *a: pytest.fail('unneeded download'))
    monkeypatch.setattr(model, 'stop', lambda *a: pytest.fail('unneeded restart'))
    model.start(arguments(), '3b-quantized', envdir)


def test_stop_targets_only_owned_model_pane(monkeypatch):
    calls = []
    p = {'id': '%7', 'pid': 100, 'dead': False}
    monkeypatch.setattr(model, 'controlled_panes', lambda: [('askiisc-cds-local:vllm', p)])
    monkeypatch.setattr(model, 'assert_owned', lambda *a: None)
    monkeypatch.setattr(model, 'run', lambda args, **kw: calls.append(args))
    monkeypatch.setattr(model, 'pane', lambda target: {**p, 'dead': True})
    monkeypatch.setattr(model, 'listeners', lambda: set())
    model.stop(1)
    assert calls == [['tmux', 'send-keys', '-t', '%7', 'C-c']]


def test_concurrent_switch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(model, 'STATE', tmp_path / 'state')
    with model.lock():
        with pytest.raises(model.SwitchError, match='in progress'):
            with model.lock():
                pytest.fail('second switch acquired the lock')


def test_start_caches_then_stops_and_waits_for_exact_checkpoint(envdir, monkeypatch, tmp_path):
    events = []
    (tmp_path / 'state').mkdir()
    monkeypatch.setattr(model, 'controlled_panes', lambda: [])
    monkeypatch.setattr(model, 'assert_owned', lambda *a: None)
    monkeypatch.setattr(model, 'cache_model', lambda *a: events.append('cache'))
    monkeypatch.setattr(model, 'stop', lambda *a: events.append('stop'))
    monkeypatch.setattr(model, 'run', lambda cmd, **kw: events.append(cmd[1]))
    n = [0]
    def pane(target):
        n[0] += 1
        return None if n[0] == 1 else {'dead': False}
    monkeypatch.setattr(model, 'pane', pane)
    results = iter([[], [{'id': 'askiisc', 'root': 'wrong-checkpoint'}],
                    [{'id': 'askiisc', 'root': model.PROFILES['8b-quantized'].repo}]])
    monkeypatch.setattr(model, 'models', lambda: next(results))
    monkeypatch.setattr(model.time, 'sleep', lambda seconds: None)
    model.start(arguments(), '8b-quantized', envdir)
    assert events[:2] == ['cache', 'stop']
    assert events[2:] == ['new-session', 'set-option', 'respawn-pane']
    assert json.loads((tmp_path / 'state' / 'selected.json').read_text())['profile'] == '8b-quantized'


def test_missing_tmux_target_can_return_success_with_empty_fields(monkeypatch):
    monkeypatch.setattr(model, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='\t\t\t\t\n'))
    assert model.pane('askiisc-model:vllm') is None


def test_tmux_prefix_match_cannot_adopt_another_session(monkeypatch):
    monkeypatch.setattr(model, 'run', lambda *a, **kw: SimpleNamespace(returncode=0,
        stdout='%9\t999\t0\taskiisc-model-other\tvllm\n'))
    assert model.pane('askiisc-model:vllm') is None
