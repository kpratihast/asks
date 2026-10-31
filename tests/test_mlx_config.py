"""MLX model selection and request-template configuration."""
import importlib.util
import json
from pathlib import Path

from askiisc_serving.settings import Settings
from test_serving import FakeModel, make, ask

ROOT = Path(__file__).resolve().parents[1]


def test_mlx_template_controls_reach_the_model_request():
    model = FakeModel('DS 252 covers cloud computing [S1].')
    client, _ = make(model, chat_template_kwargs={'enable_thinking': False})
    with client:
        assert ask(client).json()['status'] == 'answered'
    assert model.requests[0]['chat_template_kwargs'] == {'enable_thinking': False}


def test_local_profile_selects_mlx_and_explicit_environment_overrides_win(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location('local_launcher', ROOT / 'tools' / 'local.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    monkeypatch.setattr(launcher.os, "environ", dict(launcher.os.environ))
    config = tmp_path / 'local.json'
    config.write_text(json.dumps({'VLLM_URL': 'http://127.0.0.1:8000',
        'SERVED_MODEL': 'mlx-community/Qwen3.5-9B-MLX-4bit',
        'CHAT_TEMPLATE_KWARGS': {'enable_thinking': False}}))
    monkeypatch.setenv('ASKIISC_LOCAL_CONFIG', str(config))
    monkeypatch.delenv('VLLM_URL', raising=False)
    monkeypatch.delenv('SERVED_MODEL', raising=False)
    monkeypatch.delenv('CHAT_TEMPLATE_KWARGS', raising=False)
    launcher.configure()
    settings = Settings()
    assert settings.served_model == 'mlx-community/Qwen3.5-9B-MLX-4bit'
    assert settings.chat_template_kwargs == {'enable_thinking': False}
    monkeypatch.setenv('SERVED_MODEL', 'auto')
    launcher.configure()
    assert Settings().served_model == 'auto'
