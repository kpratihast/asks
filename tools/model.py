"""Switch one local vLLM Metal server between Llama checkpoints."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
SESSION = 'askiisc-model'
LEGACY = 'askiisc-cds-local:vllm'
ALIAS = 'askiisc'
PORT = 8000
URL = f'http://127.0.0.1:{PORT}'
STATE = ROOT / '.local' / 'model'


@dataclass(frozen=True)
class Profile:
    repo: str
    precision: str
    fraction: float


PROFILES = {
    '3b': Profile('mlx-community/Llama-3.2-3B-Instruct', 'BF16', 0.45),
    '8b': Profile('mlx-community/Meta-Llama-3.1-8B-Instruct-bf16', 'BF16', 0.85),
    '8b-quantized': Profile('mlx-community/Meta-Llama-3.1-8B-Instruct-4bit', '4-bit', 0.45),
    '3b-quantized': Profile('mlx-community/Llama-3.2-3B-Instruct-4bit', '4-bit', 0.25),
}
ALIASES = {'8b-q4': '8b-quantized', '8b-4bit': '8b-quantized',
           '3b-q4': '3b-quantized', '3b-4bit': '3b-quantized'}


class SwitchError(RuntimeError):
    pass


def run(args, *, check=True, capture=True):
    return subprocess.run([str(x) for x in args], text=True, check=check,
                          stdout=subprocess.PIPE if capture else None,
                          stderr=subprocess.PIPE if capture else None)


def models():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(URL + '/v1/models', timeout=2) as response:
            return json.load(response).get('data', [])
    except (OSError, ValueError, urllib.error.URLError):
        return []


def pane(target):
    result = run(['tmux', 'display-message', '-p', '-t', target,
                  '#{pane_id}\t#{pane_pid}\t#{pane_dead}\t#{session_name}\t#{window_name}'], check=False)
    if result.returncode or not result.stdout.strip():
        return None
    fields = result.stdout.strip().split('\t')
    if len(fields) != 5:
        raise SwitchError(f'Cannot inspect tmux pane {target}')
    if fields[3] not in (SESSION, LEGACY.split(':')[0]) or fields[4] != 'vllm':
        return None
    if not target.startswith('%') and target != fields[3] + ':' + fields[4]:
        return None
    return {'id': fields[0], 'pid': int(fields[1]), 'dead': fields[2] == '1'}


def processes():
    result = run(['ps', '-axo', 'pid=,ppid=,command='])
    rows = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            rows[int(fields[0])] = (int(fields[1]), fields[2])
    return rows


def descendant(pid, ancestor, table):
    seen = set()
    while pid in table and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = table[pid][0]
    return False


def listeners():
    result = run(['lsof', '-nP', '-t', f'-iTCP:{PORT}', '-sTCP:LISTEN'], check=False)
    if result.returncode not in (0, 1):
        raise SwitchError('Cannot inspect the model port with lsof')
    return {int(x) for x in result.stdout.split()}


def controlled_panes():
    # Keep the old API window and all other sessions intact.
    return [(target, p) for target in (SESSION + ':vllm', LEGACY) if (p := pane(target))]


def is_model_process(command):
    try:
        tokens = shlex.split(command)
        pos = tokens.index('--served-model-name')
        return tokens[pos + 1] == ALIAS and 'serve' in tokens and any(Path(t).name == 'vllm' for t in tokens)
    except (ValueError, IndexError):
        return False


def assert_owned(panes, table=None, port_pids=None):
    table = processes() if table is None else table
    port_pids = listeners() if port_pids is None else port_pids
    active = [p for _, p in panes if not p['dead']]
    for pid in port_pids:
        if not any(descendant(pid, p['pid'], table) for p in active):
            raise SwitchError(f'Port {PORT} belongs to an unmanaged process (PID {pid}). '
                              'Stop that server manually; it will not be killed by this script.')
    for target, p in panes:
        if p['dead']:
            continue
        children = [command for pid, (_, command) in table.items() if descendant(pid, p['pid'], table)]
        if not any(is_model_process(command) for command in children):
            raise SwitchError(f'{target} is running something other than the Ask IISc model server; '
                              'it will not be interrupted.')
    return table


def stop(timeout):
    current = controlled_panes()
    assert_owned(current)
    active = [(target, p) for target, p in current if not p['dead']]
    if not active:
        print('Model server is already stopped.', flush=True)
        return
    for target, p in active:
        print(f'Stopping model in {target} ...', flush=True)
        run(['tmux', 'send-keys', '-t', p['id'], 'C-c'])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all((p := pane(old['id'])) is None or p['dead'] for _, old in active) and not listeners():
            print('Model server stopped.', flush=True)
            return
        time.sleep(0.5)
    raise SwitchError('Graceful shutdown timed out. No replacement model was started. '
                      'Inspect the model pane; no process was force-killed.')


def command(profile, executable, fraction):
    # Override inherited model-specific environment settings for reproducible switching.
    return ['env', 'VLLM_METAL_USE_PAGED_ATTENTION=1', f'VLLM_METAL_MEMORY_FRACTION={fraction}',
            str(executable), 'serve', profile.repo, '--served-model-name', ALIAS,
            '--host', '127.0.0.1', '--port', str(PORT), '--max-model-len', '4096',
            '--max-num-seqs', '1', '--enable-prefix-caching']


def cache_model(profile, python, offline):
    print(f'Checking {profile.repo} ({profile.precision}); '
          + ('using cached files only.' if offline else 'uncached files will be downloaded.'), flush=True)
    # Exit on auth/network/disk errors BEFORE stopping the working model. Never load weights here.
    code = ('from huggingface_hub import snapshot_download; import sys; '
            'snapshot_download(sys.argv[1], local_files_only=sys.argv[2] == "1", allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"])')
    result = run([python, '-c', code, profile.repo, '1' if offline else '0'], check=False, capture=False)
    if result.returncode:
        raise SwitchError('Checkpoint preparation failed. The existing model was left running. '
                          'Check disk space/network access (or cached files with --offline).')
    # local_files_only may find an incomplete snapshot: verify every indexed weight exists.
    code = '''from huggingface_hub import snapshot_download
import json, pathlib, sys
p=pathlib.Path(snapshot_download(sys.argv[1], local_files_only=True,
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.txt"]))
indices=list(p.glob('*.safetensors.index.json'))
weights=set()
for index in indices:
    weights.update(json.loads(index.read_text())['weight_map'].values())
if not weights:
    weights={f.name for f in p.glob('*.safetensors')}
if not indices and any('-of-' in name for name in weights):
    raise SystemExit('Sharded checkpoint is missing its weight index')
if not weights or not all((p/name).is_file() for name in weights):
    raise SystemExit('Cached checkpoint is incomplete; rerun without --offline')
config=json.loads((p/'config.json').read_text())
quant=config.get('quantization') or config.get('quantization_config')
if (sys.argv[2]=='4-bit') != bool(quant):
    raise SystemExit('Checkpoint precision does not match the selected profile')
if not (p/'config.json').is_file() or not (p/'tokenizer_config.json').is_file() or not (p/'tokenizer.json').is_file():
    raise SystemExit('Cached checkpoint is missing configuration/tokenizer files')
'''
    result = run([python, '-c', code, profile.repo, profile.precision], check=False, capture=False)
    if result.returncode:
        raise SwitchError('Checkpoint is incomplete. The existing model was left running.')


@contextmanager
def lock():
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / 'switch.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SwitchError('Another model switch is in progress; wait for it to finish.') from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def start(args, key, envdir):
    profile = PROFILES[key]
    executable, python = envdir / 'bin' / 'vllm', envdir / 'bin' / 'python'
    fraction = args.memory_fraction if args.memory_fraction is not None else profile.fraction
    cmd = command(profile, executable, fraction)
    if args.dry_run:
        print(json.dumps({'profile': key, 'checkpoint': profile.repo, 'precision': profile.precision,
                          'endpoint': URL, 'alias': ALIAS, 'memory_fraction': fraction,
                          'command': shlex.join(cmd)}, indent=2))
        return
    if not executable.is_file() or not os.access(executable, os.X_OK) or not python.is_file():
        raise SwitchError(f'No vLLM environment at {envdir}; set ASKIISC_VLLM_ENV or --vllm-env.')
    assert_owned(controlled_panes())
    if not args.restart and any(m.get('id') == ALIAS and m.get('root') == profile.repo for m in models()):
        print(f'{key} ({profile.precision}) is already ready at {URL}.', flush=True)
        return
    if key == '8b':
        print('8b uses BF16 weights (~16 GB), plus inference memory. Close other heavy apps first. '
              'Use 8b-quantized for the smaller 4-bit checkpoint.', flush=True)
    cache_model(profile, python, args.offline)
    stop(args.stop_timeout)
    log = STATE / f'{key}.log'
    shell = f'exec {shlex.join(cmd)} >> {shlex.quote(str(log))} 2>&1'
    target = SESSION + ':vllm'
    existing = pane(target)
    if existing:
        if not existing['dead']:
            raise SwitchError(f'{target} is still running; refusing to replace it.')
        run(['tmux', 'respawn-pane', '-t', existing['id'], '-c', ROOT, shell])
    else:
        # A placeholder command lets us enable remain-on-exit before the server starts.
        run(['tmux', 'new-session', '-d', '-s', SESSION, '-n', 'vllm', '-c', ROOT])
        run(['tmux', 'set-option', '-t', SESSION, 'remain-on-exit', 'on'])
        # The new pane contains only our newly created shell, so replacing it is safe.
        run(['tmux', 'respawn-pane', '-k', '-t', target, '-c', ROOT, shell])
    (STATE / 'selected.json').write_text(json.dumps({'profile': key, 'checkpoint': profile.repo,
                                                  'precision': profile.precision, 'log': str(log)}, indent=2))
    print(f'Starting {key} ({profile.precision}). Log: {log}', flush=True)
    deadline = time.monotonic() + args.timeout
    next_progress = time.monotonic() + 15
    while time.monotonic() < deadline:
        if any(m.get('id') == ALIAS and m.get('root') == profile.repo for m in models()):
            print(f'Ready: {key} ({profile.precision}) at {URL}; chat: http://127.0.0.1:8100', flush=True)
            return
        p = pane(target)
        if p is None or p['dead']:
            raise SwitchError(f'Model startup exited. Inspect {log} or run ./model.sh logs.')
        if time.monotonic() >= next_progress:
            print(f'Still loading {key}; log: {log}', flush=True)
            next_progress = time.monotonic() + 15
        time.sleep(1)
    raise SwitchError(f'Readiness timed out after {args.timeout:g}s. The server may still be loading. '
                      'Check ./model.sh status or ./model.sh logs; use ./model.sh stop to stop it.')


def status():
    available = models()
    if available:
        for m in available:
            key = next((k for k, p in PROFILES.items() if p.repo == m.get('root')), 'unrecognized')
            print(json.dumps({'status': 'ready', 'profile': key, 'checkpoint': m.get('root'),
                              'alias': m.get('id'), 'context_tokens': m.get('max_model_len'), 'endpoint': URL}, indent=2))
    else:
        active = any(not p['dead'] for _, p in controlled_panes())
        print('Model server is starting or unavailable.' if active else 'Model server is stopped.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog='Examples: ./model.sh 3b | ./model.sh --model 8b-quantized | ./model.sh stop')
    parser.add_argument('action', nargs='?', help='3b, 8b, 8b-quantized, 3b-quantized, status, stop, logs')
    parser.add_argument('--model', help='Model profile (alternative to the positional argument)')
    parser.add_argument('--vllm-env', type=Path, default=Path(os.environ.get('ASKIISC_VLLM_ENV', '~/.venv-vllm-metal')).expanduser())
    parser.add_argument('--memory-fraction', type=float, help='Override the Metal memory budget (0 < value <= 1)')
    parser.add_argument('--timeout', type=float, default=600, help='Readiness timeout in seconds (default 600)')
    parser.add_argument('--stop-timeout', type=float, default=60)
    parser.add_argument('--dry-run', action='store_true', help='Print model/command without downloads or service changes')
    parser.add_argument('--offline', action='store_true', help='Require a complete cached checkpoint')
    parser.add_argument('--restart', action='store_true', help='Restart even if the requested model is ready')
    args = parser.parse_args(argv)
    if args.action and args.model:
        parser.error('Use either a positional argument or --model, not both')
    action = (args.model or args.action or 'status').lower()
    action = ALIASES.get(action, action)
    if action not in {*PROFILES, 'status', 'stop', 'logs'}:
        parser.error(f'Unknown profile/action: {action}')
    if args.memory_fraction is not None and not 0 < args.memory_fraction <= 1:
        parser.error('--memory-fraction must be in (0, 1]')
    if not args.timeout > 0 or not args.stop_timeout > 0:
        parser.error('Timeouts must be positive')
    if args.dry_run:
        if action not in PROFILES:
            parser.error('--dry-run requires a model profile')
        start(args, action, args.vllm_env.expanduser())
        return
    for dependency in ('tmux', 'lsof', 'ps'):
        if not shutil.which(dependency):
            raise SwitchError(f'Missing dependency: {dependency}')
    if action == 'status':
        status()
    elif action == 'logs':
        selected = STATE / 'selected.json'
        if not selected.exists():
            print(f'No switcher log yet. Existing server: tmux attach -t {LEGACY.split(":")[0]}')
        else:
            path = Path(json.loads(selected.read_text())['log'])
            print(path)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-60:]) if path.exists() else 'Log not created yet.')
    else:
        with lock():
            if action == 'stop':
                stop(args.stop_timeout)
            else:
                start(args, action, args.vllm_env.expanduser())


if __name__ == '__main__':
    try:
        main()
    except (SwitchError, subprocess.CalledProcessError, OSError) as exc:
        message = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            message += '\n' + (exc.stderr or '')
        print('Error: ' + message, file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('\nInterrupted. Check ./model.sh status; use ./model.sh stop if a server is still loading.', file=sys.stderr)
        raise SystemExit(130)
