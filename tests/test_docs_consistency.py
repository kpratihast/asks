"""#31: setup documentation must match the repository it ships in."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted(p for p in ROOT.rglob('*.md') if '.venv' not in p.parts and 'node_modules' not in p.parts)


def test_docs_do_not_refer_to_a_clone_folder_that_does_not_exist():
    # README clones into `asks`; the old prototype folder name `askiisc-retrieval` must not be used.
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'cd asks' in readme
    offenders = [f'{p.relative_to(ROOT)}:{n}' for p in DOCS
                 for n, line in enumerate(p.read_text(encoding='utf-8').splitlines(), 1)
                 if 'askiisc-retrieval' in line]
    assert offenders == []


def test_readme_python_version_is_the_one_ci_tests():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    workflow = (ROOT / '.github/workflows/aws-validation.yml').read_text(encoding='utf-8')
    ci_versions = set(re.findall(r"python-version:\s*'?([0-9.]+)'?", workflow))
    stated = re.search(r'Use Python ([0-9.]+)', readme)
    assert ci_versions == {'3.12'} and stated and stated.group(1) == '3.12'
    assert '3.11' not in readme.split('```')[0]


def test_models_md_documents_how_to_install_the_vllm_metal_environment():
    models = (ROOT / 'MODELS.md').read_text(encoding='utf-8')
    assert '## Install vLLM-Metal' in models
    install = models.split('## Install vLLM-Metal', 1)[1]
    assert 'install.sh' in install and '~/.venv-vllm-metal' in install


def _exports(text):
    """Every `export NAME=value` setting in a document (several may share one line)."""
    lines = [line for line in text.splitlines() if line.lstrip().startswith('export ')]
    return dict(re.findall(r'\b(ASKIISC_[A-Z_]+|SERVED_MODEL)=(\S+)', '\n'.join(lines)))


def test_gpu_runbook_named_in_pinned_issue_exists_and_is_linked():
    # #18: issue #1 tells red teams "README, then claude/GPU_RUNBOOK.md".
    assert (ROOT / 'claude/GPU_RUNBOOK.md').is_file()
    assert 'claude/GPU_RUNBOOK.md' in (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'claude/GPU_RUNBOOK.md' in (ROOT / 'SERVING.md').read_text(encoding='utf-8')


def test_serving_gpu_section_matches_the_runbook_configuration():
    # #18: SERVING.md §2 used bge-reranker-v2-m3, no gate and the external ingestion backend.
    runbook = _exports((ROOT / 'claude/GPU_RUNBOOK.md').read_text(encoding='utf-8'))
    serving = (ROOT / 'SERVING.md').read_text(encoding='utf-8')
    gpu = _exports(serving.split('## 2. On the GPU', 1)[1].split('\n## 3.', 1)[0])
    keys = ('ASKIISC_BACKEND', 'ASKIISC_EMBED_MODEL', 'ASKIISC_EMBED_DIM', 'ASKIISC_RERANKER',
            'ASKIISC_RERANK_MODEL', 'ASKIISC_RERANK_MIN', 'ASKIISC_SOURCE_DEPARTMENTS', 'SERVED_MODEL')
    assert {k: gpu.get(k) for k in keys} == {k: runbook.get(k) for k in keys}
    assert runbook['ASKIISC_RERANK_MODEL'] == 'BAAI/bge-reranker-base' and runbook['ASKIISC_RERANK_MIN'] == '0.69'
    assert runbook['ASKIISC_BACKEND'] == 'stub' and runbook['SERVED_MODEL'] == 'llama-3.1-8b'


def test_reranker_comment_does_not_point_the_gpu_deployment_at_another_reranker():
    config = (ROOT / 'askiisc_retrieval/config.py').read_text(encoding='utf-8')
    assert 'Deployment on GPU: BAAI/bge-reranker-v2-m3' not in config


def test_gpu_runbook_states_gpu_access_and_private_reporting_instead_of_placeholders():
    # #18: issue #1 and the runbook left the GPU host and the private-reporting contact unfilled.
    runbook = (ROOT / 'claude/GPU_RUNBOOK.md').read_text(encoding='utf-8')
    assert '**GPU access:**' in runbook and '**Private reporting:**' in runbook
    assert 'must supply' not in runbook and '<fill in>' not in runbook and '<team email>' not in runbook
