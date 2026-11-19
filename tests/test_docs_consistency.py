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
