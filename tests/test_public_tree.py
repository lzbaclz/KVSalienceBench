"""The public inventory guard must catch force-added generated artifacts."""
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

CHECKER = Path(__file__).resolve().parents[1] / 'scripts/check_public_tree.py'


@pytest.fixture
def checkout(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'scripts').mkdir()
    shutil.copy2(CHECKER, tmp_path / 'scripts/check_public_tree.py')
    (tmp_path / '.gitignore').write_text('*.pdf\n/experiments/results/\n/pre/\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    return tmp_path


def run_check(checkout):
    return subprocess.run([sys.executable, str(checkout / 'scripts/check_public_tree.py')],
                          cwd=checkout, capture_output=True, text=True)


def test_local_ignored_outputs_do_not_enter_public_inventory(checkout):
    (checkout / 'pre').mkdir()
    (checkout / 'pre/slides.pdf').write_bytes(b'local preview')
    assert run_check(checkout).returncode == 0


@pytest.mark.parametrize('name', ['pre/slides.pdf', 'paper_icdm/main.tex',
                                 'experiments/results/run.json', 'docs/review.pdf'])
def test_force_added_non_code_files_are_rejected(checkout, name):
    p = checkout / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('generated or editorial artifact')
    subprocess.run(['git', 'add', '-f', name], cwd=checkout, check=True)
    result = run_check(checkout)
    assert result.returncode == 1
    assert name in result.stderr
