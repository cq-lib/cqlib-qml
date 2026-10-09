"""Run tests and smoke checks outside the checkout using an installed wheel."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', type=Path, default=Path('tests'))
    parser.add_argument('--tutorials', type=Path, default=Path('docs/tutorials'))
    args = parser.parse_args()
    source = args.tests.resolve().parent
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    with tempfile.TemporaryDirectory(prefix='qml-wheel-check-') as directory:
        root = Path(directory)
        shutil.copytree(args.tests.resolve(), root / 'tests')
        shutil.copytree(args.tutorials.resolve(), root / 'docs/tutorials')
        shutil.copytree(source / 'scripts', root / 'scripts')
        shutil.copy(source / 'MANIFEST.in', root / 'MANIFEST.in')
        code = ('from pathlib import Path; import cqlib_qml; '
                f'assert not Path(cqlib_qml.__file__).resolve().is_relative_to(Path({str(source / "cqlib_qml")!r})), '
                '"Expected wheel import, got source checkout"; print(cqlib_qml.__file__)')
        subprocess.run([sys.executable, '-c', code], cwd=root, env=env, check=True)
        # Examples are checked before packaging and are not included in the wheel.
        subprocess.run([sys.executable, '-m', 'pytest', 'tests', '-q',
                        '--ignore=tests/test_torch_examples.py',
                        '--import-mode=importlib', '-p', 'no:cacheprovider'], cwd=root, env=env, check=True)
        subprocess.run([sys.executable, 'scripts/smoke_test.py'], cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
