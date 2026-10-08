"""Verify source contents and rebuild a wheel from an sdist."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='qml-sdist-check-') as directory:
        with tarfile.open(args.archive) as archive:
            names = archive.getnames()
            for suffix in ('docs/tutorials/models.md', 'docs/tutorials/encoder.md',
                           'docs/tutorials/training_contracts.md', 'cqlib_qml/_state.py',
                           'scripts/release_check.sh', 'scripts/smoke_test.py',
                           'scripts/verify_wheel.py', 'tests/test_release_contracts.py'):
                if not any(name.endswith('/' + suffix) for name in names):
                    raise ValueError(f'sdist is missing {suffix}')
            if not any('/release_notes/' in name and name.endswith('.md') for name in names):
                raise ValueError('sdist is missing release notes')
            if any('/docs/api/' in name for name in names):
                raise ValueError('Generated API HTML must not enter the sdist')
            archive.extractall(directory, filter='data')
        roots = list(Path(directory).iterdir())
        if len(roots) != 1:
            raise ValueError('Expected a single source root in sdist')
        subprocess.run([sys.executable, '-m', 'build', '--wheel', '--no-isolation',
                        '--outdir', str(args.outdir.resolve()), str(roots[0])], check=True)


if __name__ == '__main__':
    main()
