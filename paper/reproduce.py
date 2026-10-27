#!/usr/bin/env python3
"""Fetch paper assets, recalculate results, replay folds or use the extra candidate."""
import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent

def unpack(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:*') as source:
        source.extractall(destination, filter='data')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fetch = sub.add_parser('fetch', help='Install release assets or local review archives')
    fetch.add_argument('--data-dir', type=Path, required=True)
    fetch.add_argument('--asset-dir', type=Path, help='Local review archives; no network access')
    fetch.add_argument('--tag', default='paper-2026-10-27')
    fetch.add_argument('--assets', nargs='+', choices=['inputs','results','nested','candidate'], default=['inputs','results','nested','candidate'])
    results = sub.add_parser('results', help='Recompute metrics, formal tests, selections, tables and figures')
    results.add_argument('--data-dir', type=Path, required=True)
    results.add_argument('--output-dir', type=Path, required=True)
    results.add_argument('--points-only', action='store_true', help='Skip resampling and selection replay for a fast inspection')
    results.add_argument('--skip-figures', action='store_true')
    train = sub.add_parser('train', help='Run fixed or nested grouped training without a new search')
    train.add_argument('--data-dir', type=Path, required=True)
    train.add_argument('--output-dir', type=Path, required=True)
    train.add_argument('--cohort', choices=['core_dev','common_dev'], required=True)
    train.add_argument('--protocol', choices=['target','guide','atomic_template','dual_atomic'], required=True)
    train.add_argument('--outer-fold', type=int, choices=range(5), help='Omit for five outer folds')
    train.add_argument('--kind', choices=['fixed','nested','perturbation'], default='nested')
    train.add_argument('--tf-python', type=Path, help='Separate Python with TensorFlow 2.18 and tf-keras 2.18')
    train.add_argument('--seed', type=int, choices=[137,271,409], default=137)
    train.add_argument('--perturbation', choices=['history_shuffle','mapping_permutation'], default='history_shuffle')
    predict = sub.add_parser('predict-candidate', help='Use the additional audit candidate, leaving the default CLI unchanged')
    predict.add_argument('--data-dir', type=Path, required=True)
    predict.add_argument('--input', type=Path)
    predict.add_argument('--demo', action='store_true')
    predict.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'fetch':
        for asset in args.assets:
            name = f'tricasrank-paper-{asset}.tar.gz'
            if args.asset_dir:
                archive = args.asset_dir.resolve() / name
                unpack(archive, args.data_dir)
            else:
                import tempfile
                url = f'https://github.com/ZebraL415/Cas12a-Activity-Ranker/releases/download/{args.tag}/{name}'
                with tempfile.TemporaryDirectory() as tmp:
                    archive = Path(tmp) / name
                    try: urllib.request.urlretrieve(url, archive)
                    except Exception as error:
                        raise SystemExit(f'Cannot download {url}. The paper release must be published first, or use --asset-dir. {error}')
                    unpack(archive, args.data_dir)
        print(f'Installed {", ".join(args.assets)} in {args.data_dir.resolve()}')
    elif args.command == 'predict-candidate':
        if args.demo == bool(args.input): parser.error('Choose --input or --demo')
        folder = args.data_dir.resolve() / 'candidate'
        source = folder / 'fixture_input.csv' if args.demo else args.input.resolve()
        subprocess.run([sys.executable, str(folder/'predict_candidate.py'), '--input', str(source), '--output', str(args.output.resolve())], check=True)
    else:
        script = PAPER / ('results.py' if args.command == 'results' else 'train.py')
        command = [sys.executable, str(script), '--data-dir', str(args.data_dir.resolve()), '--output-dir', str(args.output_dir.resolve())]
        if args.command == 'results':
            if args.points_only: command.append('--points-only')
            if args.skip_figures: command.append('--skip-figures')
        else:
            command += ['--cohort', args.cohort, '--protocol', args.protocol, '--kind', args.kind,
                        '--seed', str(args.seed), '--perturbation', args.perturbation]
            if args.outer_fold is not None: command += ['--outer-fold', str(args.outer_fold)]
            # Keep the venv entry path: resolve() would dereference its Python
            # symlink and silently bypass the selected TensorFlow environment.
            if args.tf_python: command += ['--tf-python', str(args.tf_python.absolute())]
        subprocess.run(command, check=True)

if __name__ == '__main__': main()
