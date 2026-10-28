"""Sequential neural seeds 43/44 on the unchanged split and primary model.

Every seed is reported; no best test seed is selected. Generative primary state
and corpus are copied exactly, while neural weights and calibrated selector are
independently fit. The calibration group fold seed remains 42.
"""
import json, shutil, subprocess, sys
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--primary-run', type=Path, default=ROOT/'experiments/grouped_seed42')
    parser.add_argument('--output-parent', type=Path, default=ROOT/'experiments')
    parser.add_argument('--raw-data-dir', type=Path, default=ROOT/'data')
    options = parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    for seed in [43,44]:
        run=options.output_parent/f'grouped_model_seed{seed}'
        if run.exists():raise RuntimeError(f'Replication directory already exists: {run}. Inspect rather than overwrite.')
        run.mkdir(parents=True)
        for name in ['corpus','contextual']:shutil.copytree(options.primary_run/name,run/name)
        for stage in ['train-tagger','train-selector','evaluate']:
            command=[sys.executable,str(HERE/'run_grouped_experiment.py'),'--code-root',str(options.repo_root),'--raw-data-dir',str(options.raw_data_dir),'--output-dir',str(run),'--stage',stage,'--seed','42','--model-seed',str(seed)]
            print(json.dumps({'model_seed':seed,'stage':stage,'command':command}),flush=True)
            with (run/f'{stage}.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=options.repo_root,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(f'Seed{seed} {stage} failed; see log.')
    print('All planned seed replications completed.',flush=True)

if __name__=='__main__':main()
