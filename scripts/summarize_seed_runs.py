"""Report every neural seed, preserving one fixed corpus and no best-test choice."""
import json,hashlib,statistics
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-run', type=Path, default=ROOT/'experiments/grouped_seed42')
    parser.add_argument('--replication-parent', type=Path, default=ROOT/'experiments')
    options = parser.parse_args()
    runs=[]
    primary=options.primary_run
    manifest_hash=hashlib.sha256((primary/'corpus/split_manifest.json').read_bytes()).hexdigest()
    primary_hash=hashlib.sha256((primary/'contextual/model_full.json').read_bytes()).hexdigest()
    for seed in [42,43,44]:
        run=primary if seed==42 else options.replication_parent/f'grouped_model_seed{seed}'
        path=run/'evaluation/test_metrics.json'
        if not path.exists():continue
        if hashlib.sha256((run/'corpus/split_manifest.json').read_bytes()).hexdigest()!=manifest_hash:
            raise ValueError('Replications do not share the frozen split manifest.')
        if hashlib.sha256((run/'contextual/model_full.json').read_bytes()).hexdigest()!=primary_hash:
            raise ValueError('Replications do not share the frozen primary model.')
        m=json.loads(path.read_text(encoding='utf-8'))
        s=json.loads((run/'selector/selection_summary.json').read_text(encoding='utf-8'))
        t=json.loads((run/'edit_tagger/metrics.json').read_text(encoding='utf-8'))
        runs.append(dict(seed=seed,edit_cer=m['test_metrics']['edit_tagger']['micro_cer'],hybrid_cer=m['test_metrics']['hybrid']['micro_cer'],edit_em=m['test_metrics']['edit_tagger']['exact_match'],hybrid_em=m['test_metrics']['hybrid']['exact_match'],C=s['best_C'],threshold=s['best_threshold'],best_epoch=t['best_epoch'],absolute_cer_reduction=m['selector_effect']['absolute_cer_reduction']))
    fields=['edit_cer','hybrid_cer','edit_em','hybrid_em','absolute_cer_reduction']
    if not runs:
        raise ValueError('No completed frozen seed evaluations found.')
    payload={'split_seed':42,'planned_neural_seeds':[42,43,44],'all_seeds_complete':len(runs)==3,'runs':runs,'mean':{k:statistics.mean(r[k] for r in runs) for k in fields},'sample_sd':{k:statistics.stdev(r[k] for r in runs) if len(runs)>1 else None for k in fields},'split_manifest_sha256':manifest_hash,'primary_model_sha256':primary_hash,'selection_policy':'Every planned seed reported. Seed42 is the predefined primary run. No best test seed is selected. Each auxiliary/selector independently fit on train/dev/calibration; grouped calibration fold seed remains42.','scope':'Variability of neural initialisation and corresponding candidate selection on ONE fixed dictionary-corpus grouped split; not between-split uncertainty or external generalization.'}
    (primary/'multiseed_summary.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
