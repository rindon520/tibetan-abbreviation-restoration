"""Export separately labelled training-seed and post-hoc diagnostic tables."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--replications-root',type=Path,required=True)
    p.add_argument('--replication-pattern',default='grouped_model_seed{seed}',help='Replication directory name, with a {seed} placeholder.')
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args()
    repo,run,out=args.repo_root.resolve(),args.run_dir.resolve(),args.out_dir.resolve()
    sources=set()
    def load(path):
        sources.add(path);return json.loads(path.read_text(encoding='utf-8'))
    def write(name,rows):
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    main=load(run/'evaluation/test_metrics.json')
    summary=load(out/'summary.json')
    diagnostics_path=run/'diagnostics/candidate_diagnostics.json'
    if diagnostics_path.exists():
        d=load(diagnostics_path)
        diagrows=[
            dict(method_id='pure_neural',label='纯神经标签＋后处理',micro_cer=d['pure_neural_same_checkpoint_postprocessed_metrics']['micro_cer'],role='edit_branch',posthoc=True),
            dict(method_id='pure_rules',label='纯编辑对齐规则＋后处理',micro_cer=d['pure_rules_edit_alignment_postprocessed_metrics']['micro_cer'],role='edit_branch',posthoc=True),
            dict(method_id='neural_fallback',label='预定义神经＋规则回退',micro_cer=main['test_metrics']['edit_tagger']['micro_cer'],role='edit_branch',posthoc=False),
        ]
        ablation_path=run/'no_neural_ablation/evaluation/test_metrics.json'
        if ablation_path.exists():
            ablation=load(ablation_path)
            diagrows.extend([
                dict(method_id='no_neural_selector',label='后验无神经双路选择',micro_cer=ablation['test_metrics']['micro_cer'],role='selector',posthoc=True),
                dict(method_id='main_neural_selector',label='预定义双路选择',micro_cer=main['test_metrics']['hybrid']['micro_cer'],role='selector',posthoc=False),
            ])
            summary['no_neural_selector']=dict(scope=ablation['scope'],C=ablation['C'],threshold=ablation['threshold'],micro_cer=ablation['test_metrics']['micro_cer'],main_micro_cer=main['test_metrics']['hybrid']['micro_cer'],difference_cer_pp=(ablation['test_metrics']['micro_cer']-main['test_metrics']['hybrid']['micro_cer'])*100,edit_distance_difference=ablation['total_edit_distance']-main['test_metrics']['hybrid']['total_edit_distance'])
        write('neural_role.csv',diagrows)
        summary['neural_diagnostics']=dict(scope=d['scope'],fallback_fraction=d['fallback_fraction'],source_codepoints=d['source_codepoints'],checkpoint_epoch=d['neural_checkpoint_epoch'])
    seedrows=[]
    missing=[]
    for seed in [42,43,44]:
        folder=run if seed==42 else args.replications_root.resolve()/args.replication_pattern.format(seed=seed)
        path=folder/'evaluation/test_metrics.json'
        if not path.exists():missing.append(seed);continue
        d=load(path)
        for method,label in [('edit_tagger','编辑标注分支'),('hybrid','双路候选选择')]:
            metric=d['test_metrics'][method]
            seedrows.append(dict(model_seed=seed,method_id=method,label=label,n=metric['samples'],micro_cer=metric['micro_cer'],exact_match=metric['exact_match']))
    if missing and not args.allow_partial:
        raise RuntimeError(f'Run seeds are not all ready: {missing}. Do not present incomplete repetitions as full results.')
    if not missing:write('seed_replications.csv',seedrows)
    sha={item['path']:item for item in summary['sources']}
    for path in sources:
        if path==out/'summary.json':continue
        try:
            relative=path.relative_to(repo).as_posix()
        except ValueError:
            try:
                relative='experiment/'+path.relative_to(run).as_posix()
            except ValueError:
                relative='replications/'+path.relative_to(args.replications_root.resolve()).as_posix()
        sha[relative]=dict(path=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary['sources']=list(sorted(sha.values(),key=lambda v:v['path']))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(neural_diagnostics=diagnostics_path.exists(),seeds_complete=not missing,missing=missing),ensure_ascii=False))


if __name__=='__main__':main()
