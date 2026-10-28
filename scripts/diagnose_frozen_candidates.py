"""Post-hoc diagnostics of frozen seed42 candidates, never used for selection.

Report rule fallback, reference-based oracle, and fixed pure-neural/rule controls.
These controls do not refit candidate weights or recalibrate the main selector.
"""
import json,sys,time
import argparse
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from run_grouped_experiment import (load_jsonl,write_json,write_jsonl,load_corpus,fast_decode_batches,clean_require_torch,validate_alignment)
from train_restore import levenshtein
from evaluate_restore import compute_metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'experiments/grouped_seed42')
    options = parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    import train_edit_tagger as tagger
    import torch
    from torch.utils.data import DataLoader
    from tibetan_syllable import build_syllable_vocabulary
    from types import SimpleNamespace
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    tagger.require_torch=clean_require_torch
    run=options.run_dir.resolve()
    args=SimpleNamespace(output_dir=run,seed=42)
    manifest,records,pairs=load_corpus(args)
    p=load_jsonl(run/'evaluation/contextual_test_predictions.jsonl')
    a=load_jsonl(run/'evaluation/edit_tagger_test_predictions.jsonl')
    validate_alignment(p,a,pairs['test'])
    decisions=[d for r in a for d in r['decisions']]
    origins=Counter(d['origin'] for d in decisions)
    oracle=[x|{'prediction':y['prediction'] if levenshtein(y['prediction'],x['reference'])<levenshtein(x['prediction'],x['reference']) else x['prediction']} for x,y in zip(p,a)]
    state=torch.load(run/'edit_tagger/best.pt',map_location='cpu',weights_only=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=tagger.create_model(tagger.EditTaggerConfig(**state['config'])).to(device)
    model.load_state_dict(state['model_state'])
    chars={c:i for i,c in enumerate(state['char_vocab'])};labels={c:i for i,c in enumerate(state['labels'])}
    train_labelled,_=tagger.build_labelled_pairs(pairs['train'])
    global_counts,contexts=tagger.build_rule_counts(train_labelled)
    vocab=build_syllable_vocabulary(pairs['train'])
    labelled=[(s,r,[tagger.COPY_LABEL]*len(s)) for s,r in pairs['test']]
    loader=DataLoader(tagger.EditDataset(labelled,chars,labels,state['config']['max_positions']),batch_size=128,shuffle=False,num_workers=0,collate_fn=tagger.make_collate(state['config']['pad_char_id'],state['config']['pad_label_id']))
    neural=fast_decode_batches(model,loader,state['labels'],device,global_counts,contexts,0.,3,.5)
    rule=[]
    for source,reference in pairs['test']:
        prediction=''.join(tagger.rule_output_at(source,i,global_counts,contexts,min_count=3,min_confidence=.5) for i in range(len(source)))
        rule.append({'source':source,'reference':reference,'raw_prediction':prediction})
    info=json.loads((run/'edit_tagger/metrics.json').read_text(encoding='utf-8'))
    neural=tagger.postprocess_rows(neural,vocab,info['best_max_edit_distance'])
    rule=tagger.postprocess_rows(rule,vocab,info['best_max_edit_distance'])
    result={'scope':'post-hoc fixed-checkpoint diagnostic controls; not used to select or refit main system','samples':len(p),'fallback_counts':dict(origins),'source_codepoints':len(decisions),'fallback_fraction':origins['rule_fallback']/len(decisions),'reference_driven_two_candidate_oracle_metrics':compute_metrics(oracle),'pure_neural_same_checkpoint_postprocessed_metrics':compute_metrics(neural),'pure_rules_edit_alignment_postprocessed_metrics':compute_metrics(rule),'neural_checkpoint_epoch':state['epoch'],'parameter_count':sum(t.numel() for t in state['model_state'].values()),'constraints':'both controls use unchanged training vocabulary, dev-selected edit distance1 and min_count3/min_confidence.5; threshold0/always-rule are fixed diagnostic endpoints; oracle uses reference and cannot be deployed'}
    dest=run/'diagnostics'
    if (dest/'candidate_diagnostics.json').exists():raise RuntimeError('Diagnostics already exist; do not repeat test-based inspection.')
    write_jsonl(dest/'pure_neural_test_predictions.jsonl',neural)
    write_jsonl(dest/'pure_rules_test_predictions.jsonl',rule)
    write_json(dest/'candidate_diagnostics.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
