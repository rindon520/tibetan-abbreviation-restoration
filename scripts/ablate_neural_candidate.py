"""Post-hoc full-system no-neural auxiliary control with independent calibration.

All underlying train rules and primary model are frozen. The rule-only candidate
replaces the neural/fallback auxiliary; grouped calibration selects C/threshold.
Previously inspected test artifacts support exploratory diagnosis only; this
control never replaces the frozen seed42 main result.
"""
import json,sys,shutil,hashlib,time
import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from run_grouped_experiment import load_corpus,load_jsonl,write_json,write_jsonl,train_selector,selector_resources,validate_alignment
from train_restore import levenshtein
from evaluate_restore import compute_metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'experiments/grouped_seed42')
    options = parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    import train_edit_tagger as tagger
    from tibetan_syllable import build_syllable_vocabulary
    from train_hybrid_selector import selector_features,build_rows
    import numpy as np
    base=options.run_dir.resolve();run=base/'no_neural_ablation'
    if run.exists():raise RuntimeError('Ablation directory exists. Do not repeat test-based model selection.')
    run.mkdir()
    for name in ['corpus','contextual']:shutil.copytree(base/name,run/name)
    args=SimpleNamespace(output_dir=run,seed=42)
    manifest,records,pairs=load_corpus(args)
    train_labelled,_=tagger.build_labelled_pairs(pairs['train'])
    global_counts,contexts=tagger.build_rule_counts(train_labelled)
    vocab=build_syllable_vocabulary(pairs['train'])
    rows=[]
    started=time.perf_counter()
    for source,reference in pairs['calibration']:
        decisions=[{'index':i,'source':c,'output':tagger.rule_output_at(source,i,global_counts,contexts,min_count=3,min_confidence=.5),'confidence':0.,'origin':'rule_fallback'} for i,c in enumerate(source)]
        rows.append({'source':source,'reference':reference,'raw_prediction':''.join(d['output'] for d in decisions),'decisions':decisions})
    candidate_info=json.loads((base/'edit_tagger/metrics.json').read_text(encoding='utf-8'))
    rows=tagger.postprocess_rows(rows,vocab,candidate_info['best_max_edit_distance'])
    write_jsonl(run/'edit_tagger/calibration_predictions.jsonl',rows)
    print(json.dumps({'stage':'rule_calibration_ready','seconds':time.perf_counter()-started}),flush=True)
    train_selector(args)
    # Only after the selector is frozen, load the already-diagnosed fixed test
    # predictions. Their earlier observation is explicitly acknowledged below.
    p=load_jsonl(base/'evaluation/contextual_test_predictions.jsonl')
    a=load_jsonl(base/'diagnostics/pure_rules_test_predictions.jsonl')
    for row in a:
        source=row['source']
        row['decisions']=[{'index':i,'source':c,'confidence':0.,'origin':'rule_fallback'} for i,c in enumerate(source)]
    validate_alignment(p,a,pairs['test'])
    lm,vocabulary=selector_resources(pairs)
    state=json.loads((run/'selector/selector_model.json').read_text(encoding='utf-8'))
    features=np.asarray([selector_features(x,y,lm,vocabulary) for x,y in zip(p,a)],dtype=float)
    logits=((features-np.asarray(state['scaler_mean']))/np.asarray(state['scaler_scale']))@np.asarray(state['coef'])+state['intercept']
    probabilities=1/(1+np.exp(-np.clip(logits,-700,700)))
    chosen=probabilities>=state['threshold']
    selected=build_rows(p,a,chosen,probabilities)
    for row in selected:
        if row['selected_model']=='edit_tagger':row['selected_model']='edit_alignment_pure_rules'
    metrics=compute_metrics(selected)
    pd=np.asarray([levenshtein(x['prediction'],x['reference']) for x in p]);ad=np.asarray([levenshtein(x['prediction'],x['reference']) for x in a]);hd=np.asarray([levenshtein(x['prediction'],x['reference']) for x in selected])
    result={'scope':'post-hoc full-system diagnostic ablation; not a pre-specified primary model and never replaces seed42 by test performance','protocol':'same train/dev/calibration/test groups; same primary; replace auxiliary with frozen edit-alignment rules; independent grouped calibration C/threshold selection; final diagnostic test','prior_test_observation':'Pure-rule test candidates had already been inspected as fixed diagnostic endpoints; no C/threshold or architecture grid derived from their test errors. This is an exploratory ablation, not pristine confirmatory evidence.','C':state['C'],'threshold':state['threshold'],'test_metrics':metrics,'total_edit_distance':int(hd.sum()),'total_reference_codepoints':sum(len(x['reference']) for x in selected),'selected_auxiliary':int(chosen.sum()),'selected_better':int(((ad<pd)&chosen).sum()),'selected_worse':int(((ad>pd)&chosen).sum()),'selected_tie':int(((ad==pd)&chosen).sum()),'net_saved_edits':int((pd-hd).sum()),'frozen_main_neural_auxiliary_hybrid_cer':json.loads((base/'evaluation/test_metrics.json').read_text(encoding='utf-8'))['test_metrics']['hybrid']['micro_cer'],'seconds':time.perf_counter()-started}
    write_jsonl(run/'evaluation/test_predictions.jsonl',selected)
    write_json(run/'evaluation/test_metrics.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
