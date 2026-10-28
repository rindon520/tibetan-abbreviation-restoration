"""Reproduce the restoration pipeline with parent groups and independent calibration.

This driver reuses the published model algorithms. All original data/code/results
remain untouched. Unicode NFC is used to link split groups, never to change text.
The final test is evaluated only by the explicit evaluate stage after selection.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

def clean_require_torch():
    """Use a coherent environment without suppressing OpenMP runtime errors."""
    import torch
    from torch import nn
    return torch, nn


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class UnionFind:
    def __init__(self): self.parent = {}
    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def prepare(args):
    existing_manifest = args.output_dir / 'corpus/split_manifest.json'
    if existing_manifest.exists():
        manifest, _, _ = load_corpus(args)
        print(json.dumps({'stage': 'prepare_existing', 'trusted_segments': manifest['trusted_segments'], 'seed': manifest['seed']}), flush=True)
        return
    from train_restore import read_parallel_data
    from prepare_segment_data import split_source_segments, split_restore_segments
    data = args.raw_data_dir
    raw = read_parallel_data(data / "Source.txt", data / "Restore.txt")
    rows = []
    uf = UnionFind()
    source_parent = {}
    rejected = []
    for parent_id, (s, r) in enumerate(raw):
        ss, rr = split_source_segments(s), split_restore_segments(r)
        if not ss or len(ss) != len(rr):
            rejected.append(parent_id)
            continue
        uf.find(parent_id)
        for j, (cs, cr) in enumerate(zip(ss, rr)):
            key = unicodedata.normalize("NFC", cs)
            if key in source_parent: uf.union(parent_id, source_parent[key])
            source_parent[key] = parent_id
            rows.append({"record_id": len(rows), "parent_line_id": parent_id, "segment_in_parent": j, "source": cs, "reference": cr})
    grouped = defaultdict(list)
    for row in rows:
        row["group_id"] = f"g{uf.find(row['parent_line_id']):05d}"
        grouped[row["group_id"]].append(row["record_id"])
    gids = sorted(grouped)
    random.Random(args.seed).shuffle(gids)
    remaining = list(gids)
    splits = {}
    for split, ratio in (("test", .10), ("calibration", .10), ("dev", .10)):
        target = len(rows) * ratio
        chosen = []
        count = 0
        while remaining and count < target:
            gid = remaining[0]
            width = len(grouped[gid])
            if chosen and abs(count + width - target) > abs(count - target): break
            chosen.append(remaining.pop(0)); count += width
        splits[split] = {"record_indices": [i for gid in chosen for i in grouped[gid]], "group_ids": chosen}
    splits["train"] = {"record_indices": [i for gid in remaining for i in grouped[gid]], "group_ids": remaining}
    for split, spec in splits.items():
        spec["samples"] = len(spec["record_indices"])
        spec["groups"] = len(spec["group_ids"])
        spec["parent_lines"] = len({rows[i]["parent_line_id"] for i in spec["record_indices"]})
    assigned = [i for spec in splits.values() for i in spec["record_indices"]]
    assert sorted(assigned) == list(range(len(rows)))
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2: continue
            a, b = splits[s1]["record_indices"], splits[s2]["record_indices"]
            assert not ({rows[i]["parent_line_id"] for i in a} & {rows[i]["parent_line_id"] for i in b})
            assert not ({unicodedata.normalize('NFC', rows[i]['source']) for i in a} & {unicodedata.normalize('NFC', rows[i]['source']) for i in b})
    corpus_dir = args.output_dir / "corpus"
    write_jsonl(corpus_dir / "records.jsonl", rows)
    for fname, key in [("Source.txt", "source"), ("Restore.txt", "reference")]:
        (corpus_dir / fname).write_text("\n".join(row[key] for row in rows) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "seed": args.seed,
        "protocol": "parent_line_and_NFC_source_connected_groups_70_10_10_10",
        "group_normalization": "NFC, grouping keys only; source and reference preserved",
        "source_file_sha256": sha256(data / "Source.txt"),
        "restore_file_sha256": sha256(data / "Restore.txt"),
        "records_sha256": sha256(corpus_dir / "records.jsonl"),
        "raw_line_pairs": len(raw), "rejected_line_pairs": len(rejected),
        "trusted_segments": len(rows), "connected_groups": len(grouped),
        "splits": splits,
        "cross_split_parent_overlap": 0, "cross_split_normalized_source_overlap": 0,
        "selection_contract": {
            "train": "all generative mappings, vocabularies, ngrams and neural weights",
            "dev": "context evidence/correction thresholds and neural early stopping/fallback thresholds",
            "calibration": "selector training and grouped out-of-fold C/threshold selection only",
            "test": "final evaluation after every parameter is frozen; no iterative tuning",
        },
        "limitations": ["Parent-line groups do not establish manuscript/collection independence because provenance is absent.", "A newly repartitioned corpus previously used in development is not a pristine external test dataset.", "Trusted segment pairing filters with reference SHAD count; this does not evaluate source-only segmentation or OCR."],
    }
    write_json(corpus_dir / "split_manifest.json", manifest)
    print(json.dumps({"stage": "prepare", "splits": {s: {k: v for k, v in spec.items() if k not in ['record_indices', 'group_ids']} for s, spec in splits.items()}, "groups": len(grouped)}, ensure_ascii=False), flush=True)


def load_corpus(args):
    root = args.output_dir / "corpus"
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    if sha256(root / "records.jsonl") != manifest["records_sha256"]:
        raise ValueError("Frozen corpus records have changed.")
    if args.seed != manifest['seed']:
        raise ValueError('Requested seed differs from frozen split seed.')
    rows = load_jsonl(root / "records.jsonl")
    assignments, parent_splits, source_splits, group_splits = {}, {}, {}, {}
    for split, specification in manifest['splits'].items():
        groups = set()
        for index in specification['record_indices']:
            if not isinstance(index, int) or index < 0 or index >= len(rows) or index in assignments:
                raise ValueError('Frozen split indices are invalid or overlap.')
            assignments[index] = split
            row = rows[index]
            groups.add(row['group_id'])
            for identity, registry in ((row['parent_line_id'], parent_splits), (unicodedata.normalize('NFC', row['source']), source_splits), (row['group_id'], group_splits)):
                if identity in registry and registry[identity] != split:
                    raise ValueError('Frozen split violates parent/source/group independence.')
                registry[identity] = split
        if groups != set(specification['group_ids']):
            raise ValueError('Frozen split group metadata disagrees with records.')
    if set(assignments) != set(range(len(rows))):
        raise ValueError('Frozen splits do not cover the corpus exactly once.')
    raw_dir = getattr(args, 'raw_data_dir', None)
    if raw_dir is not None:
        for filename, hash_key in [('Source.txt', 'source_file_sha256'), ('Restore.txt', 'restore_file_sha256')]:
            original_file = Path(raw_dir) / filename
            if original_file.exists() and sha256(original_file) != manifest[hash_key]:
                raise ValueError('Original data changed after the split was frozen.')
    pairs = {s: [(rows[i]["source"], rows[i]["reference"]) for i in spec["record_indices"]] for s, spec in manifest["splits"].items()}
    return manifest, rows, pairs


def context_state(mapping, contexts, vocab, parameters):
    return {
        "model_type": "contextual_char_transducer_plus_syllable_vocabulary_corrector",
        "parameters": parameters, "base_mapping": mapping,
        "context_counts": [{"key": list(key), "outputs": dict(counter)} for key, counter in sorted(contexts.items(), key=lambda x: repr(x[0]))],
        "syllable_counts": dict(vocab["counts"]),
        "syllable_delimiters": {key: dict(val) for key, val in vocab["delimiters"].items()},
    }


def restore_context_state(payload):
    counts = Counter(payload["syllable_counts"])
    by_length = defaultdict(list)
    for key in counts: by_length[len(key)].append(key)
    vocab = {"counts": counts, "delimiters": {key: Counter(val) for key, val in payload["syllable_delimiters"].items()}, "by_length": dict(by_length), "total": sum(counts.values())}
    return payload["base_mapping"], {tuple(item["key"]): Counter(item["outputs"]) for item in payload["context_counts"]}, vocab, payload["parameters"]


def train_contextual(args, pairs):
    import train_contextual_corrector as ctx
    from train_transducer import learn_char_mapping
    from tibetan_syllable import build_syllable_vocabulary
    started = time.perf_counter()
    mapping, _ = learn_char_mapping(pairs['train'], iterations=5, max_output_length=8)
    contexts = ctx.build_context_counts(pairs['train'], mapping, max_output_length=8)
    vocab = build_syllable_vocabulary(pairs['train'])
    tuning, best = [], None
    for min_count in (1, 2, 3, 5):
        for min_confidence in (.5, .6, .7):
            for max_edit_distance in (0, 1):
                params = dict(min_count=min_count, min_confidence=min_confidence, max_edit_distance=max_edit_distance, max_syllable_chars=8)
                rows, metrics = ctx.evaluate_setting(pairs['dev'], mapping, contexts, vocab, **params)
                tuning.append(params | {"dev_micro_cer": metrics['micro_cer'], "dev_exact_match": metrics['exact_match']})
                key = ctx.tuning_key(metrics, min_count)
                if best is None or key < best['key']: best = dict(key=key, parameters=params, rows=rows, metrics=metrics)
    assert best is not None
    run = args.output_dir / "contextual"
    state = context_state(mapping, contexts, vocab, best['parameters'])
    write_json(run / "model_full.json", state)
    write_jsonl(run / "dev_predictions.jsonl", best['rows'])
    calib = ctx.predict_rows(pairs['calibration'], mapping, contexts, vocab, **best['parameters'])
    write_jsonl(run / "calibration_predictions.jsonl", calib)
    write_json(run / 'training_summary.json', {"training_seconds": time.perf_counter() - started, "iterations": 5, "max_output_length": 8, "best_parameters": best['parameters'], "tuning_on": "dev", "tuning": tuning, "dev_metrics": best['metrics'], "calibration_predictions_for_selector_only": len(calib), "train_samples": len(pairs['train'])})
    print(json.dumps({"stage": "contextual", "parameters": best['parameters'], "dev_cer": best['metrics']['micro_cer'], "seconds": time.perf_counter() - started}), flush=True)


def fast_decode_batches(model, loader, id_to_label, device, global_counts, context_counts, confidence_threshold, rule_min_count, rule_min_confidence):
    """Identical argmax/rules, one CPU copy per batch instead of per-position sync."""
    import train_edit_tagger as tagger
    torch, _ = tagger.require_torch()
    model.eval()
    rows = []
    with torch.no_grad():
        for chars, _, sources, refs in loader:
            confidence, label_ids = model(chars.to(device)).softmax(-1).max(-1)
            confidence, label_ids = confidence.cpu().tolist(), label_ids.cpu().tolist()
            for j, (source, reference) in enumerate(zip(sources, refs)):
                pieces, decisions = [], []
                for i, char in enumerate(source):
                    score = float(confidence[j][i])
                    if score < confidence_threshold:
                        output = tagger.rule_output_at(source, i, global_counts, context_counts, min_count=rule_min_count, min_confidence=rule_min_confidence)
                        origin = "rule_fallback"
                    else:
                        output = tagger.label_to_output(char, id_to_label[label_ids[j][i]])
                        origin = "neural"
                    pieces.append(output)
                    decisions.append(dict(index=i, source=char, output=output, confidence=score, origin=origin))
                rows.append(dict(source=source, reference=reference, raw_prediction=''.join(pieces), decisions=decisions))
    return rows


def train_tagger(args, pairs):
    import train_edit_tagger as tagger
    started = time.perf_counter()
    original_split, original_argv = tagger.split_pairs, sys.argv
    original_require = tagger.require_torch
    tagger.require_torch = clean_require_torch
    tagger.split_pairs = lambda *a, **kw: (pairs['train'], pairs['dev'], pairs['calibration'])
    original_decode = tagger.decode_batches
    tagger.decode_batches = fast_decode_batches
    run = args.output_dir / 'edit_tagger'
    run.mkdir(parents=True, exist_ok=True)
    model_seed = args.model_seed if args.model_seed is not None else args.seed
    sys.argv = [str(Path(tagger.__file__)), '--data-dir', str(args.output_dir/'corpus'), '--output-dir', str(run), '--seed', str(model_seed), '--epochs', str(args.epochs), '--patience', str(args.patience), '--batch-size', str(args.batch_size), '--amp']
    try:
        tagger.main()
    finally:
        sys.argv = original_argv
        tagger.split_pairs = original_split
        tagger.decode_batches = original_decode
        tagger.require_torch = original_require
    (run/'valid_predictions.jsonl').rename(run/'dev_predictions.jsonl')
    (run/'test_predictions.jsonl').rename(run/'calibration_predictions.jsonl')
    payload = json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    # The original entry point calls its second held-out split "test". It was
    # explicitly supplied calibration, so retain the metric only as calibration.
    payload['dev_samples'] = payload.pop('valid_samples')
    payload['calibration_samples'] = payload.pop('test_samples')
    payload['metrics']['dev'] = payload['metrics'].pop('valid')
    payload['metrics']['calibration'] = payload['metrics'].pop('test')
    payload['protocol'] = 'train weights; dev early stopping and tuning; calibration predictions only; final test unvisited'
    payload['training_seconds'] = time.perf_counter()-started
    payload['training_args'] = dict(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, lr=.0003, weight_decay=.01, label_smoothing=.03, gradient_clip=1., seed=model_seed, split_seed=args.seed, amp=True)
    write_json(run/'metrics.json', payload)
    print(json.dumps({'stage':'tagger_complete','best_epoch':payload['best_epoch'],'dev_cer':payload['metrics']['dev']['micro_cer'],'seconds':payload['training_seconds']}),flush=True)


def train_candidates(args):
    _, _, pairs = load_corpus(args)
    train_contextual(args, pairs)
    train_tagger(args, pairs)


def validate_alignment(primary, auxiliary, expected):
    if len(primary) != len(auxiliary) or len(primary) != len(expected): raise ValueError('Candidate count differs from frozen split.')
    for i, (p, a, pair) in enumerate(zip(primary, auxiliary, expected)):
        if (p['source'],p['reference']) != pair or (a['source'],a['reference']) != pair: raise ValueError(f'Candidate pair alignment mismatch at index {i}.')


def selector_resources(pairs):
    from train_hybrid_selector import CharacterLanguageModel
    from tibetan_syllable import split_syllable_units
    return CharacterLanguageModel(r for _,r in pairs['train']), Counter(u.core for _,r in pairs['train'] for u in split_syllable_units(r) if u.core)


def train_selector(args):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from train_hybrid_selector import selector_features, build_rows
    from train_restore import levenshtein
    from evaluate_restore import compute_metrics
    manifest, records, pairs = load_corpus(args)
    p=load_jsonl(args.output_dir/'contextual/calibration_predictions.jsonl')
    a=load_jsonl(args.output_dir/'edit_tagger/calibration_predictions.jsonl')
    validate_alignment(p,a,pairs['calibration'])
    lm, vocabulary = selector_resources(pairs)
    features=np.asarray([selector_features(x,y,lm,vocabulary) for x,y in zip(p,a)],dtype=np.float64)
    labels=np.asarray([levenshtein(y['prediction'],x['reference'])<levenshtein(x['prediction'],x['reference']) for x,y in zip(p,a)],dtype=np.int64)
    groups=np.asarray([records[i]['group_id'] for i in manifest['splits']['calibration']['record_indices']])
    if len(np.unique(labels))!=2 or np.bincount(labels).min()<5: raise ValueError('Calibration labels lack sufficient support for 5-fold selection.')
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=args.seed)
    fold_indices=list(cv.split(features,labels,groups))
    for tr,va in fold_indices:
        assert not set(groups[tr]) & set(groups[va])
        if len(np.unique(labels[tr]))!=2: raise ValueError('A grouped calibration fold has only one class.')
    best=None
    tuning=[]
    for c in (.01,.03,.1,.3,1.,3.,10.):
        probs=np.zeros(len(labels),dtype=float)
        for tr,va in fold_indices:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=2000,random_state=args.seed))
            model.fit(features[tr],labels[tr]); probs[va]=model.predict_proba(features[va])[:,1]
        for threshold in (.35,.40,.45,.50,.55,.60,.65,.70):
            chosen=probs>=threshold
            rows=build_rows(p,a,chosen,probs)
            metrics=compute_metrics(rows)
            key=(round(metrics['micro_cer'],12),-metrics['exact_match'])
            tuning.append(dict(C=c,threshold=threshold,calibration_oof_cer=metrics['micro_cer'],calibration_oof_exact=metrics['exact_match']))
            if best is None or key<best['key']: best=dict(key=key,C=c,threshold=threshold,rows=rows,metrics=metrics)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=best['C'],max_iter=2000,random_state=args.seed)).fit(features,labels)
    scaler, clf=model.steps[0][1],model.steps[1][1]
    state=dict(model_type='independent_calibration_candidate_selector',C=best['C'],threshold=best['threshold'],scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),coef=clf.coef_[0].tolist(),intercept=float(clf.intercept_[0]),features=features.shape[1],positive_label='auxiliary strictly lower reference edit distance; ties label primary',selection_protocol='five-fold StratifiedGroupKFold on independent calibration; final refit calibration',train_reference_resources_only=True)
    state.update(format_version=1,feature_count=features.shape[1],coefficients=clf.coef_[0].tolist(),language_model={'max_order':lm.max_order,'alpha':lm.alpha,'alphabet_size':lm.alphabet_size,'counts':{str(order):dict(counter) for order,counter in lm.counts.items()},'context_counts':{str(order):dict(counter) for order,counter in lm.context_counts.items()}},vocabulary=list(vocabulary))
    run=args.output_dir/'selector'
    write_json(run/'selector_model.json',state)
    write_jsonl(run/'calibration_oof_predictions.jsonl',best['rows'])
    write_json(run/'selection_summary.json',dict(best_C=best['C'],best_threshold=best['threshold'],calibration_samples=len(labels),class_counts=np.bincount(labels).tolist(),calibration_group_count=len(set(groups)),folds=[dict(train=len(tr),heldout=len(va),heldout_groups=len(set(groups[va]))) for tr,va in fold_indices],calibration_oof_metrics=best['metrics'],tuning=tuning,all_selected_before_test=True))
    print(json.dumps({'stage':'selector_frozen','C':best['C'],'threshold':best['threshold'],'calibration_oof_cer':best['metrics']['micro_cer'],'features':features.shape[1]}),flush=True)


def tagger_test_rows(args,pairs):
    import train_edit_tagger as tagger
    tagger.require_torch = clean_require_torch
    from tibetan_syllable import build_syllable_vocabulary
    from torch.utils.data import DataLoader
    torch,_=tagger.require_torch()
    requested_device = getattr(args, 'device', 'auto')
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device)
    run=args.output_dir/'edit_tagger'
    info=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    checkpoint=torch.load(run/'best.pt',map_location=device,weights_only=False)
    config=tagger.EditTaggerConfig(**checkpoint['config'])
    model=tagger.create_model(config).to(device); model.load_state_dict(checkpoint['model_state'])
    char_to_id={c:i for i,c in enumerate(checkpoint['char_vocab'])}
    label_to_id={l:i for i,l in enumerate(checkpoint['labels'])}
    labelled_train,_=tagger.build_labelled_pairs(pairs['train'])
    global_counts,context_counts=tagger.build_rule_counts(labelled_train)
    vocab=build_syllable_vocabulary(pairs['train'])
    # No target alignment is needed for inference. Dummy copy labels satisfy
    # dataset shape; the reference text is carried solely for later evaluation.
    labelled_test=[(source,reference,[tagger.COPY_LABEL]*len(source)) for source,reference in pairs['test']]
    loader=DataLoader(tagger.EditDataset(labelled_test,char_to_id,label_to_id,config.max_positions),batch_size=args.batch_size,shuffle=False,num_workers=0,collate_fn=tagger.make_collate(config.pad_char_id,config.pad_label_id))
    raw=fast_decode_batches(model,loader,checkpoint['labels'],device,global_counts,context_counts,info['best_confidence_threshold'],3,.5)
    return tagger.postprocess_rows(raw,vocab,info['best_max_edit_distance'])


def evaluate(args):
    import numpy as np
    import train_contextual_corrector as ctx
    from train_hybrid_selector import selector_features,build_rows
    from train_transducer import apply_mapping
    from tibetan_syllable import apply_syllable_vocabulary_correction
    from train_syllable_corrector import terminal_shad
    from evaluate_restore import compute_metrics
    from train_restore import levenshtein
    manifest,records,pairs=load_corpus(args)
    dest=args.output_dir/'evaluation'
    if (dest/'test_metrics.json').exists(): raise RuntimeError('Final test was already evaluated. Frozen result exists; use it rather than iterating test selection.')
    state=json.loads((args.output_dir/'contextual/model_full.json').read_text(encoding='utf-8'))
    mapping,contexts,vocab,params=restore_context_state(state)
    selector=json.loads((args.output_dir/'selector/selector_model.json').read_text(encoding='utf-8'))
    p=ctx.predict_rows(pairs['test'],mapping,contexts,vocab,**params)
    a=tagger_test_rows(args,pairs)
    validate_alignment(p,a,pairs['test'])
    lm,vocabulary=selector_resources(pairs)
    features=np.asarray([selector_features(x,y,lm,vocabulary) for x,y in zip(p,a)],dtype=float)
    standardized=(features-np.asarray(selector['scaler_mean']))/np.asarray(selector['scaler_scale'])
    logits=standardized@np.asarray(selector['coef'])+selector['intercept']
    probabilities=1/(1+np.exp(-np.clip(logits,-700,700)))
    chosen=probabilities>=selector['threshold']
    h=build_rows(p,a,chosen,probabilities)
    models={
        'identity':[r|{'prediction':r['source']} for r in p],
        'global_transducer':[r|{'prediction':apply_mapping(r['source'],mapping)} for r in p],
        'global_plus_shad':[r|{'prediction':terminal_shad(apply_mapping(r['source'],mapping))} for r in p],
        'global_plus_shad_vocab':[r|{'prediction':apply_syllable_vocabulary_correction(terminal_shad(apply_mapping(r['source'],mapping)),vocab,max_edit_distance=params['max_edit_distance'],max_syllable_chars=8)} for r in p],
        'context_raw':[r|{'prediction':r['raw_prediction']} for r in p],
        'context_plus_shad':[r|{'prediction':r['base_prediction']} for r in p],
        'contextual':p,
        'edit_raw':[r|{'prediction':r['raw_prediction']} for r in a],
        'edit_plus_shad':[r|{'prediction':r['shad_prediction']} for r in a],
        'edit_tagger':a,
        'hybrid':h,
    }
    metrics={}
    for name,rows in models.items():
        metrics[name]=compute_metrics(rows)
        metrics[name]['total_reference_codepoints']=sum(len(r['reference']) for r in rows)
        metrics[name]['total_edit_distance']=sum(levenshtein(r['prediction'],r['reference']) for r in rows)
        metrics[name]['exact_count']=sum(r['prediction']==r['reference'] for r in rows)
        write_jsonl(dest/f'{name}_test_predictions.jsonl',rows)
    pd=np.asarray([levenshtein(r['prediction'],r['reference']) for r in p]); ad=np.asarray([levenshtein(r['prediction'],r['reference']) for r in a]); hd=np.asarray([levenshtein(r['prediction'],r['reference']) for r in h])
    choice_summary=dict(test_samples=len(p),selected_auxiliary=int(chosen.sum()),selected_better=int(((ad<pd)&chosen).sum()),selected_worse=int(((ad>pd)&chosen).sum()),selected_tie=int(((ad==pd)&chosen).sum()),candidate_disagreements=sum(x['prediction']!=y['prediction'] for x,y in zip(p,a)),auxiliary_better_total=int((ad<pd).sum()),net_saved_edits=int((pd-hd).sum()),absolute_cer_reduction=metrics['contextual']['micro_cer']-metrics['hybrid']['micro_cer'],relative_cer_reduction=(metrics['contextual']['micro_cer']-metrics['hybrid']['micro_cer'])/metrics['contextual']['micro_cer'])
    # Paired bootstrap resamples independent connected parent groups, not rows.
    group_ids=[records[i]['group_id'] for i in manifest['splits']['test']['record_indices']]
    grouped=defaultdict(list)
    for i,g in enumerate(group_ids): grouped[g].append(i)
    group_values=np.asarray([[sum(pd[i]-hd[i] for i in ids),sum(len(p[i]['reference']) for i in ids),sum(int(h[i]['prediction']==h[i]['reference'])-int(p[i]['prediction']==p[i]['reference']) for i in ids),len(ids)] for ids in grouped.values()],dtype=float)
    rng=np.random.default_rng(args.seed)
    estimates=[]
    for _ in range(5000):
        sums=group_values[rng.integers(0,len(group_values),len(group_values))].sum(axis=0)
        estimates.append([sums[0]/sums[1],sums[2]/sums[3]])
    intervals=np.quantile(np.asarray(estimates),[.025,.975],axis=0)
    choice_summary.update(bootstrap_iterations=5000,bootstrap_seed=args.seed,bootstrap_unit='NFC-source-linked parent-line connected group',bootstrap_groups=len(grouped),absolute_cer_reduction_ci95=intervals[:,0].tolist(),exact_match_gain_ci95=intervals[:,1].tolist(),bootstrap_scope='conditional on one split/model seed; excludes training variability and external generalization')
    result=dict(protocol='corrected_grouped_train_dev_calibration_test',data={s:{k:v for k,v in spec.items() if k not in ['record_indices','group_ids']} for s,spec in manifest['splits'].items()},test_metrics=metrics,selector_effect=choice_summary,limitations=manifest['limitations'])
    write_json(dest/'test_metrics.json',result)
    print(json.dumps({'stage':'final_test_complete','models':{k:dict(cer=v['micro_cer'],exact=v['exact_match'],tibetan_unit_error_rate=v['micro_tibetan_unit_error_rate']) for k,v in metrics.items()},'selector_effect':choice_summary}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--raw-data-dir',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','train-candidates','train-tagger','train-selector','evaluate','all'],default='all')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--model-seed',type=int,help='Neural weight/dropout seed; split and grouped calibration seed remain --seed.')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--patience',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args()
    args.code_root=args.code_root.resolve();args.output_dir=args.output_dir.resolve()
    args.raw_data_dir=(args.raw_data_dir or args.code_root/'data').resolve()
    sys.path.insert(0,str(args.code_root))
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    import importlib.metadata
    def installed_version(distribution):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None
    environment = dict(
        python=platform.python_version(),
        dependencies={name: installed_version(name) for name in ['torch', 'numpy', 'scikit-learn', 'scipy', 'joblib', 'threadpoolctl']},
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_code_sha256={f: sha256(args.code_root/f) for f in ['train_transducer.py', 'train_contextual_corrector.py', 'train_edit_tagger.py', 'train_hybrid_selector.py', 'tibetan_syllable.py', 'evaluate_restore.py']},
        driver_sha256=sha256(Path(__file__)),
    )
    if args.stage in ['train-candidates', 'train-tagger', 'evaluate', 'all']:
        torch, _ = clean_require_torch()
        torch.set_num_threads(8)
        environment['dependencies']['torch'] = torch.__version__
        environment.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, cpu_threads=torch.get_num_threads())
    write_json(args.output_dir/f'environment_{args.stage}.json', environment)
    write_json(args.output_dir/'environment.json', environment)
    started=time.perf_counter()
    if args.stage in ['prepare','all']:prepare(args)
    if args.stage in ['train-candidates','all']:train_candidates(args)
    if args.stage=='train-tagger':
        _,_,pairs=load_corpus(args)
        train_tagger(args,pairs)
    if args.stage in ['train-selector','all']:train_selector(args)
    if args.stage in ['evaluate','all']:evaluate(args)
    print(json.dumps({'stage':'complete','seconds':time.perf_counter()-started}),flush=True)


if __name__=='__main__':main()
