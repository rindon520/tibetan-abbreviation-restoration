"""Matched character Transformer baseline on a frozen grouped corpus.

Vocabulary uses training pairs only. Development loss selects the checkpoint;
test pairs never affect vocabulary, optimisation or model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def greedy_batch(model, tokenizer, pairs, device, max_length=256, batch_size=64):
    import torch
    # Near-tied logits in this small unpretrained model can change argmax with
    # batch shape under TF32. Decode the frozen checkpoint in full FP32.
    torch.backends.cuda.matmul.allow_tf32 = False
    from train_restore import pad_sequences
    rows = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start+batch_size]
            src = pad_sequences([tokenizer.encode(s, add_eos=True) for s, _ in batch], tokenizer.pad_id).to(device)
            mask = src.eq(tokenizer.pad_id)
            memory = model.encode(src, mask)
            target = torch.full((len(batch), 1), tokenizer.bos_id, device=device, dtype=torch.long)
            finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
            for _ in range(max_length):
                hidden = model.decode(memory, mask, target)
                next_ids = model.output(hidden[:, -1]).argmax(dim=-1)
                next_ids = torch.where(finished, tokenizer.eos_id, next_ids)
                target = torch.cat([target, next_ids[:, None]], dim=1)
                finished |= next_ids.eq(tokenizer.eos_id)
                if bool(finished.all()): break
            for (source, reference), tokens in zip(batch, target.cpu().tolist()):
                eos = tokens.index(tokenizer.eos_id) if tokenizer.eos_id in tokens else len(tokens)
                rows.append({'source': source, 'reference': reference, 'prediction': tokenizer.decode(tokens[:eos])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--corpus-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo_root))
    from train_restore import (CharTokenizer, ModelConfig, PairDataset, build_vocab, create_model,
        make_collate_fn, make_scheduler, set_seed, train_one_epoch, evaluate_loss)
    from evaluate_restore import compute_metrics
    import torch
    from torch.utils.data import DataLoader
    set_seed(args.seed)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifest_path = args.corpus_dir/'split_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records_path = args.corpus_dir / 'records.jsonl'
    if hashlib.sha256(records_path.read_bytes()).hexdigest() != manifest['records_sha256']:
        raise ValueError('Frozen corpus records have changed.')
    records = [json.loads(x) for x in (args.corpus_dir/'records.jsonl').read_text(encoding='utf-8').splitlines() if x]
    pairs = {name: [(records[i]['source'], records[i]['reference']) for i in split['record_indices']]
             for name, split in manifest['splits'].items()}
    stoi, itos = build_vocab(args.output_dir/'__no_external_vocabulary__', pairs['train'])
    tokenizer = CharTokenizer(stoi, itos)
    config = ModelConfig(vocab_size=len(itos), pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id,
                         eos_id=tokenizer.eos_id, d_model=256, layers=4, heads=8, ff_dim=1024,
                         dropout=.1, max_positions=260)
    model = create_model(config).to(device)
    loaders = {name: DataLoader(PairDataset(pairs[name], tokenizer, 256, 256), batch_size=args.batch_size,
                    shuffle=(name=='train'), collate_fn=make_collate_fn(tokenizer.pad_id), num_workers=0)
               for name in ['train','dev']}
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    scheduler = make_scheduler(optimizer, 300, args.epochs*len(loaders['train']))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type=='cuda'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = {'config': asdict(config), 'seed': args.seed, 'epochs_max': args.epochs, 'patience': args.patience,
            'device': str(device), 'batch_size': args.batch_size, 'lr': 3e-4, 'weight_decay': .01,
            'warmup_steps': 300, 'checkpoint_criterion': 'full_development_token_cross_entropy',
            'decoder': 'batched_greedy', 'max_source_chars': 255, 'max_output_chars': 256,
            'training_tf32': True, 'inference_tf32': False,
            'vocabulary_scope': 'train_only', 'split_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'training_truncation_source': sum(len(s)+1>256 for s,r in pairs['train']),
            'training_truncation_target': sum(len(r)+1>256 for s,r in pairs['train'])}
    dump(args.output_dir/'run_info.json', info)
    history=[]
    best=float('inf'); stale=0; started=time.time()
    for epoch in range(1,args.epochs+1):
        begin=time.time()
        loss=train_one_epoch(model,loaders['train'],optimizer,scheduler,scaler,device,tokenizer.pad_id,True,1.)
        valid=evaluate_loss(model,loaders['dev'],device,tokenizer.pad_id)
        row={'epoch':epoch,'train_loss':loss,'dev_loss':valid,'seconds':time.time()-begin}
        history.append(row); print(json.dumps(row),flush=True)
        if valid<best:
            best=valid;stale=0
            torch.save({'model_state':model.state_dict(),'config':asdict(config),'model_config':asdict(config),'stoi':stoi,'itos':itos,
                        'epoch':epoch,'dev_loss':valid},args.output_dir/'best.pt')
        else:stale+=1
        dump(args.output_dir/'history.json',history)
        if stale>=args.patience:break
    state=torch.load(args.output_dir/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model_state'])
    results={'checkpoint_epoch':state['epoch'],'elapsed_seconds':time.time()-started}
    for split in ['dev','test']:
        rows=greedy_batch(model,tokenizer,pairs[split],device)
        for row, idx in zip(rows,manifest['splits'][split]['record_indices']):
            row.update({k:records[idx][k] for k in ['record_id','parent_line_id','group_id']})
        (args.output_dir/f'{split}_predictions.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
        results[split]=compute_metrics(rows)
        print(json.dumps({'split':split,'metrics':results[split]},ensure_ascii=False),flush=True)
    dump(args.output_dir/'metrics.json',results)


if __name__=='__main__':main()
