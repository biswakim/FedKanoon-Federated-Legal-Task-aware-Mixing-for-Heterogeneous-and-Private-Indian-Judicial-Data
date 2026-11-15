"""Evaluate an existing FEDLEGAL-India baseline checkpoint, without retraining.

Checkpoints contain ONLY trainable weights. Reconstruct the SAME initial pretrained
backbone and LoRA config using config.json before loading those weights.
"""
import argparse
import json
from pathlib import Path
import torch
import numpy as np
from .train import set_seed, ToyTokenizer, make_loaders, run_epoch, save_csv
from .data import COURTS, load_frame
from .model import TaskModel


def evaluate(checkpoint_path, data_override=None, device_override=None, output_override=None):
    checkpoint_path=Path(checkpoint_path)
    run_dir=checkpoint_path.parent
    config_path=run_dir/'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing {config_path}. Needed to reconstruct model, split and preprocessing.')
    opts=json.loads(config_path.read_text())
    if data_override: opts['data']=str(data_override)
    if device_override: opts['device']=device_override
    from argparse import Namespace
    args=Namespace(**opts)
    if args.toy and args.device.startswith('cuda'): args.device='cpu'
    set_seed(args.seed)
    frame,labels=load_frame(args.data,args.court_col,args.text_col,
        {'duration':args.duration_col,'case_type':args.type_col,'disposal':args.disposal_col}[args.task],
        args.date_col,args.task,args.split_col,args.text_already_masked)
    if args.toy:
        tokenizer=ToyTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model)
    loaders=make_loaders(frame,tokenizer,args)
    payload=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if payload['task']!=args.task or payload['labels']!=labels:
        raise ValueError('Checkpoint task/labels do not match run configuration or dataset')
    model=TaskModel(args.model,args.task,len(labels) if labels else 1,args.lora,args.toy).to(args.device)
    model.load_trainable(payload['trainable_state'])
    rows=[]
    for court in COURTS:
        metrics=run_epoch(model,loaders[court]['test'],None,args.device,args.task)
        rows.append({'court':court,'task':args.task,'seed':args.seed,'algorithm':args.algorithm,
                     'best_round':payload['round'],**metrics})
    target=Path(output_override) if output_override else run_dir
    target.mkdir(parents=True,exist_ok=True)
    save_csv(target/'per_court_test.csv',rows)
    means={key:float(np.mean([r[key] for r in rows if r[key] is not None]))
           for key in ('maelog','mae_days','spearman','f1_4','macro_f1','accuracy') if key in rows[0]}
    summary={'task':args.task,'algorithm':args.algorithm,'seed':args.seed,'best_round':payload['round'],**means}
    if args.task=='duration':
        summary['worst_court_maelog']=max(r['maelog'] for r in rows)
    else:
        summary['worst_court_macro_f1']=min(r['macro_f1'] for r in rows)
    (target/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'checkpoint':str(checkpoint_path),'output':str(target),**summary},indent=2))
    return summary


def main():
    p=argparse.ArgumentParser(description='Evaluate saved baseline best.pt on same court test splits; no training')
    p.add_argument('--checkpoint',required=True,help='outputs/<task>/seed_<seed>/best.pt')
    p.add_argument('--data',help='Override saved dataset path; MUST be same records/splits')
    p.add_argument('--device',help='e.g. cuda or cpu')
    p.add_argument('--output',help='Optional output directory; default checkpoint run directory')
    a=p.parse_args()
    evaluate(a.checkpoint,a.data,a.device,a.output)

if __name__=='__main__': main()
