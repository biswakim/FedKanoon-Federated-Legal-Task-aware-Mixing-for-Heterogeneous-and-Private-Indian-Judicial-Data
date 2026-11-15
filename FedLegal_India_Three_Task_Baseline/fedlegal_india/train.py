"""Simulated, full-participation federated training: independent task models.

NOTE: This is a FEDLEGAL-inspired adaptation. It does not invoke the original
FedLegal registry/trainer classes; see README for provenance and limitations.
"""
import argparse
import copy
import csv
import json
import random
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import COURTS, TokenDataset, collate, load_frame
from .model import TaskModel
from .metrics import evaluate_metrics

class ToyTokenizer:
    pad_token_id = 0
    def encode(self, t, add_special_tokens=False):
        import hashlib
        return [2+int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], 'little')%998 for w in t.split()]
    def num_special_tokens_to_add(self, pair=False): return 0
    def build_inputs_with_special_tokens(self, ids): return ids


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_loaders(frame, tokenizer, args):
    loaders={}
    for court in COURTS:
        loaders[court]={}
        for split in ('train','val','test'):
            sub=frame[(frame.court==court)&(frame.split==split)]
            ds=TokenDataset(sub,tokenizer,args.max_len,args.trunc)
            loaders[court][split]=DataLoader(ds,batch_size=args.batch_size,
                shuffle=split=='train',num_workers=0,collate_fn=partial(collate,pad=tokenizer.pad_token_id or 0))
    return loaders


def run_epoch(model, loader, opt, device, task, class_weight=None, reference=None, mu=0):
    train=opt is not None
    model.train(train)
    losses=[]; gold=[]; pred=[]
    for ids,mask,target in loader:
        ids,mask,target=ids.to(device),mask.to(device),target.to(device)
        with torch.set_grad_enabled(train):
            scores=model(ids,mask)
            loss=model.loss(scores,target,class_weight)
            if train and mu and reference is not None:
                # FedProx proximal penalty only for trainable parameters
                prox=sum((p-reference[n].to(device)).square().sum() for n,p in model.named_parameters() if p.requires_grad)
                loss=loss+mu*prox/2
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
                opt.step()
        losses.append((float(loss.item()),len(target)))
        gold.extend(target.detach().cpu().tolist())
        pred.extend((scores.squeeze(-1) if task=='duration' else scores.argmax(-1)).detach().cpu().tolist())
    metrics=evaluate_metrics(task,gold,pred)
    metrics['loss']=sum(a*b for a,b in losses)/max(1,sum(b for _,b in losses))
    return metrics


def weighted_average(states, sizes):
    total=sum(sizes)
    return {k:sum(s[k].float()*(n/total) for s,n in zip(states,sizes)).to(states[0][k].dtype) for k in states[0]}


def save_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(args):
    set_seed(args.seed)
    out=Path(args.output)/args.task/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    frame, labels=load_frame(args.data,args.court_col,args.text_col,
            {'duration':args.duration_col,'case_type':args.type_col,'disposal':args.disposal_col}[args.task],
            args.date_col,args.task,args.split_col,args.text_already_masked)
    if args.toy:
        tokenizer=ToyTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model)
    loaders=make_loaders(frame,tokenizer,args)
    model=TaskModel(args.model,args.task,len(labels) if labels else 1,args.lora,args.toy).to(args.device)
    # InLegalBERT frozen LoRA branch; independent task-specific linear head.
    global_state=model.trainable_state()
    best=float('inf'); bad=0; history=[]
    weights={c:len(loaders[c]['train'].dataset) for c in COURTS}
    class_weight={}
    for court in COURTS:
        if args.task=='disposal' and args.weight_disposal:
            ys=frame.loc[(frame.court==court)&(frame.split=='train'),'target'].to_numpy(dtype=int)
            cnt=np.bincount(ys,minlength=4)
            class_weight[court]=torch.tensor(len(ys)/(4*np.maximum(cnt,1)),device=args.device,dtype=torch.float32)
        else: class_weight[court]=None
    for rnd in range(1,args.rounds+1):
        states=[]; sizes=[]
        for court in COURTS:
            model.load_trainable(global_state)
            opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=args.weight_decay)
            for ep in range(args.local_epochs):
                run_epoch(model,loaders[court]['train'],opt,args.device,args.task,class_weight[court],global_state if args.algorithm=='fedprox' else None,args.mu if args.algorithm=='fedprox' else 0)
            states.append(model.trainable_state())
            sizes.append(weights[court])
        global_state=weighted_average(states,sizes)
        model.load_trainable(global_state)
        court_val={c:run_epoch(model,loaders[c]['val'],None,args.device,args.task,class_weight[c]) for c in COURTS}
        val_loss=sum(court_val[c]['loss']*len(loaders[c]['val'].dataset) for c in COURTS)/sum(len(loaders[c]['val'].dataset) for c in COURTS)
        history.append({'round':rnd,'validation_loss':val_loss,**{f'{c}_val_loss':court_val[c]['loss'] for c in COURTS}})
        print(f'[{args.task} seed={args.seed}] round={rnd} val_loss={val_loss:.4f}',flush=True)
        if val_loss < best-1e-7:
            best=val_loss;bad=0
            torch.save({'trainable_state':global_state,'task':args.task,'labels':labels,'round':rnd},out/'best.pt')
        else:
            bad+=1
            if bad>=args.patience:break
    save_csv(out/'history.csv',history)
    checkpoint=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_trainable(checkpoint['trainable_state'])
    rows=[]
    for court in COURTS:
        metrics=run_epoch(model,loaders[court]['test'],None,args.device,args.task)
        rows.append({'court':court,'task':args.task,'seed':args.seed,'algorithm':args.algorithm,
                    'best_round':checkpoint['round'],**metrics})
    save_csv(out/'per_court_test.csv',rows)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))
    means={k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in rows[0] if k in ('maelog','mae_days','spearman','f1_4','macro_f1','accuracy')}
    summary={'task':args.task,'algorithm':args.algorithm,'seed':args.seed,'best_round':checkpoint['round'],**means}
    if args.task=='duration':summary['worst_court_maelog']=max(r['maelog'] for r in rows)
    else:summary['worst_court_macro_f1']=min(r['macro_f1'] for r in rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


def cli():
    p=argparse.ArgumentParser(description='FEDLEGAL-inspired independent task FL on Indian courts')
    p.add_argument('--data',required=True)
    p.add_argument('--task',choices=['duration','case_type','disposal'],required=True)
    p.add_argument('--court-col',default='court')
    p.add_argument('--text-col',default='text')
    p.add_argument('--date-col',default='decision_date')
    p.add_argument('--split-col',default='',help='Prefer exact original project train/val/test column, if present')
    p.add_argument('--duration-col',default='duration_days')
    p.add_argument('--type-col',default='case_type')
    p.add_argument('--disposal-col',default='disposal')
    p.add_argument('--text-already-masked',action='store_true',help='Use if CSV text matches your original project masking')
    p.add_argument('--model',default='law-ai/InLegalBERT')
    p.add_argument('--lora',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--trunc',choices=['headtail','head'],default='headtail')
    p.add_argument('--max-len',type=int,default=256)
    p.add_argument('--algorithm',choices=['fedavg','fedprox'],default='fedavg')
    p.add_argument('--mu',type=float,default=0.01)
    p.add_argument('--rounds',type=int,default=30)
    p.add_argument('--local-epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--weight-decay',type=float,default=0.01)
    p.add_argument('--patience',type=int,default=4)
    p.add_argument('--weight-disposal',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--output',default='outputs')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--toy',action='store_true',help='Smoke-test ONLY: random encoder; no Hugging Face model')
    args=p.parse_args()
    if args.toy and args.device=='cuda':args.device='cpu'
    if not args.toy and not args.lora: print('WARNING: full backbone training; not matched to original LoRA budget')
    run(args)

if __name__=='__main__':cli()
