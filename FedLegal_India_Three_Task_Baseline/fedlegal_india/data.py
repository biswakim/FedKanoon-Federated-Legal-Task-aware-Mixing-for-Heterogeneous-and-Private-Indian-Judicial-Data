"""Explicit court-wise dataset loader. Never silently infer target columns."""
import json
import re
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

COURTS = ('Bombay', 'Delhi', 'Karnataka', 'Kerala', 'Madhya Pradesh')
TYPES = ('writ', 'bail', 'criminal', 'civil', 'contempt', 'arbitration', 'review', 'tax', 'motor accident', 'other')
DISPOSALS = ('allowed', 'dismissed', 'disposed', 'withdrawn')


def normalize_court(x):
    s = re.sub(r'[^a-z]', '', str(x).lower())
    d = {'bombay':'Bombay', 'delhi':'Delhi', 'karnataka':'Karnataka',
         'kerala':'Kerala','madhyapradesh':'Madhya Pradesh','mp':'Madhya Pradesh',
         'madhyapradeshhighcourt':'Madhya Pradesh'}
    s = s.replace('highcourt', '')
    if s not in d:
        raise ValueError(f'Unknown court {x!r}; configure --court-col / source data')
    return d[s]


def mask_text(s):
    """Conservative text masking; audit against preprocessing from original project."""
    s = str(s)
    s = re.sub(r'\b(?:allowed|dismissed|disposed\s+of|withdrawn|rejected|not\s+pressed|infructuous)\b', ' [OUTCOME] ', s, flags=re.I)
    s = re.sub(r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b', ' [DATE] ', s)
    s = re.sub(r'\b\d{4}\b', ' [YEAR] ', s)
    s = re.sub(r'\b\d{2,}\b', ' [NUMBER] ', s)
    return s


def load_frame(path, court_col, text_col, label_col, date_col, task, split_col='', text_already_masked=False):
    df = pd.read_parquet(path) if path.lower().endswith('.parquet') else pd.read_csv(path)
    required = [court_col, text_col, label_col] + ([date_col] if not split_col else [split_col])
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f'Missing required columns {absent}. Available: {list(df.columns)}')
    df = df.dropna(subset=[court_col, text_col, label_col]).copy()
    df['court'] = df[court_col].map(normalize_court)
    df['text'] = df[text_col].astype(str)
    if not text_already_masked:
        df['text'] = df['text'].map(mask_text)
    if task == 'duration':
        df['days'] = pd.to_numeric(df[label_col], errors='coerce')
        df = df[df['days'].notna() & df['days'].ge(0)].copy()
        df['target'] = np.log1p(df['days'].astype(float))
        labels = None
    else:
        labels = TYPES if task == 'case_type' else DISPOSALS
        mapping = {lab: i for i, lab in enumerate(labels)}
        vals = df[label_col].astype(str).str.lower().str.strip()
        vals = vals.replace({'disposed of':'disposed','motor_accident':'motor accident','motor-accident':'motor accident'})
        # 'unclear' is deliberately excluded for disposal only
        unknown = sorted(set(vals) - set(mapping))
        if unknown:
            print(f'[DATA] Dropping {sum(~vals.isin(mapping))} rows with unsupported {task} labels: {unknown[:15]}')
        df = df[vals.isin(mapping)].copy()
        df['target'] = vals.loc[df.index].map(mapping).astype(int)
    if not len(df):
        raise ValueError('No labeled rows remained after filtering')
    if split_col:
        df['split'] = df[split_col].astype(str).str.lower().str.strip().replace({'validation':'val','valid':'val'})
        invalid = set(df['split']) - {'train','val','test'}
        if invalid:
            raise ValueError(f'Invalid split labels {invalid}; expected train/val/test')
    else:
        df['_date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=False)
        if df['_date'].isna().any():
            raise ValueError(f'{df["_date"].isna().sum()} dates could not be parsed. Supply a precomputed --split-col to preserve exact original partitions.')
        df['split'] = ''
        for court, ix in df.groupby('court').groups.items():
            order = df.loc[ix].sort_values('_date', kind='stable').index
            n = len(order)
            a, b = int(.70*n), int(.85*n)
            df.loc[order[:a], 'split'] = 'train'
            df.loc[order[a:b], 'split'] = 'val'
            df.loc[order[b:], 'split'] = 'test'
    summary = df.groupby(['court','split']).size().unstack(fill_value=0)
    print('[DATA] Samples per court/split:\n', summary.to_string())
    for court in COURTS:
        if court not in set(df.court):
            raise ValueError(f'Court {court} missing')
        for sp in ('train','val','test'):
            if ((df.court == court) & (df.split == sp)).sum() == 0:
                raise ValueError(f'Empty {court} {sp} split')
    return df.reset_index(drop=True), labels


class TokenDataset(Dataset):
    def __init__(self, frame, tokenizer, max_len=256, trunc='headtail'):
        self.targets = frame.target.to_numpy()
        self.task = 'duration' if self.targets.dtype.kind == 'f' else 'classification'
        texts = frame.text.tolist()
        self.features = []
        for t in texts:
            ids = tokenizer.encode(t, add_special_tokens=False)
            budget = max_len - tokenizer.num_special_tokens_to_add(pair=False)
            if len(ids) > budget:
                if trunc == 'headtail':
                    h = budget // 2
                    ids = ids[:h] + ids[-(budget-h):]
                else:
                    ids = ids[:budget]
            ids = tokenizer.build_inputs_with_special_tokens(ids)
            if len(ids) > max_len:
                raise AssertionError('Tokenized sequence exceeds max_len')
            self.features.append(ids)
    def __len__(self): return len(self.targets)
    def __getitem__(self, i): return self.features[i], self.targets[i]


def collate(batch, pad):
    import torch
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, (tok, _) in enumerate(batch):
        ids[j, :len(tok)] = torch.tensor(tok)
        mask[j, :len(tok)] = 1
    targets = torch.tensor([y for _,y in batch])
    return ids, mask, targets
