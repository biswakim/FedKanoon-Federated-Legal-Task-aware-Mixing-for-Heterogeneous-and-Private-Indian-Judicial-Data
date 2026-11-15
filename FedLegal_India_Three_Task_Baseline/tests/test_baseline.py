import numpy as np
import pandas as pd
import torch
from fedlegal_india.data import load_frame, COURTS
from fedlegal_india.model import TaskModel
from fedlegal_india.train import weighted_average
from fedlegal_india.metrics import evaluate_metrics

def test_aggregation():
    out=weighted_average([{'x':torch.tensor([1.])},{'x':torch.tensor([3.])}],[1,3])
    assert torch.allclose(out['x'],torch.tensor([2.5]))

def test_metrics():
    assert evaluate_metrics('duration',[0,1],[0,1])['maelog']==0
    assert evaluate_metrics('case_type',[0,1],[0,1])['macro_f1']==0.2

def test_toy_model():
    m=TaskModel('', 'duration',1,toy=True)
    x=torch.tensor([[1,2,0],[2,0,0]])
    y=m(x,(x!=0).long())
    assert y.shape==(2,1)
    y.mean().backward()
    assert m.head.weight.grad is not None

def test_splits(tmp_path):
    rows=[]
    for court in COURTS:
        for i in range(12):
            rows.append({'court':court,'text':'case facts example', 'duration_days':i+1,
                'case_type':'writ','disposal':'allowed','decision_date':f'2024-01-{i+1:02d}'})
    path=tmp_path/'toy.csv'; pd.DataFrame(rows).to_csv(path,index=False)
    for task,column in [('duration','duration_days'),('case_type','case_type'),('disposal','disposal')]:
        df,_=load_frame(str(path),'court','text',column,'decision_date',task)
        assert set(df.split)=={'train','val','test'}
        assert len(df)==60


def test_duration_bin_cutoffs():
    # Four bins in the paper are <=30, 31-180, 181-730, >730.
    from fedlegal_india.metrics import evaluate_metrics
    x=np.log1p([30,180,730,731])
    assert evaluate_metrics('duration', x, x)['f1_4'] == 1.0

def test_paper_report(tmp_path):
    from fedlegal_india.paper_report import create_tables
    root=tmp_path/'runs'
    for task in ('duration','case_type','disposal'):
        folder=root/task/'seed_0'
        folder.mkdir(parents=True)
        rows=[]
        for c in COURTS:
            row={'court':c,'task':task,'algorithm':'fedavg','seed':0}
            if task=='duration': row.update({'maelog':1.4,'mae_days':120.,'spearman':0.6,'f1_4':0.4})
            else: row.update({'macro_f1':0.5 if task=='disposal' else 0.9,'accuracy':0.8})
            rows.append(row)
        pd.DataFrame(rows).to_csv(folder/'per_court_test.csv',index=False)
    result=create_tables(root,tmp_path/'paper')
    assert len(result)==1
    assert (tmp_path/'paper'/'paper_table.md').exists()
