import json
import sys
from pathlib import Path
import pandas as pd

def main(root):
    files=sorted(Path(root).glob('*/seed_*/summary.json'))
    if not files:
        raise SystemExit('No summary.json files found')
    d=pd.DataFrame([json.loads(x.read_text()) for x in files])
    d.to_csv(Path(root)/'all_seeds.csv',index=False)
    metric_cols=[c for c in ['maelog','mae_days','spearman','f1_4','macro_f1','accuracy','worst_court_maelog','worst_court_macro_f1'] if c in d]
    result=d.groupby(['algorithm','task'])[metric_cols].agg(['mean','std','count'])
    result.to_csv(Path(root)/'comparison.csv')
    print(result.to_string())
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'outputs')
