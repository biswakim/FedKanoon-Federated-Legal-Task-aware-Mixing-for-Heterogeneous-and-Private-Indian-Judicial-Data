from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedlegal_india.data import COURTS,TYPES,DISPOSALS

def generate(path):
    rows=[]
    for k,c in enumerate(COURTS):
        for i in range(30):
            rows.append({'court':c,'text':f'This petition raises a civil matter about service conditions {i} and legal reasons. The application has facts and counsel.',
                'decision_date':f'2024-{1+i//28:02d}-{1+i%28:02d}',
                'duration_days':float(10+(i*11+k*17)%400), 'case_type':TYPES[i%len(TYPES)],
                'disposal':DISPOSALS[(i+k)%len(DISPOSALS)]})
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
if __name__=='__main__':generate('data/synthetic_smoke.csv')
