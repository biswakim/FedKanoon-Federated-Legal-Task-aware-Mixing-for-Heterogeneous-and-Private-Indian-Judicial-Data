import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import spearmanr


def evaluate_metrics(task, actual, prediction):
    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    if task == 'duration':
        d = np.maximum(np.expm1(actual), 0)
        p = np.maximum(np.expm1(np.clip(prediction, -10, 15)), 0)
        bins = np.array([30., 180., 730.]) + 1e-8
        corr = (spearmanr(actual, prediction).statistic
                if len(actual)>1 and np.std(actual)>1e-12 and np.std(prediction)>1e-12 else np.nan)
        return {'maelog': float(np.mean(np.abs(actual-prediction))),
                'mae_days': float(np.mean(np.abs(d-p))),
                'spearman': float(corr) if np.isfinite(corr) else None,
                'f1_4': float(f1_score(np.digitize(d,bins,right=True),np.digitize(p,bins,right=True),average='macro',labels=[0,1,2,3],zero_division=0))}
    return {'macro_f1':float(f1_score(actual,prediction,average='macro',
            labels=list(range(10 if task=='case_type' else 4)),zero_division=0)),
            'accuracy':float(np.mean(actual==prediction))}
