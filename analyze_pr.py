# analyze_pr.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_curve, average_precision_score

data = Path("data")
y = pd.read_parquet(data/"y.parquet")["y"]
oof = pd.read_parquet(data/"oof_prob.parquet")["oof_prob"]

# alinha por índice, só interseção
idx = y.index.intersection(oof.index)
y = y.loc[idx].astype(int)
p = oof.loc[idx].astype(float)

ap = average_precision_score(y, p)
prec, rec, thr = precision_recall_curve(y, p)
f1 = np.zeros_like(thr)
for i, t in enumerate(thr):
    yhat = (p >= t).astype(int)
    tp = ((yhat==1) & (y==1)).sum()
    fp = ((yhat==1) & (y==0)).sum()
    fn = ((yhat==0) & (y==1)).sum()
    f1[i] = 0 if (2*tp+fp+fn)==0 else 2*tp/(2*tp+fp+fn)

j = int(f1.argmax()) if len(f1) else -1
print({
    "PR-AUC": float(ap),
    "best_thr": float(thr[j]) if j>=0 else None,
    "F1@best_thr": float(f1[j]) if j>=0 else None
})

# (opcional) guarda curva para plot noutro sítio
pd.DataFrame({"recall":rec[:-1], "precision":prec[:-1], "thr":thr}).to_csv(data/"pr_curve.csv", index=False)
print("Guardado:", data/"pr_curve.csv")
