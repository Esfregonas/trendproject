# metrics.py (novo)
from sklearn.metrics import average_precision_score, precision_recall_curve, f1_score

def pr_auc(y_true, y_prob):
    return average_precision_score(y_true, y_prob)

def best_threshold_by_f1(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    # o último ponto de t é None; alinhamos comprimentos
    t = t.tolist() + [1.0]
    f1 = [0 if (pp+rr)==0 else 2*pp*rr/(pp+rr) for pp, rr in zip(p, r)]
    j = max(range(len(f1)), key=f1.__getitem__)
    return t[j], f1[j]
