import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def evaluate_binary(y_true, probability, threshold=0.5):
    pred = (probability >= threshold).astype(int)
    return {"accuracy":accuracy_score(y_true,pred),"precision":precision_score(y_true,pred,zero_division=0),
            "recall":recall_score(y_true,pred,zero_division=0),"f1":f1_score(y_true,pred,zero_division=0),
            "roc_auc":roc_auc_score(y_true,probability),"pr_auc":average_precision_score(y_true,probability)}

def choose_threshold(y_true, probability, thresholds=np.arange(0.20,0.71,0.05)):
    scores=[(t,f1_score(y_true,(probability>=t).astype(int),zero_division=0)) for t in thresholds]
    return max(scores,key=lambda x:x[1])[0]
