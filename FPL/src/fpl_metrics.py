# src/metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def eval_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def eval_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def eval_report(y_true, y_pred, digits=4):
    return classification_report(y_true, y_pred, digits=digits)

def near_accuracy(true_pos, pred_pos, tol=1):
    """
    위치 근접 평가(나중에 rel_pos 적용할 때 사용)
    """
    true_pos = np.asarray(true_pos)
    pred_pos = np.asarray(pred_pos)
    return float(np.mean(np.abs(true_pos - pred_pos) <= tol))
