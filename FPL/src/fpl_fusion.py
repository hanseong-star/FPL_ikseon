# src/fusion.py
import numpy as np

def fuse_probabilities(P_shape, P_color, alpha=0.7):
    """
    P_final = alpha * P_shape + (1-alpha) * P_color
    """
    beta = 1.0 - alpha
    return alpha * P_shape + beta * P_color
