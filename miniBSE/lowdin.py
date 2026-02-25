import numpy as np

def lowdin_sqrt(S):
    eigvals, eigvecs = np.linalg.eigh(S)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

def transform_mos(C, S):
    S_half = lowdin_sqrt(S)
    return S_half @ C


