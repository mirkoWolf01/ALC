import numpy as np


def norma(x, p):
    if p == 'inf':
        max_val = 0
        for val in x:
            if abs(val) > max_val:
                max_val = abs(val)
        return max_val

    res = 0
    for i in range(len(x)):
        res += abs(x[i]) ** p
    
    return res ** (1/p)

def normaliza(X, p):
    res = []

    for v in X:
        v_arr = np.array(v)
        norm  = norma(v_arr, p)
        res.append(v_arr / norm)

    return np.array(res, dtype=object)
            

def normaMatMC(A, q, p, Np):
    n, m = A.shape
    V = []

    for i in range(Np):
        V.append(np.random.uniform(-5,5, n))

    max_val = 0
    for v in V:
        norm = norma(v, p)
        if v / norm != 1:
            continue

        mult_res = calcularAx(A, np.array(v))
        max_cand = norma(mult_res,q)

        if max_cand > max_val: max_val = max_cand

    return max_val

def normaExacta(A, p=[1, 'inf']):
    n, m = A.shape
    max_val = 0

    if p != 1 and p != 'inf':
        return None

    if(p == 1):
        for j in range(n):
            tot = 0
            for i in range(n):
                tot += abs(A[i,j])
            
            if tot > max_val: max_val = tot
    else:
        for i in range(n):
            tot = 0
            for j in range(n):
                tot += abs(A[i, j])
            
            if tot > max_val: max_val = tot
    return max_val 


def condMC(A, p):
    pass

def condExacto(A, p):
    pass

def calcularAx(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    (n, m) = a.shape
    assert len(x) == m

    res = [[0] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            res[i][0] += a[i][j] * x[j][0]

    return np.array(res)
