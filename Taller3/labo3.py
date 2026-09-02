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
    _, m = A.shape

    max_val = 0
    max_vec = None
    
    for _ in range(Np):
        v = np.random.uniform(-5,5, m)
        norm = norma(v, p)
        if norm == 0: continue

        # v_normalizado en formato columna, y mult_res en 1D.
        v_normalizado = (v / norm).reshape(m, 1)
        mult_res = (A @ np.array(v_normalizado)).reshape(-1)

        val_candidato = norma(mult_res, q)

        if val_candidato >= max_val: 
            max_val = val_candidato
            max_vec = v_normalizado


    return [max_val, max_vec]

def normaExacta(A, p=[1, 'inf']):
    n, _ = A.shape
    max_val = 0

    if p != 1 and p != 'inf': return None

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

# Calculo usando la norma inducida via montecarlo
def condMC(A, p, Np):
    Ainv = np.linalg.inv(A)

    A_norma_inducida = normaMatMC(A,p,p, Np)[0]
    Ainv_norma_inducida = normaMatMC(Ainv,p,p, Np)[0]

    if A_norma_inducida[1] is None or Ainv_norma_inducida[1] is None: 
        return -1

    return A_norma_inducida * Ainv_norma_inducida
    
# Calculo usando la norma exacta
def condExacto(A, p):
    Ainv = np.linalg.inv(A)

    A_norma_exacta = normaExacta(A,p)
    Ainv_norma_exacta = normaExacta(Ainv, p)

    return A_norma_exacta * Ainv_norma_exacta


