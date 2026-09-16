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
	v = [e.copy().astype(np.float64) for e in X]
	for e in v:
		n = norma(e, p)
		for i in range(len(e)):
			e[i] = e[i] / n
	return v
            

def normaMatMC(A, q, p, Np):
    _, m = A.shape

    max_val, max_vec = 0, None
    
    for _ in range(Np):
        v = np.random.uniform(-5,5, m)
        norm = norma(v, p)
        if norm == 0: continue

        v_normalizado = v / norm
        mult_res = A @ v_normalizado
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

    # Recordar que devuelve un par [val, vector]
    A_norma_inducida = normaMatMC(A,p,p, Np)
    Ainv_norma_inducida = normaMatMC(Ainv,p,p, Np)

    if A_norma_inducida[1] is None or Ainv_norma_inducida[1] is None: 
        return -1

    return A_norma_inducida[0] * Ainv_norma_inducida[0]
    
# Calculo usando la norma exacta
def condExacto(A, p):
    Ainv = np.linalg.inv(A)

    A_norma_exacta = normaExacta(A,p)
    Ainv_norma_exacta = normaExacta(Ainv, p)

    return A_norma_exacta * Ainv_norma_exacta


