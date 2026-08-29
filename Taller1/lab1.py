import numpy as np

def error(x,y):
    return abs(np.float64(x) - np.float64(y))

def error_relativo(x,y):
    if x == 0:
        return y
    if y == 0:
        return x

    return abs(np.float64(x) - np.float64(y)) / abs(np.float64(x))

def matricesIguales(A, B) -> bool:
    e = 1e-08

    if A.shape != B.shape: return False

    for n in range(A.shape[0]):
        for m in range(A.shape[1]):
            if error_relativo(A[n,m], B[n,m]) > e: return False

    return True

def transpuesta(a: np.ndarray) -> np.ndarray:
    (n, m) = a.shape

    res = []
    for j in range(m):
        nueva_fila = []
        for i in range(n):
            nueva_fila.append(a[i][j])
        res.append(nueva_fila)

    return np.array(res)

def esSimetrica(A: np.ndarray) -> bool:
    (n, m) = A.shape
    At = transpuesta(A)

    return np.matricesIguales(A, At)