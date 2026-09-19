import numpy as np

from alc import *

def calculaLU(A):
    return elim_gaussiana(A)

def res_tri(L, b, inferior = True):
    if L is None:
        return None

    n, m = L.shape
    xs = np.zeros(n)

    if inferior:
        for i in range(n):
            if L[i][i] == 0:
                return None
            suma = dotm(xs[:i], L[i][:i])
            xs[i] = (b[i] - suma) / L[i][i]
    else:
        for i in range(n-1, -1, -1):
            if L[i][i] == 0:
                return None
            suma = dotm(L[i, i+1:], xs[i+1:])
            xs[i] = (b[i] - suma) / L[i][i]

    return xs

def inversa(A):
    L, U, _ = calculaLU(A)

    if L is None or U is None:
        return None

    I = np.eye(len(L))
    INV = np.zeros(A.shape)

    for i in range (len(L)):
        y = res_tri(L, I[:, i])

        col_inv = res_tri(U, y, False)

        if col_inv is None:
            return None

        INV[:, i] = col_inv

    return INV

def calculaLDV(A):
    L, U, _ = calculaLU(A)

    if L is None:
        return None, None, None

    Ut = transpose(U)
    Vt, D, _ = calculaLU(Ut)

    if D is None:
        return None, None, None

    return L, D, transpose(Vt)

def esSDP(A, atol= 1e-8):
    L, D, V = calculaLDV(A)

    if not matricesIguales(transpose(L), V, atol):
        return False

    for j in range(len(D)):
        if D[j][j] > 0: continue
        return False

    return True
