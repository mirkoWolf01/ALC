import numpy as np


def error_relativo(x,y):
    if x == 0:
        return y
    if y == 0:
        return x

    return abs(np.float64(x) - np.float64(y)) / abs(np.float64(x))

def matricesIguales(A, B,  e = 1e-07) -> bool:
    if A.shape != B.shape: return False

    for n in range(A.shape[0]):
        for m in range(A.shape[1]):
            if error_relativo(A[n,m], B[n,m]) > e: return False

    return True

def transpose(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(-1, 1)

    (n, m) = a.shape

    res = []
    for j in range(m):
        nueva_fila = []
        for i in range(n):
            nueva_fila.append(a[i][j])
        res.append(nueva_fila)

    return np.array(res)

def sumar_fila_multiplo(a: np.ndarray, i: int, j: int, s):
    assert i < len(a) and j < len(a)
    a[j] += a[i] * s

def elim_gaussiana(A):
    if A is None:
        return None, None, 0

    cant_op = 0
    n, m = A.shape

    if m != n:
        print('Matriz no cuadrada')
        return None, None, 0

    L = np.eye(n)
    U = A.copy().astype(np.float64)

    for i in range(n - 1):
        col = []

        if U[i][i] == 0:
            print(f"Error: Null pivot at ({i}:{i})")
            return None, None, 0

        for j in range(i + 1, n):
            factor: np.float64 = - U[j][i] / U[i][i]

            sumar_fila_multiplo(U, i, j, factor)
            col.append(-factor)

            cant_op += 1 + 2 * (n - i - 1)

        L[i + 1:n, i] = col

    return L, U, cant_op


def calculaLU(A):
    return elim_gaussiana(A)

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
        if D[j][j] > atol: continue
        return False

    return True

if __name__ == "__main__":
    D0 = np.diag([1, -1, 1])
    print(esSDP(D0))

    D0 = np.diag([1, 1, 1e-16])
    print(esSDP(D0))

    L0 = np.array([[1, 0, 0],
                   [1, 1, 0],
                   [1, 1, 1]])
    D0 = np.diag([1, 1, 1])
    V0 = np.array([[1, 0, 0],
                   [1, 1, 0],
                   [1, 1 + 1e-3, 1]]).T
    A = L0 @ D0 @ V0
    print(esSDP(A, 1e-3))

    A = np.array([[4.0, 2.0], [2.0, 3.0]])
    print(esSDP(A))

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    print(esSDP(A))