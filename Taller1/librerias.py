import numpy as np

def esCuadrada(a: np.ndarray) -> bool:
    n = len(a)
    return a.shape == (n, n)

## Saca los que no cumplan la condicion
def mfiltrar_excluyendo(a: np.ndarray, condicion) -> np.ndarray:
    (n, m) = a.shape

    res = [[0 for _ in range(m)] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            if condicion(i, j): continue
            res[i][j] = a[i][j]

    return np.array(res)

def triangSup(a: np.ndarray) -> np.ndarray:
    return mfiltrar_excluyendo(a, lambda i, j: j <= i)

def triangInf(a: np.ndarray) -> np.ndarray:
    return mfiltrar_excluyendo(a, lambda i, j: j >= i)

def diagonal(a: np.ndarray) -> np.ndarray:
    return mfiltrar_excluyendo(a, lambda i, j: j != i)

def traza(a: np.ndarray) -> int:
    assert esCuadrada(a)

    res = 0
    for i in range (len(a)):
        res += a[i][i]

    return res

def transpuesta(a: np.ndarray) -> np.ndarray:
    (n, m) = a.shape

    res = []
    for j in range(m):
        nueva_fila = []
        for i in range(n):
            nueva_fila.append(a[i][j])
        res.append(nueva_fila)

    return np.array(res)

def esSimetrica(a: np.ndarray) -> bool:
    (n, m) = a.shape
    at = transpuesta(a)

    for i in range(n):
        for j in range(m):
            if at[i][j] == a[i][j]: continue
            return False

    return True

def calcularAx(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    (n, m) = a.shape
    assert len(x) == m

    res = [[0] for _ in range(n)]

    for i in range(n):
        for j in range(m):
            res[i][0] += a[i][j] * x[j][0]

    return np.array(res)

def intercambiarFilas(a: np.ndarray, i: int , j: int):
    assert i < len(a) and j < len(a)
    a[i], a[j] = a[j].copy(), a[i].copy()

def sumar_fila_multiplo(a: np.ndarray, i: int, j: int, s):
    assert i < len(a) and j < len(a)
    a[j] += a[i] * s

def esDiagonalmenteDominante(a: np.ndarray) -> bool:
    assert esCuadrada(a)
    n = len(a)

    for i in range(n):
        fila= a[i]
        diag = abs(fila[i])
        total = 0

        for j in range(n):
            if i == j: continue
            total += abs(fila[j])

        if total > diag: return False

    return True

def matrizCirculante(v: np.ndarray) -> np.ndarray:
    n = len(v)
    values = v.tolist()
    res = []

    if n != 0:
        for _ in range(n):
            res.append(values.copy())
            last = values.pop()
            values = [last] + values

    return np.array(res)

def matrizVandermonde(v: np.ndarray) -> np.ndarray:
    n = len(v)
    res = []

    for i in range(n):
        fila = v.copy()
        for j in range(n):
            fila[j] **= i
        res.append(fila)

    return np.array(res)

def numeroAureo(n: int) -> float:
    if n == 0: return 0

    base = np.array([[1],
                     [1]])

    fk = np.array([[1,
                       0]])

    for _ in range(n):
        fk[0][0], fk[0][1] = calcularAx(fk, base)[0][0], fk[0][0]

    return float(fk[0][0] / fk[0][1])

def matrizFibonacci(n: int) -> np.ndarray:
    fib = [-1 for _ in range(2*n)]
    fib[0] = 0
    fib[1] = 1

    for i in range(2, 2*n):
        fib[i] = fib[i-1] + fib[i-2]

    res = [[fib[i+j] for j in range(n)] for i in range(n)]
    return np.array(res)