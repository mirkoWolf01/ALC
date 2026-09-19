import numpy as np

def esCuadrada(a: np.ndarray) -> bool:
    n = len(a)
    return a.shape == (n, n)

def mfiltrar_excluyendo(a: np.ndarray, condicion) -> np.ndarray:
    (n, m) = a.shape
    res = np.zeros((n, m))

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

def esSimetrica(a: np.ndarray) -> bool:
    (n, m) = a.shape
    at = transpose(a)

    for i in range(n):
        for j in range(m):
            if at[i][j] == a[i][j]: continue
            return False

    return True


def calcularAx(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    return mmult(a, x)

def dotm(A, B):
    n = len(A)
    assert n == len(B)

    res = 0
    for i in range(n):
        res += A[i] * B[i]

    return res

def mmult(a: np.ndarray, b: np.ndarray):
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    n1, m1 = a.shape
    n2, m2 = b.shape

    assert m1 == n2, f"Dimensiones incompatibles para multiplicar: {m1} != {n2}"
    res = np.zeros((n1, m2))

    for i in range(n1):
        for j in range(m2):
            for k in range(m1):
                res[i, j] += a[i, k] * b[k, j]

    return res

def intercambiarFilas(a: np.ndarray, i: int , j: int):
    assert i < len(a) and j < len(a)
    a[i], a[j] = a[j].copy(), a[i].copy()

def sumar_fila_multiplo(a: np.ndarray, i: int, j: int, s):
    assert i < len(a) and j < len(a)
    a[j] += a[i] * s

# labo1

def error(x,y):
    return abs(np.float64(x) - np.float64(y))

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

# labo2

def rota(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def escala(s):
    n = len(s)

    res = np.zeros((n, n))

    i = np.arange(n)
    res[i, i] = s

    return res

def rota_y_escala(theta, s):
    r_matrix = rota(theta)
    s_matrix = escala(s)

    # Primero lo roto, y despues lo escalo
    # Porque se multiplica primero de derecha a izquierda
    return mmult(s_matrix, r_matrix)


def afin(theta, s ,b):
    rs_matrix = rota_y_escala(theta, s)

    res = np.zeros((3,3))
    res[0:2, 0:2] = rs_matrix
    res[:2, 2] = b
    res[2, 2] = 1

    return res

def trans_afin(v, theta, s ,b):
    af_matrix = afin(theta, s, b)
    nv = np.ones((3, 1))
    nv[:2, 0] = v

    return mmult(af_matrix, nv)[:2, 0]

# labo4

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
