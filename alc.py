import numpy as np

#region labo0 and aux
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
    if len(a) == 0:
        return np.array([])
    elif a.ndim == 1:
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

# Calcula el producto interno entre dos vectores
def dotm(v, w) -> float:
    v = flat(v) if v.ndim > 1 else v
    w = flat(w) if w.ndim > 1 else w

    n = len(v)
    assert n == len(w)

    res = 0
    for i in range(n):
        res += v[i] * w[i]

    return res

def mmult(a: np.ndarray, b: np.ndarray):
    a = a.reshape(1, -1) if a.ndim == 1 else a
    b = b.reshape(-1, 1) if b.ndim == 1 else b

    n1, m1 = a.shape
    n2, m2 = b.shape

    assert m1 == n2, f"Dimensiones incompatibles para multiplicar: {m1} != {n2}"
    res = np.zeros((n1, m2))

    for i in range(n1):
        for j in range(m2):
            for k in range(m1):
                res[i, j] += a[i, k] * b[k, j]

    return res

def flat(v: np.ndarray) -> np.ndarray:
    return v[:, 0]

def intercambiarFilas(a: np.ndarray, i: int , j: int):
    assert i < len(a) and j < len(a)
    a[i], a[j] = a[j].copy(), a[i].copy()

def sumar_fila_multiplo(a: np.ndarray, i: int, j: int, s):
    assert i < len(a) and j < len(a)
    a[j] += a[i] * s
#endregion

#region labo1
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
#endregion

#region labo2
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
#endregion

#region labo3
import numpy as np


def norma(x, p):
    x = x[:, 0] if isinstance(x, np.ndarray) and x.ndim > 1 else x

    if p == 'inf':
        max_val = 0
        for val in x:
            if abs(val) > max_val:
                max_val = abs(val)
        return max_val

    res = 0
    for i in range(len(x)):
        res += abs(x[i]) ** p

    return res ** (1 / p)


def normaliza(X, p):
    return [normalize_vector(v, p) for v in X]

def normalize_vector(v: np.ndarray, p) -> np.ndarray:
    nv = v.copy().astype(np.float64)
    norm = norma(nv, p)

    if norm == 0: return nv

    return nv / norm

def normaMatMC(A, q, p, Np):
    _, m = A.shape

    max_val, max_vec = 0, None

    for _ in range(Np):
        v = np.random.uniform(-5, 5, m)
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

    if p == 1:
        for j in range(n):
            tot = 0
            for i in range(n):
                tot += abs(A[i, j])

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
    A_norma_inducida = normaMatMC(A, p, p, Np)
    Ainv_norma_inducida = normaMatMC(Ainv, p, p, Np)

    if A_norma_inducida[1] is None or Ainv_norma_inducida[1] is None:
        return -1

    return A_norma_inducida[0] * Ainv_norma_inducida[0]


# Calculo usando la norma exacta
def condExacto(A, p):
    Ainv = np.linalg.inv(A)

    A_norma_exacta = normaExacta(A, p)
    Ainv_norma_exacta = normaExacta(Ainv, p)

    if A_norma_exacta is None or Ainv_norma_exacta is None:
        return -1

    return A_norma_exacta * Ainv_norma_exacta
#endregion

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

#region labo6
def mult_and_normalize(A:np.ndarray, v:np.ndarray) -> np.ndarray:
    w = calcularAx(A, v)
    return normalize_vector(w, 2)

# TODO: Falta que devuelva ops en vez de 0.
def metpot2k(A, tol= 1e-15, K = 1000):
    n, m = A.shape

    initial_random_v = np.array(np.random.uniform(-1.0, 1.0, n))

    v = normalize_vector(initial_random_v, 2)
    vp = mult_and_normalize(A, mult_and_normalize(A, v))

    e = dotm(v, vp)

    k, nops = 0, 0
    while abs(e - 1.0) > tol and k < K:
        v = vp
        vp = mult_and_normalize(A, mult_and_normalize(A, v))
        e = dotm(v, vp)

        k = k + 1

    # En vez de poner transpose(vp), uso el producto interno.
    l = dotm(vp, mmult(A, vp))
    return vp, l, 0

def buildHH(v, size: int, tol=  1e-15):
    I = np.eye(size,size)
    e1 = np.zeros(size)
    e1[0] = 1.0

    v = v[:, 0] if v.ndim > 1 else v
    vtemp = np.array(e1) - v
    norma_tmp = norma(vtemp,2)

    if norma_tmp < tol: return I

    column = vtemp.reshape(-1, 1)
    row = vtemp.reshape(1, -1)

    return I - 2 * mmult(column, row) * (norma_tmp**-2)


# Devuelve S, D tal que A =  S@D@S^t
def diagRH(A, tol= 1e-15, K = 1000):
    n = len(A)

    v1, l1, _ = metpot2k(A, tol, K)
    H_v1 = buildHH(v1, n, tol)

    S, D = np.zeros((n,n)), np.zeros((n,n))
    if n == 2:
        S = H_v1
        D = mmult(H_v1, mmult(A, transpose(H_v1)))
    else:
        B = mmult(H_v1, mmult(A, transpose(H_v1)))
        Ap = B[1:n, 1:n]

        Sp, Dp = diagRH(Ap, tol, K)

        D[0, 0] = l1
        D[1:n, 1:n] = Dp

        S[0, 0] = 1
        S[1:n, 1:n] = Sp

        S = mmult(H_v1, S)

    return S, D
#endregion