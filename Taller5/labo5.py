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

    return res ** (1 / p)

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


def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    n, m = A.shape

    if n != m:
        return None

    At = A.copy().T

    Q = np.zeros((n, n))
    R = np.zeros((n, n))

    n2a_1 = norma(At[0], 2)
    if n2a_1 < tol: return None
    Q[0] = (At[0] / n2a_1)
    R[0][0] = n2a_1

    for j in range(1, n):
        Q[j] = At[j]

        for k in range(j):
            R[k][j] = Q[k].T @ Q[j]
            Q[j] = Q[j] - R[k][j] * Q[k]

        nq_j = norma(Q[j], 2)
        if nq_j < tol: return None

        R[j][j] = nq_j
        Q[j] = Q[j] / nq_j

    if retorna_nops:
        return Q, R, 0

    return Q.T, R


def signo(i):
    return 1 if i >= 0 else -1

def QR_con_HH(A, tol=1e-12, extras=False):
    m, n = A.shape

    if not (m >= n):
        return None

    R = A.copy().astype(np.float64)
    Q = np.eye(m)

    extra_info = {
        "R_matrices": [],
        "Q_matrices": [],
    }

    for k in range(0, n):
        x = R[k:m, k]

        alpha = -signo(x[0]) * norma(x, 2)

        e1 = np.zeros(m-k)
        e1[0] = 1

        u = x - alpha * e1
        n_u = norma(u, 2)

        if n_u > tol:
            u = u / n_u

            H = np.eye(m-k) - 2*mmult(u, u.T)

            Hp = np.eye(m)
            Hp[k:m, k:m] = H

            R = Hp @ R
            Q = Q @ Hp.T

    if extras:
        return Q, R, extra_info

    return Q, R


def calculaQR(A, metodo='RH', tol=1e-12):
    if metodo == 'RH':
        return QR_con_HH(A, tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol)
    else:
        return None

