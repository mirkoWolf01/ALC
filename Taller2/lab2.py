import numpy as np

def rota(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def escala(s):
    n = len(s)
    res = np.diag(s)

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


def mmult(a: np.ndarray, b: np.ndarray):
    n1, m1 = a.shape
    n2, m2 = b.shape

    assert m1 == n2
    res = np.zeros((n1, m2))

    for i in range(n1):
        for j in range(m2):
            for k in range(m1):
                res[i, j] += a[i, k] * b[k, j]

    return res

