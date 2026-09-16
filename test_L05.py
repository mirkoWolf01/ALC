### Funciones L05-QR
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

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    n, m = A.shape

    if(n != m):
        return None

    At = A.copy().T

    Q = np.zeros((n,n))
    R = np.zeros((n,n))

    n2a_1 = norma(At[0],2)
    if n2a_1 < tol: return None
    Q[0] = (At[0] / n2a_1)
    R[0][0] = n2a_1

    for j in range(1, n):
        Q[j] = At[j]

        for k in range(j):
            R[k][j] = Q[k].T @ Q[j]
            Q[j] = Q[j] - R[k][j] * Q[k]

        nq_j = norma(Q[j],2)
        if nq_j < tol: return None

        R[j][j] = nq_j
        Q[j] = Q[j] / nq_j

    if retorna_nops:
        return Q, R, 0

    return Q.T, R

def signo(x):
    return 1 if x>=0 else -1

def QR_con_HH(A,tol=1e-12,extras=False):
    m,n = A.shape

    if not (m >= n):
        return None

    At = A.copy().T

    Q = np.eye(m)
    R = A.copy()


    e1 = np.zeros(n)
    e1[0] = 1

    for k in range(0, n):
        x = R[k:m][k]
        alpha = -signo(x[0]) * norma(x, 2)
        u = x - alpha * (e1 ** (m -k +1))

        if norma(u,2) > tol:
            u = normaliza(u, 2)


    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve: 
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso


    """
def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """

# Tests L05-QR:

import numpy as np

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)
