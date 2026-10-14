from alc import *

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