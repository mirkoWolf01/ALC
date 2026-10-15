from alc import *


def generate_rand_columns(size:int):
    rnumbers = np.array(np.random.uniform(0, 1, size ** 2))

    columns = []

    for i in range(size):
        columna = []
        for j in range(size):
            columna.append(rnumbers[i * size + j])
        columns.append(np.array(columna))

    return columns


def transiciones_al_azar_continuas(n):
    columnas = generate_rand_columns(n)
    columns = normaliza(columnas, 1)

    T = np.zeros((n, n))
    for i in range(n):
        T[0:n, i] = columns[i]

    return T


def transiciones_al_azar_uniformes(n, thres):
    columns = generate_rand_columns(n)
    for i in range(n):
        for j in range(n):
            columns[i][j] = 0 if columns[i][j] > thres else 1

        # Caso borde donde toda la columna es 0. Asigno un 1 en alguna posicion al azar.
        if norma(columns[i], 1) == 0:
            columns[i][np.random.randint(0, n)] = 1

    columns = normaliza(columns, 1)

    T = np.zeros((n, n))
    for i in range(n):
        T[0:n, i] = columns[i]

    return T


def nucleo(A, tol=1e-15):
    B = mmult(transpose(A), A)
    S, D = diagRH(B, tol)

    res = []
    for i in range(len(D)-1, -1, -1):
        if abs(D[i, i]) < tol:
            res.append(S[:, i])
        else: break

    return transpose(np.array(res))


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    if len(listado) == 0:
        return {}, (m_filas, n_columnas)

    assert all(len(l) == len(listado[0]) for l in listado), \
        "Las listas de creacion deben de tener el mismo tamaño"

    res = {}
    for i in range(len(listado[0])):
        if abs(listado[2][i]) < tol: continue
        key: tuple = (listado[0][i], listado[1][i])
        res[key] = listado[2][i]

    return res, (m_filas, n_columnas)


def multiplica_rala_vector(A: tuple[dict, tuple[int, int]], v):
    m = rala_to_matrix(A)
    return flat(mmult(m, v))

def rala_to_matrix(A: tuple[dict, tuple[int, int]]) -> np.ndarray:
    d, (m, n) = A

    matrix = np.zeros((m,n))

    for (i,j) in d.keys():
        matrix[i, j] = d[(i,j)]

    return matrix
