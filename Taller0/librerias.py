from idlelib.search import find

import numpy as np
import matplotlib.pyplot as plt

def esCuadrada(a: np.ndarray) -> bool:
    n = len(a)
    return a.shape == (n, n)

## Saca los que no cumplan la condicion
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
    res = ([[1] * n, list(v)]
           + [[v[i] ** j for i in range(n)] for j in range(2, n)])

    return np.array(res)

def numeroAureo(n: int) -> np.float64:
    if n == 0: return np.float64(0)

    base = np.array([[1],
                     [1]], dtype=np.float64)

    fk = np.array([[1,
                    0]], dtype=np.float64)

    for _ in range(n):
        fk[0][0], fk[0][1] = calcularAx(fk, base)[0][0], fk[0][0]

    return np.float64(fk[0][0] / fk[0][1])

def matrizFibonacci(n: int) -> np.ndarray:
    fib = [0, 1] + [0 for _ in range(2, 2*n)]

    for i in range(2, 2*n):
        fib[i] = fib[i-1] + fib[i-2]

    res = [[fib[i+j] for j in range(n)] for i in range(n)]
    return np.array(res)

def matrizHilbert(n:int) -> np.ndarray:
    res = [[np.divide(1, (i + j + 1)) for j in range(n)] for i in range(n)]

    return np.array(res)

def calcular_polinomios(polinomio: np.ndarray, value: int | float | np.float64) -> np.float64:
    pot_values = np.array([[value ** i for i in range(len(polinomio))]])

    polinomio_vert = transpuesta(np.array([polinomio]))
    res = calcularAx(pot_values, polinomio_vert)

    return np.float64(res[0])

def print_polinomio(polinomio: np.ndarray):
    show_graph(lambda x: calcular_polinomios(polinomio, x), -1, 1, np.float64)

def show_graph(function, min_range: float, max_range: float, res_types) -> None:
    x = np.linspace(min_range, max_range, num=200, dtype=res_types)
    y = [function(val) for val in x]

    plt.plot(x, y, color='red')
    plt.grid(True)
    plt.legend()

    plt.show()

def row_echelon(a: np.ndarray) -> np.ndarray:
    n, m = a.shape
    res = a.copy()

    for  i in range(n):
        resto_columna = res[i:, i]
        indice_fila_mayor = int(i + np.argmax(resto_columna))

        # Compruebo si existe un pivote mayor
        if i != indice_fila_mayor:
            intercambiarFilas(res, i, indice_fila_mayor)

        pivot = res[i, i]

        for  j in range(i+1, m):
            factor  = res[j, i] / pivot

            # La fila j entera - la fila i entera * el factor de multiplicacion
            res[j, :] = res[j, :] - res[i, :] * factor

    return res