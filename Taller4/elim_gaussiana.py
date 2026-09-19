import numpy as np
from alc import sumar_fila_multiplo

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


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()
    
    
