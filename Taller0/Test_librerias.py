import  numpy as np, librerias as lib

class TestLibrerias:

    def test_esCuadrada_da_true_para_matriz_cuadrada(self):

        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9]])

        assert lib.esCuadrada(matrix)

    def test_esCuadrada_da_false_para_matriz_no_cuadrada(self):

        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12]])

        assert not lib.esCuadrada(matrix)

    def test_triangSup_devuelve_la_triangulacion_superior_de_una_matriz_sin_diagonal(self):

        matrix = np.array([[1, 6, 3, 9],
                           [4, 5, 6, 1],
                           [7, 8, 9, 3],
                           [8, 3, 4, 7]])

        assert np.array_equal(lib.triangSup(matrix), np.triu(matrix, 1))

    def test_triangInf_devuelve_la_triangulacion_inferior_de_una_matriz_sin_diagonal(self):

        matrix = np.array([[1, 2, 3, 9],
                           [4, 5, 6, 1],
                           [7, 8, 9, 3],
                           [8, 3, 4, 7]])

        assert np.array_equal(lib.triangInf(matrix), np.tril(matrix, -1))

    def test_diagonal_devuelve_una_matriz_de_la_diagonal(self):
        matrix = np.array([[1, 2, 3, 9],
                           [4, 5, 6, 1],
                           [7, 8, 9, 3],
                           [8, 3, 4, 7]])

        assert np.array_equal(lib.diagonal(matrix), np.diag(np.diag(matrix)))

    def test_traza_devuelve_suma_diagonal(self):
        matrix = np.array([[1, 2, 3, 9],
                           [4, 5, 6, 1],
                           [7, 8, 9, 3],
                           [8, 3, 4, 7]])

        assert lib.traza(matrix) == np.sum(np.diag(matrix))

    def test_traspuesta_devuelve_la_traspuesta_de_una_matriz(self):
        matrix = np.array([[1, 2, 3, 9],
                           [4, 5, 6, 1],
                           [7, 8, 9, 3],
                           [8, 3, 4, 7]])

        res = lib.transpuesta(matrix)
        assert isinstance(res, np.ndarray)
        assert np.array_equal(res, np.transpose(matrix))

    def test_esSimetrica_da_verdadero_si_matriz_es_simetrica(self):
        matrix = np.array([[1,  2,  5],
                           [2,  3, -2],
                           [5, -2, -1]])

        assert lib.esSimetrica(matrix)

    def test_esSimetrica_da_falso_si_matriz_no_es_simetrica(self):
        matrix = np.array([[1, 2, 3, 9],
                           [4, 5, 6, 1],
                           [7, 8, 9, 3],
                           [8, 3, 4, 7]])

        assert not lib.esSimetrica(matrix)

    def test_calcularAx_devuelve_multiplicacion_vectorial_entre_A_y_x(self):
        matrix = np.array([[2,  0,  4],
                           [1,  3, 5]])

        x = np.array([[3],
                      [1],
                      [2]])

        res = lib.calcularAx(matrix, x)

        assert type(res) is  np.ndarray
        assert np.array_equal(res, matrix @ x)


    def test_intercambiarFIlas_intercambia_2_filas_de_una_matriz(self):
        matrix = np.array([[2, 0, 4],
                           [1, 3, 5],
                           [3, 1, 4],
                           [7, 4, 9]])

        res_matrix =np.array([[2, 0, 4],
                              [7, 4, 9],
                              [3, 1, 4],
                              [1, 3, 5]])

        i, j = 1, 3
        lib.intercambiarFilas(matrix, i,  j)

        assert matrix[i] is not matrix[j]
        assert np.array_equal(matrix, res_matrix)

    def test_sumaer_fila_multiplo_suma_una_fila_multiplo_a_otra_fila(self):
        matrix = np.array([[2, 0, 4],
                           [1, 3, 5],
                           [3, 1, 4],
                           [7, 4, 9]])

        res_matrix =np.array([[ 2, 0,  4],
                              [ 1, 3,  5],
                              [23, 1, 44],
                              [ 7, 4,  9]])

        i, j = 0, 2
        lib.sumar_fila_multiplo(matrix, i, j, 10)
        assert matrix[j] is not matrix[i]
        assert np.array_equal(matrix, res_matrix)

    def test_esDiagonalmenteDominante_da_verdadero_si_matriz_es_diagonalmente_dominante(self):
        matrix = np.array([[9,  2,   5],
                           [2,  -6, -2],
                           [5, -2,   7]])

        assert lib.esDiagonalmenteDominante(matrix)

    def test_esDiagonalmenteDominante_da_falso_si_matriz_no_es_diagonalmente_dominante(self):
        matrix = np.array([[8,  2,  5],
                           [2, -9, -2],
                           [5, -2, -6]])

        assert not lib.esDiagonalmenteDominante(matrix)

    def test_matrizCirculante_devuelve_la_matriz_circulante_de_v(self):
        v = np.array([9, 2, 5, 2, 1])
        res_matrix =np.array([[9, 2, 5, 2, 1],
                              [1, 9, 2, 5, 2],
                              [2, 1, 9, 2, 5],
                              [5, 2, 1, 9, 2],
                              [2, 5, 2, 1, 9]])

        assert type(lib.matrizCirculante(v)) is  np.ndarray
        assert np.array_equal(lib.matrizCirculante(v), res_matrix)

    def test_matrizVandermonde_devuelve_la_matriz_vandermonde_de_v(self):
        v = np.array([2, 3, 5, 9])
        res_matrix =np.array([[1,  1,   1,   1],
                              [2,  3,   5,   9],
                              [4,  9,  25,  81],
                              [8, 27, 125, 729]])

        assert type(lib.matrizCirculante(v)) is  np.ndarray
        assert np.array_equal(lib.matrizVandermonde(v), res_matrix)

    def test_numeroAureo_aproxima_el_numero_aureo_n_esimo(self):
        # Casos base
        assert lib.numeroAureo(0) == 0
        assert lib.numeroAureo(1) == 1

        # Casos chicos
        assert lib.numeroAureo(4) == 5/3
        assert lib.numeroAureo(7) == 21/13

        # Casos grandes
        assert lib.numeroAureo(20) == 10946/6765
        assert lib.numeroAureo(50) == 20365011074 / 12586269025

        #lib.show_graph(lib.numeroAureo, 0, 100, int)

    def test_martizFibonacci_devuelve_una_matriz_de_los_resultados_de_fibonacci(self):
        res = lib.matrizFibonacci(5)
        res_esperado = np.array([[0, 1, 1,  2,  3],
                                 [1, 1, 2,  3,  5],
                                 [1, 2, 3,  5,  8],
                                 [2, 3, 5,  8, 13],
                                 [3, 5, 8, 13, 21]])

        assert isinstance(res, np.ndarray)
        assert np.array_equal(res, res_esperado)


    def test_matrizHilbert_devuelve_la_matriz_hilbert_de_n(self):
        res = lib.matrizHilbert(5)
        res_esperado = np.array([[1.0, 1/2, 1/3, 1/4, 1/5],
                                 [1/2, 1/3, 1/4, 1/5, 1/6],
                                 [1/3, 1/4, 1/5, 1/6, 1/7],
                                 [1/4, 1/5, 1/6, 1/7, 1/8],
                                 [1/5, 1/6, 1/7, 1/8, 1/9]])


        assert isinstance(res, np.ndarray)
        assert np.allclose(res, res_esperado)

    def test_calcular_polinomios1(self):
        polinomio = np.array([-1, 1, -1, 1, -1, 1])
        assert lib.calcular_polinomios(polinomio, 3) == 182

        #lib.print_polinomio(polinomio)

    def test_calcular_polinomios2(self):
        polinomio = np.array([3, 0, 1])
        assert lib.calcular_polinomios(polinomio, 3) == 12

        #lib.print_polinomio(polinomio)

    def test_calcular_polinomios3(self):
        polinomio = np.array([-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        assert lib.calcular_polinomios(polinomio, 4) == 1048574

        #lib.print_polinomio(polinomio)


    def test_row_echelon_con_pivoteo_evita_division_por_cero(self):
        matrix = np.array([[0.0, 1.0, 1.0],
                           [1.0, 2.0, 3.0],
                           [2.0, 1.0, 4.0]])

        res = lib.row_echelon(matrix)
        res_esperado = np.array([[2.0, 1.0, 4.0],
                                 [0.0, 1.5, 1.0],
                                 [0.0, 0.0, 0.3333333]])

        assert isinstance(res, np.ndarray)
        assert np.allclose(res, res_esperado)

        elementos_debajo_diagonal = np.tril(res, -1)
        assert np.allclose(elementos_debajo_diagonal, 0.0)