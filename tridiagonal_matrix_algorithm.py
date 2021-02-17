import numpy as np
from copy import deepcopy


def tdma_solve(main, upper, bottom, res):
    main_c, upper_c, bottom_c, res_c = map(lambda x: deepcopy(x), (main, upper, bottom, res))

    for i in range(1, len(res)):
        temp = bottom_c[i - 1] / main_c[i - 1]
        main_c[i] -= temp * upper_c[i - 1]
        res_c[i] -= temp * res_c[i - 1]

    diag = main_c
    diag[-1] = res_c[-1] / main_c[-1]

    for i in range(len(res) - 2, -1, -1):
        diag[i] = (res_c[i] - upper_c[i] * diag[i + 1]) / main_c[i]

    return diag


if __name__ == '__main__':
    A = np.array([
        [6, 2, 0, 0],
        [3, -6, 4, 0],
        [0, -1, 3, 1],
        [0, 0, -1, 2]],
        dtype=float
    )

    main_diagonal = np.array([6, -6, 3, 2], dtype=float)
    upper_diagonal = np.array([2, 4, 1], dtype=float)
    bottom_diagonal = np.array([3, -1, -1], dtype=float)
    result = np.array([3, 3, 4, 2], dtype=float)

    res = tdma_solve(main_diagonal, upper_diagonal, bottom_diagonal, result)
    print(f'Given matrix A = \n{A}\nResult for matrix A:\n{res}')
    print(f'With built-in numpy function:\n{np.linalg.solve(A, result)}')
