import multiprocessing
import time

import numpy as np


def mat_mul(A, B):
    return np.matmul(A, B)


def parallel_mat_mul(A, B):
    n, k, k2, m = A.shape[0], A.shape[1], B.shape[0], B.shape[1]
    assert k == k2

    A1, A2 = A[:, : k // 2], A[:, k // 2 :]
    B1, B2 = B[: k // 2, :], B[k // 2 :, :]

    args1 = (A1, B1)
    args2 = (A2, B2)

    with multiprocessing.Pool(2) as pool:
        result1, result2 = pool.starmap(mat_mul, [args1, args2])

    C = result1 + result2
    return C


def measure_time(matmul_func, repeat=100):
    n, k, m = 1000, 1000, 1000
    A = np.random.rand(n, k)
    B = np.random.rand(k, m)

    start = time.time()
    for _ in range(repeat):
        C = matmul_func(A, B)
    end = time.time()
    print(f"Time taken: {(end - start) / repeat: .4f}")


def validate_matmul():
    n, k, m = 100, 100, 100
    A = np.random.rand(n, k)
    B = np.random.rand(k, m)
    C = parallel_mat_mul(A, B)
    assert C.shape == (n, m)
    C1 = np.matmul(A, B)
    assert np.allclose(C, C1)


if __name__ == "__main__":
    validate_matmul()

    print("Time taken for matmul:")
    measure_time(mat_mul)
    print("Time taken for parallel matmul:")
    measure_time(parallel_mat_mul)
