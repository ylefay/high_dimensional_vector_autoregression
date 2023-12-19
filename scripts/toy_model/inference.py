import numpy as np
from hd_var.routines.als.als import als_compute
from hd_var.generate import generate_A_given_rank
from hd_var.operations import rank_tensor
from hd_var.assumptions import check_ass2, check_ass1

INFERENCE_ROUTINES = [als_compute]
criterion = lambda A, prev_A, iter: iter < 10 and np.linalg.norm(A - prev_A) / (
    np.linalg.norm(prev_A) if np.linalg.norm(prev_A) > 0 else 1) > 1e-3


def main(inference_routine, dataset, check=False):
    np.random.seed(0)
    X, A, E = dataset['X'], dataset['A'], dataset['E']
    if check:
        check_ass1(A)
        check_ass2(A)
    N, T = X.shape
    P = A.shape[-1]  # cheating
    ranks = rank_tensor(A)  # cheating
    A_init = generate_A_given_rank(N, P, ranks)
    print(f'A_true:{A}')
    print(f'A_init:{A_init}')
    res = inference_routine(A_init, ranks, X, criterion)
    return res, A


if __name__ == '__main__':
    for inference_routine in INFERENCE_ROUTINES:
        dataset = np.load(f'./data/var_10000_2_3.npz')
        res, A = main(inference_routine, dataset)
        print(f'A_estimated:{res[1]}, A_true:{A}')
        print(f'rel error:{np.linalg.norm(res[1] - A) / np.linalg.norm(A)}')
