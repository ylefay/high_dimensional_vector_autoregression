import numpy as np
import jax.numpy as jnp
import jax
from hd_var.routines.als.als import als_compute
from hd_var.generate import generate_A_given_rank
from hd_var.operations import rank_tensor
from hd_var.assumptions import check_ass2, check_ass1

INFERENCE_ROUTINES = [als_compute]
jax.config.update("jax_enable_x64", True)


@jax.jit
def criterion(inps):
    prev_A, A, iter, _, _, _, _ = inps
    return (iter < 1000) & (jnp.linalg.norm(prev_A - A) / jnp.linalg.norm(prev_A) > 1e-2)


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
        dataset = np.load(f'./data/var_62_bis_2000_10_5.npz')
        res, A = main(inference_routine, dataset)
        print(f'A_estimated:{res[1]}, A_true:{A}')
        print(f'rel error:{np.linalg.norm(res[1] - A) / np.linalg.norm(A)}')
