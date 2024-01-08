import numpy as np
import jax.numpy as jnp
import jax
from hd_var.routines.mlr.als import als_compute, als_compute_closed_form
from hd_var.routines.shorr.admm import admm_compute, _admm_compute
from hd_var.generate import generate_A_given_rank
from hd_var.operations import rank_tensor
from hd_var.assumptions import check_ass2, check_ass1
from functools import partial

INFERENCE_ROUTINES = [als_compute_closed_form, als_compute, admm_compute]
INFERENCE_ROUTINES = [partial(admm_compute, iter_sor=100)]
jax.config.update("jax_enable_x64", True)


def criterion(inps):
    A, prev_A, iter, *_ = inps
    return (iter < 1000) & (jnp.linalg.norm(prev_A - A) > 1e-2)


def main(inference_routine, dataset, check=False):
    np.random.seed(0)
    y, A, E = dataset['y'], dataset['A'], dataset['E']
    if check:
        check_ass1(A)
        check_ass2(A)
    N, T = y.shape
    P = A.shape[-1]  # cheating
    ranks = rank_tensor(A)  # cheating
    A_init = generate_A_given_rank(N, P, ranks)
    res = inference_routine(A_init=A_init, ranks=ranks, y_ts=y, criterion=criterion)
    return res, A


if __name__ == '__main__':
    for inference_routine in INFERENCE_ROUTINES:
        dataset = np.load(f'./data/var_4_260_3_2.npz')
        if 'y_list' in dataset.keys():
            for i in range(len(dataset['y_list'])):
                _dataset = {'y': dataset['y_list'][i], 'A': dataset['A_list'][i], 'E': dataset['E_list'][i]}
                res, A = main(inference_routine, _dataset)
                print(f'A_true:{A}, A_estimated:{res[1]}')
                print(f'rel error:{np.linalg.norm(res[1] - A) / np.linalg.norm(A)}')
        else:
            res, A = main(inference_routine, dataset)
            print(f'A_true:{A}, A_estimated:{res[1]}')
            print(f'rel error:{np.linalg.norm(res[1] - A) / np.linalg.norm(A)}')
