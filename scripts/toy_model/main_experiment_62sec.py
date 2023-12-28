import numpy as np
import jax.numpy as jnp
import jax
from hd_var.routines.mlr.als import als_compute, als_compute_closed_form
from hd_var.routines.shorr.admm import admm_compute
from hd_var.generate import generate_A_given_rank
from hd_var.operations import rank_tensor
from hd_var.assumptions import check_ass2, check_ass1
import matplotlib.pyplot as plt
import time

inference_routine = admm_compute
cases = ['a', 'c']
gamma = [0.05, 0.1, 0.15, 0.2, 0.25]
n_gamma = len(gamma)
R = 2
T = 3000
jax.config.update("jax_enable_x64", True)


def criterion(inps):
    A, prev_A, iter, *_ = inps
    return (iter < 1000) & (jnp.linalg.norm(prev_A - A) / jnp.linalg.norm(prev_A) > 1e-3)


def main(dataset, inference_routine, t, check=False, show=False):
    np.random.seed(0)

    y_list, A_list = dataset['y_list'], dataset['A_list']
    n_sample, N, T = y_list.shape

    n_sample = 10

    error_list = np.zeros(n_sample)

    for i in range(n_sample):
        y, A = y_list[i], A_list[i]

        y_reshaped = y[:, :t]
        if check:
            # check_ass1(A)
            check_ass2(A)
        P = A.shape[-1]  # cheating
        ranks = rank_tensor(A)  # cheating
        A_init = generate_A_given_rank(N, P, ranks)
        res = inference_routine(A_init=A_init, ranks=ranks, y_ts=y_reshaped, criterion=criterion)
        error_list[i] = np.linalg.norm(res[1] - A)

        if (i / n_sample * 100) % 20 == 0 and show:
            print(i / n_sample * 100, "%")

    return error_list


if __name__ == '__main__':
    start = time.time()

    for case in cases:
        dataset = np.load(
            f'C:\\Users\\schoo\\OneDrive\\Bureau\\Mines\\Cours\\3A\\MVA\\Semestre 1\\Séries temporelles\\Projet\\project_git\\high_dimensional_vector_autoregressive\\scripts\\toy_model\\Experiments\\data\\exp62sec\\data_{T}_{case}_{R}.npz')
        y_list, A_list = dataset['y_list'], dataset['A_list']
        n_sample, N, T = y_list.shape
        r1, r2, r3 = rank_tensor(dataset['A_list'][0])
        P = dataset['A_list'][0].shape[-1]

        frob_norm = np.zeros(n_gamma)

        for i in range(n_gamma):
            gam = gamma[i]
            if case == 'b':
                S = 2 * 2 * 2
            else:
                S = 3 * 3 * 2
            t = int((S * np.log(P * (N ** 2))) / gam)
            error_list = main(dataset, inference_routine, t, show=True)
            frob_norm[i] = np.mean(error_list ** 2)
            print(gam, 'done')

        plt.plot(gamma, frob_norm, label=f'case {case}')
        plt.scatter(gamma, frob_norm)

    # plt.ylabel('Squared Bias (10e-3)')
    # plt.xlabel('T')
    # plt.title(f'r3 = {r3}, n_sample = {n_sample}')
    plt.legend()
    plt.show()
    # plt.savefig(f'C:\\Users\\schoo\\OneDrive\\Bureau\\Mines\\Cours\\3A\\MVA\\Semestre 1\\Séries temporelles\\Projet\\project_git\\high_dimensional_vector_autoregressive\\scripts\\toy_model\\Experiments\\images\\plot_fig3_{r3}_{n_sample}.png')

    end = time.time()
    times = end - start
    hours, remainder = divmod(times, 3600)
    minutes, secondes = divmod(remainder, 60)
    print(round(hours), 'h', round(minutes), 'min', round(secondes), 'sec')
