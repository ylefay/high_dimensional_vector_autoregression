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

PATH = r"C:\Users\schoo\OneDrive\Bureau\Mines\Cours\3A\MVA\Semestre 1\Séries temporelles\Projet\project_git\high_dimensional_vector_autoregressive\scripts\toy_model\Experiments\data\exp62"
PATH = "./data/exp62/"
INFERENCE_ROUTINES = [als_compute_closed_form, admm_compute]
NAME_INFERENCE_ROUTINES = ['ALS', 'ADMM']
jax.config.update("jax_enable_x64", True)
T = [2000, 2500, 3000, 3500, 4000]


def criterion(inps):
    A, prev_A, iter, *_ = inps
    return (iter < 1000) & (jnp.linalg.norm(prev_A - A) > 1e-2)


def main(dataset, inference_routine, t, check=False, show=False):
    np.random.seed(0)

    y_list, A_list = dataset['y_list'], dataset['A_list']
    n_sample, N, T = y_list.shape
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
            print(t, i / n_sample * 100, "%")

    return error_list


if __name__ == '__main__':
    start = time.time()
    dataset = np.load(
        f"{PATH}data_10_5_4000_10_3_3_2.npz")
    n_sample = dataset['y_list'].shape[0]
    r1, r2, r3 = rank_tensor(dataset['A_list'][0])
    n_T = len(T)

    for k in range(len(INFERENCE_ROUTINES)):
        inference_routine = INFERENCE_ROUTINES[k]
        bias = np.zeros(n_T)
        bias_squared = np.zeros(n_T)
        for i in range(n_T):
            t = T[i]
            error_list = main(dataset, inference_routine, t, show=True)
            bias[i] = np.mean(error_list)
            bias_squared[i] = np.mean(error_list ** 2)
            print(f'{t} done')
        plt.plot(T, 10 ** 3 * bias_squared, label=NAME_INFERENCE_ROUTINES[k])
        plt.scatter(T, 10 ** 3 * bias_squared)

    plt.ylabel('Squared Bias (10e-3)')
    plt.xlabel('T')
    plt.title(f'r3 = {r3}, n_sample = {n_sample}')
    plt.legend()
    # plt.savefig(f'C:\\Users\\schoo\\OneDrive\\Bureau\\Mines\\Cours\\3A\\MVA\\Semestre 1\\Séries temporelles\\Projet\\project_git\\high_dimensional_vector_autoregressive\\scripts\\toy_model\\Experiments\\images\\plot_fig3_{r3}_{n_sample}.png')
    plt.show()

    end = time.time()
    times = end - start
    hours, remainder = divmod(times, 3600)
    minutes, secondes = divmod(remainder, 60)
    print(round(hours), 'h', round(minutes), 'min', round(secondes), 'sec')
