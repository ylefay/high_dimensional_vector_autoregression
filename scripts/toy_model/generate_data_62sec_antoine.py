import numpy as np
from hd_var.generate import generate
from hd_var.assumptions import check_ass2, check_ass1
from hd_var.generate import generate_A_given_case

PATH = "C:\\Users\\schoo\\OneDrive\\Bureau\\Mines\\Cours\\3A\\MVA\\Semestre 1\\Séries temporelles\\Projet\\project_git\\high_dimensional_vector_autoregressive\\scripts\\toy_model\\Experiments\\data\\exp62sec\\"
PATH = "./data/exp62sec/"


def main(ranks, T, N, P, cov, check=False, save=False):
    while True:
        try:
            A = generate_A_given_case(ranks, case)
            if check:
                # check_ass1(A)
                check_ass2(A)
            y, A, E = generate(A, T, P, N, cov)
            break
        except RuntimeWarning:
            pass
        except AssertionError:
            pass
            # print("Non Stationnaire")
    if save:
        np.savez(f'{PATH}data/var_62_{T}_{N}_{P}.npz', y=y, A=A, E=E)
    return y, A, E


if __name__ == '__main__':
    T = 3000  # Length of the time series
    sigma = 1.0  # Variance of the innovations, assuming diagonal noise
    case = 'c'
    n_samples = 10
    if case == 'a':
        N, P = 10, 5
    elif case == 'b':
        pass
    elif case == 'c':
        N, P = 20, 5
    elif case == 'd':
        N, P = 10, 10
    cov = np.eye(N, ) * sigma  # Covariance matrix of the innovations
    ranks = [2, 2, 2]
    y_list = np.zeros((n_samples, N, T))
    A_list = np.zeros((n_samples, N, N, P))
    E_list = np.zeros((n_samples, N, T))
    for i in range(n_samples):
        y_list[i], A_list[i], E_list[i] = main(ranks=ranks, T=T, N=N, P=P, cov=cov,
                                               check=True)
    np.savez(f'{PATH}data_{N}_{P}_{T}_{n_samples}_{ranks[0]}_{ranks[1]}_{ranks[2]}.npz',
             y_list=y_list, A_list=A_list, E_list=E_list)
