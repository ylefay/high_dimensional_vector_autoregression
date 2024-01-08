import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from hd_var.rank_selection import rank_selection, NN_compute
from hd_var.routines.mlr.als import als_compute_closed_form
from hd_var.routines.shorr.penalization import lambda_optimal
from hd_var.routines.shorr.admm import admm_compute
from hd_var.generate import generate_A_given_rank
from hd_var.utils import normalise_y, estimate_noise_variance, differentiate, integrate_series, predict

PATH = r"C:\\Users\\schoo\\OneDrive\\Bureau\\Mines\\Cours\\3A\\MVA\\Semestre 1\\Séries temporelles\\Projet\\project_git\\high_dimensional_vector_autoregressive\\scripts\\report\\results\\realdata\\"
PATH = "./results/realdata/"

case = 1

if case == 1:
    Path = f"{PATH}2022-12.csv"
    df = pd.read_csv(Path)
    id_group2 = [81, 82, 83, 84, 85, 86, 87, 179, 180, 181, 227, 228, 229, 230]
    df_group2 = df.iloc[:, id_group2]
    df = df_group2
    df_without_missing_values = df.dropna(axis=1)  # Remove columns with NA value
    df_without_head = df_without_missing_values.iloc[2:]  # Remove 2 first lines
    df_final = df_without_head.iloc[:, 1:]  # Remove date
    y = np.array(df_final).T

if case == 2:
    Path_data_clean = r"C:\Users\schoo\OneDrive\Bureau\Mines\Cours\3A\MVA\Semestre 1\Séries temporelles\Projet\project_git\data\real_data\Data_paper\data_clean.xlsx"

    df1 = pd.read_excel(Path_data_clean, sheet_name='Sheet1')
    df2 = pd.read_excel(Path_data_clean, sheet_name='Sheet2')

    columns_to_keep = ['GDP251', 'GDP252', 'GDP253', 'GDP256', 'GDP263', 'GDP264', 'GDP265', 'GDP270', 'PMCP', 'PMDEL',
                       'PMI', 'PMNO', 'PMNV', 'PMP', 'IPS10', 'UTL11', 'HSFR', 'BUSLOANS', 'CCINRV', 'FM1', 'FM2',
                       'FMRNBA', 'FMRRA', 'FSPIN', 'FYFF', 'FYGT10', 'SEYGT10', 'CES002', 'LBMNU', 'LBOUT', 'LHEL',
                       'LHUR', 'CES275R', 'CPIAUCSL', 'GDP273', 'GDP276', 'PSCCOMR', 'PWFSA', 'EXRUS', 'HHSNTN']

    name_df1 = df1.loc[:, df1.columns == 'Name']
    name_df2 = df2.loc[:, df2.columns == 'Name']

    index_df1_to_keep = []
    for k in range(len(name_df1)):
        name = name_df1.iloc[k][0]
        if [name] in name_df2.values:
            index_df1_to_keep.append(k)

    df1_quarter = df1.iloc[index_df1_to_keep]

    columns_to_keep_df1 = []
    for col in df1_quarter.columns.tolist():
        if (col.rstrip() in columns_to_keep):
            columns_to_keep_df1.append(col)
    df1_final = df1_quarter.loc[:, columns_to_keep_df1]

    columns_to_keep_df2 = []
    for col in df2.columns.tolist():
        if (col.rstrip() in columns_to_keep):
            columns_to_keep_df2.append(col)
    df2_final = df2.loc[:, columns_to_keep_df2]

    df1_np = np.array(df1_final)
    df2_np = np.array(df2_final)

    df_full = np.concatenate((df1_np, df2_np), axis=1)

    y = df_full.T

if case == 3:
    data = np.load(f"{PATH}/var_4_10000_3_2.npz", allow_pickle=True)
    y = data['y'][:, 5:200]
    A_true = data['A']


def plot_line_array(y, k):
    '''
    Plot the kth component of the time series using numpy array
    '''
    y_line = y[k, :]
    plt.plot(y_line)
    plt.show()


plot_line_array(normalise_y(y), 0)
N = y.shape[0]
# %%
ORDER = 1
normalise = True
n_points_to_predict = 8
if normalise:
    y = normalise_y(y, -n_points_to_predict)
y_without_end = y[:, :-n_points_to_predict]
y_after_diff_without_end = y_without_end
constants = np.zeros((ORDER, N))
for ord in range(ORDER):
    constants[ord] = y_after_diff_without_end[:, 0]
    y_after_diff_without_end = differentiate(y_after_diff_without_end)

_, T = y_after_diff_without_end.shape
P = 4  # hand-chosen
lam = 1e-2 * np.log(N * N * P)  # hand-chosen

# %% rank estimation and noise estimation
repeat = 10
noise_var = np.zeros((N, N))
ranks_estimated = np.zeros(3)
for k in range(repeat):
    A_estimated_NN = NN_compute(y_after_diff_without_end, P, lam, A_init="random")[0]  # depends upon an initial guess.
    _ranks_estimated = rank_selection(A_estimated_NN, T)
    ranks_estimated += [int(_ranks_estimated[i]) for i in range(3)]
    noise_var += estimate_noise_variance(y_after_diff_without_end,
                                         A_estimated_NN)  # performing at the same time a noise estimation
    print(_ranks_estimated)
ranks_estimated = (ranks_estimated / repeat).astype(int)
print(ranks_estimated)
noise_var /= repeat

# %% penalisation for SHORR estimate
pen_l = lambda_optimal(N, P, T, cov=noise_var)
admm = partial(admm_compute, pen_l=pen_l)
# %%
ranks = [1, 2, 1]  # hand-chosen
inference_routine = als_compute_closed_form  # admm #als_compute_closed_form
it = 0
while True and it < 1:
    try:
        it += 1
        A_rand = generate_A_given_rank(N, P, ranks)
        _, A_est, _ = inference_routine(A_init=A_rand, ranks=ranks, y_ts=y_after_diff_without_end)
        y_after_diff_pred = predict(y_after_diff_without_end, A_est, n_points_to_predict, cov=noise_var)
        for i in range(N):
            plt.title(i)
            plt.plot(y_after_diff_pred[i, :])
            plt.show()
        y_int_pred = y_after_diff_pred
        for ord in range(ORDER):
            y_int_pred = integrate_series(constants[ORDER - 1 - ord], y_int_pred)
        print('success')
        break
    except ValueError:
        pass

# %%
for j in range(N):
    y_line_pred = y_int_pred[j, :]
    y_line = y[j, :]
    plt.plot(y_line, label="real")
    plt.plot(y_line_pred, label="pred")
    plt.title(f'{j}')
    plt.legend()
    plt.show()

# %%
error = np.linalg.norm(y_int_pred - y)
print(error)  # SHORR 1.1708 ok : 1245 # mlr : 2.7642

# %%
column_names = ['time', 'serie']
time = np.linspace(0, y.shape[1] - 1, y.shape[1]).reshape(-1, 1).T
result_to_save = np.concatenate((time, y_int_pred)).T

# %%
folder = f'{PATH}result_mlr.npz'
np.savez(folder, y_pred=result_to_save)

# %%
data = np.load(f'{PATH}result_mlr.npz')
time, y_pred = data['y_pred'][:, 0], data['y_pred'][:, 1:]

# %%
for i in range(N):
    time = np.linspace(0, y.shape[1] - 1, y.shape[1])[-n_points_to_predict - 1:].reshape(-1, 1).T
    y_save = y_pred[:, i][-n_points_to_predict - 1:].reshape(-1, 1).T
    result_to_save = np.concatenate((time, y_save)).T
    folder = f'{PATH}series_mlr_{i}.csv'
    np.savetxt(folder, result_to_save, delimiter=',', header=','.join(column_names), comments='', fmt='%.6f')
