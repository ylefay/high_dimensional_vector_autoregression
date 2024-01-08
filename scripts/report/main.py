import pandas as pd
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from hd_var.rank_selection import rank_selection, NN_compute
from hd_var.routines.mlr.als import als_compute_closed_form
from hd_var.routines.shorr.penalization import lambda_optimal
from hd_var.routines.shorr.admm import admm_compute
from hd_var.generate import generate_A_given_rank

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
    df_without_head = df_without_missing_values.iloc[2:]  # Remove 2 first lines of informations
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


def plot_line_array(y, k):
    '''
    Plot the kth component of the time series using numpy array
    '''
    y_line = y[k, :]
    plt.plot(y_line)
    plt.show()


def differentiate(y):
    '''
    Return the series y_t - y_t-1 
    '''
    y_diff = np.diff(y, axis=-1)
    return y_diff


def integrate_series(y_0, y_diff):
    N, T_1 = y_diff.shape
    T = T_1 + 1
    y = np.zeros((N, T))
    y[:, 1:] = np.cumsum(y_diff, axis=-1)
    y += np.outer(y_0, np.ones(T))
    return y


def estimate_noise_variance(y, A):
    """
    Using an estimator A, compute the noise variance
    Assuming centred data.
    """
    N, P = A.shape[0], A.shape[2]
    T = y.shape[1]
    y_hat = np.zeros((N, T))
    for t in range(P, T):
        y_hat[:, t] = np.sum(np.array([np.dot(A[:, :, i], y[:, t - 1 - i]) for i in range(P)]), axis=0)
    noise = y[:, P:] - y_hat[:, P:]
    noise_variance = np.cov(noise)
    return noise_variance


def predict(y, A, n_times):
    N, P = A.shape[0], A.shape[2]
    T = y.shape[1]
    y_out = np.concatenate((y, np.zeros((N, n_times))), axis=1)
    for k in range(n_times):
        y_out[:, T + k] = np.sum(np.array([np.dot(A[:, :, i], y_out[:, T + k - 1 - i]) for i in range(P)]), axis=0)
    return y_out


def normalise_y(y):
    return (y - np.min(y, axis=-1, keepdims=True)) / (
            np.max(y, axis=-1, keepdims=True) - np.min(y, axis=-1, keepdims=True))


plot_line_array(normalise_y(y), 0)

# %%
n_points_to_predict = 8
y_n = normalise_y(y)
y_without_end = y[:, :-n_points_to_predict]
y_without_end_n = normalise_y(y_without_end)
y_diff_without_end = differentiate(y_without_end_n)
y_diff_diff_without_end = differentiate(y_diff_without_end)

N, T = y_diff_diff_without_end.shape
P = 4  # hand-chosen
lam = 1e-2 * np.log(N * N * P)  # hand-chosen

# %% rank estimation and noise estimation
repeat = 1
noise_var = np.zeros((N, N))
for k in range(repeat):
    A_estimated_NN = NN_compute(y_diff_diff_without_end, P, lam, A_init="random")[0]  # depends upon an initial guess.
    ranks_estimated = rank_selection(A_estimated_NN, T)
    ranks_estimated = [int(ranks_estimated[i]) for i in range(len(ranks_estimated))]
    noise_var += estimate_noise_variance(y_diff_diff_without_end,
                                         A_estimated_NN)  # performing at the same time a noise estimation
    print(ranks_estimated)
noise_var /= repeat

# %% penalisation for SHORR estimate
pen_l = lambda_optimal(N, P, T, cov=noise_var)
admm = partial(admm_compute, pen_l=pen_l)
# %%
ranks = [2, 2, 2]  # Those ranks have previously been estimated.
inference_routine = als_compute_closed_form  # admm #als_compute_closed_form
it = 0
while True and it < 1:
    try:
        it += 1
        A_rand = generate_A_given_rank(N, P, ranks)
        _, A_est, _ = inference_routine(A_init=A_rand, ranks=ranks, y_ts=y_diff_diff_without_end)
        y_diff_diff_pred = predict(y_diff_diff_without_end, A_est, n_points_to_predict)
        for i in range(N):
            plt.title(i)
            plt.plot(y_diff_diff_pred[i, :])
            plt.show()
        y_diff_pred = integrate_series(y_diff_without_end[:, 0], y_diff_diff_pred)
        y_pred = integrate_series(y_without_end_n[:, 0], y_diff_pred)
        print('success')
        break
    except ValueError:
        pass

# %%
for j in range(6):
    y_line_pred = y_pred[j, :]
    y_line = y_n[j, :]
    plt.plot(y_line, label="real")
    plt.plot(y_line_pred, label="pred")
    plt.title(f'{j}')
    plt.legend()
    plt.show()

# %%
error = np.linalg.norm(y_pred - y_n)
error = np.sum((y_pred - y_n) ** 2)
print(error)  # SHORR 1.1708 ok : 1245 # mlr : 2.7642

# %%
column_names = ['time', 'serie']
time = np.linspace(0, y.shape[1] - 1, y.shape[1]).reshape(-1, 1).T
result_to_save = np.concatenate((time, y_n)).T

# %%
folder = f'{PATH}series.csv'
np.savetxt(folder, result_to_save, delimiter=',', header=','.join(column_names), comments='', fmt='%.6f')

# %%
data = np.load(f'{PATH}result_shorr.npz')
y_pred = data['y_pred']

# %%
for i in range(6):
    time = np.linspace(0, y.shape[1] - 1, y.shape[1])[-n_points_to_predict - 1:].reshape(-1, 1).T
    y_save = y_pred[i, :][-n_points_to_predict - 1:].reshape(-1, 1).T
    '''
    time = np.linspace(0,y.shape[1]-1,y.shape[1]).reshape(-1,1).T
    y_save = y_normalize[i,:].reshape(-1,1).T
    '''
    result_to_save = np.concatenate((time, y_save)).T
    folder = f'{PATH}series_shorr_{i}.csv'
    np.savetxt(folder, result_to_save, delimiter=',', header=','.join(column_names), comments='', fmt='%.6f')
