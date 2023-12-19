from jax.scipy.optimize import minimize


def minimize_matrix_input(f, init_matrix):
    shape = init_matrix.shape

    def _f(flatten_matrix):
        return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten(), method='BFGS', options={'maxiter': 5})

    return minimization.x.reshape(shape), minimization.fun
