from jax.scipy.optimize import minimize


def minimize_matrix_input(f, init_matrix, args=()):
    """
    Wrapper around the scipy minimize function to handle matrix input.
    """
    shape = init_matrix.shape

    if args != ():
        def _f(flatten_matrix, *args):
            return f(flatten_matrix.reshape(shape), *args)
    else:
        def _f(flatten_matrix):
            return f(flatten_matrix.reshape(shape))

    minimization = minimize(_f, init_matrix.flatten(), method='BFGS', options={'maxiter': 100}, args=args)

    return minimization.x.reshape(shape), minimization.fun
