from hd_var.operations import product, mode_flatten
import jax
import jax.numpy as jnp


def test_product():
    OP_key = jax.random.PRNGKey(0)
    random_tensor = jax.random.normal(OP_key, (3, 4, 5))
    random_matrix = jax.random.normal(OP_key, (6, 3))
    assert product(random_tensor, random_matrix, 0).shape == (6, 4, 5)


def test_matricization():
    X = jnp.array([[[[1, 13]], [[4, 16]], [[7, 19]], [[10, 22]]], [[[2, 14]], [[5, 17]], [[8, 20]], [[11, 23]]],
                   [[[3, 15]], [[6, 18]], [[9, 21]], [[12, 24]]]])
    X1 = jnp.array([[1, 4, 7, 10, 13, 16, 19, 22], [2, 5, 8, 11, 14, 17, 20, 23], [3, 6, 9, 12, 15, 18, 21, 24]])
    X2 = jnp.array([[1, 2, 3, 13, 14, 15], [4, 5, 6, 16, 17, 18], [7, 8, 9, 19, 20, 21], [10, 11, 12, 22, 23, 24]])
    X3 = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]])
    assert jnp.all(mode_flatten(X, 0) == X1)
    assert jnp.all(mode_flatten(X, 1) == X2)
    assert jnp.all(mode_flatten(X, 2) == X3)
