from hd_var.operations import product, mode_flatten
import jax
import jax.numpy


def test_product():
    OP_key = jax.random.PRNGKey(0)
    random_tensor = jax.random.normal(OP_key, (3, 4, 5))
    random_matrix = jax.random.normal(OP_key, (6, 3))
    assert product(random_tensor, random_matrix, 0).shape == (6, 4, 5)
