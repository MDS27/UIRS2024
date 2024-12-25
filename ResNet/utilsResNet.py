import os
import scipy.io
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, grad, tree_map
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from flax.training import checkpoints

def get_dataset():
    data = scipy.io.loadmat("data/allen_cahn.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()
    x_star = data["x"].flatten()

    return u_ref, t_star, x_star

@partial(jit, static_argnums=(0,))
def jacobian_fn(apply_fn, params, *args):
    J = grad(apply_fn, argnums=0)(params, *args)
    J, _ = ravel_pytree(J)
    return J

@partial(jit, static_argnums=(0,))
def ntk_fn(apply_fn, params, *args):
    J = jacobian_fn(apply_fn, params, *args)
    K = jnp.dot(J, J)
    return K

def save_checkpoint(state, workdir, keep=5, name=None):
    if not os.path.isdir(workdir):
        os.makedirs(workdir)

    if jax.process_index() == 0:
        # Get the first replica's state and save it.
        state = jax.device_get(tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step=step, keep=keep)


def flatten_pytree(pytree):
    return ravel_pytree(pytree)[0]