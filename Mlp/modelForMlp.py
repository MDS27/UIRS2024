import jax
from jax import lax, jit, grad, vmap, pmap, random, tree_map, jacfwd, jacrev
import jax.numpy as jnp
from jax.tree_util import tree_map, tree_reduce, tree_leaves
from evaluatorMlp import BaseEvaluator
from utilsMlp import flatten_pytree
import archMlp
from functools import partial
from typing import Dict
from flax.training import train_state
from flax import jax_utils
import optax

class TrainState(train_state.TrainState):
    weights: Dict
    momentum: float

    def apply_weights(self, weights, **kwargs):
        running_average = (
            lambda old_w, new_w: old_w * self.momentum + (1 - self.momentum) * new_w
        )
        weights = tree_map(running_average, self.weights, weights)
        weights = lax.stop_gradient(weights)

        return self.replace(
            step=self.step,
            params=self.params,
            opt_state=self.opt_state,
            weights=weights,
            **kwargs,
        )

def _create_train_state(config, params=None, weights=None):
    arch = archMlp.Mlp(**config.arch)
    x = jnp.ones(config.input_dim)

    lr = optax.exponential_decay(
        init_value=config.optim.learning_rate,
        transition_steps=config.optim.decay_steps,
        decay_rate=config.optim.decay_rate,
        staircase=config.optim.staircase,
    )

    if config.optim.warmup_steps > 0:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=config.optim.learning_rate,
            transition_steps=config.optim.warmup_steps,
        )

        lr = optax.join_schedules([warmup, lr], [config.optim.warmup_steps])

    tx = optax.adam(
        learning_rate=lr, b1=config.optim.beta1, b2=config.optim.beta2, eps=config.optim.eps
    )

    if config.optim.grad_accum_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.optim.grad_accum_steps)

    if params is None:
        params = arch.init(random.PRNGKey(config.seed), x)

    if weights is None:
        weights = dict(config.weighting.init_weights)

    state = TrainState.create(
        apply_fn=arch.apply,
        params=params,
        tx=tx,
        weights=weights,
        momentum=config.weighting.momentum,
    )
    return jax_utils.replicate(state)

class ForwardIVP:
    def __init__(self, config):
        self.config = config
        self.state = _create_train_state(config)
        if config.weighting.use_causal:
            self.tol = config.weighting.causal_tol
            self.num_chunks = config.weighting.num_chunks
            self.M = jnp.triu(jnp.ones((self.num_chunks, self.num_chunks)), k=1).T

    def u_net(self, params, *args):
        raise NotImplementedError()

    def r_net(self, params, *args):
        raise NotImplementedError()

    def losses(self, params, batch, *args):
        raise NotImplementedError()

    def compute_diag_ntk(self, params, batch, *args):
        raise NotImplementedError()

    @partial(jit, static_argnums=(0,))
    def loss(self, params, weights, batch, *args):
        losses = self.losses(params, batch, *args)
        weighted_losses = tree_map(lambda x, y: x * y, losses, weights)
        loss = tree_reduce(lambda x, y: x + y, weighted_losses)
        return loss

    @partial(jit, static_argnums=(0,))
    def compute_weights(self, params, batch, *args):
        grads = jacrev(self.losses)(params, batch, *args)
        grad_norm_dict = {}
        for key, value in grads.items():
            flattened_grad = flatten_pytree(value)
            grad_norm_dict[key] = jnp.linalg.norm(flattened_grad)
        mean_grad_norm = jnp.mean(jnp.stack(tree_leaves(grad_norm_dict)))

        w = tree_map(
            lambda x: (mean_grad_norm / (x + 1e-5 * mean_grad_norm)), grad_norm_dict
        )
        return w

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def update_weights(self, state, batch, *args):
        weights = self.compute_weights(state.params, batch, *args)
        weights = lax.pmean(weights, "batch")
        state = state.apply_weights(weights=weights)
        return state

    @partial(pmap, axis_name="batch", static_broadcasted_argnums=(0,))
    def step(self, state, batch, *args):
        grads = grad(self.loss)(state.params, state.weights, batch, *args)
        grads = lax.pmean(grads, "batch")
        state = state.apply_gradients(grads=grads)
        return state

class AllenCahn(ForwardIVP):
    def __init__(self, config, u0, t_star, x_star):
        super().__init__(config)

        self.u0 = u0
        self.t_star = t_star
        self.x_star = x_star

        self.t0 = t_star[0]
        self.t1 = t_star[-1]

        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))
        self.r_pred_fn = vmap(vmap(self.r_net, (None, None, 0)), (None, 0, None))

    def u_net(self, params, t, x):
        z = jnp.stack([t, x])
        _, u = self.state.apply_fn(params, z)
        return u[0]

    def r_net(self, params, t, x):
        u = self.u_net(params, t, x)
        u_t = grad(self.u_net, argnums=1)(params, t, x)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x)
        return u_t + 5 * u**3 - 5 * u - 0.0001 * u_xx

    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):

        t_sorted = batch[:, 0].sort()
        r_pred = vmap(self.r_net, (None, 0, 0))(params, t_sorted, batch[:, 1])

        r_pred = r_pred.reshape(self.num_chunks, -1)
        l = jnp.mean(r_pred**2, axis=1)
        w = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ l)))
        return l, w

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):

        u_pred = vmap(self.u_net, (None, None, 0))(params, self.t0, self.x_star)
        ics_loss = jnp.mean((self.u0 - u_pred) ** 2)

        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
            res_loss = jnp.mean((r_pred) ** 2)

        loss_dict = {"ics": ics_loss, "res": res_loss}
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star, self.x_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error

class AllenCanhEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error


    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)


        if self.config.logging.log_nonlinearities:
            layer_keys = [
                key
                for key in state.params["params"].keys()
                if key.endswith(
                    tuple(
                        [f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]
                    )
                )
            ]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params["params"][key]["alpha"]

        return self.log_dict
