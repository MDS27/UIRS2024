from typing import  Callable, Tuple, Optional, Union, Dict
import flax
from flax import linen as nn
from flax.core.frozen_dict import freeze

import jax
from jax import random, jit, vmap
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, zeros, constant

def _weight_fact(init_fn, mean, stddev):
    def init(key, shape):
        key1, key2 = random.split(key)
        w = init_fn(key1, shape)
        g = mean + normal(stddev)(key2, (shape[-1],))
        g = jnp.exp(g)
        v = w / g
        return g, v

    return init

# периодические вложения
class PeriodEmbs(nn.Module):
    period: Tuple[float]  # Периоды для разных осей
    axis: Tuple[int]  # Оси, к которым применяются периодические вложения
    trainable: Tuple[bool]  # Указывает, являются ли периоды обучаемыми

    def setup(self):
        period_params = {}
        for idx, is_trainable in enumerate(self.trainable):
            if is_trainable:
                period_params[f"period_{idx}"] = self.param(
                    f"period_{idx}", constant(self.period[idx]), ()
                )
            else:
                period_params[f"period_{idx}"] = self.period[idx]
        self.period_params = freeze(period_params)

    # Применяет периодические вложения к указанным осям.
    @nn.compact
    def __call__(self, x):
        y = []
        for i, xi in enumerate(x):
            if i in self.axis:
                idx = self.axis.index(i)
                period = self.period_params[f"period_{idx}"]
                y.extend([jnp.cos(period * xi), jnp.sin(period * xi)])
            else:
                y.append(xi)
        return jnp.hstack(y)

# Вложение фурье
class FourierEmbs(nn.Module):
    embed_scale: float
    embed_dim: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", normal(self.embed_scale), (x.shape[-1], self.embed_dim // 2)
        )
        y = jnp.concatenate(
            [jnp.cos(jnp.dot(x, kernel)), jnp.sin(jnp.dot(x, kernel))], axis=-1
        )
        return y


class Embedding(nn.Module):
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.periodicity:
            x = PeriodEmbs(**self.periodicity)(x)

        if self.fourier_emb:
            x = FourierEmbs(**self.fourier_emb)(x)

        return x

# Полносвязные слои
class Dense(nn.Module):
    features: int
    kernel_init: Callable = glorot_normal()
    bias_init: Callable = zeros
    reparam: Union[None, Dict] = None

    @nn.compact
    def __call__(self, x):
        if self.reparam is None:
            kernel = self.param(
                "kernel", self.kernel_init, (x.shape[-1], self.features)
            )
        # факторизация
        elif self.reparam["type"] == "weight_fact":
            g, v = self.param(
                "kernel",
                _weight_fact(
                    self.kernel_init,
                    mean=self.reparam["mean"],
                    stddev=self.reparam["stddev"],
                ),
                (x.shape[-1], self.features),
            )
            kernel = g * v

        bias = self.param("bias", self.bias_init, (self.features,))

        y = jnp.dot(x, kernel) + bias

        return y


# Описывает блок PirateNet
class PIModifiedBottleneck(nn.Module):
    hidden_dim: int
    output_dim: int
    activation: str
    nonlinearity: float
    reparam: Union[None, Dict]

    def setup(self):
        self.activation_fn = jnp.tanh

    @nn.compact
    def __call__(self, x, u, v):
        identity = x
        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)
        x = x * u + (1 - x) * v
        x = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)
        x = x * u + (1 - x) * v
        x = Dense(features=self.output_dim, reparam=self.reparam)(x)
        x = self.activation_fn(x)
        alpha = self.param("alpha", constant(self.nonlinearity), (1,))
        x = alpha * x + (1 - alpha) * identity
        return x

#  Архитектура PirateNet
class PirateNet(nn.Module):
    # Значения по умолчанию, будут переопределены значениями конфигурации
    arch_name: Optional[str] = "PirateNet"
    num_layers: int = 2
    hidden_dim: int = 256
    out_dim: int = 1
    activation: str = "tanh"
    nonlinearity: float = 0.0
    periodicity: Union[None, Dict] = None
    fourier_emb: Union[None, Dict] = None
    reparam: Union[None, Dict] = None
    pi_init: Union[None, jnp.ndarray] = None

    def setup(self):
        self.activation_fn = jnp.tanh

    @nn.compact
    def __call__(self, x):
        # Вложения Фурье
        embs = Embedding(periodicity=self.periodicity, fourier_emb=self.fourier_emb)(x)
        x = embs

        u = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        u = self.activation_fn(u)

        v = Dense(features=self.hidden_dim, reparam=self.reparam)(x)
        v = self.activation_fn(v)

        # Создание блоков
        for _ in range(self.num_layers):
            x = PIModifiedBottleneck(
                hidden_dim=self.hidden_dim,
                output_dim=x.shape[-1],
                activation=self.activation,
                nonlinearity=self.nonlinearity,
                reparam=self.reparam,
            )(x, u, v)

        if self.pi_init is not None:
            kernel = self.param("pi_init", constant(self.pi_init), self.pi_init.shape)
            y = jnp.dot(x, kernel)

        else:
            y = Dense(features=self.out_dim, reparam=self.reparam)(x)

        return x, y


