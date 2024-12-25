import os
import time

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.tree_util import tree_map

import ml_collections
import wandb

from samplerPirate import UniformSampler
from loggingPirate import Logger
from utilsPirate import save_checkpoint

import modelForAllenCahn
from utilsPirate import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Создание отчета на w&b
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name)

    logger = Logger()

    # Загрузка данных и эталонных решений
    u_ref, t_star, x_star = get_dataset()
    # Начальные условия
    u0 = u_ref[0, :]
    t0 = t_star[0]
    t1 = t_star[-1]
    x0 = x_star[0]
    x1 = x_star[-1]

    # Определение диапазона значений для данных
    dom = jnp.array([[t0, t1], [x0, x1]])

    # Создание сэмплера для использования обучения по батчам
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Создание временной модели

    model = modelForAllenCahn.AllenCahn(config, u0, t_star, x_star)
    state = jax.device_get(tree_map(lambda x: x[0], model.state))
    params = state.params

    # Инициализация последнего слоя с учетом начальных условий
    t = t_star[::10]
    x = x_star
    u = u0

    tt, xx = jnp.meshgrid(t, x, indexing="ij")
    inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None]])
    u = jnp.tile(u.flatten(), (t.shape[0], 1))

    feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

    coeffs, residuals, rank, s = jnp.linalg.lstsq(feat_matrix, u.flatten(), rcond=None)
    print("least square residuals: ", residuals)
    # Запись результата в конфигурацию
    config.arch.pi_init = coeffs.reshape(-1, 1)
    del model, state, params

    # Создание физичиски информированной модели
    model = modelForAllenCahn.AllenCahn(config, u0, t_star, x_star)

    evaluator = modelForAllenCahn.AllenCanhEvaluator(config, model)
    # Обучение модели
    for step in range(config.training.max_steps):
        start_time = time.time()

        batch = next(res_sampler)

        model.state = model.step(model.state, batch)

        if config.weighting.scheme == "ntk":
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)
                wandb.log(log_dict, step)

                end_time = time.time()

                logger.log_iter(step, start_time, end_time, log_dict)

        # Сохранение
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                    step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model