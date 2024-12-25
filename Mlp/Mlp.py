import ml_collections
import jax.numpy as jnp

def get_config():
    config = ml_collections.ConfigDict()

    config.mode = "train"

    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Uirs"
    wandb.name = "MlpAllen9Layers"
    wandb.tag = None

    config.use_pi_init = False

    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 9
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1.0, "embed_dim": 256})
    arch.reparam = None
    arch.pi_init = None

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.staircase = False
    optim.warmup_steps = 5000
    optim.grad_accum_steps = 0

    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 100000
    training.batch_size_per_device = 1024

    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = True
    weighting.causal_tol = 1.0
    weighting.num_chunks = 16

    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 1000
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_nonlinearities = False
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False

    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = None
    saving.num_keep_ckpts = 2

    config.input_dim = 2

    config.seed = 42

    return config
