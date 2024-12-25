''' Подключение к wandb: wandb login API'''
import os
from absl import app
from absl import flags
from ml_collections import config_flags
import jax
import trainMlp

jax.config.update("jax_default_matmul_precision", "highest")

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", ".", "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./Mlp.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    trainMlp.train_and_evaluate(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)