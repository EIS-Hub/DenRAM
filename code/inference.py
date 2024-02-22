import jax
import jax.numpy as jnp
import numpy as np
import os

from hyperparameters import SimArgs
from utils_training import run_epoch
import utils
import parameters
import data

def check_save_folder(sim_params: SimArgs):
    base_path = '../simulations'  # 'check_params'

    # building a simulation ID based on the parameters
    r_mu_lognormal_str = f'{int(sim_params.r_mu_lognormal // 1e9)}e9'
    r_std_normal_str = f'{int(sim_params.r_std_normal * 10)}e-1'
    noise_std_str = f'{int(sim_params.noise_std * 100)}'
    tm_str = f'{int(sim_params.tau_mem * 1000)}e-3'
    sim_id = (f'{sim_params.n_in}_{sim_params.n_delays}_bnil{noise_std_str}'
              f'_c2_lr2e-4_maxD{sim_params.max_delay}_log{r_mu_lognormal_str}_'
              f'std{r_std_normal_str}_tm{tm_str}')
    path_to_load = os.path.join(base_path, sim_id, f'hw_aware')

    # raise an error if not os.path.exists(path_to_load):
    if not os.path.exists(path_to_load):
        raise ValueError(f'No simulation found at {path_to_load}')

    return path_to_load

def add_bias_to_delays(delays: jnp.ndarray) -> jnp.ndarray:
    biased_delays = jnp.hstack((delays, jnp.zeros((20, 1))))
    return biased_delays


def load_trained_params(sim_params: SimArgs) -> [jnp.ndarray, jnp.ndarray]:
    path_to_load = check_save_folder(sim_params)
    path_to_w = os.path.join(path_to_load, f'w_{sim_params.seed}.npy')
    path_to_delays = os.path.join(path_to_load, f'dd_{sim_params.seed}.npy')
    if not os.path.exists(path_to_w):
        raise ValueError(f'No weights found at {path_to_w}')
    if not os.path.exists(path_to_delays):
        raise ValueError(f'No delays found at {path_to_delays}')
    w = np.load(os.path.join(path_to_load, f'w_{sim_params.seed}.npy'))
    delays = np.load(os.path.join(path_to_load, f'dd_{sim_params.seed}.npy'))
    if delays.shape[1] < w.shape[1]:
        delays = add_bias_to_delays(delays)
    return w, delays



def infer(args: SimArgs):
    key = jax.random.PRNGKey(args.seed)
    utils.log_params(args)
    # load the weights and delays
    w, delays = load_trained_params(args)

    train_loader, val_loader, test_loader = data.get_data_loaders(args)

    sim_len = 150 if args.truncation else 280
    max_discrete_delay = parameters.safe_max_delay(delays)
    hyperparams = (args.tau_mem, max_discrete_delay, args.timestep, args.n_in,
                   args.noise_std, sim_len)

    key, w, _, train_loss, train_acc = (
        run_epoch(key, w, train_loader, hyperparams, args, delays)
    )
    key, w, _, val_loss, val_acc = (
        run_epoch(key, w, val_loader, hyperparams, args, delays)
    )
    key_test = jax.random.PRNGKey(0)
    _, w, _, test_loss, test_acc = (
        run_epoch(key_test, w, test_loader, hyperparams, args, delays)
    )

    print(f'{"best_val_train_acc":<15} | '
          f'{"best_val_acc":<15} | '
          f'{"best_val_test_acc":<15}')
    print(f'{train_acc:.4f}{"":<9} | '
          f'{val_acc:.4f}{"":<9} | '
          f'{test_acc:.4f}')