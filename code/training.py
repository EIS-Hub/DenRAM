import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import os
import time

import data, parameters
import utils
from hyperparameters import SimArgs
from utils_training import run_epoch

jnp.set_printoptions(formatter={'float': lambda x: f'{x:.2f}'})


def train(args: SimArgs):
    key = jax.random.PRNGKey(args.seed)
    utils.log_params(args)
    path_to_save = utils.create_save_folder(args)

    key, w_original = parameters.weight_generation(key, args,
                                                   visualize_plot=False)
    with jax.disable_jit():
        key, discrete_delays = parameters.delay_generation(key, args,
                                                           path_to_save)

    path_to_save_weights = os.path.join(path_to_save, f'w_{args.seed}.npy')

    key, biased_w, biased_delays = parameters.add_bias(key, w_original,
                                                       discrete_delays, args)

    sim_len = 150 if args.truncation else 280
    max_discrete_delay = parameters.safe_max_delay(discrete_delays)
    hyperparams = (args.tau_mem, max_discrete_delay, args.timestep, args.n_in,
                   args.noise_std, sim_len)

    train_loader, val_loader, test_loader = data.get_data_loaders(args)

    piecewise_lr = optimizers.piecewise_constant(
        [33, 66], [args.lr, args.lr / 5, args.lr / 20])
    opt_init, opt_update, get_params = optimizers.adam(step_size=piecewise_lr)
    opt_state = opt_init(biased_w)

    print(f'Training with noise: {args.noise_std}')
    print(f'{"epoch" : <6} | '
          f'{"mean_train_loss" : <15} | '
          f'{"mean_train_acc" : <15} | '
          f'{"mean_val_acc" : <15} | '
          f'{"mean_test_acc" : <15} | '
          f'{"epoch_duration": <15}')

    best_val_acc = 0.00001;
    max_patience = 25;
    patience = 0  # Early stopping

    for e in range(args.nb_epochs):
        epoch_start = time.time()
        key, opt_state, train_loss, train_acc = (
            run_epoch(key, train_loader, hyperparams, args, biased_delays,
                      opt_state, get_params, e, opt_update)
        )
        key, _, val_loss, val_acc = (
            run_epoch(key, val_loader, hyperparams, args, biased_delays,
                      opt_state, get_params)
        )
        time_epoch = time.time()

        if val_acc > best_val_acc:
            if (val_acc - best_val_acc) / best_val_acc < 0.01:
                patience += 1
                if patience == max_patience:
                    print('Early stopping')
                    break
            else:
                best_epoch_id = e
                best_val_acc = val_acc
                best_val_train_acc = train_acc
                w_best = get_params(opt_state)
                opt_state_best = opt_state
                patience = 0
                key_test = jax.random.PRNGKey(0)
                _, _, test_loss, test_acc = (
                    run_epoch(key_test, test_loader, hyperparams, args,
                              biased_delays, opt_state, get_params)
                )
                print(f'{e : <6} | '
                      f'{train_loss:.4f}{"":<9} | '
                      f'{train_acc:.4f}{"":<9} | '
                      f'{val_acc:.4f}{"":<9} | '
                      f'{test_acc:.4f}{"":<9} | '
                      f'{time_epoch - epoch_start:.2f}')
                jnp.save(path_to_save_weights, w_best)
        else:
            if e % 10 == 0:
                print(f'{e:<6} | '
                      f'{train_loss:.4f}{"":<9} | '
                      f'{train_acc:.4f}{"":<9} | '
                      f'{val_acc:.4f}{"":<9} | '
                      f'{"":<15} | '
                      f'{time_epoch - epoch_start:.2f}')
            patience += 1
            if patience == max_patience:
                print('Early stopping')
                break

    key_test = jax.random.PRNGKey(0)
    _, _, test_loss, test_acc = (
        run_epoch(key_test, test_loader, hyperparams, args,
                  biased_delays, opt_state_best, get_params)
    )
    print(f'{"epoch":<6} | '
          f'{"best_val_train_acc":<15} | '
          f'{"best_val_acc":<15} | '
          f'{"best_val_test_acc":<15}')
    print(f'{best_epoch_id:<6} | '
          f'{best_val_train_acc:.4f}{"":<9} | '
          f'{best_val_acc:.4f}{"":<9} | '
          f'{test_acc:.4f}')

    print(f'weights: saved at {path_to_save_weights}')

    # check if results.csv exists and create it if not
    csv_path = os.path.join('../simulations', 'results.csv')
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w') as f:
            f.write('n_in,n_delays,noise_std,seed,max_delay,tau_mem,r_mu,r_std,test_acc\n')
    # write results to csv
    with open(csv_path, 'a') as f:
        f.write(f'{args.n_in},{args.n_delays},{args.noise_std},{args.seed},'
                f'{args.max_delay},{args.tau_mem},{args.r_mu_lognormal},'
                f'{args.r_std_normal},{test_acc}\n')
    print(f'results saved to: {csv_path}')
