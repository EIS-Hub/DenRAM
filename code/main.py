import numpy as np
import argparse

import hyperparameters
from training import train

np.set_printoptions(threshold=100000000)


if __name__ == '__main__':
    # recover parsed arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_in', type=int, default=700)
    parser.add_argument('--n_delays', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--r_mu', type=float, default=500e9)
    parser.add_argument('--r_std', type=float, default=0.4)
    parser.add_argument('--tau_mem', type=float, default=20e-3)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--nb_epochs', type=int, default=100)
    args = parser.parse_args()
    sim_params = hyperparameters.SimArgs(
        args.n_in, args.n_delays, args.seed, args.r_mu, args.r_std,
        args.tau_mem, args.noise_std, args.nb_epochs
    )
    print('Training')
    train(sim_params)

