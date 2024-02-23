import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import lognorm, norm
from hyperparameters import SimArgs


# Weights
def normal_visualisation(data):
    # Fit normal distribution to the log of data
    mu, std = norm.fit(data)

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.hist(data, bins=50, density=True, alpha=0.6, color='b')

    # Plot fitted normal distribution
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    pdf_fitted = norm.pdf(x, mu, std)
    ax.plot(x, pdf_fitted, 'r', linewidth=2,
            label=f'Fitted normal\nmu={mu:.4f}, std={std:.4f}')

    ax.set_title('Fitting Normal Distribution')
    ax.set_xlabel('x')
    ax.set_ylabel('Normalized density')
    ax.legend()


def weight_generation(key: jnp.array, sim_params: SimArgs,
                      visualize_plot: bool = True) -> (jnp.array, jnp.array):
    key, subkey = jax.random.split(key)
    normal_dist = jax.random.normal(subkey,
                                    shape=(sim_params.n_out,
                                           sim_params.n_in*sim_params.n_delays
                                           )
                                    )
    w = normal_dist * sim_params.w_scale / jnp.sqrt(sim_params.n_in)
    if visualize_plot:
        normal_visualisation(w.flatten())
    if sim_params.pos_w:
        w = jnp.abs(w)
    return key, w


# Delays
def hrs_normal_params(sim_params: SimArgs) -> (float, float):
    # Needed to compute accurately the mu of the underlying normal distribution
    r_std_lognormal = (sim_params.r_mu_lognormal *
                       jnp.sqrt((jnp.exp(sim_params.r_std_normal ** 2) - 1)))
    r_mu_normal = jnp.log(sim_params.r_mu_lognormal) - 0.5 * jnp.log(
        1 + (r_std_lognormal / sim_params.r_mu_lognormal) ** 2)
    return r_mu_normal


def hrs_generation(key: jnp.array, sim_params: SimArgs) \
        -> (jnp.array, jnp.array):
    key, subkey = jax.random.split(key)
    r_mu_normal = hrs_normal_params(sim_params)
    normal_dist = jax.random.normal(subkey,
                                    shape=(sim_params.n_out,
                                           sim_params.n_in*sim_params.n_delays
                                           )
                                    )
    hrs_normal = r_mu_normal + sim_params.r_std_normal * normal_dist
    hrs_lognormal = jnp.exp(hrs_normal)
    return key, hrs_lognormal


def lognormal_visualisation(data: jnp.array):
    # fitting
    shape, loc, scale = lognorm.fit(data, floc=0)
    print(shape, loc, scale)

    # plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    axes[0].hist(data, bins=50, density=True, alpha=0.6, color='g',
                 label='Data')

    xmin = min(data)
    xmax = max(data)
    x = np.linspace(xmin, xmax, 100)
    pdf_fitted = lognorm.pdf(x, shape, loc=loc, scale=scale)
    axes[0].plot(x, pdf_fitted, 'r', linewidth=2,
                 label=f'Fitted log-normal\nshape={shape:.2f}, loc={loc:.2f}, '
                       f'scale={scale:.2f}')
    axes[0].grid()
    axes[0].legend()

    axes[1].hist(data,
                 bins=np.logspace(np.log10(min(data)),
                                  np.log10(max(data)), 50
                                  ),
                 density=True, alpha=0.6, color='g', label='Data')
    axes[1].set_xscale('log')
    axes[1].plot(x, pdf_fitted, 'r', linewidth=2,
                 label=f'Fitted log-normal\nshape={shape:.2f}, loc={loc:.2f}, '
                       f'scale={scale:.2f}')
    axes[1].grid()
    axes[1].legend()
    plt.show()


def lognormal_delay_generation(key: jnp.array, sim_params: SimArgs,
                               visualize_plot: bool = True) \
        -> (jnp.array, jnp.array):
    key, hrs = hrs_generation(key, sim_params)
    continuous_delays = sim_params.cap * hrs
    discrete_delays = jnp.round(continuous_delays * (1 / sim_params.timestep))
    discrete_delays = jnp.clip(discrete_delays, 0, sim_params.max_delay)
    if visualize_plot:
        lognormal_visualisation(hrs.flatten())
        lognormal_visualisation(continuous_delays.flatten())
        lognormal_visualisation(discrete_delays.flatten())
    return key, discrete_delays


def uniform_delay_generation(key: jnp.array, sim_params: SimArgs) \
        -> (jnp.array, jnp.array):
    key, subkey = jax.random.split(key)
    discrete_delays = (
        jax.random.randint(subkey, (sim_params.n_out,
                                    sim_params.n_in*sim_params.n_delays),
                           minval=0,
                           maxval=sim_params.max_delay + 1))
    return key, discrete_delays


def pattern_delay_generation(sim_params: SimArgs) -> jnp.array:
    delays = jnp.round(
        jnp.arange(sim_params.max_delay,
                   step=sim_params.max_delay/sim_params.n_delays
                   )
    ).astype(jnp.uint8)
    return jnp.tile(delays, (sim_params.n_out, sim_params.n_in))

def delay_generation(key: jnp.array, sim_params: SimArgs, path_to_save: str,
                     visualize_plot: bool = False) -> (jnp.array, jnp.array):
    if sim_params.delay_distribution == 'lognormal':
        key, delays = lognormal_delay_generation(key, sim_params,
                                                 visualize_plot)
    elif sim_params.delay_distribution == 'uniform':
        key, delays = uniform_delay_generation(key, sim_params)
    elif sim_params.delay_distribution == 'pattern':
        delays = pattern_delay_generation(sim_params)
    else:
        raise ValueError(f'Unknown delay distribution '
                         f'{sim_params.delay_distribution}')
    print(f'discrete delays\n'
          f' - mean {jnp.mean(delays):.2f} timesteps '
          f'({jnp.mean(delays)*sim_params.timestep*1000:.0f} ms)')
    path_to_save_delays = os.path.join(path_to_save,
                                       f'd_{sim_params.seed}.npy')
    jnp.save(path_to_save_delays, delays)
    print(f' - saved at {path_to_save_delays}')
    return key, delays


def add_bias(key, w, delays, sim_params):
    key, subkey_bias = jax.random.split(key)
    bias = (jax.random.normal(subkey_bias, shape=(20, 1))
            * sim_params.w_scale / jnp.sqrt(sim_params.n_in))
    biased_w = jnp.hstack((w, bias))
    biased_delays = jnp.hstack((delays, jnp.zeros((20, 1))))
    return key, biased_w, biased_delays


def safe_max_delay(delays):
    # making sure the max delay is greater than 0 as this value will be used
    # later as a dimension
    max_delay = int(jnp.max(delays)) if int(jnp.max(delays)) != 0 else 1
    return max_delay