import os


def log_params(sim_params):
    print(f'Simulation parameters:')
    print(f'\ttimestep = {sim_params.timestep}')
    print(f'\ttau_mem = {sim_params.tau_mem}')
    print(f'\tbatch_size = {sim_params.batch_size}')
    print(f'\tlr = {sim_params.lr}')
    print(f'\tdelay_distribution = {sim_params.delay_distribution}')
    if sim_params.delay_distribution == 'lognormal':
        print(f'\t--> clipping delays to max_delay = {sim_params.max_delay}')
    print(f'\tr_mu = {sim_params.r_mu_lognormal}')
    print(f'\tr_std = {sim_params.r_std_normal}')


def create_save_folder(sim_params):
    base_path = '../simulations'  # 'check_params'
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f'folder created: {base_path}')

    # building a simulation ID based on the parameters
    r_mu_lognormal_str = f'{int(sim_params.r_mu_lognormal // 1e9)}e9'
    r_std_normal_str = f'{int(sim_params.r_std_normal * 10)}e-1'
    noise_std_str = f'{int(sim_params.noise_std * 100)}'
    tm_str = f'{int(sim_params.tau_mem * 1000)}e-3'
    lr_str = f'{int(sim_params.lr * 1e4)}e-4'
    sim_id = (f'{sim_params.n_in}_{sim_params.n_delays}_bnil{noise_std_str}'
              f'_lr{lr_str}_maxD{sim_params.max_delay}_log{r_mu_lognormal_str}_'
              f'std{r_std_normal_str}_tm{tm_str}')
    path_to_save = os.path.join(base_path, sim_id, f'hw_aware')

    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
        print(f'folder created: {path_to_save}')

    return path_to_save
