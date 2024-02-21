class SimArgs:
    def __init__(self, n_in: int, n_delays: int, seed: int, r_mu: float, r_std: float, tau_mem: float,
                 noise_std: float):
        # archi
        self.n_in = n_in
        self.n_out = 20
        self.n_delays = n_delays
        # delays
        self.delay_distribution = 'lognormal'
        self.max_delay = 200
        self.r_mu_lognormal = r_mu
        self.r_std_normal = r_std
        self.cap = 1e-12
        self.r_std_absolute = True
        self.r_std_factor = None
        # weight
        self.w_scale = 0.3
        self.pos_w = True # use only positive weights at initizialization
        self.noise_std = noise_std
        # neuron model
        self.tau_mem = tau_mem
        self.v_thr = 1
        # data
        self.nb_rep = 1
        self.timestep = 0.005 # 280 timesteps
        self.truncation = True # to use only 150 instead of 280 timesteps
        # training
        self.lr = 0.0002
        self.nb_epochs = 5
        self.batch_size = 64
        self.seed = seed
