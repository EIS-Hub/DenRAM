class SimArgs:
    def __init__(self, n_in: int, n_delays: int, seed: int, r_mu: float,
                 r_std: float, tau_mem: float, noise_std: float,
                 nb_epochs: int):
        # archi
        self.n_in = n_in  # number of input channels
        self.n_out = 20
        self.n_delays = n_delays  # number of delays per connection between any two neurons
        # delays
        self.delay_distribution = 'lognormal'  # distribution of delays
        self.max_delay = 200
        self.r_mu_lognormal = r_mu  # mean of the lognormal distribution
        self.r_std_normal = r_std  # std of the underlying normal distribution
        self.cap = 1e-12
        # weight
        self.w_scale = 0.3  # scaling used to initialize the weights
        self.pos_w = True  # use only positive weights at initialization
        self.noise_std = noise_std
        # neuron model
        self.tau_mem = tau_mem
        self.v_thr = 1
        # data
        self.nb_rep = 1
        self.timestep = 0.005  # frequency period (to get 280 timesteps)
        self.truncation = True  # to use only 150 instead of 280 timesteps
        # training
        self.lr = 0.0002
        self.nb_epochs = nb_epochs
        self.batch_size = 64
        self.seed = seed
