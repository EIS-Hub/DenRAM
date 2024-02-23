import jax
import jax.numpy as jnp

def lif_delay_forward(state, input_spikes):
    w, delay_buffer, out_spikes, I_in, V_mem, delays = state[0]
    tau_mem, max_delay, timestep, _, _, _ = state[1]
    # Implementing the FIFO replacement mechanism for the delay buffer
    delay_buffer = jnp.concatenate(
        [jax.lax.expand_dims(input_spikes, (0,)), delay_buffer[:-1,:]],
        0
    )
    delayed_input = delay_buffer[
        delays.astype(jnp.int32),
        jnp.append(
            jnp.arange(delay_buffer.shape[1]-1).repeat(
                int((w.shape[1]-1)/(delay_buffer.shape[1]-1))
            ),
        delay_buffer.shape[1]-1)
    ]
    I_in = jnp.einsum('ij,ij->i', w, delayed_input)
    V_mem = jnp.exp(-timestep/tau_mem) * V_mem + I_in
    V_mem = jnp.maximum(0, V_mem) # HW constraint
    updated_state0 = (w, delay_buffer, out_spikes, I_in, V_mem, delays)
    return (updated_state0, state[1]), (I_in, V_mem, out_spikes)


@jax.custom_jvp
def add_noise(w, key, noise_std):
    """ Adds noise only for forward pass
    """
    noisy_w = jnp.where(
        w != 0.0,
        w + jax.random.normal(key, w.shape) * jnp.max(jnp.abs(w)) * noise_std,
        w)

    return noisy_w


@add_noise.defjvp
def add_noise_jvp(primals, tangents):
    weight, key, noise_std = primals
    x_dot, y_dot, z_dot = tangents
    primal_out = add_noise(weight, key, noise_std)
    tangent_out = x_dot
    return primal_out, tangent_out


# vmap(scan)
def prediction_per_sample(w, delays, single_input, hyperparams):
    _, max_delay, _, n_in, _, _ = hyperparams
    V_mem = jnp.zeros((w.shape[0],),)
    I_in = jnp.zeros((w.shape[0],))
    out_spikes = jnp.zeros((w.shape[0],), dtype='float32')

    delay_buffer = jnp.zeros((max_delay, n_in+1), dtype='uint8')

    state = ((w, delay_buffer, out_spikes, I_in, V_mem, delays), hyperparams)
    _, (I_in, V_mem, out_spikes) = jax.lax.scan(lif_delay_forward, state,
                                                single_input)
    return out_spikes, V_mem, I_in


v_prediction_per_sample = jax.vmap(prediction_per_sample,
                                   in_axes=(None, None, 0, None))
j_v_prediction_per_sample = jax.jit(v_prediction_per_sample,
                                    static_argnums=(3,))


def loss_fn(w, delays, batch_spikes, hyperparams, batch_lbls, key_noise):
    _, max_delay, _, _, noise_std, _ = hyperparams
    w_noisy = add_noise(w, key_noise, noise_std)
    # We need to pad the inputs with tails of zeros to account for the delays
    # this is due to the jax.lax.scan function which uses the size of the input
    # to determine the number of iterations
    pad_len = max_delay
    padding = jnp.zeros(
        (pad_len, batch_spikes.shape[0], batch_spikes.shape[2])
    )
    swap = lambda x: jnp.swapaxes(x, 0, 1)
    batch_spikes_padded = swap(jnp.concatenate([swap(batch_spikes), padding],
                                               dtype=jnp.uint8))
    # print(batch_spikes_padded.shape)
    out_spikes, V_mem, _ = j_v_prediction_per_sample(w_noisy, delays,
                                                     batch_spikes_padded,
                                                     hyperparams)
    out = V_mem.max(axis=1)
    # computing loss
    logit = jax.nn.softmax(out, axis=1)
    loss = -jnp.mean((jnp.log(logit[jnp.arange(batch_spikes.shape[0]),
                                    batch_lbls])))
    # computing acc
    pred = jnp.argmax(out, axis=1)
    acc = jnp.count_nonzero(pred == batch_lbls)/len(batch_lbls)
    return loss, acc


loss_fn_jit = jax.jit(loss_fn, static_argnums=(3,))


def update_opt(w, delays, input_spikes, hyperparams, gt_y, key_noise):
    (loss, acc), grad = \
        jax.value_and_grad(loss_fn, has_aux=True)(w, delays, input_spikes,
                                                  hyperparams, gt_y, key_noise)
    return (loss, acc), grad


update_opt_jit = jax.jit(update_opt, static_argnums=(3,))


def run_epoch(key, loader, hyperparams, sim_params, biased_delays,
              opt_state, get_params, e=None, opt_update=None):
    sim_len = hyperparams[5]
    local_loss = []
    local_acc = []
    for idx, batch in enumerate(loader):
        w = get_params(opt_state)
        batch_in = batch[0]
        biased_inputs = jnp.concatenate(
            (batch_in, jnp.ones((sim_params.batch_size, sim_len, 1))),
            axis=2, dtype=jnp.uint8)
        batch_labels = batch[1]
        key, subkey_noise = jax.random.split(key)
        if e is not None:
            (loss, acc), grads = update_opt_jit(w, biased_delays,
                                                biased_inputs, hyperparams,
                                                batch_labels, subkey_noise)
            opt_state = opt_update(e, grads, opt_state)
        else:
            loss, acc = loss_fn_jit(w, biased_delays, biased_inputs,
                                    hyperparams, batch_labels, subkey_noise)
        local_loss.append(loss)
        local_acc.append(acc)
    mean_local_loss = jnp.mean(jnp.array(local_loss))
    mean_local_acc = jnp.mean(jnp.array(local_acc))
    return key, opt_state, mean_local_loss, mean_local_acc
