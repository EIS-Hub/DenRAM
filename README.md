# Welcome to DenRAM

## Repo Content

- `requirements`: directory containing the requirements files for macOS and
  Linux
- `code`: directory containing the code


- The following folders will be generated at the first execution of the code:
  - `datasets/`: directory containing the audio spikes dataset [SHD](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/).
      - `audiospikes_256/`: directory containing the audio spikes dataset with 256 
  input channels
      - `audiospikes_700/`: directory containing the audio spikes dataset with 64
  input channels
  - `simulations/`: directory containing the simulation results including
    - for each model, refered with a unique id containing the parameters
    (eg. `700_16_bnil0_lr2e-4_maxD200_log600e9_std4e-1_tm20e-3`):
      - trained weights
      - associated random delays
    - `results.csv`: file containing the simulation results of all runs

## Installation Instructions
- Installation takes about 10 minutes

### System Requirements (tested on)
- MacBook Pro (M2 Pro) with macOS Sonoma 14.3.1 (23D60)
- Linux Ubuntu 22.04 LTS + cuda 11 (NVIDIA A6000 x2)

### Python Requirements
- Python 3.10.2
- We recommand creating a virtual environment: `python3 -m venv denram_venv`
#### macOS (Apple Silicon M3 Chip)
- requirements_macos.txt : `pip install -r requirements/requirements_macos.txt`
#### Linux (Ubuntu 20.04)
- requirements_linux.txt : `pip3 install -r requirements/requirements_linux.txt`
- install jax: \
`pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- install PyTorch: \
`pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118`

`
## Running the Code

- change the working directory to `code/`: `cd code/`

### Training
- run a single training: \
`python3 main3.py --n_in $n_in --n_delays $n_delays --seed $seed --r_mu $r_mu --r_std $r_std --tau_mem $tau_mem --noise_std $noise_std`
- Default values for the parameters when running: `python3 main3.py`:
    - `$n_in`: 700
    - `$n_delays`: 16
    - `$seed`: 42
    - `$r_mu`: 600e9
    - `$r_std`: 0.4
    - `$tau_mem`: 20e-3
    - `$noise_std`: 0.1

## Reproducing the Results

- change the working directory to `code/`: `cd code/` (if not already done)

### Training
- reproduce the results of Fig. 4d (accuracy vs noise_std, D1 and D2): 
```bash
chmod a+x reproduce_fig4d_D1andD2_training.sh
./reproduce_fig4d_D1andD2_training.sh
```
### Plotting the results
- executing `./reproduce_fig4d_D1andD2_training.sh` will generate the file 
`simulations/results.csv` containing the simulation results. It will 
automatically plot the graph. 
- If you want to plot the graph again, you can simply execute
`python3 reproduce_fig4d_D1andD2.py`.

### Time Benchmarking (batch size = 64)
- macOS: 
  - 700 inputs, 16 delays: 750 s / epoch 
  - 256 inputs, 16 delays: 240 s / epoch
- Linux: 
  - 700 inputs, 16 delays: 21 s / epoch 
  - 256 inputs, 16 delays: 7 s / epoch


## Pseudocode
Training
```
1: generate random delays D according to a lognormal distribution
2: generate random weights W according to a normal distribution
3: for each epoch do
4:   for each batch in the training set do
5:     delay the input spikes
6:     compute the membrane potential over time V
7:     recover the maximum membrane potential V_max
8:     compute the loss L
9:     compute the gradients G
10:    update the weights W = W - lr * G
11:   end for
12: for each batch in the validation set do
13:   delay the input spikes
14:   compute the membrane potential over time V
15:   recover the maximum membrane potential V_max
16:   compute the loss L_val
17:   compute the accuracy A
18: end for
19: if L_val < best_validation_loss then
20:   update best_validation_loss
21:   save the weights
22: end if
23: for each batch in the test set do
24:   delay the input spikes
25:   compute the membrane potential over time V
26:   recover the maximum membrane potential V_max
27:   compute the accuracy A
28: end for
29: end for
```


