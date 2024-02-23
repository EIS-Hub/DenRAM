#!/bin/bash

nb_epochs=100

# D2 training

# for training with noise injection 0% to 20% we use the optimal parameters
# found in the grid search using 10% noise injection
seed_values=(42 50 9)
n_in_values=(256)
n_delays_values=(16)
r_mu_values=(500e9)
r_std_values=(0.4)
tau_mem_values=(15e-3)
noise_std_values=(0.0 0.05 0.1 0.15 0.2)

total=$((${#seed_values[@]} * ${#n_in_values[@]} * ${#n_delays_values[@]} * ${#r_mu_values[@]} * ${#r_std_values[@]} * ${#tau_mem_values[@]} * ${#noise_std_values[@]}))
total=$((total * 2)) # D1 and D2

## Initialize current simulation counter
current=0

## Loop over each parameter, starting with seed
for noise_std in "${noise_std_values[@]}"; do
    for n_in in "${n_in_values[@]}"; do
        for n_delays in "${n_delays_values[@]}"; do
            for r_mu in "${r_mu_values[@]}"; do
                for r_std in "${r_std_values[@]}"; do
                    for tau_mem in "${tau_mem_values[@]}"; do
                        for seed in "${seed_values[@]}"; do
                            # Increment current simulation counter
                            ((current++))

                            # Echo current simulation number in red
                            echo -e "\e[31mSimulation $current / $total\e[0m"

                            # Execute the command with the current set of parameters
                            python3 main.py --n_in $n_in --n_delays $n_delays --seed $seed --r_mu $r_mu --r_std $r_std --tau_mem $tau_mem --noise_std $noise_std --nb_epochs $nb_epochs
                        done
                    done
                done
            done
        done
    done
done

python3 main.py --n_in 256 --n_delays 16 --seed 42 --r_mu 500e9 --r_std 0.4 --tau_mem 15e-3 --noise_std 0.0 --nb_epochs 1
# D1 training

# for training with noise injection 0% to 20% we use the optimal parameters
# found in the grid search using 10% noise injection
seed_values=(42 50 9)
n_in_values=(700)
n_delays_values=(16)
r_mu_values=(500e9)
r_std_values=(0.4)
tau_mem_values=(20e-3)
noise_std_values=(0.0 0.05 0.1 0.15 0.2)

# Loop over each parameter, starting with seed
for noise_std in "${noise_std_values[@]}"; do
    for n_in in "${n_in_values[@]}"; do
        for n_delays in "${n_delays_values[@]}"; do
            for r_mu in "${r_mu_values[@]}"; do
                for r_std in "${r_std_values[@]}"; do
                    for tau_mem in "${tau_mem_values[@]}"; do
                        for seed in "${seed_values[@]}"; do
                            # Increment current simulation counter
                            ((current++))

                            # Echo current simulation number in red
                            echo -e "\e[31mSimulation $current / $total\e[0m"

                            # Execute the command with the current set of parameters
                            python3 main.py --n_in $n_in --n_delays $n_delays --seed $seed --r_mu $r_mu --r_std $r_std --tau_mem $tau_mem --noise_std $noise_std --nb_epochs $nb_epochs
                        done
                    done
                done
            done
        done
    done
done

cp ../simulations/results.csv ../reproduction/results.csv

# generate the plots
python3 plot_fig4d_D1andD2.py
