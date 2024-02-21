import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
px = 1 / plt.rcParams['figure.dpi']  # pixel in inches

fig_size = (700*px, 500*px)
label_fontsize = 1200*px
fig_size = (700*px, 600*px)
legend_fontsize = 1200*px
xy_label_fontsize = 1200*px
tick_fontsize = 1200*px
tick_size = 500*px
tick_width = 200*px
grid_line_width = 100*px

def plot(df256, df700):
    noise = np.array(df256.index)*100
    noise = noise.astype(int)

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.errorbar(noise, df700['best_val_test_acc_mean']*100,
                yerr=df700['best_val_test_acc_std']*100,
                fmt='o-', color='tab:green', label='D1: 700 in, 16 delays',
                capsize=5, capthick=2)
    ax.errorbar(noise, df256['best_val_test_acc_mean']*100,
                yerr=df256['best_val_test_acc_std']*100,
                fmt='o-', color='tab:olive', label='D2: 256 in, 16 delays',
                capsize=5, capthick=2)
    ax.set_xlabel('Noise (% of max{|W|})', size=xy_label_fontsize)
    ax.set_ylabel('Test Accuracy [%]', size=xy_label_fontsize)
    ax.tick_params(labelsize=tick_fontsize, direction='in', size=tick_size,
                   width=tick_width)
    ax.set_xticks(noise)
    ax.set_ylim(20, 102)
    ax.legend(loc='lower left', fontsize=legend_fontsize, frameon=False, ncol=1)
    ax.grid(linestyle='-', linewidth=grid_line_width, alpha=0.5)
    plt.tight_layout()
    plt.savefig('acc_vs_noise.pdf', format='pdf', transparent=True)
    plt.savefig('acc_vs_noise.jpg', format='jpg', transparent=True)
    plt.show()


def recover_mean_and_std_vs_noise_std(df, n_in):
    df = df[df['n_in'] == n_in]
    df = df[['noise_std', 'test_acc']]
    df = df.groupby('noise_std').agg(['mean', 'std'])
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    print(df)
    return df


def generate_figure():
    df = pd.read_csv('../simulations/results.csv')
    df256 = recover_mean_and_std_vs_noise_std(df, 256)
    df700 = recover_mean_and_std_vs_noise_std(df, 700)
    # assert that they have the same number of noise levels
    assert len(df256) == len(df700), (
         f'Different number of noise levels for 256 and 700 inputs '
         f'respectively {len(df256)} and {len(df700)}'
    )
    # assert that the noise levels are the same
    assert np.all(df256.index == df700.index), (
        f'Noise levels are different for 256 and 700 inputs, ' 
        f'respectively {df256.index} and {df700.index}'
    )
    plot(df256, df700)


if __name__ == '__main__':
    generate_figure()