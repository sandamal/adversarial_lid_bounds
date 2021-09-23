'''
    Author: Sandamal on 20/4/21
    Description: Plot LID line against delta
'''

import numpy as np
import pandas as pd

np.random.seed(2021)

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})

attack = 'fgsm'
k = 100
nq = 50
dataset_name = 'mnist'
# dataset_name = 'cifar'

file_names = ['lb', 'ub', 'original_lb', 'original_ub']
closer_data_frames = {}
farther_data_frames = {}

for file in file_names:
    summary_results = f'../results/{k}/bound_{dataset_name}_{attack}_k_{k}_nq_{nq}_{file}.csv'
    # summary_results = f'../results/k_{k}_cond_split/bound_{dataset_name}_{attack}_k_{k}_nq_{nq}_{file}.csv'
    df = pd.read_csv(summary_results)

    farther_df = df[df['perturb_farther'] == True]
    closer_df = df[df['perturb_farther'] == False]

    farther_df = farther_df.groupby('delta').mean()
    closer_df = closer_df.groupby('delta').mean()

    farther_data_frames[file] = farther_df
    closer_data_frames[file] = closer_df

fig, ax = plt.subplots()

original_ub_df = farther_data_frames.get('original_ub')
x_ticks = original_ub_df.index
lid_b_values = original_ub_df['lid_b'].values.reshape(-1, 1)
x_tick_values = x_ticks.values.reshape(-1, 1)
ax.plot(x_ticks, original_ub_df['original_ub'], ls='--', linewidth=2, label='mean upper-bound', color='tab:red')

original_lb_df = farther_data_frames.get('original_lb')
x_ticks = original_lb_df.index
x_tick_values = np.vstack((x_tick_values, x_ticks.values.reshape(-1, 1)))
lid_b_values = np.vstack((lid_b_values, original_lb_df['lid_b'].values.reshape(-1, 1)))
ax.plot(x_ticks, original_lb_df['original_lb'], ls='--', linewidth=2, label='mean lower-bound', color='tab:green')

lid_b_values = lid_b_values.squeeze()
x_tick_values = x_tick_values.squeeze()
sorted_indices = x_tick_values.argsort()
ax.plot(x_tick_values[sorted_indices], lid_b_values[sorted_indices], ls='-', linewidth=2, label='mean LID estimate', color='tab:blue')

ax.set_xlabel(r'$\delta$')
ax.set_ylabel('LID and its bounds')
# ax.set_ylim([0, 200])
ax.set_xlim([0, 2])

# plt.legend(ncol=3, loc='upper center', fancybox=True, shadow=True, framealpha=1, bbox_to_anchor=(0.5, 1.22), fontsize='x-small')
plt.legend(ncol=1, fancybox=True, shadow=False, framealpha=0.3, fontsize='small')

plt.grid()
# plt.show()
# plt.savefig(f'../figures/bound_summary_{dataset_name}_{attack}_k_{k}_nq_{nq}.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'../figures/bound_summary_{dataset_name}_{attack}_k_{k}_nq_{nq}.png', format='png', dpi=300, bbox_inches='tight')
plt.close()
