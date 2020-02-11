
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# home = '/data/'
home = os.path.expanduser('~')

# the raw df
av_df = pd.read_csv(os.path.join(home,'Dropbox/loki_0/simple_rt_experiment_probabilityC/analysis/aggregated_data/av_est.csv'))
av_df = av_df.rename(columns = {'ID':'subj_id'})
av_df = av_df.rename(columns = {'epoch_n':'epoch_number'})

av_df['lambda_val'] = np.nan
av_df['p_optimal'] = np.nan

av_df.loc[av_df.condition == 'hc', 'p_optimal'] = 0.65
av_df.loc[av_df.condition == 'lc', 'p_optimal'] = 0.85
av_df.loc[(av_df.condition == 'hc') | (av_df.condition == 'lc'), 'lambda_val'] = 20
# lambda = 20, shift every 20 trials on avg. for conflict conditions

av_df.loc[av_df.condition == 'hv', 'lambda_val'] = 10
av_df.loc[av_df.condition == 'lv', 'lambda_val'] = 30
av_df.loc[(av_df.condition == 'hv') | (av_df.condition == 'lv'), 'p_optimal'] = 0.75
# p(r)=0.75 for vol. conditions

av_df.to_csv(os.path.join(home, 'Dropbox/loki_0/simple_rt_experiment_probabilityC/analysis/aggregated_data/av_est.csv'), index=False)

# df averaged by subject and epoch trial (all conditions)
mean_av_df = av_df.groupby(['subj_id', 'shifted_epoch_trial'])[['a_est_z', 'v_est_z']].agg('mean').reset_index()
mean_av_df = mean_av_df.loc[(mean_av_df.shifted_epoch_trial <= 8) & (mean_av_df.shifted_epoch_trial >= -1)]



def plot_a_v_time(data, fig, ax,  conditional=False, savefig=None, all_subs=False, linestyle='-', legend=True, home=home):

    # jtplot.style(context='talk', fscale=1.3, spines=False, gridlines='--', )


    # hack to get hue to work with lineplots and markers ...

    fig_path=os.path.join(home, 'Dropbox/loki_0/simple_rt_experiment_probabilityC/analysis/figures/av_time_plots')

    import itertools
    n_plotted_trials = data.shifted_epoch_trial.nunique()

    palette_seed = sns.color_palette('Greens', n_colors=200)[80::10]

    assert len(palette_seed) >= n_plotted_trials, 'check n_colors for color palette'

    palette = itertools.cycle(palette_seed)

    sns.lineplot(data=data, x='a_est_z', y='v_est_z', hue='shifted_epoch_trial', palette=palette_seed[:n_plotted_trials], marker='o');

    x = data.a_est_z
    y = data.v_est_z

    for i in range(len(data)):
        plt.plot(x.values[i:i+2], y.values[i:i+2], color=next(palette), linestyle=linestyle, linewidth=2.5, label=data.subj_id.unique()[0])

    if conditional is True:
        plt.title('subject ' + str(int(data.subj_id.unique()[0])) + ': ' +
                '$\lambda =$ ' + str(int(data.lambda_val.unique()[0])) + ' p = ' + str(data.p_optimal.unique()[0]), fontsize=20)
        fig_name = (str(int(data.subj_id.unique()[0])) + '_' + str(int(data.condition.unique()[0])) +'_a_v_time_color.png')
    if all_subs is True:
        fig_name = ('all_conditions_all_subs_a_v_time_color.png')
    else:
        plt.title('subject ' + str(int(data.subj_id.unique()[0])), fontsize=20)
        fig_name = (str(int(data.subj_id.unique()[0])) +'_all_conditions_a_v_time_color.png')


    plt.xlabel(r'$\hat{a}$')
    plt.ylabel(r'$\hat{v}$')

    if legend:
        legend = ax.legend()
        legend.texts[0].set_text("epoch trial")
    else:
        ax.get_legend().remove()


    if savefig:

        plt.savefig(os.path.join(fig_path, fig_name))


    return fig, ax



fig, ax = plt.subplots()
lines = ["-","--","-.",":"]

import itertools
linecycler = itertools.cycle(lines)

for subj_id in mean_av_df.subj_id.unique():

    fig, ax = plt.subplots()


    sub_data = mean_av_df.loc[mean_av_df.subj_id == subj_id].reset_index().copy()

    plot_a_v_time(sub_data, fig, ax, conditional=False, savefig=True, all_subs=False, legend=True)

# ax.legend(ax.lines[:4], ('s1', 's2', 's3', 's4'))
