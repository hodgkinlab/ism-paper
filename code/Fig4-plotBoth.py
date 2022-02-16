import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
# pd.set_option('display.max_rows', 999999)
# pd.set_option('display.max_columns', 999999)
# pd.set_option('display.expand_frame_repr', False)
options = {'strings_to_formulas': False, 'strings_to_urls': False}  # for saving fit results to excel file
rc = {
	'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
	'xtick.labelsize': 16, 'ytick.labelsize': 16,
	'figure.figsize': (6, 5),
	'axes.grid': False, 'grid.linestyle': ':', 'axes.grid.which': 'both',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 5.5, 'lines.linewidth': 1.0,
	'errorbar.capsize': 2.5
}
sns.set(context='paper', style='white', rc=rc)

exp1_xlsx = './out/Fig4/results.xlsx'
rss1 = pd.read_excel(exp1_xlsx, sheet_name='efficiency', index_col=0)
par_fit1 = pd.read_excel(exp1_xlsx, sheet_name='pars_efficiency', index_col=0)

exp2_xlsx = './out/Fig4/repeat/results.xlsx'
rss2 = pd.read_excel(exp2_xlsx, sheet_name='efficiency', index_col=0)
par_fit2 = pd.read_excel(exp2_xlsx, sheet_name='pars_efficiency', index_col=0)

idx1 = np.argmin(np.abs(rss1.index.to_numpy() - par_fit1['best-fit'].values[0]*100))
idx2 = np.argmin(np.abs(rss2.index.to_numpy() - par_fit2['best-fit'].values[0]*100))



## Plot RSS
cp = sns.color_palette()
fig1, ax1 = plt.subplots(tight_layout=True)
ax1.plot(rss1.index, rss1['RSS'], '-', color='k', lw=1.5, label='Exp 1')
ax1.fill_between(rss1.index, rss1['Lower95%'], rss1['Upper95%'], color='k', alpha=0.3)
ax1.errorbar([par_fit1['best-fit']*100], [rss1['RSS'].to_numpy()[idx1]], 
              xerr=[[(par_fit1['best-fit']-par_fit1['Lower95%']).values[0]*100],
                    [(par_fit1['Upper95%']-par_fit1['best-fit']).values[0]*100]],
              fmt='o', color='k', mec='k', 
              label=f"${par_fit1['best-fit'].values[0]*100:.3f}\pm_{{{(par_fit1['best-fit']-par_fit1['Lower95%']).values[0]*100:.3f}}}^{{{(par_fit1['Upper95%']-par_fit1['best-fit']).values[0]*100:.3f}}}$%")

ax1.plot(rss2.index, rss2['RSS'], '-.', color=cp[5], lw=1.5, label='Exp 2 (Repeat)')
ax1.fill_between(rss2.index, rss2['Lower95%'], rss2['Upper95%'], color=cp[5], alpha=0.3)
ax1.errorbar([par_fit2['best-fit']*100], [rss2['RSS'].to_numpy()[idx2]], 
              xerr=[[(par_fit2['best-fit']-par_fit2['Lower95%']).values[0]*100],
                    [(par_fit2['Upper95%']-par_fit2['best-fit']).values[0]*100]],
              fmt='X', color=cp[5], mec='k',
              label=f"${par_fit2['best-fit'].values[0]*100:.3f}\pm_{{{(par_fit2['best-fit']-par_fit2['Lower95%']).values[0]*100:.3f}}}^{{{(par_fit2['Upper95%']-par_fit2['best-fit']).values[0]*100:.3f}}}$%")
ax1.legend(ncol=2, fontsize=14, handletextpad=0.3, columnspacing=0.1)
ax1.set(xlabel="%Efficiency", xlim=(0,100),
        ylabel="RSS", ylim=(0,None))


## Import AID+ and IgG1+ measurements
aid_xlsx1 = './data/Fig4/Fig4I_AIDproportion_division.xlsx'
aid_xlsx2 = './data/Fig4/repeat/IgG_IgE_data.xlsx'
aid_df1 = pd.read_excel(aid_xlsx1, usecols="A:D", index_col=0)
aid_df2 = pd.read_excel(aid_xlsx2, sheet_name='AID Div', index_col=0)

igg1_xlsx1 = './data/Fig4/Fig4J_IgG1switching_division.xlsx'
igg1_xlsx2 = './data/Fig4/repeat/IgG_IgE_data.xlsx'
igg1_df1 = pd.read_excel(igg1_xlsx1, usecols="A:D", index_col=0)
igg1_df2 = pd.read_excel(igg1_xlsx2, sheet_name='IgG1 Div', index_col=0)

## Import model fits
exp_aid1 = pd.read_excel(exp1_xlsx, sheet_name='AID', index_col=0)
exp_aid2 = pd.read_excel(exp2_xlsx, sheet_name='AID', index_col=0)

exp_igg1_1 = pd.read_excel(exp1_xlsx, sheet_name='IgG1', index_col=0)
exp_igg1_2 = pd.read_excel(exp2_xlsx, sheet_name='IgG1', index_col=0)

gens = igg1_df1.index


## Plot AID+
cp = sns.color_palette("Paired")
fig2, ax2 = plt.subplots(tight_layout=True)
ax2.errorbar(gens, aid_df1.mean(axis=1), yerr=aid_df1.sem(axis=1), fmt='o--', color='b', mfc='none', label="Exp 1")
ax2.plot(gens, exp_aid1['AID+'], '.-', color='b', label="AID+")
ax2.fill_between(gens, exp_aid1['Lower95%'], exp_aid1['Upper95%'], color='b', ec=None, alpha=0.3)

ax2.errorbar(gens, aid_df2.mean(axis=1)/100, yerr=aid_df2.sem(axis=1)/100, fmt='o-.', color=cp[3], mfc='none', label="Exp 2 (Repeat)")
ax2.plot(gens, exp_aid2['AID+'], '.-', color=cp[3], label="AID+")
ax2.fill_between(gens, exp_aid2['Lower95%'], exp_aid2['Upper95%'], color=cp[3], ec=None, alpha=0.5)
ax2.set(ylabel="Proportion AID+", xlabel="Division", ylim=(0, None))
ax2.legend(loc='upper left', ncol=2, handletextpad=0.5, columnspacing=0.5, fontsize=14)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))


## Plot IgG1+
fig3, ax3 = plt.subplots(tight_layout=True)
ax3.errorbar(gens, igg1_df1.mean(axis=1), yerr=igg1_df1.sem(axis=1), fmt='o--', color=cp[9], mfc='none', label="Exp 1")
ax3.plot(gens, exp_igg1_1['IgG1+'], '.-', c=cp[9], label="IgG1+")
ax3.fill_between(gens, exp_igg1_1['Lower95%'], exp_igg1_1['Upper95%'], color=cp[9], ec=None, alpha=0.3)

ax3.errorbar(gens, igg1_df2.mean(axis=1)/100, yerr=igg1_df2.sem(axis=1)/100, fmt='o-.', color=cp[5], mfc='none', label="Exp 2 (Repeat)")
ax3.plot(gens, exp_igg1_2['IgG1+'], '.-', c=cp[5], label="IgG1+")
ax3.fill_between(gens, exp_igg1_2['Lower95%'], exp_igg1_2['Upper95%'], color=cp[5], ec=None, alpha=0.5)
ax3.set(ylabel="Proportion IgG1+", xlabel="Division", ylim=(0, None))
ax3.legend(loc='upper left', ncol=2, handletextpad=0.5, columnspacing=0.5, fontsize=14)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))


## Save plots
fig1.savefig("./out/Fig4/repeat/compare_rss.pdf", dpi=300)
fig2.savefig("./out/Fig4/repeat/compare_AID.pdf", dpi=300)
fig3.savefig("./out/Fig4/repeat/compare_IgG1.pdf", dpi=300)

# plt.show()