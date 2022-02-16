import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', 999999)
# pd.set_option('display.max_columns', 999999)
# pd.set_option('display.expand_frame_repr', False)
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
rc = {
	'font.size': 16, 'axes.titlesize': 16, 'axes.labelsize': 16,
	'xtick.labelsize': 16, 'ytick.labelsize': 16,
	'axes.grid': False, 'grid.linestyle': ':', 'axes.grid.which': 'both',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 5.5, 'lines.linewidth': 1.0,
	'errorbar.capsize': 2.5
}
sns.set(context='paper', style='white', rc=rc)

## Import data
iy1_pre = pd.read_excel("./data/FigS4/predivision_singlecellqPCR_Iy1.xlsx", usecols="A:C")
iy1_pos = pd.read_excel("./data/FigS4/POSTdivision_singlecellqPCR_Iy1.xlsx", usecols="A:C")

## Organise into Pre- and Post-division
iy1_pre['Sort'] = 'Pre-division'
iy1_pre_lo = iy1_pre[iy1_pre['Sort enrichment']=='Lo'].copy()
iy1_pre_hi = iy1_pre[iy1_pre['Sort enrichment']=='Hi'].copy()

iy1_pos['Sort'] = 'Post-division'
iy1_pos_lo = iy1_pos[iy1_pos['Sort enrichment']=='Lo'].copy()
iy1_pos_hi = iy1_pos[iy1_pos['Sort enrichment']=='Hi'].copy()

df = pd.concat([iy1_pre, iy1_pos], ignore_index=True, sort=False)
df.replace(to_replace=dict(Hi='High'), inplace=True)
df.replace(to_replace=dict(Lo='Low'), inplace=True)


## Original plot (recreated)
# fig1, ax1 = plt.subplots(figsize=(5,6), tight_layout=True)
# ax1.errorbar([0, 1], [np.mean(iy1_pre_hi['Iy1']), np.mean(iy1_pos_hi['Iy1'])], 
# 			 yerr=sps.sem(iy1_pos_hi['Iy1']), color='r', fmt='o-', label='High')
# ax1.errorbar([0, 1], [np.mean(iy1_pre_lo['Iy1']), np.mean(iy1_pos_lo['Iy1'])], 
# 			 yerr=sps.sem(iy1_pre_lo['Iy1']), color='b', fmt='o-', label='Low')
# ax1.set(xlabel="", xlim=(-0.5, 1.5), 
# 		xticks=(0,1), xticklabels=("Pre-division", "Post-division"),
# 		ylabel=r"Iy1 $\log_2$(RQ)", ylim=(0, 9))
# ax1.legend(fontsize=16)


## Violin plot
fig2, ax2 = plt.subplots(figsize=(5,6), tight_layout=True)
sns.violinplot(x='Sort', y='Iy1', hue='Sort enrichment', 
			   data=df, split=True, inner=None, palette=['r', 'b'], ax=ax2)
# sns.swarmplot(x='Sort', y='Iy1', hue='Sort enrichment', dodge=True,
# 			  data=df, palette=['r', 'b'], edgecolor='k', linewidth=1, ax=ax2, clip_on=False)
ax2.set(xlabel="", ylabel=r"Iy1 $\log_2$(RQ)", ylim=(0,None))
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles[:2], labels[:2], loc='upper right', fontsize=16)


# fig1.savefig("./out/FigS4/original (recreated).pdf", dpi=300)
fig2.savefig("./out/FigS4/suggested.pdf", dpi=300)

# plt.show()