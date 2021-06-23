import sys, tqdm
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
	'figure.figsize': (8, 6),
	'axes.grid': False, 'grid.linestyle': ':', 'axes.grid.which': 'both',
	'axes.axisbelow': True, 'axes.titlepad': 0,
	'axes.spines.top': False, 'axes.spines.right': False,
	'axes.spines.left': True, 'axes.spines.bottom': True,
	'ytick.left': True, 'xtick.bottom': True,
	'lines.markersize': 5.5, 'lines.linewidth': 1.0,
	'errorbar.capsize': 2.5
}
sns.set(context='paper', style='white', rc=rc)

## Import IgG1 data
df = pd.read_excel("./data/Fig1/Fig1G_ClonalIgG1data.xlsx"); data_key = 'IgG1'
gens = np.unique(df['Division'])
cp = sns.color_palette("Spectral", n_colors=np.unique(df['Family']).size)


## Sort by family mean
df[f'mean_{data_key}'] = df.groupby('Family')[data_key].transform('mean')
df_sort = df.sort_values(by=f'mean_{data_key}', ascending=False).copy().reset_index(drop=True)
rank_idx = np.unique(df_sort['Family'], return_index=True)[1]
rank = [df_sort['Family'][index] for index in sorted(rank_idx)]



fig2_1, ax2_1 = plt.subplots(figsize=(5, 8), tight_layout=True)
sns.stripplot(x='Family', y=data_key, hue='Division', data=df_sort, order=rank, linewidth=1, jitter=True, clip_on=False, ax=ax2_1)
ax2_1.set_ylabel(data_key)
ax2_1.set_xlabel("Clones")
ax2_1.set_xticks([])
ax2_1.legend(title='Division', title_fontsize=16, fontsize=16)


fig2_2, ax2_1 = plt.subplots(figsize=(5, 8), tight_layout=True)
sns.stripplot(x='Division', y=data_key, hue='Family', data=df, palette=cp, linewidth=1, jitter=0.3, ax=ax2_1)
ax2_1.get_legend().remove()


fig2, ax2 = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1,2]}, sharey=True, figsize=(8, 8))
sns.stripplot(x='Family', y=data_key, hue='Division', data=df_sort, 
			  order=rank, linewidth=1, jitter=True, clip_on=False, ax=ax2[0])
ax2[0].set_ylabel(data_key)
ax2[0].set_xlabel("Clones")
ax2[0].set_xticks([])
ax2[0].legend(title='Div.', title_fontsize=16, fontsize=16, handletextpad=0.5, columnspacing=0.5)
sns.stripplot(x='Division', y=data_key, hue='Family', data=df, 
			  palette=cp, linewidth=1, clip_on=False, jitter=0.3, ax=ax2[1])
ax2[1].get_yaxis().set_visible(False)
ax2[1].spines['left'].set_visible(False)
ax2[1].get_legend().remove()
fig2.tight_layout(rect=(0, 0, 1, 1))
fig2.subplots_adjust(wspace=0.05, hspace=0)


fig2_1.savefig(f"./out/Fig1/Fig1_{data_key}_v1-1.pdf", dpi=300)
fig2_2.savefig(f"./out/Fig1/Fig1_{data_key}_v1-2.pdf", dpi=300)
fig2.savefig(f"./out/Fig1/Fig1_{data_key}_v2.pdf", dpi=300)
sys.exit()  # uncomment this to run codes below, which is not to be used in the paper



# ------ BINARISING THE DATA (not to be used in the paper)
from sklearn.cluster import KMeans
## Perform permutation test for clonal dependencies (statistical test for Fig.1G)
## Null hypothesis: Each cell's expression level is independent of its: (i) generation and clone; (ii) clone only
## => Rejection occured (p < 0.05)
## The idea of binarising the data is to confirm if we observe the same statistical feature on a simulated dataset

## Binarise IgG1+
def jitter(arr, pos):
	if pos: return 1 + np.random.uniform(low=-0.15, high=0.15, size=len(arr))
	else: return 0 + np.random.uniform(low=-0.15, high=0.15, size=len(arr))
## Method 1: Binarise by threshold
# threshold = 600
# pos_mask1 = df_sort[df_sort['IgG1']>threshold].index
# neg_mask1 = df_sort[df_sort['IgG1']<=threshold].index
# df_sort.loc[pos_mask1, 'IgG1'] = jitter(df_sort.loc[pos_mask1, 'IgG1'].to_numpy(), 1)
# df_sort.loc[neg_mask1, 'IgG1'] = jitter(df_sort.loc[neg_mask1, 'IgG1'].to_numpy(), 0)
# pos_mask2 = df[df['IgG1']>threshold].index
# neg_mask2 = df[df['IgG1']<=threshold].index
# df.loc[pos_mask2, 'IgG1'] = jitter(df.loc[pos_mask2, 'IgG1'].to_numpy(), 1)
# df.loc[neg_mask2, 'IgG1'] = jitter(df.loc[neg_mask2, 'IgG1'].to_numpy(), 0)

## Method 2: Binarise by K-means-clustering (K=2). [NB: Not so much different to Method 1]
X1 = df[['Family', 'IgG1']].to_numpy()
y_pred1 = KMeans(n_clusters=2).fit_predict(X1)
kms_df1 = pd.DataFrame({'Family': X1[:,0], 'IgG1': X1[:,1], 'Group': y_pred1})
pos_mask1 = kms_df1[kms_df1['Group']==1].index
neg_mask1 = kms_df1[kms_df1['Group']==0].index
df.loc[df.index[pos_mask1], 'IgG1'] = jitter(df.iloc[pos_mask1]['IgG1'].to_numpy(), 1)
df.loc[df.index[neg_mask1], 'IgG1'] = jitter(df.iloc[neg_mask1]['IgG1'].to_numpy(), 0)

X2 = df_sort[['Family', 'IgG1']].to_numpy()
y_pred2 = KMeans(n_clusters=2).fit_predict(X2)
kms_df2 = pd.DataFrame({'Family': X2[:,0], 'IgG1': X2[:,1], 'Group': y_pred2})
pos_mask2 = kms_df2[kms_df2['Group']==1].index
neg_mask2 = kms_df2[kms_df2['Group']==0].index
df_sort.loc[df_sort.index[pos_mask2], 'IgG1'] = jitter(df_sort.iloc[pos_mask2]['IgG1'].to_numpy(), 1)
df_sort.loc[df_sort.index[neg_mask2], 'IgG1'] = jitter(df_sort.iloc[neg_mask2]['IgG1'].to_numpy(), 0)


fig3_1, ax3_1 = plt.subplots(figsize=(5, 8), tight_layout=True)
sns.stripplot(x='Family', y=data_key, hue='Division', data=df_sort, order=rank, linewidth=1, jitter=True, ax=ax3_1)
ax3_1.set_ylabel(data_key)
ax3_1.set_xlabel("Clones")
ax3_1.set_yticks([0, 1])
ax3_1.set_yticklabels(['-ve', '+ve'])
ax3_1.set_xticks([])
ax3_1.legend(title='Division', title_fontsize=16, fontsize=16)


fig3_2, ax3_2 = plt.subplots(figsize=(5, 8), tight_layout=True)
sns.stripplot(x='Division', y=data_key, hue='Family', data=df, palette=cp, linewidth=1, jitter=0.3, ax=ax3_2)
ax3_2.set_yticks([0, 1])
ax3_2.set_yticklabels(['-ve', '+ve'])
ax3_2.get_legend().remove()


fig3, ax3 = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1,2]}, sharey=True, figsize=(8, 8))
sns.stripplot(x='Family', y=data_key, hue='Division', data=df_sort, 
			  order=rank, linewidth=1, jitter=True, clip_on=False, ax=ax3[0])
ax3[0].set_ylabel(data_key)
ax3[0].set_xlabel("Clones")
ax3[0].set_yticks([0, 1])
ax3[0].set_yticklabels(['-ve', '+ve'])
ax3[0].set_xticks([])
ax3[0].legend(loc='center left', title='Div.', title_fontsize=16, fontsize=16, handletextpad=0.5, columnspacing=0.5)
sns.stripplot(x='Division', y=data_key, hue='Family', data=df, 
			  palette=cp, linewidth=1, clip_on=False, jitter=0.3, ax=ax3[1])
ax3[1].get_yaxis().set_visible(False)
ax3[1].spines['left'].set_visible(False)
ax3[1].get_legend().remove()
fig3.tight_layout(rect=(0, 0, 1, 1))
fig3.subplots_adjust(wspace=0.05, hspace=0)


fig3_1.savefig(f"./out/Fig1/Fig1_{data_key}_v1-1(bin).pdf", dpi=300)
fig3_2.savefig(f"./out/Fig1/Fig1_{data_key}_v1-2(bin).pdf", dpi=300)
fig3.savefig(f"./out/Fig1/Fig1_{data_key}_v2(bin).pdf", dpi=300)



### Permutation test
## NB: Permutation test on binarised + simulated data is in "Fig1G_MC.py"

### Hypothesis testing
df = pd.read_excel("./data/Fig1/Fig1G_ClonalIgG1data.xlsx"); data_key = 'IgG1'
stat = np.nanmean(np.unique(df.groupby('Family')[data_key].transform('var')))

B = 250000
permute_stat1, permute_stat2 = [], []
c_df1 = df.copy()
c_df2 = df.copy()
for _ in tqdm.tqdm(range(B), desc="Permutation"):
	_select1 = df[data_key].copy()
	c_df1[data_key] = np.random.permutation(_select1)
	permute_stat1.append(np.nanmean(np.unique(c_df1.groupby('Family')[data_key].transform('var'))))

	for igen in np.unique(c_df2['Division']):
		_select2 = df[df['Division']==igen][data_key].copy()
		c_df2.loc[c_df2['Division']==igen, data_key] = np.random.permutation(_select2)
	permute_stat2.append(np.nanmean(np.unique(c_df2.groupby('Family')[data_key].transform('var'))))
permute_stat1 = np.array(permute_stat1)
pval1 = (1 + permute_stat1[permute_stat1<=stat].size) / (B + 1)

permute_stat2 = np.array(permute_stat2)
pval2 = (1 + permute_stat2[permute_stat2<=stat].size) / (B + 1)

fig4, ax4 = plt.subplots(figsize=(7,5), tight_layout=True)
ax4.axvline(np.log10(stat), color='k', ls='--', label="Data statistic")
sns.distplot(np.log10(permute_stat1), kde=False, norm_hist=True, color='b', ax=ax4, label=f"Test 1: $p={pval1:.6f}$")
sns.distplot(np.log10(permute_stat2), kde=False, norm_hist=True, color='r', ax=ax4, label=f"Test 2: $p={pval2:.6f}$")
ax4.set_xlabel(rf"$\log_{{{10}}}$ Average clonal variance ({data_key})")
ax4.legend(fontsize=14)
ax4.set_ylabel("Distribution")
fig4.savefig(f"./out/Fig1/Fig1_permTests_{data_key}.pdf", dpi=300)

plt.show()
