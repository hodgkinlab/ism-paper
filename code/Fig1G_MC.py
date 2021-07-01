import sys, time, tqdm
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.cluster import KMeans
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
rng = np.random.RandomState(seed=55192505)
# pd.set_option('display.max_rows', 999999)
# pd.set_option('display.max_columns', 999999)
# pd.set_option('display.expand_frame_repr', False)
options = {'strings_to_formulas': False, 'strings_to_urls': False}  # for saving fit results to excel file
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


class cell:
	def __init__(self, pAID, pIy1, pEff, gen, AID, IgG1, gmax, fam_id):
		self.gen = gen
		self.AID = AID
		self.IgG1 = IgG1

		####################################################################################
		#				Determine AID state (Inherit or Become AID+)					   #
		####################################################################################
		if self.AID==1:  # Check if current cell inherits AID+ state from its mother cell
			pass         # stay AID+
		else:
			self.AID = rng.binomial(n=1, p=pAID[gen])

		####################################################################################
		#					Determine Iy1 state (Independent to AID)					   #
		####################################################################################
		## Iy1 and Iy2b are autonomously and independently expressed (no correlation observed experimentally)
		Iy1 = rng.binomial(n=1, p=pIy1)
		s = rng.binomial(n=1, p=pEff)
		
		####################################################################################
		#							Isotype switching rules								   #
		####################################################################################
		if self.AID==1:  					  # Switching occurs only if AID+
			if self.IgG1==0:  				  # When current cell is IgG1- (or IgM+)
				if s*Iy1==0: pass    	 	  # Nothing happens
				elif s*Iy1==1: self.IgG1 = 1  # Both AID+ and Iy1+ = IgG1+						
			elif self.IgG1==1:  			  # When current cell is IgG1+
				self.IgG1 = 1   			  # stay in IgG1+ state (i.e. nothing happens)

		####################################################################################
		#					Create two daughters cells if not destiny					   #
		####################################################################################
		if self.gen < gmax:
			self.l = cell(pAID, pIy1, pEff, self.gen+1, self.AID, self.IgG1, gmax, fam_id)
			self.r = cell(pAID, pIy1, pEff, self.gen+1, self.AID, self.IgG1, gmax, fam_id)
		else:
			####################################################################################
			#				Collect results if current gen. = g_max						   #
			####################################################################################
			tree_array.append([gen, Iy1, self.AID, self.IgG1, fam_id])
			self.l = None
			self.r = None

def run_simulation(n, gens, gens_freq, pAID=None, pIy1=None, pEff=None):
	if any(prob is None for prob in [pAID, pIy1]): raise Exception("Probability not set!")
	if not (0 <= pEff <= 1): raise Exception(f"Efficiency must be [0,1]: current value = {pEff:.3f}")

	for id in tqdm.trange(n, desc="Simulation", leave=True):
		gmax = rng.choice(gens, p=gens_freq)
		cell(pAID=pAID, pIy1=pIy1, pEff=pEff, gen=0, AID=0, IgG1=0, gmax=gmax, fam_id=id)

def aid_hazard(data):  # convert overall proportion to hazard probability
	divs, nreps = data.shape
	prob = 0.0*data[:]
	for g in range(divs):
		if g == 0: prob[0,:] = data[0,:]
		else: prob[g,:] = (data[g,:] - data[g-1,:])/(1. - data[g-1,:])
	return prob

# def hazard_to_prop(pAID):  # convert hazard probability to overall proportion
# 	divs = pAID.shape[0]
# 	prop = 0.0*pAID
# 	prop[0] = pAID[0]
# 	for g in range(divs)[1:]:
# 		if g==1: prop[1] = pAID[1] * (1. - pAID[0]) + pAID[0]
# 		else: prop[g] = pAID[g]*(1. - prop[g-1]) + prop[g-1]
# 	return prop

def calc_positive_prop(df, group, key):  # calculate proportion of Iy1+
	total = df.groupby(group).size().reset_index(name='count')
	pos = df.groupby(group)[key].apply(lambda x: (x>0).sum()).reset_index(name='count')
	prop = pos.to_numpy()[:,1] / total.to_numpy()[:,1]
	return total, pos, prop

if __name__ == "__main__":
	gens = np.arange(0, 7, step=1)

	## Import AID+ measurements
	aid_df = pd.read_excel("./data/Fig2-IgG1/Fig2E_AIDproportion_division.xlsx", usecols="A:D", index_col=0)
	## Use empirical mean as AID+ probability (per division)
	data = aid_df.to_numpy()
	c_data = aid_hazard(data)
	c_data[c_data<0] = 0
	pAID = np.nanmean(c_data, axis=1)  

	## Import Iy1+ measurements
	iy1_df = pd.read_csv("./data/Fig2-IgG1/Fig2F_Iy1GLT_division_singlecellqPCR.csv", usecols=range(0,3), index_col=0)
	iy1_df.reset_index(drop=True, inplace=True)
	## Use empirical mean  as Iy1+ probability (constant probability across division)
	_, _, prop_pIy1 = calc_positive_prop(iy1_df, group='Division', key='Iy1')  # calculate proportion Iy1+ per division
	pIy1 = np.mean(prop_pIy1)

	## Import fitted pe1 (IgG1) efficiency
	pe1_df = pd.read_excel("./out/Fig2-IgG1/Fig2_results.xlsx", sheet_name='pars_efficiency', index_col=0)
	pe1, pe1_low, pe1_upp = pe1_df['best-fit'].values[0], pe1_df['Lower95%'].values[0], pe1_df['Upper95%'].values[0]

	## Import Fig1G IgG1+
	_igg1_df = pd.read_excel("./data/Fig1/Fig1G_ClonalIgG1data.xlsx")
	gens_size = _igg1_df.groupby('Division').size().to_numpy()
	gens_freq = gens_size/np.sum(gens_size)  # obtain frequency of cells observed per generation


	## Monte-Carlo simulation settings
	n = 149	  	  # number of families (matching with data)
	B = 250000    # number of permutations

	## Run MC simulation with optimal efficiency
	tree_array = []  # store AID, Iy1, Iy2b states of all cells in generation = g_max
	run_simulation(n=n, gens=gens, gens_freq=gens_freq, pAID=pAID, pIy1=pIy1, pEff=pe1)
	model_df = pd.DataFrame(tree_array, columns=['Division', 'Iy1', 'AID', 'IgG1', 'Family'])
	model_df.drop(['Iy1', 'AID'], axis=1, inplace=True)

	## Binarise data for comparison purpose
	df = pd.read_excel("./data/Fig1/Fig1G_ClonalIgG1data.xlsx"); data_key = 'IgG1'
	## Method 1: Binarise by threshold
	# threshold = 600
	# pos_mask2 = df[df['IgG1']>threshold].index
	# neg_mask2 = df[df['IgG1']<=threshold].index
	# df.loc[pos_mask2, 'IgG1'] = 1
	# df.loc[neg_mask2, 'IgG1'] = 0

	## Method 2: Binarise by K-means-clustering (K=2). [NB: Not so much different to Method 1]
	X1 = df[['Family', 'IgG1']].to_numpy()
	y_pred1 = KMeans(n_clusters=2).fit_predict(X1)
	kms_df1 = pd.DataFrame({'Family': X1[:,0], 'IgG1': X1[:,1], 'Group': y_pred1})
	pos_mask1 = kms_df1[kms_df1['Group']==1].index
	neg_mask1 = kms_df1[kms_df1['Group']==0].index
	df.loc[df.index[pos_mask1], 'IgG1'] = 1
	df.loc[df.index[neg_mask1], 'IgG1'] = 0
	stat = np.nanmean(np.unique(df.groupby('Family')[data_key].transform('var')))
	model_stat = np.nanmean(np.unique(model_df.groupby('Family')[data_key].transform('var')))

	## Permutation test
	permute_stat1, permute_stat2 = [], []
	c_df1 = df.copy()
	c_df2 = df.copy()
	m_permute_stat1, m_permute_stat2 = [], []
	c_mdf1 = model_df.copy()
	c_mdf2 = model_df.copy()
	for _ in tqdm.tqdm(range(B), desc="Permutation"):
		_select1 = df[data_key].copy()
		c_df1[data_key] = np.random.permutation(_select1)
		permute_stat1.append(np.nanmean(np.unique(c_df1.groupby('Family')[data_key].transform('var'))))
		_m_select1 = model_df[data_key].copy()
		c_mdf1[data_key] = np.random.permutation(_m_select1)
		m_permute_stat1.append(np.nanmean(np.unique(c_mdf1.groupby('Family')[data_key].transform('var'))))

		for igen in np.unique(c_df2['Division']):
			_select2 = df[df['Division']==igen][data_key].copy()
			c_df2.loc[c_df2['Division']==igen, data_key] = np.random.permutation(_select2)
			_m_select2 = model_df[model_df['Division']==igen][data_key].copy()
			c_mdf2.loc[c_mdf2['Division']==igen, data_key] = np.random.permutation(_m_select2)
		permute_stat2.append(np.nanmean(np.unique(c_df2.groupby('Family')[data_key].transform('var'))))
		m_permute_stat2.append(np.nanmean(np.unique(c_mdf2.groupby('Family')[data_key].transform('var'))))
	permute_stat1 = np.array(permute_stat1)
	pval1 = (1 + permute_stat1[permute_stat1<=stat].size) / (B + 1)
	m_permute_stat1 = np.array(m_permute_stat1)
	m_pval1 = (1 + m_permute_stat1[m_permute_stat1<=model_stat].size) / (B + 1)
	permute_stat2 = np.array(permute_stat2)
	pval2 = (1 + permute_stat2[permute_stat2<=stat].size) / (B + 1)
	m_permute_stat2 = np.array(m_permute_stat2)
	m_pval2 = (1 + m_permute_stat2[m_permute_stat2<=model_stat].size) / (B + 1)


	figg, axx = plt.subplots(nrows=2, figsize=(7,5), sharex=True)
	axx[0].axvline(stat, color='k', ls='--', label="Data statistic")
	sns.distplot(permute_stat1, kde=False, norm_hist=True, color='b', ax=axx[0], label=f"Test 1: $p={pval1:.6f}$")
	sns.distplot(permute_stat2, kde=False, norm_hist=True, color='r', ax=axx[0], label=f"Test 2: $p={pval2:.6f}$")
	axx[1].axvline(model_stat, color='k', ls='--', label="Data statistic (model)")
	sns.distplot(m_permute_stat1, kde=False, norm_hist=True, color='blue', ax=axx[1], label=f"Test 1: $p={m_pval1:.6f}$")
	sns.distplot(m_permute_stat2, kde=False, norm_hist=True, color='red', ax=axx[1], label=f"Test 2: $p={m_pval2:.6f}$")
	for axis in axx:
		axis.legend(frameon=False, fontsize=12)
		axis.set_ylabel("Distribution")
	axx[1].set_xlabel(rf"Average clonal variance ({data_key})")
	axx[0].set_title(f"Binarised data ($B={B}$)", x=0.01, ha='left', weight='bold')
	axx[1].set_title(r"Simulated data (IgM $\rightarrow$ IgG1)", x=0.01, ha='left', weight='bold')
	figg.tight_layout(rect=(0, 0, 1, 1))
	figg.subplots_adjust(wspace=0, hspace=0.2)

	figg.savefig(f"./out/Fig2-IgG1/Fig2_data+MC_permTests.pdf", dpi=300)

	plt.show()
