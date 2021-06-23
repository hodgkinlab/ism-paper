import sys, time, tqdm
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import lmfit as lmf
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
rng = np.random.RandomState(seed=70392595)
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

SITER = 100
BITER = 10000

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


def residual(pars, x, pAID, pIy1, data=None):
	vals = pars.valuesdict()
	pe = vals['pe']

	## Mean equation (counts)
	exp_AIDpos = [pAID[0]]
	exp_Iy1pos = [pIy1]
	exp_G1 = [pAID[0]*pe*pIy1]
	for igen in x[1:]:
		exp_AIDpos.append(2.**igen * (1. - np.prod(1. - pAID[:igen+1])))
		exp_Iy1pos.append(2.**igen * pIy1)
		exp_G1.append(2.*exp_G1[igen-1] + (exp_AIDpos[igen] - 2.*exp_G1[igen-1])*pe*pIy1)
	exp_AIDpos = np.array(exp_AIDpos)/2**x
	exp_Iy1pos = np.array(exp_Iy1pos)/2**x
	exp_G1 = np.array(exp_G1)/2**x

	## Mean equation (proportion)
	# exp_AIDpos = [pAID[0]]
	# exp_Iy1pos = [pIy1]
	# exp_G1 = [pAID[0]*pe*pIy1]
	# for igen in x[1:]:
	# 	exp_AIDpos.append(1. - np.prod(1. - pAID[:igen+1]))
	# 	exp_Iy1pos.append(pIy1)
	# 	exp_G1.append(
	# 		exp_G1[0]*(1. - pe*pIy1)**igen + \
	# 		pe*pIy1*np.sum( [(1. - pe*pIy1)**i * (1. - np.prod(1. - pAID[:igen+1-i])) for i in range(igen)] )
	# 	)

	if data is None:
		return exp_AIDpos, exp_Iy1pos, exp_G1
	else:
		_, nreps = data.shape
		resid = 0.0*data[:]
		for i in range(nreps):
			resid[:,i] = data[:,i] - exp_G1
		return resid.flatten()

def bootstrap(pars, x, data_pAID, data_pIy1, data_IgG1):
	b_pes = []
	b_aid, b_iy1, b_igg1 = [], [], []
	for i in tqdm.trange(BITER, desc='Bootstrap', leave=True):
		b_pars = pars.copy()

		## Input 1: Resample AID data
		b_prop_AID = []
		for row in data_pAID:
			b_prop_AID.append(rng.choice(row, size=len(row), replace=True))
		b_prop_AID = np.array(b_prop_AID)
		b_AID = aid_hazard(b_prop_AID)
		b_AID[b_AID<0] = 0
		b_pAID = np.nanmean(b_AID, axis=1)

		## Input 2: Resample Iy1 data
		c_df = data_pIy1.copy()
		b_df = c_df.groupby('Division').apply(lambda x: x.sample(frac=1, replace=True)).reset_index(drop=True)
		_, _, b_prop_Iy1 = calc_positive_prop(b_df, group='Division', key='Iy1')
		b_pIy1 = np.mean(b_prop_Iy1)

		## Output: Resample IgG1 data
		b_prop_IgG1 = []
		for row in data_IgG1:
			b_prop_IgG1.append(rng.choice(row, size=len(row), replace=True))
		b_prop_IgG1 = np.array(b_prop_IgG1)


		b_candidates = {'result': [], 'residual': []}  # store fitted parameter and its residual
		for j in tqdm.trange(SITER, leave=False):
			# Random initial values
			b_pars['pe'].set(value=rng.uniform(0, 1))
			try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
				mini_lm = lmf.Minimizer(residual, b_pars, fcn_args=(x, b_pAID, b_pIy1, b_prop_IgG1))
				res_lm = mini_lm.minimize(method='leastsq')  # Levenberg-Marquardt algorithm

				result = res_lm
				resid = res_lm.chisqr

				b_candidates['result'].append(result)
				b_candidates['residual'].append(resid)
			except ValueError as ve:
				pass
		try:  # skip empty set
			_results = pd.DataFrame(b_candidates)
			_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual
			best_result = _results.iloc[0]['result']
			b_pes.append(best_result.params.valuesdict()['pe'])

			curves = residual(best_result.params, x, b_pAID, b_pIy1, data=None)
			b_aid.append(curves[0])
			b_iy1.append(curves[1])
			b_igg1.append(curves[2])
		except IndexError as ie:
			pass
	return b_aid, b_iy1, b_igg1, b_pes


	

if __name__ == "__main__":
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
	## Use empirical mean as Iy1+ probability (constant probability across division)
	## This is equivalent to using a linear regression
	_, _, prop_pIy1 = calc_positive_prop(iy1_df, group='Division', key='Iy1')  # calculate proportion Iy1+ per division
	pIy1 = np.mean(prop_pIy1)

	igg1_df = pd.read_excel("./data/Fig2-IgG1/Fig2M_IgG1switching_division.xlsx", usecols="A:D", index_col=0)
	igg1_data = igg1_df.to_numpy()

	
	gens = np.arange(0, 7, step=1)


	## Optimisation (find best-fit efficiency)
	pars = lmf.Parameters()
	pars.add('pe', value=0.5, min=0, max=1, vary=True)
	candidates = {'result': [], 'residual': []}  # store fitted parameter and its residual
	for s in tqdm.trange(SITER, desc="Init. Fit"):
		# Random initial values
		pars['pe'].set(value=rng.uniform(0, 1))
		try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
			mini_lm = lmf.Minimizer(residual, pars, fcn_args=(gens, pAID, pIy1, igg1_data))
			res_lm = mini_lm.minimize(method='leastsq')  # Levenberg-Marquardt algorithm

			result = res_lm
			resid = res_lm.chisqr

			candidates['result'].append(result)
			candidates['residual'].append(resid)
		except ValueError as ve:
			pass
	fit_results = pd.DataFrame(candidates)
	fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual
	best_result = fit_results.iloc[0]['result']
	print(lmf.fit_report(best_result))
	fit_pe = best_result.params.valuesdict()['pe']
	best_aid, best_iy1, best_igg1 = residual(best_result.params, gens, pAID, pIy1, data=None)

	## Bootstrap
	boots_aid, boots_iy1, boots_igg1, boots_pe = bootstrap(pars, gens, data, iy1_df, igg1_data)
	aid_low95, aid_upp95 = np.percentile(boots_aid, q=[2.5, 97.5], axis=0)
	iy1_low95, iy1_upp95 = np.percentile(boots_iy1, q=[2.5, 97.5], axis=0)
	igg1_low95, igg1_upp95 = np.percentile(boots_igg1, q=[2.5, 97.5], axis=0)
	pe_low95, pe_upp95 = np.percentile(boots_pe, q=[2.5, 97.5])


	## Compute RSS
	rss = []
	delta_pe = 1000
	pes = np.linspace(0.0, 1.0, num=delta_pe)
	copy_pars = pars.copy()
	for _pe in pes:
		copy_pars['pe'].set(value=_pe)
		_rss = np.sum(residual(copy_pars, gens, pAID, pIy1, data=igg1_data)**2)
		rss.append(_rss)

	b_rss = []
	for b in tqdm.trange(BITER, desc='RSS', leave=True):
		## Input 1: Resample AID data
		b_prop_AID = []
		for row in data:
			b_prop_AID.append(rng.choice(row, size=len(row), replace=True))
		b_prop_AID = np.array(b_prop_AID)
		b_AID = aid_hazard(b_prop_AID)
		b_AID[b_AID<0] = 0
		b_pAID = np.nanmean(b_AID, axis=1)

		## Input 2: Resample Iy1 data
		c_df = iy1_df.copy()
		b_df = c_df.groupby('Division').apply(lambda x: x.sample(frac=1, replace=True, random_state=rng)).reset_index(drop=True)
		_, _, b_prop_Iy1 = calc_positive_prop(b_df, group='Division', key='Iy1')
		b_pIy1 = np.mean(b_prop_Iy1)

		## Output: Resample IgG1 data
		b_prop_IgG1 = []
		for row in igg1_data:
			b_prop_IgG1.append(rng.choice(row, size=len(row), replace=True))
		b_prop_IgG1 = np.array(b_prop_IgG1)

		tmp_rss = []
		for _pe in pes:
			copy_pars['pe'].set(value=_pe)
			_rss = np.sum(residual(copy_pars, gens, b_pAID, b_pIy1, data=b_prop_IgG1)**2)
			tmp_rss.append(_rss)
		b_rss.append(tmp_rss)
	rss_low95, rss_upp95 = np.nanpercentile(b_rss, q=[2.5, 97.5], axis=0)
	data_rss = np.sum(residual(best_result.params, gens, pAID, pIy1, data=igg1_data)**2)
	pe_95range = np.linspace(pe_low95, pe_upp95, num=delta_pe)
	rss_95range = []
	for _pe95 in pe_95range:
		copy_pars['pe'].set(value=_pe95)
		rss_95range.append(np.sum(residual(copy_pars, gens, pAID, pIy1, data=igg1_data)**2))


	# Plot RSS
	fig1, ax1 = plt.subplots(tight_layout=True)
	ax1.plot(pes*100, rss, 'k-', lw=2)
	ax1.fill_between(pes*100, rss_low95, rss_upp95, color='k', ec=None, alpha=0.3)
	ax1.errorbar([fit_pe*100], [data_rss],
				 xerr=[[(fit_pe - pe_low95)*100], [(pe_upp95 - fit_pe)*100]],
				 fmt='o', color='red', mec='k', zorder=10, 
				 label=f"${fit_pe*100:.3f}\pm_{{{(fit_pe - pe_low95)*100:.3f}}}^{{{(pe_upp95 - fit_pe)*100:.3f}}}$%")
	ax1.plot(pe_95range*100, rss_95range, '-', color='red', zorder=10)
	ax1.legend(fontsize=16, handletextpad=0, columnspacing=0)
	ax1.set_ylabel("RSS", fontsize=16)
	ax1.set_xlabel("%Efficiency", fontsize=16)
	ax1.set(xlim=(0, 100), ylim=(0, None))		


	## Plot AID+
	fig2, ax2 = plt.subplots(tight_layout=True)
	ax2.errorbar(gens, data.mean(axis=1), yerr=sps.sem(data, axis=1), fmt='o-', color='k', mfc='none', label="Experiment")
	ax2.plot(gens, best_aid, 'bo--', label="Model")
	ax2.fill_between(gens, aid_low95, aid_upp95, color='b', ec=None, alpha=0.3)
	ax2.set(ylabel="Proportion AID+", xlabel="Division", ylim=(0, None))
	ax2.legend(loc='upper left', fontsize=16)
	ax2.xaxis.set_major_locator(MaxNLocator(integer=True))


	figg, axx = plt.subplots(tight_layout=True)
	axx.plot(np.unique(iy1_df['Division']), prop_pIy1, 'ko-', mfc='none', label="Experiment")
	axx.plot(gens, best_iy1, 'ro--', label="Model")
	axx.fill_between(gens, iy1_low95, iy1_upp95, color='r', ec=None, alpha=0.3)
	axx.set(ylabel="Proportion Iy1+", xlabel="Division", ylim=(0, 1))
	axx.legend(fontsize=16)
	axx.xaxis.set_major_locator(MaxNLocator(integer=True))


	## Plot IgG1+
	fig3, ax3 = plt.subplots(tight_layout=True)
	ax3.errorbar(igg1_df.index, igg1_df.mean(axis=1), yerr=igg1_df.sem(axis=1), 
				 fmt='o-', color='k', mfc='none', label="Experiment")
	ax3.plot(gens, best_igg1, 'o--', c='purple', label="Model")
	ax3.fill_between(gens, igg1_low95, igg1_upp95, color='purple', ec=None, alpha=0.3)
	ax3.set(ylabel="Proportion IgG1+", xlabel="Division", ylim=(0, None))
	ax3.legend(loc='upper left', fontsize=16)
	ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

	## Save plots
	fig1.savefig('./out/Fig2-IgG1/Fig2K-M_RSS.pdf', dpi=300)
	fig2.savefig('./out/Fig2-IgG1/Fig2K-M_AID.pdf', dpi=300)
	figg.savefig('./out/Fig2-IgG1/Fig2K-M_Iy1.pdf', dpi=300)
	fig3.savefig('./out/Fig2-IgG1/Fig2K-M_IgG1.pdf', dpi=300)

	## Save results to excel
	with pd.ExcelWriter("./out/Fig2-IgG1/Fig2_results.xlsx", engine='xlsxwriter', options=options, mode='w') as writer:
		eff_xlsx = pd.DataFrame(data = {
								'Efficiency (%)': pes*100,
								'RSS': rss,
								'Lower95%': rss_low95,
								'Upper95%': rss_upp95
							})
		eff_xlsx.to_excel(writer, sheet_name="efficiency", index=False)

		pe_xlsx = pd.DataFrame(
			data = {'best-fit': [fit_pe],
					'Lower95%': [pe_low95],
					'Upper95%': [pe_upp95]},
			index = ['pe']
		)
		pe_xlsx.to_excel(writer, sheet_name='pars_efficiency')

		aid_xlsx = pd.DataFrame(
			data = {'Division': gens,
					'AID+': best_aid,
					'Lower95%': aid_low95,
					'Upper95%': aid_upp95}
		)
		aid_xlsx.to_excel(writer, sheet_name="AID", index=False)

		iy1_xlsx = pd.DataFrame(
			data = {'Division': gens,
					'Iy1+': best_iy1,
					'Lower95%': iy1_low95,
					'Upper95%': iy1_upp95}
		)
		iy1_xlsx.to_excel(writer, sheet_name="Iy1", index=False)

		igg1_xlsx = pd.DataFrame(
			data = {'Division': gens,
					'IgG1+': best_igg1,
					'Lower95%': igg1_low95,
					'Upper95%': igg1_upp95}
		)
		igg1_xlsx.to_excel(writer, sheet_name="IgG1", index=False)
	
	plt.show()
	

