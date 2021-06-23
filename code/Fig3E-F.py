import sys, time, tqdm
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import lmfit as lmf
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
rng = np.random.RandomState(seed=93644869)
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
	'errorbar.capsize': 2.5,
	'legend.title_fontsize': 14
}
sns.set(context='paper', style='white', rc=rc)

SITER = 200
BITER = 10000

def aid_hazard(prop_AID):  # convert overall proportion to hazard probability
	divs = prop_AID.shape[0]
	prob = 0.0*prop_AID[:]
	for g in range(divs):
		if g == 0: prob[0] = prop_AID[0]
		else: prob[g] = (prop_AID[g] - prop_AID[g-1])/(1. - prop_AID[g-1])
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


def residual(pars, x, pAID, pIy1, pIy2b, data1=None, data2=None, weight=None, ext=False):
	ndata, _, nreps = data1.shape
	resid1 = 0*data1[:]
	resid2 = 0*data2[:]

	all_exp_AIDpos = []
	all_exp_G1 = []
	all_exp_G2 = []

	vals = pars.valuesdict()
	pe1 = vals['pe1']
	pe2 = vals['pe2']
	for i in range(ndata):
		p_p1 = pe1*pIy1[i]
		p_p2 = pe2*pIy2b	

		## Mean equation (counts)
		exp_AIDneg = [1. - pAID[0]]
		exp_AIDpos = [pAID[0]]
		exp_G1 = [exp_AIDpos[0]*(p_p1*p_p2 + p_p1*(1. - p_p2))]
		exp_G2 = [exp_AIDpos[0]*(1. - p_p1)*p_p2]
		for igen in x[1:]:
			exp_AIDneg.append(2**igen * np.prod(1. - pAID[:igen+1]))
			exp_AIDpos.append(2**igen - exp_AIDneg[igen])
			exp_G1.append(
				2. * exp_G1[igen-1]*(1. - p_p1*p_p2 - p_p1*(1. - p_p2) - (1. - p_p1)*p_p2) - \
				2. * exp_G2[igen-1]*(p_p1*p_p2 + p_p1*(1. - p_p2)) + \
				exp_AIDpos[igen]*(p_p1*p_p2 + p_p1*(1. - p_p2))
			)
			exp_G2.append(
				2. * exp_G2[igen-1]*(1. - (1. - p_p1)*p_p2) + \
				exp_AIDpos[igen]*((1. - p_p1)*p_p2)
			)
		exp_AIDpos = np.array(exp_AIDpos)/2**x
		exp_G1 = np.array(exp_G1)/2**x
		exp_G2 = np.array(exp_G2)/2**x

		all_exp_AIDpos.append(exp_AIDpos * weight[i].mean(axis=1))
		all_exp_G1.append(exp_G1 * weight[i].mean(axis=1))
		all_exp_G2.append(exp_G2 * weight[i].mean(axis=1))

		## TODO: Mean equation (proportion i.e. multiply 2^-g)

		for j in range(nreps):
			resid1[i,:,j] = (data1[i,:,j] - exp_G1) * np.sqrt(weight[i,:,j])
			resid2[i,:,j] = (data2[i,:,j] - exp_G2) * np.sqrt(weight[i,:,j])
	rss = np.concatenate((resid1.ravel(), resid2.ravel()))

	if ext:
		return all_exp_AIDpos, all_exp_G1, all_exp_G2
	else:
		return rss

def bootstrap(pars, x, data_pAID, data_pIy1_Iy2b, data_IgG1, pe1_95CI, data_IgG2b, weight, conc):
	pe1_low, pe1_upp = pe1_95CI[0], pe1_95CI[1]

	b_pes = []
	b_aid, b_igg1, b_igg2 = [], [], []
	for i in tqdm.trange(BITER, desc='Bootstrap', leave=True):
		b_pars = pars.copy()	

		## Input 1: Resample AID data (no replicate)
		# b_prop_AID = []
		# for row in data_pAID:
		# 	b_prop_AID.append(rng.choice(row, size=len(row), replace=True))
		# b_prop_AID = np.array(b_prop_AID)
		# b_AID = aid_hazard(b_prop_AID)
		# b_AID[b_AID<0] = 0
		# b_pAID = np.nanmean(b_AID, axis=1)

		## Input 2: Resample Iy1 data
		b_pIy1 = []
		for c in conc:
			c_df = data_pIy1_Iy2b[data_pIy1_Iy2b['IL-4 concentration (U/mL)']==f'IL4_{c}'].copy()
			b_iy1_df = c_df.groupby('Division').apply(lambda x: x.sample(frac=1, replace=True)).reset_index(drop=True)
			_, _, prop_pIy1 = calc_positive_prop(b_iy1_df, group='Division', key='Iy1')
			b_pIy1.append(np.mean(prop_pIy1))

		## Input 3: Resample Iy2b data
		b_iy2b_df = data_pIy1_Iy2b.groupby('Division').apply(lambda x: x.sample(frac=1, replace=True)).reset_index(drop=True)
		_, _, b_pIy2b = calc_positive_prop(b_iy2b_df, group='Division', key='Iy2b')
		b_pIy2b = np.mean(b_pIy2b)

		## Output 1 & 2: Resample IgG1 & IgG2b data
		b_all_igg1, b_all_igg2 = [], []
		for icc in range(len(conc)):
			_tmp_igg1, _tmp_igg2 = [], []
			for row1, row2 in zip(data_IgG1[icc], data_IgG2b[icc]):
				_tmp_igg1.append(rng.choice(row1, size=len(row1), replace=True))
				_tmp_igg2.append(rng.choice(row2, size=len(row2), replace=True))
			b_all_igg1.append(_tmp_igg1)
			b_all_igg2.append(_tmp_igg2)
		b_all_igg1 = np.array(b_all_igg1)
		b_all_igg2 = np.array(b_all_igg2)

		b_pars['pe1'].set(value=rng.uniform(pe1_low, pe1_upp))

		b_candidates = {'result': [], 'residual': []}  # store fitted parameter and its residual
		for j in tqdm.trange(SITER, leave=False):
			# Random initial values
			b_pars['pe2'].set(value=rng.uniform(0, 1))
			try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
				mini_lm = lmf.Minimizer(residual, b_pars, fcn_args=(x, data_pAID, b_pIy1, b_pIy2b, b_all_igg1, b_all_igg2, weight))
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

			b_pes.append(best_result.params.valuesdict()['pe2'])
			curves = residual(best_result.params, x, data_pAID, b_pIy1, b_pIy2b, b_all_igg1, b_all_igg2, weight, ext=True)
			b_aid.append(curves[0])
			b_igg1.append(curves[1])
			b_igg2.append(curves[2])
		except IndexError as ie:
			pass
	return np.array(b_aid), np.array(b_igg1), np.array(b_igg2), np.array(b_pes)




if __name__ == "__main__":
	conc = [200, 100, 50, 20, 10]  # IL-4 concentration

	## Import AID+ data
	aid_df = pd.read_excel("./data/Fig3/fig3_AID_division.xlsx", index_col=0)
	aid_df.columns = conc
	## Empirical mean for expressing AID+
	data = aid_df.mean(axis=1).to_numpy()/100  # taking mean across IL-4
	pAID = aid_hazard(data)
	pAID[pAID<0] = 0

	## Import Iy1+, Iy2b+ data
	iy1_iy2_df = pd.read_excel("./data/Fig3/Fig3D_singlecellqPCR_Iy1Iy2b_IL4_DIV.xlsx", usecols="A:E", index_col=0)
	iy1_iy2_df.reset_index(drop=True, inplace=True)
	## Empirical mean for expressing Iy1+, Iy2b+
	pIy1 = []
	for c in conc:
		conc_dat = iy1_iy2_df[iy1_iy2_df['IL-4 concentration (U/mL)']==f'IL4_{c}'].copy()
		_, _, prop_pIy1 = calc_positive_prop(conc_dat, group='Division', key='Iy1')
		pIy1.append(np.mean(prop_pIy1))
	pIy1 = np.array(pIy1)

	_, _, prop_pIy2b = calc_positive_prop(iy1_iy2_df, group='Division', key='Iy2b')
	pIy2b = np.mean(prop_pIy2b)

	## Import IgG1, IgG2b data
	igg1_df = pd.read_excel("./data/Fig3/fig3_IgG1_division.xlsx", index_col=0)
	igg1_df.columns = [c for c in conc for _ in (1,2,3)]
	igg2_df = pd.read_excel("./data/Fig3/fig3_IgG2b_division.xlsx", index_col=0)
	igg2_df.columns = [c for c in conc for _ in (1,2,3)]

	## Import cell frequency data
	freq_df = pd.read_excel('./data/Fig3/divisionfreq_Fig3.xlsx', index_col=0)
	freq_df.columns = [c for c in conc for _ in (1,2,3)]

	## Import fitted pe1 (IgG1) efficiency
	pe1_df = pd.read_excel("./out/Fig2-IgG1/Fig2_results.xlsx", sheet_name='pars_efficiency', index_col=0)
	pe1, pe1_low, pe1_upp = pe1_df['best-fit'].values[0], pe1_df['Lower95%'].values[0], pe1_df['Upper95%'].values[0]

	
	gens = np.arange(0, 7, step=1)


	## Organise IgG1, IgG2b dataset for joint fitting
	all_freq = []
	all_igg1, all_igg2 = [], []
	pars = lmf.Parameters()
	pars.add('pe1', value=pe1, min=0, max=1, vary=False)
	for icc, c in enumerate(conc):
		pars.add(f'pe2_{icc+1}', value=0.5, min=0, max=1, vary=True)
		igg1_data = igg1_df.loc[:,c].to_numpy()/100
		igg2_data = igg2_df.loc[:,c].to_numpy()/100
		freq = freq_df.loc[:,c].to_numpy()/100

		all_igg1.append(igg1_data)
		all_igg2.append(igg2_data)
		all_freq.append(freq)
	all_igg1 = np.array(all_igg1)
	all_igg2 = np.array(all_igg2)
	all_freq = np.array(all_freq)


	## Optimisation: shared IgG2b efficiency parameter
	s_pars = lmf.Parameters()
	s_pars.add('pe1', value=pe1, min=0, max=1, vary=False)
	s_pars.add('pe2', value=0.5, min=0, max=1, vary=True)
	candidates = {'result': [], 'residual': []}  # store fitted parameter and its residual
	for s in tqdm.trange(SITER, desc="Init. Fit"):
		# Random initial values
		# s_pars['pe1'].set(value=rng.uniform(pe1_low, pe1_upp))
		s_pars['pe2'].set(value=rng.uniform(0, 1))
		try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
			mini_lm = lmf.Minimizer(residual, s_pars, fcn_args=(gens, pAID, pIy1, pIy2b, all_igg1, all_igg2, all_freq))
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
	curves = residual(best_result.params, gens, pAID, pIy1, pIy2b, all_igg1, all_igg2, all_freq, ext=True)
	_, b_igg1, b_igg2, b_pe2 = \
		bootstrap(s_pars, gens, pAID, iy1_iy2_df, all_igg1, (pe1_low, pe1_upp), all_igg2, all_freq, conc)



	cp = sns.color_palette("Paired")
	markers = ['o', 'P', 'D', 's', 'X']
	igg1_colors = np.flip(["#000000", "#331104", "#5e1507", "#8d1306", "#BF0000"])
	igg2_colors = np.flip(["#000000", "#14142b", "#1a2052", "#1a2d7c", "#0D3AA9"])
	
	fit_pars = []
	fit_pars_low95, fit_pars_upp95 = [], []
	fit_pars_lows, fit_pars_upps = [], []
	igg1_avg, igg1_lows, igg1_upps = [], [], []
	igg2_avg, igg2_lows, igg2_upps = [], [], []

	fig2_1, ax2_1 = plt.subplots(tight_layout=True)
	fig2_2, ax2_2 = plt.subplots(tight_layout=True)
	fig2_3, ax2_3 = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(8,8))
	with pd.ExcelWriter("./out/Fig3/Fig3_results.xlsx", engine='xlsxwriter', options=options, mode='w') as writer:
		for icc, c in enumerate(conc):			
			igg1_avg.append(np.mean(curves[1][icc]))
			igg1_low95, igg1_upp95 = np.percentile(b_igg1[:,icc,:], q=[2.5, 97.5], axis=0)
			igg1_lows.append(np.mean(igg1_low95))
			igg1_upps.append(np.mean(igg1_upp95))

			igg2_avg.append(np.mean(curves[2][icc]))
			igg2_low95, igg2_upp95 = np.percentile(b_igg2[:,icc,:], q=[2.5, 97.5], axis=0)
			igg2_lows.append(np.mean(igg2_low95))
			igg2_upps.append(np.mean(igg2_upp95))

			pe2_low95, pe2_upp95 = np.percentile(b_pe2, q=[2.5, 97.5])
			best = best_result.params['pe2'].value
			fit_pars.append(best)
			fit_pars_low95.append(best - pe2_low95)
			fit_pars_upp95.append(pe2_upp95 - best)
			fit_pars_lows.append(pe2_low95)
			fit_pars_upps.append(pe2_upp95)


			igg1_data = igg1_df.loc[:,c].to_numpy()/100 * all_freq[icc]
			ax2_1.errorbar(gens, igg1_data.mean(axis=1), yerr=sps.sem(igg1_data, axis=1), 
						   fmt=f'{markers[icc]}:', color=igg1_colors[icc], mfc='none', label=f"{c}U")
			ax2_1.plot(gens, curves[1][icc], marker=markers[icc], c=igg1_colors[icc], ms=4, label=f"Model")
			ax2_1.fill_between(gens, igg1_low95, igg1_upp95, color=igg1_colors[icc], ec=None, alpha=0.2)
			ax2_3[icc][0].errorbar(gens, igg1_data.mean(axis=1), yerr=sps.sem(igg1_data, axis=1), 
								   fmt=f'{markers[icc]}:', color=igg1_colors[icc], mfc='none', label="Data")
			ax2_3[icc][0].plot(gens, curves[1][icc], marker=markers[icc], c=igg1_colors[icc], label=f"Model")
			ax2_3[icc][0].fill_between(gens, igg1_low95, igg1_upp95, color=igg1_colors[icc], ec=None, alpha=0.2)

			igg2_data = igg2_df.loc[:,c].to_numpy()/100 * all_freq[icc]
			ax2_2.errorbar(gens, igg2_data.mean(axis=1), yerr=sps.sem(igg2_data, axis=1), 
						   fmt=f'{markers[icc]}:', color=igg2_colors[icc], mfc='none', label=f"{c}U")
			ax2_2.plot(gens, curves[2][icc], marker=markers[icc], c=igg2_colors[icc], ms=4, label=f"Model")
			ax2_2.fill_between(gens, igg2_low95, igg2_upp95, color=igg2_colors[icc], ec=None, alpha=0.2)
			ax2_3[icc][1].errorbar(gens, igg2_data.mean(axis=1), yerr=sps.sem(igg2_data, axis=1), 
						   		   fmt=f'{markers[icc]}:', color=igg2_colors[icc], mfc='none', label="Data")
			ax2_3[icc][1].plot(gens, curves[2][icc], marker=markers[icc], color=igg2_colors[icc], label=f"Model")
			ax2_3[icc][1].fill_between(gens, igg2_low95, igg2_upp95, color=igg2_colors[icc], ec=None, alpha=0.2)

			igg1_xlsx = pd.DataFrame(
				data = {'Division': gens,
						'IgG1': curves[1][icc],
						'Lower95%': igg1_low95,
						'Upper95%': igg1_upp95,
						'Data': igg1_data.mean(axis=1),
						'SEM': sps.sem(igg1_data, axis=1)}
			)
			igg2_xlsx = pd.DataFrame(
				data = {'Division': gens,
						'IgG2b': curves[2][icc],
						'Lower95%': igg2_low95,
						'Upper95%': igg2_upp95,
						'Data': igg2_data.mean(axis=1),
						'SEM': sps.sem(igg2_data, axis=1)}
			)
			igg1_xlsx.to_excel(writer, sheet_name=f"IgG1 ({c}U)", index=False)
			igg2_xlsx.to_excel(writer, sheet_name=f"IgG2b ({c}U)", index=False)
		handles1, labels1 = ax2_1.get_legend_handles_labels()
		ax2_1.legend(reversed(handles1), reversed(labels1), ncol=2, loc='upper left', 
					 handletextpad=0.5, columnspacing=0.5, fontsize=13)
		ax2_1.set(ylabel=r"IgG1+ (density)", xlabel="Division", ylim=(0, None))
		ax2_1.xaxis.set_major_locator(MaxNLocator(integer=True))

		handles2, labels2 = ax2_2.get_legend_handles_labels()
		ax2_2.legend(reversed(handles2), reversed(labels2), ncol=2, loc='upper left', 
					  handletextpad=0.5, columnspacing=0.5, fontsize=13)
		ax2_2.set(ylabel=r"IgG2b+ (density)", xlabel="Division", ylim=(0, None))
		ax2_2.xaxis.set_major_locator(MaxNLocator(integer=True))

		## 
		max1, max2 = 0, 0
		for axis1, axis2 in zip(ax2_3[:,0], ax2_3[:,1]):
			ymin1, ymax1 = axis1.get_ylim()
			if ymax1 > max1:
				max1 = ymax1
			ymin2, ymax2 = axis2.get_ylim()
			if ymax2 > max2:
				max2 = ymax2
		for axis1, axis2, c in zip(ax2_3[:,0], ax2_3[:,1], conc):
			axis1.legend(loc='upper left', title=f"{c}U IL-4", frameon=False, fontsize=12)
			axis1.set_ylabel("Density")
			axis1.set_ylim(bottom=0, top=max1)
			# axis1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
			# axis1.yaxis.major.formatter._useMathText = True
			axis1.xaxis.set_major_locator(MaxNLocator(integer=True))
			
			axis2.legend(loc='upper left', title=f"{c}U IL-4", frameon=False, fontsize=12)
			axis2.set_ylim(bottom=0, top=max2)
			axis2.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax2_3[0,0].set_title("IgG1+", x=0.01, ha='left', weight='bold')
		ax2_3[4,0].set_xlabel("Division")
		ax2_3[0,1].set_title("IgG2b+", x=0.01, ha='left', weight='bold')
		ax2_3[4,1].set_xlabel("Division")
		fig2_3.tight_layout(rect=(0, 0, 1, 1))
		fig2_3.subplots_adjust(wspace=0.05, hspace=0.05)

		

		fig1_1, ax1_1 = plt.subplots(tight_layout=True)
		proc_igg1_data, proc_igg1_sem = [], []
		proc_igg2_data, proc_igg2_sem = [], []
		for c in conc:
			igg1_data = igg1_df.loc[:,c].to_numpy()/100 * all_freq[icc]
			proc_igg1_data.append(np.mean(igg1_data))
			proc_igg1_sem.append(sps.sem(np.mean(igg1_data, axis=0)))

			igg2_data = igg2_df.loc[:,c].to_numpy()/100 * all_freq[icc]
			proc_igg2_data.append(np.mean(igg2_data))
			proc_igg2_sem.append(sps.sem(np.mean(igg2_data, axis=0)))
		ax1_1.plot(conc, igg1_avg, 'o--', color='red', label="IgG1+")
		ax1_1.fill_between(conc, igg1_lows, igg1_upps, color='red', alpha=0.2)
		ax1_1.plot(conc, igg2_avg, 'o--', color='blue', label="IgG2b+")
		ax1_1.fill_between(conc, igg2_lows, igg2_upps, color='blue', alpha=0.2)

		ax1_1.errorbar(conc, proc_igg1_data, yerr=proc_igg1_sem, fmt='o-', mfc='none', color='r', label="Data")
		ax1_1.errorbar(conc, proc_igg2_data, yerr=proc_igg2_sem, fmt='o-',  mfc='none', color='b', label="Data")
		ax1_1.legend(loc='upper left', fontsize=14, ncol=2, scatteryoffsets=[.5], handletextpad=0.5, columnspacing=0.5)
		ax1_1.set_xscale('log')
		ax1_1.set(xlabel="IL-4 Concentration (U/mL)", 
				  ylabel=r"Avg. of IgG1+/IgG2b+ density", xlim=(1, 1000), ylim=(0,None))				


		fig1_2, ax1_2 = plt.subplots(tight_layout=True)
		model_igg1_avg, model_igg1_lows, model_igg1_upps = [], [], []
		model_igg2_avg, model_igg2_lows, model_igg2_upps = [], [], []
		_proc_igg1_data, _proc_igg1_sem = [], []
		_proc_igg2_data, _proc_igg2_sem = [], []
		for icc, c in enumerate(conc):
			igg1_data = igg1_df.loc[:,c].to_numpy()/100
			_proc_igg1_data.append(np.mean(igg1_data))
			_proc_igg1_sem.append(sps.sem(np.mean(igg1_data, axis=0)))
			model_igg1_avg.append(np.mean(curves[1][icc]/(all_freq[icc].mean(axis=1))))
			igg1_low95, igg1_upp95 = np.percentile(b_igg1[:,icc,:], q=[2.5, 97.5], axis=0)
			model_igg1_lows.append(np.mean(igg1_low95/(all_freq[icc].mean(axis=1))))
			model_igg1_upps.append(np.mean(igg1_upp95/(all_freq[icc].mean(axis=1))))

			igg2_data = igg2_df.loc[:,c].to_numpy()/100
			_proc_igg2_data.append(np.mean(igg2_data))
			_proc_igg2_sem.append(sps.sem(np.mean(igg2_data, axis=0)))
			model_igg2_avg.append(np.mean(curves[2][icc]/(all_freq[icc].mean(axis=1))))
			igg2_low95, igg2_upp95 = np.percentile(b_igg2[:,icc,:], q=[2.5, 97.5], axis=0)
			model_igg2_lows.append(np.mean(igg2_low95/(all_freq[icc].mean(axis=1))))
			model_igg2_upps.append(np.mean(igg2_upp95/(all_freq[icc].mean(axis=1))))
		ax1_2.plot(conc, model_igg1_avg, 'o--', color='red', label="IgG1+")
		ax1_2.fill_between(conc, model_igg1_lows, model_igg1_upps, color='red', alpha=0.2)
		ax1_2.plot(conc, model_igg2_avg, 'o--', color='blue', label="IgG2b+")
		ax1_2.fill_between(conc, model_igg2_lows, model_igg2_upps, color='blue', alpha=0.2)

		ax1_2.errorbar(conc, _proc_igg1_data, yerr=_proc_igg1_sem, fmt='o-', mfc='none', color='r', label="Data")
		ax1_2.errorbar(conc, _proc_igg2_data, yerr=_proc_igg2_sem, fmt='o-',  mfc='none', color='b', label="Data")
		ax1_2.legend(loc='upper left', fontsize=14, ncol=2, scatteryoffsets=[.5], handletextpad=0.5, columnspacing=0.5)
		ax1_2.set_xscale('log')
		ax1_2.set(xlabel="IL-4 Concentration (U/mL)", 
				  ylabel=r"Proportion IgG1+/IgG2b+", xlim=(1, 1000), ylim=(0,None))


		## Plot efficiency
		pe1_best = best_result.params['pe1'].value
		fig3, ax3 = plt.subplots(tight_layout=True)
		ax3.plot(conc, [pe1_best*100]*len(conc), 'o-', color='red', label="IgG1+")
		ax3.fill_between(conc, pe1_low*100, pe1_upp*100, color='r', alpha=0.3)
		ax3.plot(conc, np.array(fit_pars)*100, 'o-', color='blue', label="IgG2b+")
		ax3.fill_between(conc, np.array(fit_pars_lows)*100, np.array(fit_pars_upps)*100, color='b', alpha=0.3)

		ax3.set_xscale('log')
		ax3.set(xlabel="IL-4 Concentration (U/mL)", ylabel="%Efficiency", xlim=(1, 1000), ylim=(0,100))
		ax3.legend(fontsize=16)


		fig1_1.savefig('./out/Fig3/Fig3E-F_Overall_v1.pdf', dpi=300)
		fig1_2.savefig('./out/Fig3/Fig3E-F_Overall_v2.pdf', dpi=300)
		fig2_1.savefig('./out/Fig3/Fig3E-F_IgG1_DIV.pdf', dpi=300)
		fig2_2.savefig('./out/Fig3/Fig3E-F_IgG2b_DIV.pdf', dpi=300)
		fig2_3.savefig('./out/Fig3/Fig3E-F_IgG1_IgG2b_DIV.pdf', dpi=300)
		fig3.savefig('./out/Fig3/Fig3E-F_Efficiency.pdf', dpi=300)

		igg1_den_il4 = pd.DataFrame(
			data = {'IL-4 Concentration (U/mL)': conc,
					'Model (shared eff)': igg1_avg,
					'Model Lower95%': igg1_lows,
					'Model Upper95%': igg1_upps,
					'Data': proc_igg1_data,
					'SEM': proc_igg1_sem}
		)
		igg1_il4 = pd.DataFrame(
			data = {'IL-4 Concentration (U/mL)': conc,
					'Model (shared eff)': model_igg1_avg,
					'Model Lower95%': model_igg1_lows,
					'Model Upper95%': model_igg1_upps,
					'Data': _proc_igg1_data,
					'SEM': _proc_igg1_sem}
		)
		igg2_den_il4 = pd.DataFrame(
			data = {'IL-4 Concentration (U/mL)': conc,
					'Model (shared eff)': igg2_avg,
					'Model Lower95%': igg2_lows,
					'Model Upper95%': igg2_upps,
					'Data': proc_igg2_data,
					'SEM': proc_igg2_sem}
		)
		igg2_il4 = pd.DataFrame(
			data = {'IL-4 Concentration (U/mL)': conc,
					'Model (shared eff)': model_igg2_avg,
					'Model Lower95%': model_igg2_lows,
					'Model Upper95%': model_igg2_upps,
					'Data': _proc_igg2_data,
					'SEM': _proc_igg2_sem}
		)
		pe_xlsx = pd.DataFrame(
			data = {'IL-4 Concentration (U/mL)': conc,
					'Model pe2 (shared)': fit_pars,
					'Model Lower95%': fit_pars_lows,
					'Model Upper95%': fit_pars_upps}
		)
		igg1_den_il4.to_excel(writer, sheet_name="Avg IgG1 density", index=False)
		igg1_il4.to_excel(writer, sheet_name="Avg IgG1 prop", index=False)
		igg2_den_il4.to_excel(writer, sheet_name="Avg IgG2b density", index=False)
		igg2_il4.to_excel(writer, sheet_name="Avg IgG2b prop", index=False)
		pe_xlsx.to_excel(writer, sheet_name="pars", index=False)
	plt.show()
	

