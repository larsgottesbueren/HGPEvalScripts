import performance_profiles
import relative_runtimes_plot
import runtime_plot
import effectiveness_tests
import speedup_plots
import commons
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sb
import glob
import itertools

import combine_performance_profile_and_relative_slowdown as cpprs
import runtime_share

def get_thread_numbers(df):
	thread_list = sorted(list(df.threads.unique()))
	if 1 in thread_list:
		thread_list.remove(1)
	else:
		print("no sequential runs found. cannot compute speedups. abort")
		print(df)
		exit()
	return thread_list

def flows_speedup_plots(options, out_dir, algorithm):
	paper_width = options['width'] / 2
	aspect_ratio = 2.25
	height = paper_width / aspect_ratio

	df = pd.read_csv('plain_flow_runtimes.csv')
	print(df.algorithm.unique())
	df = df[df.algorithm == algorithm]
	thread_list = get_thread_numbers(df)
	color_mapping = commons.construct_new_color_mapping(thread_list)

	fig, ax = plt.subplots(figsize=(paper_width, height))
	speedup_plots.scalability_plot(df, "time", ax, thread_colors=color_mapping, 
	                               show_rolling_gmean=True, show_scatter=True, alpha=0.5,
	                               xscale='log', yscale='log', display_labels=False,
	                               seed_aggregator="median", window_size=5)
	ax.set_xlabel("sequential time for " + algorithm + " [s]")
	ax.set_ylabel("speedup")

	ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
	ax.set_yticks([0.25, 0.5, 1,2,4,8,16,32,64])
	ax.set_yticklabels([0.25, 0.5, 1,2,4,8,16,32,64])

	num_legend_entries = len(thread_list)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[num_legend_entries:], labels[num_legend_entries:], ncol=3, title='threads', loc='lower center', bbox_to_anchor=(0.5, -0.84), frameon=False)
	

	plt.savefig(out_dir + "plain_flow_speedups_" + algorithm + ".pdf", bbox_inches='tight')

def flowcutter_speedup_plots(options, out_dir):
	paper_width = options['width'] / 2
	aspect_ratio = 2.25
	height = paper_width / aspect_ratio

	df = pd.read_csv('parPR-flowcutter_setB_scalability.csv')
	thread_list = get_thread_numbers(df)
	color_mapping = commons.construct_new_color_mapping(thread_list)
			
	fig, ax = plt.subplots(figsize=(paper_width, height))
	speedup_plots.scalability_plot(df, "time", ax, thread_colors=color_mapping, 
	                               show_rolling_gmean=True, show_scatter=True, alpha=0.5,
	                               xscale='log', yscale='log', display_labels=False,
	                               seed_aggregator="median", window_size=5)
	ax.set_xlabel("sequential time for FlowCutter [s]")
	ax.set_ylabel("speedup")

	ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
	ax.set_yticks([0.25, 0.5, 1, 2,4,8,16,32,64])

	num_legend_entries = len(thread_list)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles[num_legend_entries:], labels[num_legend_entries:], ncol=3, title='threads', loc='lower center', bbox_to_anchor=(0.5, -0.84), frameon=False)

	plt.savefig(out_dir + "flowcutter_speedups.pdf", bbox_inches='tight')

def flows_relative_runtime_plots(options, out_dir):
	df = pd.read_csv('plain_flow_runtimes.csv')
	df = df[df.threads == 1]
	df["k"] = 2
	df["epsilon"] = 0.03
	algos = commons.infer_algorithms_from_dataframe(df)
	# algos.remove("ParPR-Block")
	colors = commons.construct_new_color_mapping(algos)
	
	relative_runtimes_plot.plot(out_dir + "plain_flow", df, "SeqPR", algos=algos, colors=colors, field="time", seed_aggregator="mean", figsize=options['half_figsize'])

def flowcutter_speedup_numbers():
	df = pd.read_csv('parPR-flowcutter_setB_scalability.csv')
	speedup_plots.print_speedups(df, field="time", seed_aggregator="mean")

def flows_speedup_numbers():
	df = pd.read_csv('plain_flow_runtimes.csv')
	df = df[df.algorithm == "ParPR-RL"]
	speedup_plots.print_speedups(df, field="time", seed_aggregator="mean")

def mt_kahypar_speedup_plots(options, out_dir):
	paper_width = options['width']
	aspect_ratio = 0.92 * 1.5
	height = paper_width / aspect_ratio
	fig, axes = plt.subplots(2, 2, sharey=True, figsize=(paper_width, height))

	df = commons.read_files(list(glob.glob("mt_kahypar_d_f_*_scaling.csv")))
	
	thread_list = sorted(list(df.threads.unique()))
	if 1 in thread_list:
		thread_list.remove(1)
	else:
		print("no sequential runs :(")
		return
	color_mapping = commons.construct_new_color_mapping(thread_list)

	for ax, (i, (k_lb, k_ub)) in zip(axes.ravel(), enumerate([(2,2), (8,16), (64,64)])):
		print(i, k_lb)
		speedup_plots.scalability_plot(df=df[(df.k >= k_lb) & (df.k <= k_ub)], field="flowTime", ax=ax, thread_colors=color_mapping, display_labels=False, display_legend=False, seed_aggregator="mean",
		                               xscale='log', yscale='log', show_rolling_gmean=True, alpha=0.5, filter_tiny_outlier_threshold = 1.0, window_size=10)
		ax.set_xlabel("sequential time [s]")
		if k_lb == k_ub:
			title_string = R'$k = ' + str(k_lb) + R'$'
		else:
			title_string = R'$k \in \{' + str(k_lb) + R',' + str(k_ub) + R'\}$'
		ax.set_title(title_string)
		ax.set_ylabel("")
		
		ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		ax.set_yticks([2,4,8,16,32,64])


	speedup_plots.scalability_plot(df=df, field="totalPartitionTime", ax=axes[1][1], thread_colors=color_mapping, display_labels=False, display_legend=False, seed_aggregator="median",
		                           xscale='log', yscale='log', show_rolling_gmean=True, alpha=0.5, filter_tiny_outlier_threshold = 1.0)
	axes[1][1].set_xlabel("sequential time [s]")
	axes[1][1].set_title("Mt-KaHyPar-D-F")
	axes[1][1].set_ylabel("")
	axes[1][1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
	axes[1][1].set_yticks([2,4,8,16,32,64])	
		
	for row in range(2):
		for col in range(2):
			ax = axes[row][col]
			if col != 0:
				ax.yaxis.set_ticks_position('none')

	handles, labels = axes[0][0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), frameon=False, ncol=3, title="threads")
	
	for row in range(2):
		axes[row][0].set_ylabel('speedup')


	plt.subplots_adjust(wspace=0.025, hspace=0.5)
	plt.savefig(out_dir + "mt-kahypar-d-f-speedups.pdf", bbox_inches='tight', pad_inches=0.0)

def increasing_threads(options, out_dir):
	df = commons.read_files(list(glob.glob("mt_kahypar_d_f_*_scaling.csv")))
	# df["algorithm"] = df["algorithm"] + " " + df["threads"].astype(str)
	algos = ["Mt-KaHyPar-D-F " + str(i) for i in [1,4,16,64]]
	color_mapping_algos = ["Mt-KaHyPar-D-F " + str(i) for i in [4,16,64, 1]]
	colors = commons.construct_new_color_mapping(color_mapping_algos)
	
	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig, algos=algos, colors=colors)
	fig.savefig(out_dir + "increasing_threads.pdf", bbox_inches="tight", pad_inches=0.0)

def main_setA(options, out_dir):
	mt_kahypar_file_list = ["mt-kahypar-d-f-setA.csv", "mt-kahypar-q-f-setA.csv", "mt-kahypar-d-setA.csv", "mt-kahypar-q-setA.csv"]
	others_file_list = ["hmetis_r_setA.csv", "kahypar_ca_setA.csv", "kahypar_hfc_setA.csv", "patoh_d_setA.csv", "patoh_q_setA.csv"]

	df = commons.read_files(mt_kahypar_file_list)
	df = df[df.threads == 10]
	
	df2 = commons.read_files(others_file_list)
	df = pd.concat([df, df2])

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-Q-F", algos=["Mt-KaHyPar-D-F", "Mt-KaHyPar-Q-F", "Mt-KaHyPar-D", "KaHyPar-HFC"], time_limit=28800)
	fig.savefig(out_dir + "setA.pdf", bbox_inches="tight", pad_inches=0.0)

	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig, algos=["Mt-KaHyPar-D-F", "Mt-KaHyPar-Q-F", "Mt-KaHyPar-D", "KaHyPar-HFC"])
	fig.savefig(out_dir + "setA_reduced_algoset.pdf", bbox_inches="tight", pad_inches=0.0)

def main_setB(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-f-64.csv", "mt-kahypar-q-f-64.csv", "mt-kahypar-d-64.csv", "mt-kahypar-q-64.csv",
							 "bipart-64.csv", "zoltan-mt-bench.csv", "patoh-d-mt-bench.csv"])

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-D-F")
	fig.savefig(out_dir + "setB.pdf", bbox_inches="tight", pad_inches=0.0)


def effectiveness_tests_plot(options, out_dir):
	mt_kahypar_file_list = ["mt-kahypar-d-setA.csv", "mt-kahypar-d-f-setA.csv", "mt-kahypar-q-f-setA.csv",]
	others_file_list = []#["hmetis_r_setA.csv", "kahypar_ca_setA.csv", "kahypar_hfc_setA.csv"]
	df = commons.read_files(mt_kahypar_file_list)
	df = df[df.threads == 10]
	#df2 = commons.read_files(others_file_list)
	#df = pd.concat([df, df2])

	for algo_tuple in itertools.product(["Mt-KaHyPar-D-F", "Mt-KaHyPar-Q-F"]
	                                    , ["Mt-KaHyPar-D"]#, "Mt-KaHyPar-D-F"]
	                                    ):
		algos = list(algo_tuple)
		virt_df = effectiveness_tests.create_virtual_instances(df, algos, num_repetitions=20)
		
		fig = plt.figure(figsize=options['half_figsize'])
		performance_profiles.infer_plot(virt_df, fig)
		fig.savefig(out_dir + "effectiveness-tests_" + algos[0] + "_" + algos[1] + ".pdf", bbox_inches="tight", pad_inches=0.0)


def refinement_stats(options, out_dir):
	fraction_tuples = [	
		('flows_moves', 'flows_num_refinements'),
		('flows_num_improvements', 'flows_moves'),
		('flows_zero_gain_improvement', 'flows_num_improvements'),

		('flows_incorrect_gains', 'flows_moves'),
		('flows_gain_reverts', 'flows_moves'),
		('flows_balance_reverts', 'flows_moves'),
		('flows_actual_gain_sum', 'flows_expected_gain_sum'),
	]

	labels = [
		'claimed improvement',
		'actual improvement',
		'zero gain',

		'gain wrong',
		'negative gain revert',
		'balance revert',
		'gain sum',
	]

	df = pd.read_csv('mt-kahypar-d-f-refinement-stats.csv')

	for (a,b), name in zip(fraction_tuples, labels):
		df[name] = df[a] / df[b]
		
	df = df.select_dtypes(['number'])
	unrolled = df.melt(id_vars=['k','epsilon','seed','threads'])

	fig, ax = plt.subplots(figsize=options['figsize'])
	import event_frequency
	event_frequency.plot(unrolled, fig, ax, fields=labels)
	fig.savefig(out_dir + "refinement_stats.pdf", bbox_inches='tight', pad_inches=0.0)

def runtime_shares(options, out_dir):
	fig = plt.figure(figsize=options['figsize'])

	mapper = {	'flow_refinement_scheduler':'flows', 'fm':'FM refinement', 'label_propagation':'LP refinement', 'preprocessing':'preprocessing', 
				'coarsening':'coarsening', 'initial_partitioning':'initial'
				}
	df = commons.read_file('mt-kahypar-d-f-refinement-stats.csv')
	df.rename(columns=mapper, inplace=True)
	fields = list(mapper.values())
	df = df[df.seed == 0]
	df = runtime_share.clean(df)
	runtime_share.plot(df, fields=fields, sort_field="flows", fig=fig, tfield='totalPartitionTime')
	fig.axes[0].spines['top'].set_visible(False)
	fig.savefig(out_dir + "runtime_shares.pdf", bbox_inches='tight', pad_inches=0.0)

def gain_contributions(options, out_dir):
	fig = plt.figure(figsize=options['figsize'])
	df = commons.read_file('mt-kahypar-d-f-refinement-stats.csv')
	mapper = {	'flows_actual_gain_sum' : 'Flows gain',
				'lp_actual_gain_sum' : 'LP gain'
				}
	df.rename(columns=mapper, inplace=True)
	df = df[df.seed == 0]
	df['total gain'] = df['initial_km1'] - df['km1']
	df['FM gain'] = df['total gain'] - df['Flows gain'] - df['LP gain']
	df = df[(df['FM gain'] >= 0) & (df['total gain'] > 0)].copy()
	runtime_share.plot(df, fields=["LP gain", "FM gain", "Flows gain"], sort_field="LP gain", fig=fig, tfield='total gain')
	fig.axes[0].spines['top'].set_visible(False)
	fig.axes[0].set_ylabel('fractional gain contribution')
	fig.savefig(out_dir + "gain_contributions.pdf", bbox_inches='tight', pad_inches=0.0)

def read_sql_bases():
	import sqlite3
	df = pd.DataFrame()
	for t in [1,4,16,64]:
		con = sqlite3.connect("runtime_share_databases/mt_kahypar_d_f_" + str(t) + ".db")
		t_df = pd.read_sql_query("SELECT * from ex1", con)
		df = pd.concat([df, t_df])
	df.rename(columns={'partitionTime' : 'totalPartitionTime', "num_threads" : "threads"}, inplace=True)
	return df

def runtime_shares_flows_by_threads(options, out_dir):
	import event_frequency
	df = read_sql_bases()

	fields = ['apply_moves', 'region_growing', 'construct_flow_network', 'hyper_flow_cutter']
	names = ['apply moves', 'grow region', 'assemble', 'FlowCutter']
	actual_total = 'flow_refinement_scheduler'

	df['total'] = [sum(x) for x in zip(*[df[f] for f in fields])]	# unfortunately the scheduler time cannot be taken as divisor
	for f, name in zip(fields, names):
		df[name] = df[f] / df['total']
	df = df.select_dtypes(['number'])

	thread_colors = {
				1  : 'tab:blue',
				4  : 'tab:green',
				16 : 'tab:red',
				64 : 'tab:orange',
			}

	fig, axes = plt.subplots(ncols=3, figsize=options['figsize'])
	for i, (k_lb, k_ub) in enumerate([(2,2), (8,16), (64,64)]):
		unrolled = df[(df.k >= k_lb) & (df.k <= k_ub)].melt(id_vars=['k','epsilon','seed','threads'])
		event_frequency.plot(unrolled, fig, axes[i], fields=names, hue='threads', colors=thread_colors)
		
		if k_lb == k_ub:
			title_string = R'$k = ' + str(k_lb) + R'$'
		else:
			title_string = R'$k \in \{' + str(k_lb) + R',' + str(k_ub) + R'\}$'
		axes[i].set_title(title_string)

	axes[0].set_ylabel('running time share')
	for ax in axes[1:]:
		ax.set_ylabel('')
		ax.yaxis.set_ticklabels([])

	if True:
		handles, labels = axes[0].get_legend_handles_labels()
		handles, labels = handles[:4], labels[:4]
		for ax in axes:
			ax.legend().remove()
		fig.legend(handles, labels, ncol=4, title='threads', loc='lower center', bbox_to_anchor=(0.5, -0.25), frameon=False)

	fig.savefig(out_dir + "flow_phase_stats.pdf", bbox_inches='tight', pad_inches=0.0)

def refinement_stats_by_threads(options, out_dir):
	import event_frequency
	df = read_sql_bases()
	thread_colors = {
				1  : 'tab:blue',
				4  : 'tab:green',
				16 : 'tab:red',
				64 : 'tab:orange',
			}

	mapper = {
		'num_flow_improvement'							: 'flows_num_improvements',
		'num_flow_refinements'							: 'flows_num_refinements',	
		'failed_updates_due_to_balance_constraint'		: 'flows_balance_reverts',
		'failed_updates_due_to_conflicting_moves'		: 'flows_gain_reverts',
		'correct_expected_improvement'					: 'flows_correct_gains',
		'zero_gain_improvement'							: 'flows_zero_gain_improvement',
	}

	df.rename(columns=mapper, inplace=True)
	df['flows_moves'] = df['flows_num_improvements'] + df['flows_balance_reverts'] + df['flows_gain_reverts']
	df['flows_incorrect_gains'] = df['flows_moves'] - df['flows_correct_gains']

	fraction_tuples = [	
		('flows_moves', 'flows_num_refinements'),
		('flows_num_improvements', 'flows_moves'),
		('flows_zero_gain_improvement', 'flows_num_improvements'),

		('flows_incorrect_gains', 'flows_moves'),
		('flows_gain_reverts', 'flows_moves'),
		('flows_balance_reverts', 'flows_moves'),
	]

	labels = [
		'claimed improvement',
		'actual improvement',
		'zero gain',

		'gain wrong',
		'negative gain revert',
		'balance revert',
	]

	for (a,b), name in zip(fraction_tuples, labels):
		df[name] = df[a] / df[b]
		
	df = df.select_dtypes(['number'])
	unrolled = df.melt(id_vars=['k','epsilon','seed','threads'])

	df2 = pd.read_csv('mt-kahypar-d-f-refinement-stats.csv')
	df2['gain_sum'] = df2['flows_actual_gain_sum'] / df2['flows_expected_gain_sum']
	df2 = df2.select_dtypes(['number'])
	unrolled2 = df2.melt(id_vars=['k','epsilon','seed','threads'])

	unrolled = pd.concat([unrolled, unrolled2])

	fig, ax = plt.subplots(figsize=options['figsize'])
	import event_frequency
	event_frequency.plot(unrolled, fig, ax, fields=labels + ['gain_sum'], hue='threads', colors=thread_colors)

	
	ax.set_ylabel('frequency')
	fig.savefig(out_dir + "refinement_stats_by_threads.pdf", bbox_inches='tight', pad_inches=0.0)


def main_sea20(options, out_dir):
	prefix = 'kahypar-hfc-sea20/km1_'
	suffix = '.csv'
	files = ['hmetis_k', 'hmetis_r', 'hype', 'kahypar-hfc', 'kahypar-hfc-eco', 'kahypar-mf', 'mondriaan', 'patoh_d', 'patoh_q', 'zoltan_algd']
	df = commons.read_files([prefix + f + suffix for f in files])

	colors = commons.construct_color_mapping_with_default_colors(commons.infer_algorithms_from_dataframe(df))
	

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="KaHyPar-MF", colors=colors, time_limit=28800, width_ratios=[0.65, 0.35], ncol=5)
	fig.savefig(out_dir + "sea20_setA.pdf", bbox_inches="tight", pad_inches=0.0)


	fig, axes = plt.subplots(ncols=2, figsize=options['figsize'])
	for ax in axes:	# this is necessary when using subplots instead of gridspec
		ax.set_axis_off()
	handles, labels = performance_profiles.infer_plot(df, fig, algos=['KaHyPar-HFC', 'KaHyPar-MF'], colors=colors, external_subplot=axes[0], display_legend=False)
	axes[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)
	handles, labels = performance_profiles.infer_plot(df, fig, algos=['KaHyPar-HFC-Eco', 'KaHyPar-MF'], colors=colors, external_subplot=axes[1], display_legend=False)
	axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)
	plt.subplots_adjust(wspace=0.25)
	fig.savefig(out_dir + 'sea20_hfc_vs_mf.pdf', bbox_inches="tight", pad_inches=0.0)

def runtime_share_sea20(options, out_dir):
	df = pd.read_csv('kahypar-hfc-sea20/runtime_per_phase.csv')
	df.phase = df.phase.apply(str.lower)
	fields = list(df.phase.unique())

	fig, ax = plt.subplots(figsize=(options['width'] * 0.8, options['height'] * 0.9))	
	ax.set_axisbelow(True)
	ax.grid(b=True, axis='y', which='major', ls='dashed')

	boxprops = dict(fill=True, edgecolor='black', linewidth=0.9)
	rem_props = dict(linestyle='-', linewidth=0.9)

	box_plot = sb.boxplot(y="time", x="algorithm", hue="phase", data=df,
	                      order=["KaHyPar-MF", "KaHyPar-HFC", "KaHyPar-HFC-Eco"],
	                      hue_order=fields,
	                      width=0.9, 
	                      showfliers=True, 
	                      fliersize=2,
	                      boxprops=boxprops, whiskerprops=rem_props, medianprops=rem_props,
	                      #flierprops = { "rasterize" : True },
	                      capprops=rem_props
	                      )
	# TODO rasterize fliers`?
	plt.legend(bbox_to_anchor=(1.03, 0.5), loc='center left', borderaxespad=0.)
	ax.set_yscale('symlog', linthresh=1e-1)
	ax.set_ylim(bottom=-0.02, top=1e4 * 6.5)
	#ax.set_yscale('log', base=10)
	
	ax.set_ylabel(R'time per pin [$\mu$s]')

	ax.xaxis.label.set_visible(False)
	fig.savefig(out_dir + "sea20_runtime_per_phase.pdf", bbox_inches="tight", pad_inches=0.0)

	'''
	fields = [str.lower(x) for x in df.phase.unique()]
	print(df.algorithm.unique())
	df = df.pivot(index=["algorithm","graph","k","seed"],columns="phase", values="time").reset_index()
	df.rename(columns=str.lower, inplace=True)

	fig = plt.figure(figsize=options['figsize'])
	runtime_share.plot(df[(df.algorithm=='KaHyPar-HFC-Eco') & (df.seed==0)].copy(), fields=fields, sort_field="flow refinement", fig=fig, tfield=None)
	print('plot done')
	fig.savefig(out_dir + 'sea20_runtime_share.pdf')
	'''

def remove_timeouts(df):
	return df[~df.algorithm.str.contains('Timeout')]

def replace_timeouts(df):
	s = df.algorithm.str.contains('Timeout')
	df.loc[s, 'seed'] = df.loc[s, 'imbalance'].astype(int)
	df.loc[s, 'totalPartitionTime'] = 28800
	df.loc[s, 'timeout'] = 'yes'
	df.loc[s, 'epsilon'] = 1.0 		# imbalance can't be recovered :(
	df.loc[s, 'algorithm'] = df.loc[s, 'algorithm'].str.replace('Timeout:', '')
	return df 

def rename_algos(df):
	df["algorithm"].replace(to_replace={"HMetis-R" : "hMetis-R"}, inplace=True)
	df["algorithm"] = df["algorithm"].str.replace("HyperFlowCutter", "HFC")
	# df["algorithm"] = df["algorithm"].str.replace("bin", "")
	# df["algorithm"] = df["algorithm"].str.replace("lib", "")

def explore_esa19(options, out_dir):
	files = list(glob.glob("esa19/*.results"))
	files.remove("esa19/kahypar_and_hmetis.results")
	df = commons.read_files(files)
	df = df[(df.algorithm != "HFC-100") & (~df.algorithm.str.contains('bin'))]
	df = replace_timeouts(df)
	rename_algos(df)
	df = df[df.epsilon.isin([0.0, 0.03])]
	df.to_csv('esa19_both_imbalances.csv', index=False)

def esa19(options, out_dir):
	df = pd.read_csv('esa19_both_imbalances.csv')

	algos = ["hMetis-R", "KaHyPar-MF", "ReBaHFC-D","ReBaHFC-Q", "PaToH-D","PaToH-Q", "KaHyPar-EVO", "HFC-100"]
	colors = commons.construct_color_mapping_with_default_colors(algos)


	for eps in [0.03, 0.0]:
		if eps == 0.03:
			continue
		fig = plt.figure(figsize=options['figsize'])
		my_algos = algos.copy()
		if eps == 0.03:
			my_algos = my_algos[:-2]
			print(my_algos)

		cpprs.combined_pp_rs(df[df.epsilon==eps], fig, baseline="PaToH-D", algos=my_algos, colors=colors, time_limit=28801, width_ratios=[0.5, 0.5], ncol=4)
		fig.savefig(out_dir + "esa19_setA_" + str(eps) +  ".pdf", bbox_inches="tight", pad_inches=0.0)
	
	return

	for algo_tuple in [
						('ReBaHFC-D', 'PaToH-D'), 
						('ReBaHFC-Q', 'PaToH-Q'), 
						('ReBaHFC-D', 'PaToH-Q')
						]:
		my_algos = list(algo_tuple)
				
		fig = plt.figure(figsize=options['half_figsize'])
		performance_profiles.infer_plot(df[df.epsilon==0.03], fig, algos=my_algos, colors=colors, )
		fig.savefig(out_dir + "esa19_direct_" + my_algos[0] + "_" + my_algos[1] + ".pdf", bbox_inches="tight", pad_inches=0.0)


	for algo_tuple in itertools.product(["KaHyPar-MF", "hMetis-R", "PaToH-Q"]
                                    , ["ReBaHFC-D", "ReBaHFC-Q"]#, "Mt-KaHyPar-D-F"]
                                    ):
		my_algos = list(algo_tuple)
		virt_df = effectiveness_tests.create_virtual_instances(df[df.epsilon == 0.03], my_algos, num_repetitions=20)
		
		fig = plt.figure(figsize=options['half_figsize'])
		performance_profiles.infer_plot(virt_df, fig, colors=colors)
		fig.savefig(out_dir + "esa19_effectiveness-tests_" + my_algos[0] + "_" + my_algos[1] + ".pdf", bbox_inches="tight", pad_inches=0.0)

def bulk(options, out_dir):
	df = commons.read_file('bulk_piercing.csv')
	df['timeout'] = df['time_limit_exceeded']
	df["algorithm"] = df.apply(lambda x : "Bulk" if x["bulk_piercing"] else "NoBulk", axis=1)
	df["km1"] = df[["flow","flowbound"]].min(axis=1)
	df["imbalance"] = 0.029
	df["epsilon"] = 0.03
	df["totalPartitionTime"] = df["time"]
	print(df[df.bulk_piercing].time.max())
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Bulk", time_limit=7200)
	fig.savefig(out_dir + "bulk_piercing.pdf", bbox_inches="tight", pad_inches=0.0)

def taus(options, out_dir):
	import sqlite3
	df = pd.DataFrame()
	for f in ["0_5", 1, 2, 4, "max"]:
		con = sqlite3.connect("tau_dbs/tau_" + str(f) + ".db")
		t_df = pd.read_sql_query("SELECT * from ex1", con)
		df = pd.concat([df, t_df])
	
	df.rename(columns={'partitionTime' : 'totalPartitionTime', "num_threads" : "threads"}, inplace=True)
	df["tau"] = df['flow_parallel_searches_multiplier']
	df["algorithm"] = df.apply(lambda x : R'$\tau=' + str(x['flow_parallel_searches_multiplier']) + R'$', axis=1)
	df["algorithm"] = df.apply(lambda x : R'$\tau=\infty$' if "2048" in x['algorithm'] else x['algorithm'], axis=1)

	df['timeout'] = 'no'
	df['failed'] = 'no'
	
	colors = commons.construct_new_color_mapping(commons.infer_algorithms_from_dataframe(df))
	fig, outer_grid = plt.subplots(nrows=1, ncols=2, figsize=(options['width'], options['height']))
	for ax in outer_grid.ravel():	# this is necessary when using subplots instead of gridspec
		ax.set_axis_off()
	handles, labels = performance_profiles.infer_plot(df[df.tau <= 2], fig, external_subplot=outer_grid[0], colors=colors, display_legend=False)
	handles2, labels2 = performance_profiles.infer_plot(df[df.tau >= 1], fig, external_subplot=outer_grid[1], colors=colors, display_legend=False)
	
	fig.legend(handles + handles2[2:], labels + labels2[2:], loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=5)

	plt.subplots_adjust(wspace=0.25)
	fig.savefig(out_dir + 'taus.pdf', bbox_inches="tight", pad_inches=0.0)
	
	mapper = {
		'num_flow_improvement'							: 'flows_num_improvements',
		'num_flow_refinements'							: 'flows_num_refinements',	
		'failed_updates_due_to_balance_constraint'		: 'flows_balance_reverts',
		'failed_updates_due_to_conflicting_moves'		: 'flows_gain_reverts',
		'correct_expected_improvement'					: 'flows_correct_gains',
		'zero_gain_improvement'							: 'flows_zero_gain_improvement',
	}

	df.rename(columns=mapper, inplace=True)
	df['flows_moves'] = df['flows_num_improvements'] + df['flows_balance_reverts'] + df['flows_gain_reverts']
	df['flows_incorrect_gains'] = df['flows_moves'] - df['flows_correct_gains']

	fraction_tuples = [	
		('flows_moves', 'flows_num_refinements'),
		('flows_num_improvements', 'flows_moves'),
		('flows_zero_gain_improvement', 'flows_num_improvements'),

		('flows_incorrect_gains', 'flows_moves'),
		('flows_gain_reverts', 'flows_moves'),
		('flows_balance_reverts', 'flows_moves'),
	]

	labels = ['claimed improvement', 'actual improvement', 'zero gain', 'gain wrong', 'negative gain revert', 'balance revert',]

	for (a,b), name in zip(fraction_tuples, labels):
		df[name] = df[a] / df[b]
		
	df = df.select_dtypes(['number'])
	unrolled = df.melt(id_vars=['k','epsilon','seed','tau'])
	colors = commons.construct_new_color_mapping(list(df.tau.unique()))
	import event_frequency
	for lb,ub in [(8,16), (64,64)]:
		fig, ax = plt.subplots(figsize=options['figsize'])
		print(unrolled[(unrolled.k >= lb) & (unrolled.k <= ub)])
		event_frequency.plot(unrolled[(unrolled.k >= lb) & (unrolled.k <= ub)], fig, ax, fields=labels, hue='tau', colors=colors)
		if lb == ub:
			ax.set_title(R'$k=' + str(lb)+ R'$')
		else:
			ax.set_title(R'$k \in \{' + str(lb) + "," + str(ub) + R'\}$')
		ax.set_ylabel('frequency')
		handles, legend_labels = ax.get_legend_handles_labels()
		print(handles, legend_labels)
		ax.legend(handles=handles[:5], labels=[0.5,1,2,4,R'$\infty$'], title=R'$\tau$')
		#ax.legend(title=R'$\tau$')
		fig.savefig(out_dir + "tau_stats_" + str(lb) + ".pdf", bbox_inches='tight', pad_inches=0.0)
		plt.close()

def flow_phase_times(options, out_dir):
	fig = plt.figure(figsize=options['figsize'])
	df = commons.read_file('flowcutter_phase_time.csv')
	df.rename(columns={'mbc_time' : 'balance'}, inplace=True)
	fields = ["global relabel", "discharge",  "update", "source cut", "saturate", "assimilate", "pierce"]
	runtime_share.plot(df, fields=fields, sort_field="global relabel", fig=fig, tfield='time')
	fig.axes[0].spines['top'].set_visible(False)
	fig.savefig(out_dir + "flowcutter_phase_runtime_shares.pdf", bbox_inches='tight', pad_inches=0.0)


	fig = plt.figure(figsize=options['figsize'])
	df = commons.read_file('flow_phase_time.csv')
	fields = ["global relabel", "discharge",  "update", "saturate"]
	runtime_share.plot(df, fields=fields, sort_field="global relabel", fig=fig, tfield='time')
	fig.axes[0].spines['top'].set_visible(False)
	fig.savefig(out_dir + "flow_phase_runtime_shares.pdf", bbox_inches='tight', pad_inches=0.0)

def run_all(options, out_dir):
	print('flows')
	main_setA(options, out_dir)
	return
	esa19(options, out_dir)
	runtime_shares(options, out_dir)
	flow_phase_times(options, out_dir)
	bulk(options, out_dir)
	main_setB(options, out_dir)
	taus(options, out_dir)
	refinement_stats_by_threads(options, out_dir)
	mt_kahypar_speedup_plots(options, out_dir)

	main_sea20(options, out_dir)


	flowcutter_speedup_numbers()
	flows_speedup_numbers()
	
	runtime_share_sea20(options, out_dir)

	gain_contributions(options, out_dir)
	

	flows_speedup_plots(options, out_dir, "ParPR-RL")
	flows_speedup_plots(options, out_dir, "ParPR-Block")
	flows_relative_runtime_plots(options, out_dir)
	flowcutter_speedup_plots(options, out_dir)

	increasing_threads(options, out_dir)
	effectiveness_tests_plot(options, out_dir)

	runtime_shares_flows_by_threads(options, out_dir)
	refinement_stats(options, out_dir)
	
	plt.close('all')
	
	return
	explore_esa19(options, out_dir)	
