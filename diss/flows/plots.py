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
	aspect_ratio = 0.92
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
		speedup_plots.scalability_plot(df=df[(df.k >= k_lb) & (df.k <= k_ub)], field="flowTime", ax=ax, thread_colors=color_mapping, display_labels=False, display_legend=False, seed_aggregator="median",
		                               xscale='log', yscale='log', show_rolling_gmean=True, alpha=0.5, filter_tiny_outlier_threshold = 1.0)
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


	plt.subplots_adjust(wspace=0.025, hspace=0.3)
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
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-D-F", time_limit=28800)
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
	import runtime_share
	fig = plt.figure(figsize=options['figsize'])

	mapper = {	'fm':'FM refinement', 'label_propagation':'LP refinement', 'preprocessing':'preprocessing', 
				'coarsening':'coarsening', 'initial_partitioning':'initial', 'flow_refinement_scheduler':'flows'
				}
	df = commons.read_file('mt-kahypar-d-f-refinement-stats.csv')
	df.rename(columns=mapper, inplace=True)
	fields = list(mapper.values())
	df = df[df.seed == 0]
	df = runtime_share.clean(df)
	runtime_share.plot(df, fields=fields, sort_field="Flows", fig=fig, tfield='totalPartitionTime')
	fig.axes[0].spines['top'].set_visible(False)
	fig.savefig(out_dir + "runtime_shares.pdf", bbox_inches='tight', pad_inches=0.0)

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

	fig, ax = plt.subplots(figsize=options['figsize'])
	import event_frequency
	event_frequency.plot(unrolled, fig, ax, fields=labels, hue='threads', colors=thread_colors)
	ax.set_ylabel('frequency')
	fig.savefig(out_dir + "refinement_stats_by_threads.pdf", bbox_inches='tight', pad_inches=0.0)

def run_all(options, out_dir):

	flows_speedup_plots(options, out_dir, "ParPR-RL")
	flows_speedup_plots(options, out_dir, "ParPR-Block")
	flows_relative_runtime_plots(options, out_dir)
	flowcutter_speedup_plots(options, out_dir)

	exit()
	increasing_threads(options, out_dir)
	effectiveness_tests_plot(options, out_dir)
	main_setA(options, out_dir)
	main_setB(options, out_dir)

	effectiveness_tests_plot(options, out_dir)
	mt_kahypar_speedup_plots(options, out_dir)

	runtime_shares(options, out_dir)
	refinement_stats_by_threads(options, out_dir)
	runtime_shares_flows_by_threads(options, out_dir)
	refinement_stats(options, out_dir)

	flowcutter_speedup_numbers()
	flows_speedup_numbers()
	
