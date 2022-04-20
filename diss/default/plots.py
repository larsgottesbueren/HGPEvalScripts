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

def mt_kahypar_speedup_plots(options, out_dir):
	paper_width = options['width']
	aspect_ratio = 0.92
	height = paper_width / aspect_ratio
	fig, axes = plt.subplots(3, 2, sharey=True, figsize=(paper_width, height))

	df = commons.read_file("scalability.csv")
	df = df[df.algorithm == "Mt-KaHyPar-D"]
	# df = df[df.threads != 128]
	thread_list = sorted(list(df.threads.unique()))
	if 1 in thread_list:
		thread_list.remove(1)
	else:
		print("no sequential runs :(")
		return
	color_mapping = commons.construct_new_color_mapping(thread_list)

	fields = ["totalPartitionTime", "preprocessingTime", "coarseningTime", "ipTime", "lpTime", "fmTime"]
	name_map = {
		"totalPartitionTime" : "total",
		"preprocessingTime" : "preprocessing",
		"coarseningTime" : "coarsening",
		"ipTime" : "initial partitioning",
		"lpTime" : "LP refinement",
		"fmTime" : "FM refinement",
	}
	for ax, (i, field) in zip(axes.ravel(), enumerate(fields)):
		print(i, field)
		speedup_plots.scalability_plot(df=df, field=field, ax=ax, thread_colors=color_mapping, display_labels=False, display_legend=False, seed_aggregator="median",
		                               xscale='log', yscale='log', show_rolling_gmean=True, alpha=0.5, filter_tiny_outlier_threshold = 1.0)
		ax.set_xlabel(name_map[field] + ". seq time [s]")
		ax.set_ylabel("")
		
		ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		ax.set_yticks([2,4,8,16,32,64,128])
		
		if field == "label_propagation_time":
			# ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
			ax.set_xticks([1, 10, 100, 1000])

	for row in range(3):
		for col in range(2):
			ax = axes[row][col]
			if col != 0:
				ax.yaxis.set_ticks_position('none')

	handles, labels = axes[0][0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), frameon=False, ncol=3, title="threads")

	for row in range(3):
		axes[row][0].set_ylabel('speedup')


	plt.subplots_adjust(wspace=0.025, hspace=0.4)
	plt.savefig(out_dir + "mt-kahypar-d-speedups.pdf", bbox_inches='tight', pad_inches=0.0)

def increasing_threads(options, out_dir):
	df = commons.read_file("scalability.csv")
	df["algorithm"] = df["algorithm"] + " " + df["threads"].astype(str)
	algos = ["Mt-KaHyPar-D " + str(i) for i in [1,4,16,64]]
	color_mapping_algos = ["Mt-KaHyPar-D " + str(i) for i in [4,16,64, 1]]
	colors = commons.construct_new_color_mapping(color_mapping_algos)
	
	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig, algos=algos, colors=colors)
	fig.savefig(out_dir + "increasing_threads.pdf", bbox_inches="tight", pad_inches=0.0)

def main_setA(options, out_dir):
	mt_kahypar_file_list = ["mt-kahypar-d-setA.csv"]
	others_file_list = ["hmetis_r_setA.csv", "kahypar_ca_setA.csv", "kahypar_hfc_setA.csv", "patoh_d_setA.csv", "patoh_q_setA.csv"]

	# main comparison
	df = commons.read_files(mt_kahypar_file_list)
	df = df[df.threads == 10]
	
	df2 = commons.read_files(others_file_list)
	df = pd.concat([df, df2])

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-D", time_limit=28800)
	fig.savefig(out_dir + "setA.pdf", bbox_inches="tight", pad_inches=0.0)

	
def main_setB(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "bipart-64.csv", "zoltan-mt-bench.csv", "patoh-d-mt-bench.csv"])
	
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-D")
	fig.savefig(out_dir + "setB.pdf", bbox_inches="tight", pad_inches=0.0)

def print_speedups():
	df = commons.read_file("scalability.csv")
	fields = ["totalPartitionTime", "coarseningTime", "ipTime", "preprocessingTime", "lpTime", "fmTime"]
	for field in fields:
		print(field)
		speedup_plots.print_speedups(df=df, field=field, seed_aggregator="median", min_sequential_time = 0)
	
def effectiveness_tests_plot(options, out_dir):
	mt_kahypar_file_list = ["mt-kahypar-d-setA.csv"]
	others_file_list = ["hmetis_r_setA.csv", "kahypar_ca_setA.csv", "kahypar_hfc_setA.csv"]
	df = commons.read_files(mt_kahypar_file_list)
	df = df[df.threads == 10]
	df2 = commons.read_files(others_file_list)
	df = pd.concat([df, df2])
		
	width = options["width"] / 2
	aspect_ratio = 1.65
	height = width / aspect_ratio
	figsize=(width, height)

	for algo_tuple in itertools.product(["Mt-KaHyPar-D"], ["hMetis-R", "KaHyPar-CA", "KaHyPar-HFC"]):
		algos = list(algo_tuple)
		virt_df = effectiveness_tests.create_virtual_instances(df, algos, num_repetitions=20)
		fig = plt.figure(figsize=options['half_figsize'])
		performance_profiles.infer_plot(virt_df, fig)
		fig.savefig(out_dir + "effectiveness-tests_" + algos[0] + "_" + algos[1] + ".pdf", bbox_inches="tight", pad_inches=0.0)

def graph_experiments(options, out_dir):
	df = commons.read_files(["graph_experiments.csv"])
	df["algorithm"].replace(to_replace={"Mt-KaHyPar-Graph-D" : "Mt-KaHyPar-D"}, inplace=True)

	algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.construct_new_color_mapping(algos)

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-D", colors=colors, algos=algos)
	fig.savefig(out_dir + 'graphs.pdf', bbox_inches='tight', pad_inches=0.0)

def large_k(options, out_dir):
	colors = commons.default_color_mapping()
	colors["Mt-KaHyPar-S"] = colors["hMetis-R"]
	df = commons.read_files(["large-k.csv"])
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-S", colors=colors)
	fig.savefig(out_dir + 'large-k.pdf', bbox_inches='tight', pad_inches=0.0)

def refinement_stats(options, out_dir):
	df = pd.read_csv('mt-kahypar-d-refinement-stats.csv')

	fraction_tuples = [
		('lp_incorrect_gains', 'lp_moves'),
		('lp_gain_reverts', 'lp_moves'),
		('lp_balance_reverts', 'lp_moves'),
		('lp_actual_gain_sum', 'lp_expected_gain_sum'),

		('attributed_incorrect_gains', 'attributed_moves'),
		('attributed_reverts', 'attributed_moves'),
		('attributed_actual_gain_sum', 'attributed_expected_gain_sum'),

		('rollback_incorrect_gains', 'rollback_moves'),
		('rollback_reverts', 'rollback_moves'),
		('rollback_actual_gain_sum', 'rollback_expected_gain_sum'),
	]

	fig = plt.figure(figsize=options['figsize'])


def runtime_share(options, out_dir):
	import runtime_share
	fig = plt.figure(figsize=options['figsize'])
	
	mapper = {	'fmTime':'FM', 'lpTime':'LP', 'preprocessingTime':'Preprocessing', 
				'coarseningTime':'Coarsening', 'ipTime':'Initial'
				}

	df = commons.read_file('mt-kahypar-d-64_binary.csv')
	df.rename(columns=mapper, inplace=True)
	fields = list(mapper.values())
	df = df[df.seed == 0]
	df = runtime_share.clean(df)
	runtime_share.plot(df, fields=fields, sort_field="FM", fig=fig)
	fig.axes[0].spines['top'].set_visible(False)
	fig.savefig(out_dir + "runtime_shares.pdf", bbox_inches='tight', pad_inches=0.0)

def mt_metis_design_choices(options, out_dir):
	# no negative gains

	# static assignment over localization

	# message queues
	print("not implemented yet")

def run_all(options, out_dir):

	runtime_share(options, out_dir)
	exit()
	refinement_stats(options, out_dir)

	increasing_threads(options, out_dir)

	main_setB(options, out_dir)
	main_setA(options, out_dir)
	graph_experiments(options, out_dir)
	
	mt_kahypar_speedup_plots(options, out_dir)
	print_speedups()

	effectiveness_tests_plot(options, out_dir)
	
	large_k(options, out_dir)

	mt_metis_design_choices(options, out_dir)
