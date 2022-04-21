import performance_profiles
import relative_runtimes_plot
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

def max_batch_size(options, out_dir):
	width = options["width"] / 2
	aspect_ratio = 1.65
	height = width / aspect_ratio
	figsize=(width, height)

	df = commons.read_files(list(glob.glob("max-batch-size/*.csv")))
	all_algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.construct_new_color_mapping(all_algos)

	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df[(df.max_batch_size == 1) | (df.max_batch_size == 100)], fig, colors=colors)
	fig.savefig(out_dir + "max_batch_size_1_100.pdf", bbox_inches="tight", pad_inches=0.0)

	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df[(df.max_batch_size == 100) | (df.max_batch_size == 200) | (df.max_batch_size == 1000)], 
	                                fig, colors=colors)
	fig.savefig(out_dir + "max_batch_size_100_200_1000.pdf", bbox_inches="tight", pad_inches=0.0)

	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df[(df.max_batch_size == 200) | (df.max_batch_size == 1000) | (df.max_batch_size == 10000)],
									fig, colors=colors)
	fig.savefig(out_dir + "max_batch_size_200_1000_10000.pdf", bbox_inches="tight", pad_inches=0.0)


def mt_kahypar_speedup_plots(options, out_dir):
	paper_width = options['width']
	aspect_ratio = 0.92
	height = paper_width / aspect_ratio
	fig, axes = plt.subplots(3, 2, sharey=True, figsize=(paper_width, height))

	df = commons.read_files(list(glob.glob("mt_kahypar_q_*_scaling.csv")))
	df = df[df.algorithm == "Mt-KaHyPar-Q"]
	# df = df[df.threads != 128]
	thread_list = sorted(list(df.threads.unique()))
	if 1 in thread_list:
		thread_list.remove(1)
	else:
		print("no sequential runs :(")
		return
	color_mapping = commons.construct_new_color_mapping(thread_list)

	fields = ["totalPartitionTime", "coarsening_time", "initial_partitioning_time", "batch_uncontraction_time", "label_propagation_time", "fm_time"]
	name_map = {
		"totalPartitionTime" : "total",
		"preprocessingTime" : "preprocessing",
		"coarsening_time" : "coarsening",
		"initial_partitioning_time" : "initial partitioning",
		"batch_uncontraction_time" : "uncoarsening",
		"label_propagation_time" : "LP refinement",
		"fm_time" : "FM refinement",
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
	plt.savefig(out_dir + "mt-kahypar-q-speedups.pdf", bbox_inches='tight', pad_inches=0.0)

def increasing_threads(options, out_dir):
	df = commons.read_files(list(glob.glob("mt_kahypar_q_*_scaling.csv")))
	df["algorithm"] = df["algorithm"] + " " + df["threads"].astype(str)
	algos = ["Mt-KaHyPar-Q " + str(i) for i in [1,4,16,64]]
	color_mapping_algos = ["Mt-KaHyPar-Q " + str(i) for i in [4,16,64, 1]]
	colors = commons.construct_new_color_mapping(color_mapping_algos)
	
	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig, algos=algos, colors=colors)
	fig.savefig(out_dir + "increasing_threads.pdf", bbox_inches="tight", pad_inches=0.0)

	
def main_setA(options, out_dir):
	mt_kahypar_file_list = ["mt-kahypar-d-setA.csv", "mt-kahypar-q-setA.csv"]
	others_file_list = ["hmetis_r_setA.csv", "kahypar_ca_setA.csv", "kahypar_hfc_setA.csv", "patoh_d_setA.csv", "patoh_q_setA.csv"]

	df = commons.read_files(mt_kahypar_file_list)
	df = df[df.threads == 10]
	
	df2 = commons.read_files(others_file_list)
	df = pd.concat([df, df2])

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-Q", time_limit=28800)
	fig.savefig(out_dir + "setA.pdf", bbox_inches="tight", pad_inches=0.0)

def main_setB(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "mt-kahypar-q-64.csv", "bipart-64.csv", "zoltan-mt-bench.csv", "patoh-d-mt-bench.csv"])

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-D")
	fig.savefig(out_dir + "setB.pdf", bbox_inches="tight", pad_inches=0.0)

def main_async(options, out_dir):
	file_list = ["async/mt_kahypar_async_optimized_64.csv", "async/mt_kahypar_d_64.csv", "async/mt_kahypar_q_64.csv", "async/bipart_64.csv", "async/zoltan_64.csv", "async/patoh_d_big.csv"]
	df = commons.read_files(file_list)
	algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.default_color_mapping()
	colors["Mt-KaHyPar-Async"] = colors["hMetis-R"]
	instances = commons.infer_instances_from_dataframe(df)

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-Async", colors=colors)
	fig.savefig(out_dir + "async-setB.pdf", bbox_inches="tight", pad_inches=0.0)

def print_speedups():
	df = commons.read_files(list(glob.glob("mt_kahypar_q_*_scaling.csv")))
	fields = ["totalPartitionTime", "coarsening_time", "initial_partitioning_time", "batch_uncontraction_time", "label_propagation_time", "fm_time"]
	for field in fields:
		print(field)
		speedup_plots.print_speedups(df=df, field=field, seed_aggregator="median", min_sequential_time = 0)
	
def effectiveness_tests_plot(options, out_dir):
	mt_kahypar_file_list = ["mt-kahypar-d-setA.csv", "mt-kahypar-q-setA.csv"]
	others_file_list = ["hmetis_r_setA.csv", "kahypar_ca_setA.csv", "kahypar_hfc_setA.csv"]
	df = commons.read_files(mt_kahypar_file_list)
	df = df[df.threads == 10]
	df2 = commons.read_files(others_file_list)
	df = pd.concat([df, df2])

	for algo_tuple in itertools.product(["Mt-KaHyPar-Q"]
	                                    , ["hMetis-R", "KaHyPar-CA", "KaHyPar-HFC", "Mt-KaHyPar-D"]
	                                    ):
		algos = list(algo_tuple)
		virt_df = effectiveness_tests.create_virtual_instances(df, algos, num_repetitions=20)
		
		fig = plt.figure(figsize=options['half_figsize'])
		performance_profiles.infer_plot(virt_df, fig)
		fig.savefig(out_dir + "effectiveness-tests_" + algos[0] + "_" + algos[1] + ".pdf", bbox_inches="tight", pad_inches=0.0)


def runtime_share(options, out_dir):
	import runtime_share
	fig = plt.figure(figsize=options['figsize'])
	mapper = {	'fm_time':'FM refinement', 'label_propagation_time':'LP refinement', 'preprocessing_time':'preprocessing', 
				'coarsening_time':'coarsening', 'initial_partitioning_time':'initial',
				"batch_uncontraction_time" : "uncoarsening",
				}
	df = commons.read_file('mt_kahypar_q_64_scaling.csv')
	df.rename(columns=mapper, inplace=True)
	fields = list(mapper.values())
	df = df[df.seed == 0]
	df = runtime_share.clean(df)
	runtime_share.plot(df, fields=fields, sort_field="FM refinement", fig=fig, tfield='totalPartitionTime')
	fig.axes[0].spines['top'].set_visible(False)
	fig.savefig(out_dir + "runtime_shares.pdf", bbox_inches='tight', pad_inches=0.0)

def refinement_stats(options, out_dir):
	df = pd.read_csv('mt-kahypar-q-refinement-stats.csv')

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

	labels = [
		'LP gain wrong',
		'LP gain revert',
		'LP balance revert',
		'LP gain',

		'FM attributed gain wrong',
		'FM attributed revert',
		'FM attributed gain',
		
		'FM rollback gain wrong',
		'FM rollback revert',
		'FM rollback gain'
	]

	for (a,b), name in zip(fraction_tuples, labels):
		df[name] = df[a] / df[b]
	
	df = df.select_dtypes(['number'])
	unrolled = df.melt(id_vars=['k','epsilon','seed','threads'])

	fig, ax = plt.subplots(figsize=options['figsize'])
	import event_frequency
	event_frequency.plot(unrolled, fig, ax, fields=labels)
	fig.savefig(out_dir + "refinement_stats.pdf", bbox_inches='tight', pad_inches=0.0)


def run_all(options, out_dir):
	max_batch_size(options, out_dir)
	mt_kahypar_speedup_plots(options, out_dir)
	increasing_threads(options, out_dir)
	main_setB(options, out_dir)
	main_setA(options, out_dir)
	main_async(options, out_dir)
	print_speedups()
	effectiveness_tests_plot(options, out_dir)
	runtime_share(options, out_dir)
	refinement_stats(options, out_dir)



