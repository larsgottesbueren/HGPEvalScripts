import performance_profiles
import relative_runtimes_plot
import speedup_plots
import commons
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sb
import glob

def infer(df, colors, figsize):
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=colors, figsize=figsize)
	performance_profiles.legend_inside(fig, ncol=1)
	return fig

def max_batch_size(options, out_dir):
	width = options["width"] / 2
	aspect_ratio = 1.65
	height = width / aspect_ratio
	figsize=(width, height)

	df = commons.read_files(list(glob.glob("max-batch-size/*.csv")))
	all_algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.construct_new_color_mapping(all_algos)

	fig = infer(df[(df.max_batch_size == 1) | (df.max_batch_size == 100)], colors, figsize)
	fig.savefig(out_dir + "max_batch_size_1_100.pdf", bbox_inches="tight", pad_inches=0.0)

	fig = infer(df[(df.max_batch_size == 100) | (df.max_batch_size == 200) | (df.max_batch_size == 1000)], colors, figsize)
	fig.savefig(out_dir + "max_batch_size_100_200_1000.pdf", bbox_inches="tight", pad_inches=0.0)

	fig = infer(df[(df.max_batch_size == 200) | (df.max_batch_size == 1000) | (df.max_batch_size == 10000)], colors, figsize)
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
	
	width = options["width"] / 2
	aspect_ratio = 1.65
	height = width / aspect_ratio
	figsize=(width, height)

	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=colors, figsize=figsize)
	performance_profiles.legend_inside(fig, ncol=1)
	fig.savefig(out_dir + "increasing_threads.pdf", bbox_inches="tight", pad_inches=0.0)

def main_setA(options, out_dir):
	prefix = "setA-"
	time_limit = 28800
	mt_kahypar_file_list = ["mt-kahypar-d-setA.csv", "mt-kahypar-q-setA.csv"]
	others_file_list = ["hmetis_r_setA.csv", "kahypar_ca_setA.csv", "kahypar_hfc_setA.csv", "patoh_d_setA.csv", "patoh_q_setA.csv"]

	width = options["width"] * 0.8
	aspect_ratio = 2.7
	height = width / aspect_ratio
	figsize = (width, height)

	# main comparison
	df = commons.read_files(mt_kahypar_file_list)
	df = df[df.threads == 10]
	
	df2 = commons.read_files(others_file_list)
	df = pd.concat([df, df2])

	algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.default_color_mapping()
	instances = commons.infer_instances_from_dataframe(df)
		
	## separate plots
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=colors, figsize=figsize, width_scale=2.0)
	fig.savefig(out_dir + prefix + "mt-kahypar-q_performance-profile.pdf", bbox_inches="tight", pad_inches=0.0)
	
	width = options['width'] * 0.5
	aspect_ratio = 1.2
	height = width / aspect_ratio
	figsize = (width, height)
	fig, ax = plt.subplots(figsize=figsize)
	relative_runtimes_plot.construct_plot(df=df, ax=ax, baseline_algorithm="Mt-KaHyPar-Q", colors=colors, algos=algos, 
	                                      seed_aggregator="mean", field="totalPartitionTime", time_limit=time_limit)
	sb.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
	fig.savefig(out_dir + prefix + "mt-kahypar-q_relative_slowdown.pdf", bbox_inches='tight', pad_inches=0.0)

def main_setB(options, out_dir):
	prefix = "setB-"
	time_limit = 7200
	file_list = ["mt-kahypar-d-64.csv", "mt-kahypar-q-64.csv", "bipart-mt-bench.csv", "zoltan-mt-bench.csv"] # add Patoh-D
	width = options["width"] * 0.8 
	aspect_ratio = 2.7
	height = width / aspect_ratio
	figsize = (width, height)

	# main comparison
	df = commons.read_files(file_list)
	algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.default_color_mapping()
	instances = commons.infer_instances_from_dataframe(df)
		
	## separate plots
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=colors, figsize=figsize, width_scale=2.0)
	fig.savefig(out_dir + prefix + "mt-kahypar-q_performance-profile.pdf", bbox_inches="tight", pad_inches=0.0)

	## TODO one plot with fewer algos
	
	width = options['width'] * 0.5
	aspect_ratio = 1.2
	height = width / aspect_ratio
	figsize = (width, height)
	fig, ax = plt.subplots(figsize=figsize)
	relative_runtimes_plot.construct_plot(df=df, ax=ax, baseline_algorithm="Mt-KaHyPar-Q", colors=colors, algos=algos, 
	                                      seed_aggregator="mean", field="totalPartitionTime", time_limit=time_limit)
	sb.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
	fig.savefig(out_dir + prefix + "mt-kahypar-q_relative_slowdown.pdf", bbox_inches='tight', pad_inches=0.0)

def main_async(options, out_dir):
	prefix = "setB-async-"
	time_limit = 7200
	file_list = ["async/mt_kahypar_async_optimized_64.csv", "async/mt_kahypar_d_64.csv", "async/mt_kahypar_q_64.csv", "async/bipart_64.csv", "async/zoltan_64.csv", "async/patoh_d_big.csv"]
	width = options["width"] * 0.8 
	aspect_ratio = 2.7
	height = width / aspect_ratio
	figsize = (width, height)

	# main comparison
	df = commons.read_files(file_list)
	algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.default_color_mapping()
	colors["Mt-KaHyPar-Async"] = colors["hMetis-R"]
	instances = commons.infer_instances_from_dataframe(df)
		
	## separate plots
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=colors, figsize=figsize, width_scale=2.0)
	fig.savefig(out_dir + prefix + "mt-kahypar-q_performance-profile.pdf", bbox_inches="tight", pad_inches=0.0)
	
	width = options['width'] * 0.6
	aspect_ratio = 1.4
	height = width / aspect_ratio
	figsize = (width, height)
	fig, ax = plt.subplots(figsize=figsize)
	relative_runtimes_plot.construct_plot(df=df, ax=ax, baseline_algorithm="Mt-KaHyPar-Async", colors=colors, algos=algos, 
	                                      seed_aggregator="mean", field="totalPartitionTime", time_limit=time_limit)
	sb.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
	fig.savefig(out_dir + prefix + "mt-kahypar-q_relative_slowdown.pdf", bbox_inches='tight', pad_inches=0.0)

def print_speedups():
	df = pd.read_csv('scalability.csv')
	fields = ["partitionTime", "preprocessingTime", "coarseningTime", "ipTime", "lpTime"]#, "fmTime"]
	for field in fields:
		print(field)
		speedup_plots.print_speedups(df=df, field=field, seed_aggregator="median", min_sequential_time = 0)
	


def run_all(options, out_dir):
	# max_batch_size(options, out_dir)
	# mt_kahypar_speedup_plots(options, out_dir)
	# increasing_threads(options, out_dir)
	main_setB(options, out_dir)
	main_setA(options, out_dir)
	main_async(options, out_dir)
	# print_speedups()
