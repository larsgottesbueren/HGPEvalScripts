import performance_profiles
import relative_runtimes_plot
import speedup_plots
import commons
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sb

def price_of_determinism(options, out_dir):
	width = options["width"] / 2
	aspect_ratio = 1.65
	height = width / aspect_ratio
	figsize=(width, height)

	df = commons.read_and_convert("component_comparison_preprocessing.csv")
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=commons.construct_new_color_mapping(algos), figsize=figsize)
	performance_profiles.legend_inside(fig, ncol=1)
	fig.savefig(out_dir + "component_comparison_preprocessing.pdf", bbox_inches="tight", pad_inches=0.0)

	df = commons.read_and_convert("component_comparison_refinement.csv")
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=commons.construct_new_color_mapping(algos), figsize=figsize)
	performance_profiles.legend_inside(fig, ncol=1)
	fig.savefig(out_dir + "component_comparison_refinement.pdf", bbox_inches="tight", pad_inches=0.0)

	df = commons.read_and_convert("component_comparison_coarsening.csv")
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=commons.construct_new_color_mapping(algos), figsize=figsize)
	performance_profiles.legend_inside(fig, ncol=1)
	fig.savefig(out_dir + "component_comparison_coarsening.pdf", bbox_inches="tight", pad_inches=0.0)
	# relative_runtimes_plot.plot("component_comparison_coarsening", df, "Mt-KaHyPar-SDet", colors= commons.construct_new_color_mapping(algos), field="coarseningTime")


def parameter_study(options, out_dir):
	third_width = options["width"] / 2
	aspect_ratio = 1.65
	height = third_width / aspect_ratio
	figsize=(third_width, height)

	# refinement
	df = commons.read_and_convert("deterministic_parameterstudy_synclp_subrounds.csv")
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=commons.construct_new_color_mapping(algos), figsize=figsize)
	fig.savefig(out_dir + "refinement_subrounds.pdf", bbox_inches="tight", pad_inches=0.0)

	# coarsening
	df = commons.read_and_convert("deterministic_parameterstudy_coarsening_subrounds.csv")
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=commons.construct_new_color_mapping(algos), figsize=figsize)
	fig.savefig(out_dir + "coarsening_subrounds.pdf", bbox_inches="tight", pad_inches=0.0)

	# preprocessing
	df = commons.read_and_convert("deterministic_parameterstudy_prepro_subrounds.csv")
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=commons.construct_new_color_mapping(algos), figsize=figsize)
	fig.savefig(out_dir + "preprocessing_subrounds.pdf", bbox_inches="tight", pad_inches=0.0)

	# preprocessing vs no preprocessing 
	df = commons.read_and_convert("no_preprocessing.csv")
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=commons.construct_new_color_mapping(algos), figsize=figsize)
	performance_profiles.legend_below(fig, ncol=1)
	fig.savefig(out_dir + "preprocessing_impact.pdf", bbox_inches="tight", pad_inches=0.0)

def bipart_speedup_plots(options, out_dir):
	paper_width = options['width'] * 0.7
	aspect_ratio = 2.25
	height = paper_width / aspect_ratio

	df = pd.read_csv('bipart-mt-bench.csv')
	df = df[df.algorithm == "BiPart"]
	time_limit = 7200
	df = df[(df.graph != 'kmer_P1a.mtx.hgr') | (df.k != 64)]
	df = df[df.threads != 128]
	
	fig, ax = plt.subplots(figsize=(paper_width, height))

	thread_list = sorted(list(df.threads.unique()))
	if 1 in thread_list:
		thread_list.remove(1)
	else:
		print("no sequential runs found. cannot compute speedups. abort")
		exit()
	color_mapping = commons.construct_new_color_mapping(thread_list)
	speedup_plots.scalability_plot(df=df, field="totalPartitionTime", ax=ax, thread_colors=color_mapping, 
	                               show_rolling_gmean=False, show_scatter=True, display_legend=True, seed_aggregator=None,
	                               xscale='log', display_labels=False)
	ax.set_xlabel("sequential time for BiPart [s]")
	ax.set_ylabel("speedup")
	plt.savefig(out_dir + "bipart_speedups.pdf", bbox_inches='tight', pad_inches=0.0)


def mt_kahypar_speedup_plots(options, out_dir):
	paper_width = options['width']
	aspect_ratio = 0.92
	height = paper_width / aspect_ratio
	fig, axes = plt.subplots(3, 2, sharey=True, figsize=(paper_width, height))

	df = pd.read_csv('scalability.csv')
	df = df[df.algorithm == "Mt-KaHyPar-SDet"]
	# df = df[df.threads != 128]
	thread_list = sorted(list(df.threads.unique()))
	if 1 in thread_list:
		thread_list.remove(1)
	else:
		print("no sequential runs :(")
		return
	color_mapping = commons.construct_new_color_mapping(thread_list)

	fields = ["partitionTime", "preprocessingTime", "coarseningTime", "ipTime", "lpTime"]
	name_map = {
		"partitionTime" : "total",
		"preprocessingTime" : "preprocessing",
		"coarseningTime" : "coarsening",
		"ipTime" : "initial partitioning",
		"lpTime" : "refinement"
	}
	for ax, (i, field) in zip(axes.ravel(), enumerate(fields)):
		print(i, field)
		speedup_plots.scalability_plot(df=df, field=field, ax=ax, thread_colors=color_mapping, display_labels=False, display_legend=False, seed_aggregator=None,
		                               xscale='log', yscale='log', show_rolling_gmean=True)
		ax.set_xlabel(name_map[field] + ". seq time [s]")
		ax.set_ylabel("")
		
		ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
		ax.set_yticks([4,8,16,32,64])
		
		if field == "lpTime":
			# ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
			ax.set_xticks([1, 10, 100])

	for row in range(3):
		for col in range(2):
			ax = axes[row][col]
			if col != 0:
				ax.yaxis.set_ticks_position('none')

	handles, labels = axes[0][0].get_legend_handles_labels()

	legend_ax = axes[-1][-1]
	legend_ax.legend(handles, labels, loc='center', ncol=2, title='threads')
	legend_ax.set_axis_off()
	
	for row in range(3):
		axes[row][0].set_ylabel('speedup')


	plt.subplots_adjust(wspace=0.025, hspace=0.4)
	plt.savefig(out_dir + "mt-kahypar-sdet_speedups.pdf", bbox_inches='tight', pad_inches=0.0)


def main(options, out_dir):
	width = options["width"] * 0.8
	aspect_ratio = 2.7
	height = width / aspect_ratio
	figsize = (width, height)

	# main comparison
	df = commons.read_files(["speed_deterministic.csv", "speed_non_deterministic.csv", "bipart-mt-bench.csv", "default.csv", "zoltan-mt-bench.csv"])
	df = df[df.timeout=="no"]
	df = df[df.threads == 64].copy()
	algos = commons.infer_algorithms_from_dataframe(df)
	colors = commons.default_color_mapping()
	# reorder algorithms for legend after color mapping
	algos = ["Mt-KaHyPar-D", "Mt-KaHyPar-SDet", "Mt-KaHyPar-S", "Zoltan", "BiPart"]
	instances = commons.infer_instances_from_dataframe(df)
		
	## separate plots
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=colors, figsize=figsize, width_scale=2.0)
	fig.savefig(out_dir + "sdet_comparison_performance_profile.pdf", bbox_inches="tight", pad_inches=0.0)
	
	aspect_ratio = 1.2
	width = options["width"] / 2
	height = width / aspect_ratio
	figsize = (width, height)
	fig, ax = plt.subplots(figsize=figsize)
	relative_runtimes_plot.construct_plot(df=df, ax=ax, baseline_algorithm="Mt-KaHyPar-SDet", colors=colors, algos=algos, 
	                                      seed_aggregator="mean", field="totalPartitionTime", time_limit=7200)
	sb.move_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=2)
	fig.savefig(out_dir + "sdet_comparison_relative_slowdown.pdf", bbox_inches='tight', pad_inches=0.0)

def print_speedups():
	df = pd.read_csv('scalability.csv')
	fields = ["partitionTime", "preprocessingTime", "coarseningTime", "ipTime", "lpTime"]#, "fmTime"]
	for field in fields:
		print(field)
		speedup_plots.print_speedups(df=df, field=field, seed_aggregator="median", min_sequential_time = 0)
	


def run_all(options, out_dir):
	#price_of_determinism(options, out_dir)
	#parameter_study(options, out_dir)
	#bipart_speedup_plots(options, out_dir)
	#mt_kahypar_speedup_plots(options, out_dir)
	#main(options, out_dir)
	print_speedups()
