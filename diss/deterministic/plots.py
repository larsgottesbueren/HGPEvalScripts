import performance_profiles
import relative_runtimes_plot
import speedup_plots
import commons
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.ticker
import seaborn as sb

def price_of_determinism(options, out_dir):
	df = commons.read_and_convert("component_comparison_preprocessing.csv")
	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig, algos=algos, colors=colors)
	fig.savefig(out_dir + "component_comparison_preprocessing.pdf", bbox_inches="tight", pad_inches=0.0)

	df = commons.read_and_convert("component_comparison_refinement.csv")
	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig, algos=algos, colors=colors)
	fig.savefig(out_dir + "component_comparison_refinement.pdf", bbox_inches="tight", pad_inches=0.0)

	df = commons.read_and_convert("component_comparison_coarsening.csv")
	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig, algos=algos, colors=colors)
	fig.savefig(out_dir + "component_comparison_coarsening.pdf", bbox_inches="tight", pad_inches=0.0)
	# relative_runtimes_plot.plot("component_comparison_coarsening", df, "Mt-KaHyPar-SDet", colors= commons.construct_new_color_mapping(algos), field="coarseningTime")


def parameter_study(options, out_dir):
	fig, outer_grid = plt.subplots(nrows=2, ncols=2, figsize=(options['width'], 2 * options['height']))
	for ax in outer_grid.ravel():	# this is necessary when using subplots instead of gridspec
		ax.set_axis_off()
	
	# refinement
	df = commons.read_and_convert("deterministic_parameterstudy_synclp_subrounds.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[0][1], display_legend=False)
	outer_grid[0][1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)

	# coarsening
	df = commons.read_and_convert("deterministic_parameterstudy_coarsening_subrounds.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[1,0], display_legend=False)
	outer_grid[1][0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)

	# preprocessing
	df = commons.read_and_convert("deterministic_parameterstudy_prepro_subrounds.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[0,0], display_legend=False)
	outer_grid[0][0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)
	
	# preprocessing vs no preprocessing 
	df = commons.read_and_convert("no_preprocessing.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[1,1], display_legend=False)
	outer_grid[1][1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)

	plt.subplots_adjust(wspace=0.22, hspace=0.6)
	fig.savefig(out_dir + "parameter_study.pdf", bbox_inches="tight", pad_inches=0.0)

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
	df = commons.read_files(["speed_deterministic.csv", "speed_non_deterministic.csv", "bipart-64.csv", "default.csv", "zoltan-mt-bench.csv"])
	df = df[df.timeout=="no"]
	df = df[df.threads == 64].copy()
	# reorder algorithms for legend after color mapping
	algos = ["Mt-KaHyPar-D", "Mt-KaHyPar-SDet", "Mt-KaHyPar-S", "Zoltan", "BiPart"]

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Mt-KaHyPar-SDet", algos=algos)
	fig.savefig(out_dir + 'deterministic.pdf', bbox_inches='tight', pad_inches=0.0)

def print_speedups():
	df = pd.read_csv('scalability.csv')
	fields = ["partitionTime", "preprocessingTime", "coarseningTime", "ipTime", "lpTime"]#, "fmTime"]
	for field in fields:
		print(field)
		speedup_plots.print_speedups(df=df, field=field, seed_aggregator="median", min_sequential_time = 0)
	

def runtime_share(options, out_dir):
	import runtime_share
	fig = plt.figure(figsize=options['figsize'])
	mapper = {	'lpTime':'LP', 'preprocessingTime':'Preprocessing', 
				'coarseningTime':'Coarsening', 'ipTime':'Initial'
				}
	df = commons.read_file('speed_deterministic.csv')
	df.rename(columns=mapper, inplace=True)
	fields = list(mapper.values())
	df = df[df.seed == 0]
	df = runtime_share.clean(df)
	runtime_share.plot(df, fields=fields, sort_field="Coarsening", fig=fig, tfield='totalPartitionTime')
	fig.axes[0].spines['top'].set_visible(False)
	fig.savefig(out_dir + "runtime_shares.pdf", bbox_inches='tight', pad_inches=0.0)


def run_all(options, out_dir):
	price_of_determinism(options, out_dir)
	parameter_study(options, out_dir)
	bipart_speedup_plots(options, out_dir)
	mt_kahypar_speedup_plots(options, out_dir)
	main(options, out_dir)
	runtime_share(options, out_dir)
	print_speedups()
