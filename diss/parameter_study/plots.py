import commons
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import performance_profiles
import relative_runtimes_plot
import combine_performance_profile_and_relative_slowdown as cpprs

def coarsening(options, out_dir):
	fig, outer_grid = plt.subplots(nrows=1, ncols=2, figsize=(options['width'], options['height']))
	for ax in outer_grid.ravel():	# this is necessary when using subplots instead of gridspec
		ax.set_axis_off()
	
	df = commons.read_file("coarsening_limit.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[0], display_legend=False)
	outer_grid[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)

	df = commons.read_and_convert("shrink_factor.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[1], display_legend=False)
	outer_grid[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)

	plt.subplots_adjust(wspace=0.25)
	fig.savefig(out_dir + "coarsening.pdf", bbox_inches="tight", pad_inches=0.0)

def initial(options, out_dir):
	fig, outer_grid = plt.subplots(nrows=1, ncols=2, figsize=(options['width'], options['height']))
	for ax in outer_grid.ravel():	# this is necessary when using subplots instead of gridspec
		ax.set_axis_off()
	
	df = commons.read_file("adaptive_ip.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[0], display_legend=False)
	outer_grid[0].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=1)

	colors = commons.construct_new_color_mapping(commons.infer_algorithms_from_dataframe(df))
	relative_runtimes_plot.plot(out_dir + "adaptive_ip_", df, colors=colors, baseline_algorithm="Adaptive-Flat-IP", figsize=options['half_figsize'], field='ipTime')

	df = commons.read_and_convert("flat_ip_runs.csv")
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=outer_grid[1], display_legend=False)
	outer_grid[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, ncol=2)

	plt.subplots_adjust(wspace=0.25)
	fig.savefig(out_dir + "initial.pdf", bbox_inches="tight", pad_inches=0.0)


def refinement(options, out_dir):
	df = commons.read_file('fm_seeds.csv')

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="FM-Seeds-1", time_limit=7200, runtime_field='fmTime')
	fig.savefig(out_dir + "fm_seeds.pdf", bbox_inches="tight", pad_inches=0.0)

	return
	fig, outer_grid = plt.subplots(nrows=2, ncols=1, figsize=(options['width'], 2 * options['height']))
	for ax in outer_grid.ravel():	# this is necessary when using subplots instead of gridspec
		ax.set_axis_off()
	
	plt.subplots_adjust(wspace=0, hspace=0.6)
	fig.savefig(out_dir + "refinement.pdf", bbox_inches="tight", pad_inches=0.0)



def run_all(options, out_dir):
	refinement(options, out_dir)
	return
	initial(options, out_dir)
	coarsening(options, out_dir)


	for file in glob.glob("*.csv"):
		stem = Path(file).stem
		print(stem)

		df = commons.read_and_convert(file)
		fig = plt.figure(figsize=options['half_figsize'])
		performance_profiles.infer_plot(df, fig)
		performance_profiles.legend_below(fig, ncol=2)
		fig.savefig(out_dir + stem + ".pdf", bbox_inches="tight", pad_inches=0.0)
