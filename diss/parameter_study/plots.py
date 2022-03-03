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

from pathlib import Path



def infer(df, figsize, colors=None):
	algos = commons.infer_algorithms_from_dataframe(df)
	if colors == None:
		colors = commons.construct_new_color_mapping(algos)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1")
	fig = performance_profiles.plot(ratios_df, colors=colors, figsize=figsize)
	# performance_profiles.legend_inside(fig, ncol=1)
	return fig

def run_all(options, out_dir):
	width = options["width"] / 2
	aspect_ratio = 1.65
	height = width / aspect_ratio
	figsize=(width, height)

	for file in glob.glob("*.csv"):
		stem = Path(file).stem
		print(stem)

		df = commons.read_and_convert(file)
		fig = infer(df, figsize)
		fig.savefig(out_dir + stem + ".pdf", bbox_inches="tight", pad_inches=0.0)
