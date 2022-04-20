import commons
import glob
import combine_performance_profile_and_relative_slowdown as cpprs
from pathlib import Path

def run_all(options, out_dir):
	for file in glob.glob("*.csv"):
		stem = Path(file).stem
		print(stem)

		df = commons.read_and_convert(file)
		fig = plt.figure(figsize=options['half_figsize'])
		performance_profiles.infer_plot(fig, df)
		performance_profiles.legend_below(fig, ncol=2)
		fig.savefig(out_dir + stem + ".pdf", bbox_inches="tight", pad_inches=0.0)
