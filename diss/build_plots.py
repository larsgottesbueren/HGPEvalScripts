import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import performance_profiles
import relative_runtimes_plot
import speedup_plots
import commons
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker

import numpy as np
import scipy

def print_speedups(file):
	df = pd.read_csv(file)
	fields = ["preprocessingTime", "fmTime", "coarseningTime", "ipTime", "lpTime"]
	for field in fields:
		print(field)
		speedup_plots.print_speedups(df=df, field=field, seed_aggregator="median", min_sequential_time = 0)

def print_gmean(file, seed_aggregator="mean"):
	df = pd.read_csv(file)
	keys =["graph", "k", "epsilon"]
	time_limit = 7200
	def aggregate_func(series):
		f = list(filter(lambda x : x < time_limit, series))
		if len(f) == 0:
			return time_limit + 1
		if seed_aggregator == "mean":
			return np.mean(f)
		elif seed_aggregator == "median":
			return np.median(f)

	for field in ["preprocessingTime"]:
		x = df.groupby(keys + ["algorithm"])[field].agg(aggregate_func).reset_index()
		print(field, "gmean time", scipy.stats.gmean(x[field]))
        
#print("old run")
#print_gmean('old_default_run.csv')
#print("new run")
#print_gmean('current_run_nok4_32.csv')
#exit()

#print("---- pre deterministic ---------")
#print_speedups("scaling_pre_deterministic.csv")
#print("\n----- default on determinism branch (older state ish) -----")
#print_speedups("investigate_scaling.csv")
#print("\n------ default main ----------")
#print_speedups("default/scalability.csv")
#exit()

options = {
	'width' 	: 5.795, 		# inches. full paper width (two columns)
	'default_aspect_ratio' : 2.65
}
options['height'] = options['width'] / options['default_aspect_ratio']
options['figsize'] = (options['width'], options['height'])
options['half_figsize'] = (options['width'] / 2, options['height'])

plt.rc('text', usetex=True)
plt.rc('font', family='libertine')
plt.rc('font', size=7)
plt.rcParams['text.latex.preamble'] = R'\usepackage{pifont}'

out_dir = "/home/gottesbueren/diss/plots"
out_dir = os.getcwd()




# print_speedups()
this_dir = os.getcwd();

import components.plots
os.chdir('components')
# components.plots.run_all(options, out_dir + "/components/")
os.chdir(this_dir)

import deterministic.plots
os.chdir('deterministic')
# deterministic.plots.run_all(options, out_dir + "/deterministic/")
os.chdir(this_dir)

import nlevel.plots
os.chdir('nlevel')
#nlevel.plots.run_all(options, out_dir + "/nlevel/")
os.chdir(this_dir)

import default.plots
os.chdir('default')
default.plots.run_all(options, out_dir + "/default/")
os.chdir(this_dir)

# import flows.plots
os.chdir('flows')
#flows.plots.run_all(options, out_dir + "/flows/")
os.chdir(this_dir)

import parameter_study.plots
os.chdir('parameter_study')
# parameter_study.plots.run_all(options, out_dir + "/parameter_study/")
os.chdir(this_dir)
