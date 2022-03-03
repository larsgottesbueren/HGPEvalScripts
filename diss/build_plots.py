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

options = {
	'width' 	: 5.795, 		# inches. full paper width (two columns)
}

plt.rc('text', usetex=True)
plt.rc('font', family='libertine')
plt.rc('font', size=8)
plt.rcParams['text.latex.preamble'] = R'\usepackage{pifont}'

out_dir = "/home/gottesbueren/diss/plots"
out_dir = os.getcwd()


# print_speedups()
this_dir = os.getcwd();

import deterministic.plots
os.chdir('deterministic')
#deterministic.plots.run_all(options, out_dir + "/deterministic/")
os.chdir(this_dir)

import nlevel.plots
os.chdir('nlevel')
nlevel.plots.run_all(options, out_dir + "/nlevel/")
os.chdir(this_dir)

# import default.plots
os.chdir('default')
#default.plots.run_all(options, out_dir + "/default/")
os.chdir(this_dir)

# import flows.plots
os.chdir('flows')
#default.plots.run_all(options, out_dir + "/flows/")
os.chdir(this_dir)
