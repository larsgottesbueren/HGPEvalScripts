import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt

options = {
	'width' 	: 5.795, 		# inches. full paper width (two columns)
	'default_aspect_ratio' : 2.9 #2.65
}
options['height'] = options['width'] / options['default_aspect_ratio']
options['figsize'] = (options['width'], options['height'])
options['half_figsize'] = (options['width'] / 2, options['height'])

plt.rc('text', usetex=True)
plt.rc('font', family='libertine')
plt.rc('font', size=7)
plt.rcParams['text.latex.preamble'] = R'\usepackage{pifont}'

# out_dir = "/home/gottesbueren/diss/plots"
out_dir = os.getcwd()

# print_speedups()
this_dir = os.getcwd();

import components.plots
os.chdir('components')
#components.plots.run_all(options, out_dir + "/components/")
os.chdir(this_dir)

import deterministic.plots
os.chdir('deterministic')
#deterministic.plots.run_all(options, out_dir + "/deterministic/")
os.chdir(this_dir)

import nlevel.plots
os.chdir('nlevel')
#nlevel.plots.run_all(options, out_dir + "/nlevel/")
os.chdir(this_dir)

import default.plots
os.chdir('default')
default.plots.run_all(options, out_dir + "/default/")
os.chdir(this_dir)

import flows.plots
os.chdir('flows')
# flows.plots.run_all(options, out_dir + "/flows/")
os.chdir(this_dir)

import parameter_study.plots
os.chdir('parameter_study')
#parameter_study.plots.run_all(options, out_dir + "/parameter_study/")
os.chdir(this_dir)
