import scales
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats.mstats

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=11.5)

def plot(df, colors, algo_order, plot_name=''):
	fig, ax = plt.subplots(figsize=(7,3.5))
	boxprops = dict(fill=False,edgecolor='black', linewidth=0.9, zorder=2)
	rem_props = dict(linestyle='-', linewidth=0.9, color='black')
	
	strip_plot = sb.stripplot(y="totalPartitionTime", x="algorithm", data=df, 
	                          jitter=0.3, size=1.5, edgecolor="gray", alpha=0.4, 
	                          ax=ax, zorder=1, 
	                          palette=colors, order=algo_order
	                          )
	box_plot = sb.boxplot(y="totalPartitionTime", x="algorithm", data=df, 
	                      width=0.5, showfliers=False, 
	                      palette=colors, order=algo_order,
	                      boxprops=boxprops, whiskerprops=rem_props, medianprops=rem_props, meanprops=rem_props, flierprops=rem_props, capprops=rem_props, 
	                      ax=ax, zorder=2)
	
	plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", rotation_mode="anchor")
	#ax.set_yscale('cuberoot')
	#ax.set_yscale('fifthroot')
	ax.set_yscale('log')
	#ax.set(yticks=[0, 1, 20, 75, 250, 750, 2500, 10000, 28800])
	ax.set_ylabel('Time [s]')
	ax.xaxis.label.set_visible(False)
	
	if plot_name == '':
		file_name = 'runtime_plot.pdf'
	else:
		file_name = plot_name + '_runtime_plot.pdf'

	# TODO make function return the figure, then let caller save
	fig.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
	#fig.savefig("runtime_plot.pgf", bbox_inches="tight", pad_inches=0.0)
	#plt.close()


def aggregate_dataframe_by_arithmetic_mean_per_instance(df):
	return df.groupby(["graph", "k", "epsilon", "algorithm"]).mean()["totalPartitionTime"].reset_index(level="algorithm")	


def print_gmean_times(df):
	algos = df.algorithm.unique()
	for algo in algos:
		print("Algo", algo, "gmean time", scipy.stats.mstats.gmean( df[df.algorithm==algo]["totalPartitionTime"] ))

if __name__ == '__main__':
	import sys, commons
	plot_name = sys.argv[1]
	files = sys.argv[2:]

	df = pd.concat(map(pd.read_csv, files))
	commons.conversion(df)
	df = df[df.failed == "no"]

	averaged_runtimes = aggregate_dataframe_by_arithmetic_mean_per_instance(df)

	algos = commons.infer_algorithms_from_dataframe(df)
	plot(averaged_runtimes, colors=commons.construct_new_color_mapping(algos), algo_order=algos, plot_name=plot_name)
