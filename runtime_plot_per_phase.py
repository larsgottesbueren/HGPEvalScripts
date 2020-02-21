import scales
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats.mstats
import color_scheme

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=14)

def plot(df):
	fig, ax = plt.subplots(figsize=(6.4, 3))
	ax.set_axisbelow(True)
	ax.grid(b=True, axis='y', which='major', ls='dashed')

	boxprops = dict(fill=True, edgecolor='black', linewidth=0.9)
	rem_props = dict(linestyle='-', linewidth=0.9)

	box_plot = sb.boxplot(y="time", x="algorithm", hue="phase", data=df,
	                      order=["KaHyPar-MF", "KaHyPar-HFC*", "KaHyPar-HFC"],
	                      hue_order=["Preprocessing", "Coarsening", "Initial partition", "Local search", "Flow refinement"],
	                      width=0.9, 
	                      showfliers=True, 
	                      fliersize=2,
	                      boxprops=boxprops, whiskerprops=rem_props, medianprops=rem_props,
	                      #flierprops = { "rasterize" : True },
	                      capprops=rem_props
	                      )
	# TODO rasterize fliers
	plt.legend(bbox_to_anchor=(1.03, 0.5), loc='center left', borderaxespad=0.)
	ax.set_yscale('symlog', linthreshy=1e-2)
	
	ax.set_ylabel(R'Time per pin [$\mu$s]')

	ax.xaxis.label.set_visible(False)
	fig.savefig("runtime_plot_per_phase.pdf", bbox_inches="tight", pad_inches=0.0)
	#fig.savefig("runtime_plot_per_phase.pgf", bbox_inches="tight", pad_inches=0.0)

df = pd.read_csv("runtime_per_phase.csv")
plot(df)
