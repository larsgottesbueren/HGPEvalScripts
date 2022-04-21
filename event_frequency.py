import scales
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats.mstats
from collections import defaultdict

def plot(df, fig, ax, fields, hue=None, colors=None):

	boxprops = dict(linewidth=0.9, zorder=2, fill=False)
	rem_props = dict(linestyle='-', linewidth=0.9)
	ax.grid(True)
	
	hue_order = None
	if hue != None:
		hue_order = list(sorted(df[hue].unique()))
		assert(colors != None)

	scatter_col = None
	if colors == None:
		base_col = 'seagreen'
		boxprops['edgecolor'] = base_col
		boxprops['facecolor'] = 'honeydew'
		boxprops['fill'] = True
		rem_props['color'] = base_col
		scatter_col = base_col


	box_plot = sb.boxplot(data=df, x='variable', y='value', hue=hue, hue_order=hue_order,
	                      #width=0.4, 
	                      showfliers=False,
	                      boxprops=boxprops,
	                      palette=colors,
	                      whiskerprops=rem_props, medianprops=rem_props, meanprops=rem_props, flierprops=rem_props, capprops=rem_props, 
	                      ax=ax, zorder=2,
	                      order=fields
	                      )

	if True:
		strip_plot = sb.stripplot(data=df, x='variable', y='value', hue=hue, hue_order=hue_order,
		                          jitter=0.3, dodge=True, size=2.5, edgecolor="gray", alpha=0.2, color=scatter_col,
		                          ax=ax, zorder=1,
		                          palette=colors,
		                          order=fields
		                          )

	
	plt.setp(ax.xaxis.get_majorticklabels(), rotation=20, ha="right", rotation_mode="anchor")

	ax.set_ylabel('frequency or (actual/expected gain)')
	ax.set_xlabel('')

	if hue != None:
		num_legend_entries = len(df[hue].unique())
				
		handles, labels = ax.get_legend_handles_labels()
		lgd = ax.legend(handles[0:num_legend_entries], labels[0:num_legend_entries], title=hue)
	

