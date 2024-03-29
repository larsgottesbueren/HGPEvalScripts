import pandas as pd
from collections import *
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import copy
import commons

import scales

import scipy.stats
import math
import numpy as np

from matplotlib.ticker import (AutoMinorLocator, FixedLocator)

#import tikzplotlib

max_objective = 2147483647
#time_limit = 28800
no_algo_solved_ratio = max_objective-1
imbalanced_ratio = max_objective-2
timeout_ratio = max_objective-3

performance_profile_fraction_scaling = 1

# Returns for each algorithm the tau's at which the number of instances solved within tau*best jumps
def performance_profiles(algos, instances, input_df, time_limit=28800, objective="km1"):
	df = input_df[input_df.algorithm.isin(algos)].copy()

	instance_grouper = ["graph", "k", "epsilon"]
	in_time = df[(df.timeout == 'no') & (df.totalPartitionTime < time_limit)]
	balanced = in_time[
		 (in_time.imbalance <= in_time.epsilon + 1e-5) &
		(in_time.failed == "no")
		].copy()
	
	
	in_time_index = in_time.groupby(instance_grouper + ["algorithm"]).mean(numeric_only=True)
	balanced_index = balanced.groupby(instance_grouper + ["algorithm"]).mean(numeric_only=True)
	'''
	in_time_index = in_time.groupby(instance_grouper + ["algorithm"]).min()
	balanced_index = balanced.groupby(instance_grouper + ["algorithm"]).min()
	'''
	best_per_instance = balanced_index[objective].groupby(level=instance_grouper).min()

	ratios = defaultdict(list)
	n = len(instances)
	unsolved = set()
	solved = defaultdict(list)

	for instance in instances:
		G,k,eps = instance

		if not instance in best_per_instance:
			unsolved.add(instance) 
			# print("no algo solved", instance)
			# for algo in algos:
			# 	ratios[algo].append(no_algo_solved_ratio)
			continue

		best = best_per_instance.loc[instance]

		for algo in algos:
			key = G,k,eps,algo
			if key in in_time_index.index:
				if key not in balanced_index.index:
					r = imbalanced_ratio
				else:
					solved[algo].append(instance)
					obj = balanced_index.loc[key][objective]
					if best != 0:
						r = obj / best
						if r < 1:
							print("r < 1", G, k, algo, obj, best, r)
					else:
						if obj == 0:
							r = 1
						else:
							r = obj + 1
			else:
				# print("timeout", G,k,algo)
				r = timeout_ratio
			ratios[algo].append(r)
			if r > 1.05 and "parriba" in algo:
				print(algo, G, k, r)

	max_ratio = max( max(ratios[algo]) for algo in algos )
	print("max ratio = ", max_ratio)
	for algo in algos:
		print(algo, "solved", len(solved[algo]), "instances. gmean performance ratio", scipy.stats.gmean(ratios[algo]), "max ratio=", max(ratios[algo]))

	last_drawn_ratio = min(imbalanced_ratio, max_ratio)
	
	output = []
	for algo in algos:
		ratios[algo].sort()
		num_best = sum(map(lambda ratio : 1 if ratio == 1 else 0, ratios[algo]))
		print(algo, num_best)
		for i, (r_prev, r) in enumerate(zip(ratios[algo], ratios[algo][1:])):
			if r_prev != r:
				output.append((algo, i * performance_profile_fraction_scaling / n, r_prev))		# last occurence of r_prev

		output.append((algo, (len(ratios[algo])-1) * performance_profile_fraction_scaling / n, ratios[algo][-1]))
		# output.append((algo, len(ratios[algo]) * performance_profile_fraction_scaling / n, last_drawn_ratio))	# draw the step function to the rightmost x-value

	output_df = pd.DataFrame(data=output, columns=["algorithm", "fraction", "ratio"])
	return output_df


def plot(df, colors, fig, external_subplot, display_legend=True, title=None, 
         grid=True, width_scale=1.0):
	algos = df.algorithm.unique()
	
	max_ratio = df.ratio.max()
	max_actual_ratio = df[df.ratio < timeout_ratio].ratio.max()
	show_timeout_tick = max_ratio >= timeout_ratio
	show_imbalanced_tick = max_ratio >= imbalanced_ratio

	base = math.ceil(math.log10(max_actual_ratio))
	if (math.log10(max_actual_ratio).is_integer()):
		base += 1
	if base == 1:
		base += 1
	remapped_timeout_ratio = 10 ** base
	remapped_imbalanced_ratio = 10 ** (base + 1)
	print(max_actual_ratio)
	if show_timeout_tick:
		print("do remap. new timeout ratio", remapped_timeout_ratio)
		df.ratio.replace(to_replace={timeout_ratio : remapped_timeout_ratio, imbalanced_ratio : remapped_imbalanced_ratio}, inplace=True)
	

	last_drawn_ratio = df.ratio.max()
	bb = [0.995, 1.1, 2, last_drawn_ratio * 1.05]
	ymax = 1.01 * performance_profile_fraction_scaling

	nbuckets = len(bb) - 1
	for i in range(len(bb) - 1):
		if bb[i] > max_ratio:
			nbuckets = i
			break
	print("nbuckets=", nbuckets, "max_ratio=", max_ratio)
	if nbuckets < 1:
		print("Warning nbuckets = 0. Aborting")
		return

	width_ratios = [1.0 / nbuckets for i in range(nbuckets)]
	if nbuckets == 3:
		width_ratios = [0.4, 0.3, 0.3]
	elif nbuckets == 2:
		width_ratios = [0.65, 0.35]


	# fig = plt.figure(figsize=figsize)
	
	gs = grd.GridSpecFromSubplotSpec(nrows=1, ncols=nbuckets, subplot_spec=external_subplot, wspace=0.0, hspace=0.0, width_ratios=width_ratios)
	axes = [plt.Subplot(fig, gs[i]) for i in range(nbuckets)]

	for algo in algos:
		algo_df = df[df.algorithm == algo]
		for i, ax in enumerate(axes):
			#print(algo_df[(algo_df.ratio >= bb[i]) & (algo_df.ratio <= bb[i+1])])
			ax.plot(
			        algo_df["ratio"], algo_df["fraction"],
			        color=colors[algo], lw=2.2, label=algo,
			        drawstyle='steps-post'
			        )


	ncol = 2
	if width_scale > 1.0:
		ncol = 3
	
	handles, labels = axes[0].get_legend_handles_labels()

	if display_legend:
		fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=ncol)
	
	#if display_legend == "Yes":
	#	axes[-1].legend(fancybox=True, framealpha=1.0, fontsize=legend_fontsize)
		#if len(algos) < 5:
		#	plt.legend(ncol=1, fancybox=True, framealpha=1.0, loc='lower right')
		#else:
		#	if width_scale > 1.0:
		#		ncols = 5
		#	else:
		#		ncols = 2
		#	axes[0].legend(ncol=ncols, fancybox=False, frameon=False, loc='upper left', bbox_to_anchor=(-0.12,-0.15))
	#elif display_legend == "Externalize":
	#	fig_leg = plt.figure()
	#	fig_leg.legend(*axes[0].get_legend_handles_labels(), loc='center', ncol=5, fancybox=True, framealpha=1.0)
	#	fig_leg.savefig(display_legend + ".pdf", bbox_inches="tight")
	#	plt.close(fig_leg)
	#elif display_legend == "wide":
	#	handles, labels = axes[0].get_legend_handles_labels()
	#	fig.legend(handles, labels, loc=(0.3, 0.3), fancybox=True, framealpha=1.0, fontsize=legend_fontsize)
	#elif display_legend == "lower right":
	#	handles, labels = axes[0].get_legend_handles_labels()
	#	fig.legend(handles, labels, loc=(0.55, 0.2), fancybox=True, framealpha=0.8, fontsize=legend_fontsize)
		# axes[1].legend(fancybox=True, framealpha=1.0, fontsize=legend_fontsize)
	#elif display_legend == "below":
	#	ncols = 2
	#	axes[0].legend(ncol=ncols, fancybox=False, frameon=False, loc='upper left', bbox_to_anchor=(-0.12,-0.19))

	for i in range(nbuckets):
		axes[i].set_xlim(bb[i], bb[i+1])
		axes[i].set_ylim(-0.01, ymax)
		if grid:
			axes[i].grid(visible=True, axis='both', which='major', ls='dashed')

	if nbuckets > 1:
		x0 = [1, 1.05, 1.1]
		
		axes[0].xaxis.set_minor_locator(AutoMinorLocator(5))
		axes[0].set_xticks(x0)
		axes[0].set_xticklabels(x0)
		
		axes[1].xaxis.set_minor_locator(AutoMinorLocator(3))
		ax1_xticks = [1.4, 1.7, 2]
		if nbuckets == 3 and 2 in axes[2].get_xticks():
			ax1_xticks.remove(2)
		axes[1].set_xticks(ax1_xticks)
		axes[1].set_xticklabels(ax1_xticks)

	
	if last_drawn_ratio >= 500 or show_timeout_tick:
		axes[nbuckets - 1].set_xscale('log')
		#axes[nbuckets - 1].set_xscale('fifthroot')
	if show_timeout_tick:
		ticks = [10 ** i for i in range(1, base)]
		tick_labels = [r'$10^' + str(i) + r'$' for i in range(1, base)]
		ticks.append(remapped_timeout_ratio)
		tick_labels.append(R'\ding{99}')
		if show_imbalanced_tick:
			ticks.append(remapped_imbalanced_ratio)
			tick_labels.append(R'\ding{56}')

	
		axes[nbuckets - 1].set_xticks(ticks)
		axes[nbuckets - 1].set_xticklabels(tick_labels)
	
	if nbuckets == 3 and 2.5 in axes[2].get_xticks() and max_ratio >= 4:
		axes[2].set_xticks(np.arange(4.0, last_drawn_ratio, step=2.0))


	axes[0].set_ylabel('fraction of instances')
	for ax in axes:			# generate the ticks for each bucket, so that the grid looks good!
		ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
	for ax in axes[1:]:
		ax.set_yticklabels(ax.get_yticklabels(), visible=False)
		ax.yaxis.set_ticks_position('none')
	if nbuckets == 2:
		axes[0].set_xlabel('performance ratio', x=1.05)
	elif nbuckets == 3:
		axes[0].set_xlabel('performance ratio', x=1.15)
	else:
		axes[0].set_xlabel('performance ratio')

	if title != None:
		axes[nbuckets-1].text(0.5, 0.2, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='white'))

	for ax in axes:
		fig.add_subplot(ax)

	return handles, labels
	# return fig

	#fig.tight_layout()	
	# fig.savefig(plotname + "_performance_profile.pdf", bbox_inches="tight", pad_inches=0.0)
	#fig.savefig(plotname + "_performance_profile.pgf", bbox_inches="tight", pad_inches=0.0)
	#tikzplotlib.save(plotname + "_performance_profile.tikz")
	# plt.close(fig)

def infer_plot(df, fig, external_subplot=None, colors=None, algos=None, instances=None, display_legend=True, objective="km1"):
	if algos == None:
		algos = commons.infer_algorithms_from_dataframe(df)
	if instances == None:
		instances = commons.infer_instances_from_dataframe(df)
	if colors == None:
		colors = commons.infer_color_mapping(algos)
	if external_subplot == None:
		outer_grid = grd.GridSpec(nrows=1, ncols=1, wspace=0.0, hspace=0.0, width_ratios=[1.0])
		external_subplot = outer_grid[0]
	ratios_df = performance_profiles(algos, instances, df, objective=objective)
	handles, labels = plot(ratios_df, fig=fig, external_subplot=external_subplot, colors=colors, display_legend=display_legend)
	if display_legend:
		legend_inside(fig, ncol=1)
	return handles, labels

def legend_below(fig, ncol):
	sb.move_legend(fig, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=ncol)

def legend_inside(fig, ncol):
	sb.move_legend(fig, loc=(0.3, 0.3), fancybox=True, framealpha=1.0, frameon=True, ncol=ncol)



if __name__ == '__main__':
	import sys, commons
	plot_name = sys.argv[1]
	files = sys.argv[2:]
	df = commons.read_files(files)
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	ratios_df = performance_profiles(algos, instances, df, plot_name, objective="km1")
	plot(plot_name, ratios_df, colors=commons.construct_new_color_mapping(algos))
