import pandas as pd
from collections import *
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import copy

import scales

#import tikzplotlib

max_flow = 2147483647
time_limit = 28800
timeout_ratio = 2000
no_algo_solved_ratio = timeout_ratio
imbalanced_ratio = 16000
do_not_plot_ratio = max_flow


performance_profile_fraction_scaling = 1


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('pgf', rcfonts=False)
plt.rc('font', size=13)

plt.rcParams['text.latex.preamble'] = R'\usepackage{pifont}'
plt.rcParams['pgf.preamble'] = R'\usepackage{pifont}'

# Returns for each algorithm the tau's at which the number of instances solved within tau*best jumps
def performance_profiles(algos, instances, input_df, plotname, instance_grouper = ["graph", "k", "epsilon"], objective="km1"):
	df = input_df[input_df.algorithm.isin(algos)].copy()

	in_time = df[(df.timeout == 'no') & (df.totalPartitionTime < time_limit)]
	balanced = in_time[(in_time.imbalance <= in_time.epsilon) & (in_time.failed == "no")].copy()
	in_time_index = in_time.groupby(instance_grouper + ["algorithm"]).mean()
	balanced_index = balanced.groupby(instance_grouper + ["algorithm"]).mean()

	best_per_instance = balanced_index[objective].groupby(level=instance_grouper).min()

	ratios = defaultdict(list)
	n = len(instances)
	unsolved = set()
	solved = defaultdict(list)

	for instance in instances:
		G,k,eps = instance

		if not instance in best_per_instance:
			unsolved.add(instance) 
			print("no algo solved", instance)
			for algo in algos:
				ratios[algo].append(no_algo_solved_ratio)
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
						if r >= timeout_ratio:
							#print("Warning. Performance ratio greater than timeout ratio. Will not be in plot!", r, algo, G, k, eps)
							r = do_not_plot_ratio
						if r < 1:
							print("r < 1", G, k, algo, obj, best, r)
					else:
						if obj == 0:
							r = 1
						else:
							r = obj + 1
							if r >= timeout_ratio:
								#print("Warning. Performance ratio greater than timeout ratio. Best = 0. Will not be in plot!", r, algo, G, k, eps)
								r = do_not_plot_ratio
			else:
				r = timeout_ratio


				#print("only timeouts", algo, G, k, eps)
			if r != do_not_plot_ratio:
				ratios[algo].append(r)

	print(len(unsolved), "instances unsolved")
	max_ratio = max( max(ratios[algo]) for algo in algos )
	print("max ratio = ", max_ratio)
	print("Plot name", plotname)
	for algo in algos:
		print(algo, "solved", len(solved[algo]), "instances")


	last_drawn_ratio = min(imbalanced_ratio, max_ratio)
	
	output = []
	for algo in algos:
		ratios[algo].sort()
		last_ratio = 0.95
		for i, r in enumerate(ratios[algo]):
			if last_ratio != r:
				if last_ratio > 0.95:
					output.append((algo, i * performance_profile_fraction_scaling / n, last_ratio))
					output.append((algo, i * performance_profile_fraction_scaling / n, r))		# first occurence of r
				last_ratio = r

		output.append((algo, len(ratios[algo]) * performance_profile_fraction_scaling / n, last_ratio))
		output.append((algo, len(ratios[algo]) * performance_profile_fraction_scaling / n, last_drawn_ratio))	# draw the step function to the rightmost x-value

		# interesting_ratios = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2]
		# for ir in interesting_ratios:
		# 	total = next((i for i, x in enumerate(ratios[algo]) if x > ir), n)
		# 	print(algo, ir, total, total / n)
		
		# for cat, ir in zip(["timeout", "imbalanced"], [timeout_ratio, imbalanced_ratio]):
		# 	c = ratios[algo].count(ir)
		# 	print(algo, cat, c, c/n)

	output_df = pd.DataFrame(data=output, columns=["algorithm", "fraction", "ratio"])
	output_df.to_csv(plotname + "_performance_profile.csv", index=False)
	return output_df


def plot(plotname, colors, display_legend="Yes", title=None, grid=True, width_scale=1.0):
	df = pd.read_csv(plotname + "_performance_profile.csv")
	algos = df.algorithm.unique()
	
	max_ratio = df.ratio.max()
	last_drawn_ratio = min(imbalanced_ratio, max_ratio)
	bb = [0.995, 1.1, 2, last_drawn_ratio + 800]
	ymax = 1.01 * performance_profile_fraction_scaling

	nbuckets = len(bb) - 1
	for i in range(len(bb) -1):
		if bb[i] > max_ratio:
			nbuckets = i
			break
	print("nbuckets=", nbuckets, "max_ratio=", max_ratio)
	if nbuckets < 1:
		print("Warning nbuckets = 0. Aborting")
		return

	linewidth = 5.53248027778


	#w = default_plotwidth * width_scale
	golden_ratio = 1.61803398875
	#h = default_plotwidth / golden_ratio

	w = linewidth * width_scale
	h = 3.406

	fig = plt.figure(figsize=(w,h))	# this is full line width (+ golden ratio). for pgf we do half, but might have to set ticks manually; also font size?
	gs = grd.GridSpec(nrows=1, ncols=nbuckets, wspace=0.0, hspace=0.0, width_ratios=[1.0/ nbuckets for i in range(nbuckets)])
	axes = [plt.subplot(gs[i]) for i in range(nbuckets)]
	for ax in axes[1:]:
		ax.set_yticklabels(ax.get_yticklabels(), visible=False)
		ax.yaxis.set_ticks_position('none')
	

	for algo in algos:
		algo_df = df[df.algorithm == algo]
		for ax in axes:
			ax.plot(algo_df["ratio"], algo_df["fraction"], color=colors[algo], lw=2.2, label=algo)
			
	if display_legend == "Yes":
		if len(algos) < 5:
			plt.legend(ncol=1, fancybox=True, framealpha=1.0, loc='lower right')
		else:
			if width_scale > 1.0:
				ncols = 5
			else:
				ncols = 2
			axes[0].legend(ncol=ncols, fancybox=False, frameon=False, loc='upper left', bbox_to_anchor=(-0.12,-0.15))
	elif display_legend != "No":
		fig_leg = plt.figure(figsize=(w,h))
		fig_leg.legend(*axes[0].get_legend_handles_labels(), loc='center', ncol=2, fancybox=True, framealpha=1.0)
		fig_leg.savefig(display_legend + ".pdf", bbox_inches="tight")
		fig_leg.savefig(display_legend + ".pgf", bbox_inches="tight")
		plt.close(fig_leg)

	for i in range(nbuckets):
		axes[i].set_xlim(bb[i], bb[i+1])
		axes[i].set_ylim(0, ymax)
		if grid:
			axes[i].grid(b=True, axis='both', which='major', ls='dashed')

	if nbuckets > 1:
		if width_scale > 1.0:
			x0 = [1, 1.025, 1.05, 1.075, 1.1]
			axes[0].set_xticks(x0)
			axes[0].set_xticklabels(x0)
			x1 = [1.2, 1.4, 1.6, 1.8]
			axes[1].set_xticks(x1)
			axes[1].set_xticklabels(x1)
		else:
			axes[0].set_xticks([1, 1.05, 1.1])
			axes[1].set_xticks([1.5])

	if max_ratio >= timeout_ratio:
		#axes[nbuckets - 1].set_xscale('fifthroot')
		axes[nbuckets - 1].set_xscale('log')

		ticks = [bb[nbuckets-1], 10, 100]
		tick_labels = copy.copy(ticks)
		ticks.append(timeout_ratio)
		tick_labels.append(R'\ding{99}')
		if max_ratio >= imbalanced_ratio:
			ticks.append(imbalanced_ratio)
			tick_labels.append(R'\ding{56}')
		axes[nbuckets - 1].set_xticks(ticks)
		axes[nbuckets - 1].set_xticklabels(tick_labels)

	axes[0].set_ylabel('Fraction of instances')
	axes[nbuckets//2].set_xlabel('Performance ratio')

	if title != None:
		axes[nbuckets-1].text(0.5, 0.2, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='white'))

	#fig.tight_layout()	
	fig.savefig(plotname + "_performance_profile.pdf", bbox_inches="tight", pad_inches=0.0)
	#fig.savefig(plotname + "_performance_profile.pgf", bbox_inches="tight", pad_inches=0.0)
	#tikzplotlib.save(plotname + "_performance_profile.tikz")
	plt.close(fig)

if __name__ == '__main__':
	import sys, commons
	plot_name = sys.argv[1]
	files = sys.argv[2:]
	df = pd.concat(map(pd.read_csv, files))

	commons.conversion(df)
	algos = commons.infer_algorithms_from_dataframe(df)
	instances = commons.infer_instances_from_dataframe(df)
	performance_profiles(algos, instances, df, plot_name)
	plot(plot_name, colors=commons.construct_new_color_mapping(algos))
