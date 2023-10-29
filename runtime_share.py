import scales
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats.mstats
import commons


def aggregate(df, seed_aggregator="mean"):
	keys = ["graph", "k", "epsilon"]
	if seed_aggregator == "median":
		return df.groupby(keys).median().reset_index()
	elif seed_aggregator == "mean":
		return df.groupby(keys + ["threads"]).mean().reset_index()
	return df

def reorder(df, sort_field):
	df.sort_values(by=[sort_field + "_fraction"], inplace=True)

def clean(df):
	df = df[(df.timeout == 'no') & (df.failed == 'no')]
	return df

def compute_fractions(df, fields, tfield):
	for f in fields:
		df[f + "_fraction"] = df[f] / df[tfield]

def plot(df, fields, sort_field, fig, tfield="totalPartitionTime"):
	if tfield == None:
		tfield = 'total_420'
		df[tfield] = [sum(x) for x in zip(*[df[f] for f in fields])]
	compute_fractions(df, fields, tfield)
	reorder(df, sort_field=sort_field)
	totals = [sum(x) for x in zip(*[df[f] for f in fields])]
	num_instances = len(df)
	x_values = [i for i in range(1, num_instances + 1)]
	colors = commons.construct_new_color_mapping(fields + ["other"])
	prev = [0.0 for x in range(num_instances)]
	for f in fields:
		fractions = [i/j for i,j in zip(df[f], df[tfield])]
		sb.barplot(x=x_values, y=fractions, bottom=prev, label=f, color=colors[f])
		prev = [p + f for p,f in zip(prev, fractions)]

	others = [(at - t)/at for t,at in zip(totals, df[tfield])]
	if any(x > 0 for x in others):
		sb.barplot(x=x_values, y=others, bottom=prev, label="other", color=colors["other"])
	#plt.legend()
	plt.legend(loc='lower right')

	fig.axes[0].set_xlabel('instances')
	fig.axes[0].set_ylabel('running time share')

	step_size = 10
	if num_instances > 150:
		step_size = 50
	if num_instances > 800:
		step_size = 100
	plt.xticks(range(0, num_instances, step_size))

	return fig

if __name__ == '__main__':
	import sys
	filename = sys.argv[1]
	fields = ["fmTime", "lpTime", "preprocessingTime", "coarseningTime", "ipTime"]
	df = pd.read_csv(filename)
	df = df[df.seed == 0].copy()
	fig = plt.figure()
	plot(df, fields, "fmTime", fig)
	fig.savefig("runtime_shares.pdf")
