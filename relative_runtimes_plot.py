import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import commons
import scipy.stats

time_limit = 28800  # adapt later

def compute_relative_runtimes(df, algos, baseline_algorithm, field, seed_aggregator):
    keys =["graph", "k", "epsilon"]

    if seed_aggregator == "mean":
        df = df.groupby(keys + ["algorithm"]).mean()[field].reset_index()   # arithmetic mean over seeds per instance
    elif seed_aggregator == "median":
        df = df.groupby(keys + ["algorithm"]).median()[field].reset_index()

    for algo in algos:
        print(algo, "gmean time", scipy.stats.gmean(df[df.algorithm == algo][field]))

    baseline_df = df[df.algorithm == baseline_algorithm].copy()
    baseline_df.set_index(keys, inplace=True)
    df = df[df.algorithm != baseline_algorithm].copy()

    def relative_time(row):
        baseline_key = tuple([row[key] for key in keys])
        if baseline_key in baseline_df.index:
            return row[field] / baseline_df.loc[baseline_key][field]
        else:
            return row[field] / time_limit
    
    df["relative_time"] = df.apply(relative_time, axis='columns')
    df.sort_values(by=["relative_time"], inplace=True)
    return df

def get_stats(df, baseline_algorithm, algos, field="totalPartitionTime"):
    df = compute_relative_runtimes(df, algos, baseline_algorithm, field)
    my_algos = algos.copy()
    print(my_algos, baseline_algorithm)
    my_algos.remove(baseline_algorithm)
    for algo in my_algos:
        print(algo)
        print("faster than baseline on", df[df.relative_time < 1.0])
        print("max", df.relative_time.max())
        print("min", df.relative_time.min())

def construct_plot(df, ax, baseline_algorithm, colors, algos=None, ylabel_fontsize=None, seed_aggregator="mean", field='totalPartitionTime'):
    n_instances = len(commons.infer_instances_from_dataframe(df))
    if algos == None:
        algos = commons.infer_algorithms_from_dataframe(df)

    df = compute_relative_runtimes(df, algos, baseline_algorithm, field, seed_aggregator)

    algos.remove(baseline_algorithm)
    for algo in algos:
        algo_df = df[df.algorithm == algo]
        n_instances_solved_by_algo = len(algo_df)
        sb.lineplot(y=algo_df["relative_time"], x=range(n_instances_solved_by_algo), label=algo, color=colors[algo], ax=ax)

    if ylabel_fontsize == None:
        ax.set_ylabel('slowdown rel. to ' + baseline_algorithm)
    elif ylabel_fontsize == "DropAlgo":
        ax.set_ylabel('relative slowdown')
    else:
        ax.set_ylabel('slowdown rel. to ' + baseline_algorithm, fontsize=ylabel_fontsize)
    ax.set_xlabel('instances')
    ax.grid(axis='y', which='both', ls='dashed')
    ax.set_yscale('log')

    step = 500
    if n_instances < 50:
        step = 10
    elif n_instances < 100:
        step = 20
    elif n_instances <= 200:
        step = 50
    elif n_instances <= 800:
        step = 100
    custom_ticks = list(range(0, n_instances, step))
    if n_instances % step != 0:
        custom_ticks.append(n_instances)

    plt.xticks(custom_ticks)


def plot(plotname, df, baseline_algorithm, colors, algos=None, figsize=None, ylabel_fontsize=None, seed_aggregator="mean", field='totalPartitionTime', legend=None):
    fig, ax = plt.subplots(figsize=figsize)
    construct_plot(df=df, ax=ax, baseline_algorithm=baseline_algorithm, colors=colors, algos=algos, ylabel_fontsize=ylabel_fontsize, seed_aggregator=seed_aggregator, field=field)
    if legend == "NoLegend":
        ax.legend().set_visible(False)
    fig.savefig(plotname + "_relative_slowdown.pdf", bbox_inches='tight')
    #fig.savefig(plotname + ".pdf", bbox_inches='tight')

if __name__ == '__main__':
    import sys
    plotname = sys.argv[1]
    baseline_algorithm = sys.argv[2]
    files = sys.argv[3:]
    df = commons.read_files(files)
    algos = commons.infer_algorithms_from_dataframe(df)
    plot(plotname + "_total", df, baseline_algorithm, colors= commons.construct_new_color_mapping(algos), field="totalPartitionTime")
    #plot(plotname + "_lp", df, baseline_algorithm, colors= commons.construct_new_color_mapping(algos), field="lpTime")
    #plot(plotname + "_coarsening", df, baseline_algorithm, colors= commons.construct_new_color_mapping(algos), field="coarseningTime")
    #plot(plotname + "_initial", df, baseline_algorithm, colors= commons.construct_new_color_mapping(algos), field="ipTime")
    #plot(plotname + "_preprocessing", df, baseline_algorithm, colors= commons.construct_new_color_mapping(algos), field="preprocessingTime")
