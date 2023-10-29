import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import commons
import scipy.stats
import math
import copy

def compute_relative_runtimes(input_df, algos, baseline_algorithm, field, seed_aggregator, time_limit):
    #print(input_df[input_df['failed'] == 'yes'])
    #df = input_df[input_df['failed'] == 'no']
    df = input_df
    keys =["graph", "k", "epsilon"]

    def aggregate_func(series):
        f = list(filter(lambda x : x < time_limit, series))
        if len(f) == 0:
            return time_limit + 1
        if seed_aggregator == "mean":
            return np.mean(f)
        elif seed_aggregator == "median":
            return np.median(f)

    df = df.groupby(keys + ["algorithm"])[field].agg(aggregate_func).reset_index()
        
    for algo in algos:
        print(algo, "gmean time", round(scipy.stats.gmean(df[df.algorithm == algo][field]), 2))

    baseline_df = df[df.algorithm == baseline_algorithm].copy()
    baseline_df.set_index(keys, inplace=True)
    df = df[df.algorithm != baseline_algorithm].copy()

    baseline_timeout = -500
    algo_timeout = -1000

    def relative_time(row):
        baseline_key = tuple([row[key] for key in keys])
        if baseline_key in baseline_df.index:
            if row[field] >= time_limit:
                if baseline_df.loc[baseline_key][field] > time_limit:
                    return 1
                else:
                    return algo_timeout
            if baseline_df.loc[baseline_key][field] > time_limit:
                return baseline_timeout
            if row[field] == 0 or baseline_df.loc[baseline_key][field] == 0:
                return 1
            return row[field] / baseline_df.loc[baseline_key][field]
        else:
            print("relative_runtimes_plot: missing entry in baseline_df", row)
            return row[field] / time_limit
    
    
    df["relative_time"] = df.apply(relative_time, axis='columns')

    min_ratio, max_ratio = df[df.relative_time > 0].relative_time.min(), df.relative_time.max()

    base = math.ceil(math.log10(max_ratio))
    if (math.log10(max_ratio).is_integer()):
        base += 1
    algo_timeout_ratio = 10 ** base

    base = math.floor(math.log10(min_ratio))
    if (math.log10(min_ratio).is_integer()):
        base -= 1
    baseline_timeout_ratio = 10 ** base
    # print(min_ratio, baseline_timeout_ratio, max_ratio, algo_timeout_ratio)

    df["relative_time"].replace(to_replace={baseline_timeout : baseline_timeout_ratio, algo_timeout : algo_timeout_ratio}, inplace=True)
    show_algo_timeout = df["relative_time"].max() == algo_timeout_ratio
    show_baseline_timeout = df["relative_time"].min() == baseline_timeout_ratio

    df.sort_values(by=["relative_time"], inplace=True)

    return df, show_algo_timeout, show_baseline_timeout, algo_timeout_ratio, baseline_timeout_ratio

def get_stats(df, baseline_algorithm, algos, field="totalPartitionTime"):
    df = compute_relative_runtimes(df, algos, baseline_algorithm, field)[0]
    my_algos = algos.copy()
    print(my_algos, baseline_algorithm)
    my_algos.remove(baseline_algorithm)
    for algo in my_algos:
        print(algo)
        print("faster than baseline on", df[df.relative_time < 1.0])
        print("max", df.relative_time.max())
        print("min", df.relative_time.min())

def construct_plot(df, ax, baseline_algorithm, colors, algos=None, ylabel_fontsize=None, seed_aggregator="mean", field='totalPartitionTime', time_limit=7200):
    n_instances = len(commons.infer_instances_from_dataframe(df))
    if algos == None:
        algos = commons.infer_algorithms_from_dataframe(df)

    df, show_algo_timeout, show_baseline_timeout, algo_timeout_ratio, baseline_timeout_ratio = compute_relative_runtimes(df, algos, baseline_algorithm, field, seed_aggregator, time_limit)

    algos.remove(baseline_algorithm)
    for algo in algos:
        algo_df = df[df.algorithm == algo]
        n_instances_solved_by_algo = len(algo_df)
        sb.lineplot(y=algo_df["relative_time"], x=range(n_instances_solved_by_algo), label=algo, color=colors[algo], ax=ax, lw=2.2)

    if ylabel_fontsize == None:
        ax.set_ylabel('slowdown rel. to ' + baseline_algorithm)
    elif ylabel_fontsize == "DropAlgo":
        ax.set_ylabel('relative slowdown')
    else:
        ax.set_ylabel('slowdown rel. to ' + baseline_algorithm, fontsize=ylabel_fontsize)
    ax.set_xlabel('instances')
    ax.grid(axis='y', which='both', ls='dashed')
    # ax.set_yscale('log')

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

    # plt.xticks(custom_ticks)
    
    ticks = [10 ** i for i in range(int(math.log10(baseline_timeout_ratio)) + 1, int(math.log10(algo_timeout_ratio)))] # no -1 in the end since it's exclusive!
    tick_labels = copy.copy(ticks)
    if show_baseline_timeout:
        ticks.insert(0, baseline_timeout_ratio)
        tick_labels.insert(0, R'\ding{99}')
    if show_algo_timeout:
        ticks.append(algo_timeout_ratio)
        tick_labels.append(R'\ding{99}')
    #ax.set_yticks(ticks)
    #ax.set_yticklabels(tick_labels)
    

def plot(plotname, df, baseline_algorithm, colors, algos=None, figsize=None, ylabel_fontsize=None, seed_aggregator="mean", field='totalPartitionTime', time_limit=7200):
    fig, ax = plt.subplots(figsize=figsize)
    construct_plot(df=df, ax=ax, baseline_algorithm=baseline_algorithm, colors=colors, algos=algos, ylabel_fontsize=ylabel_fontsize, seed_aggregator=seed_aggregator, field=field)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False, ncol=2)
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
