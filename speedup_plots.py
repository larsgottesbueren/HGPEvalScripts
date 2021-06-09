import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.gridspec as grd
import copy
import itertools
import scipy.stats


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('pgf', rcfonts=False)
plt.rc('font', size=13)

plt.rcParams['text.latex.preamble'] = R'\usepackage{pifont}'
plt.rcParams['pgf.preamble'] = R'\usepackage{pifont}'


lookup_keys = ["graph", "k"]

def compute_speedups(df, field):
    seq_df = df[df["threads"] == 1].copy()
    seq_df.set_index(lookup_keys, inplace=True)

    def seq_runtime(row):
        key = tuple([row[key] for key in lookup_keys])
        return seq_df.loc[key][field]
        
    speedups = df[df["threads"] != 1].copy()
    speedups['sequential_time'] = speedups.apply(seq_runtime, axis=1)
    speedups['speedup'] = speedups.apply(lambda row : row['sequential_time'] / row[field] , axis=1)
    return speedups


def compute_windowed_gmeans(speedups):
    speedups.sort_values(by=["threads","sequential_time"], inplace=True)
    # take rolling window of size 30, min window size is 1, start new calculation for each thread-count
    # then only take the speedup column, and apply geometric mean to each window
    speedups["rolling_gmean_speedup"] = speedups.rolling(window=30, min_periods=1, on="threads")["speedup"].apply(scipy.stats.gmean)

def scalability_plot(df, algorithm, field, ax, show_scatter=True, show_rolling_gmean=True):
    speedups = compute_speedups(df[df.algorithm == algorithm], field)
    compute_windowed_gmeans(speedups)
    
    color_palette = sb.color_palette()
    thread_colors = dict(zip(speedups.threads.unique(), color_palette))
    if show_scatter:
        sb.scatterplot(ax=ax, x="sequential_time", y="speedup", hue="threads", data=speedups, palette=thread_colors,legend=True, edgecolor="gray", alpha=0.8, s=20)
    if show_rolling_gmean:
        sb.lineplot(ax=ax, x='sequential_time', y='rolling_gmean_speedup', data=speedups, hue='threads', palette=thread_colors, linewidth=2.4, legend=(not show_scatter))
    ax.grid()
    ax.set(xscale="log")
    ax.set_yscale('log', base=2)
    ax.set_ylabel('(rolling gmean) speedup')    
    ax.set_xlabel('sequential time for ' + field + ' [s]')


if __name__ == "__main__":
    fields = ["partitionTime", "preprocessingTime", "coarseningTime", "ipTime", "lpTime"]

    df = pd.read_csv('bipart-mt-bench.csv')
    time_limit = 7200
    timeout_configs = df[df.totalPartitionTime > 7200]
    print("timeout configs", timeout_configs)
    df = df[(df.graph != 'kmer_P1a.mtx.hgr') | (df.k != 64)]
    fig, ax = plt.subplots()
    scalability_plot(df, "BiPart", "totalPartitionTime", ax, show_scatter=False)
    ax.set_xlabel("sequential time for BiPart [s]")
    fig.suptitle('BiPart Speedups')
    plt.savefig("bipart_speedups.pdf")



