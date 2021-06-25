import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.gridspec as grd
import copy
import itertools
import scipy.stats

import commons

def compute_speedups(df, field):
    lookup_keys = ["graph", "k"]
    if "seed" in df.columns:
        #lookup_keys.append("seed")
        df = df.groupby(["graph","k","threads"]).mean()[field].reset_index()

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
    speedups["rolling_gmean_speedup"] = speedups.groupby('threads')["speedup"].transform(lambda x : x.rolling(window=30, min_periods=1).apply(scipy.stats.gmean))

def scalability_plot(df, algorithm, field, ax, thread_colors=None, 
                     show_scatter=True, show_rolling_gmean=True, 
                     display_labels=True,
                     xscale=None, yscale=None):
    
    
    speedups = compute_speedups(df[df.algorithm == algorithm], field)
    compute_windowed_gmeans(speedups)
    
    if thread_colors == None:
        thread_list = list(df.threads.unique())
        if 1 in thread_list:
            thread_list.remove(1)
        print(thread_list)
        thread_colors = commons.construct_new_color_mapping(thread_list)

    if show_scatter:
        sb.scatterplot(ax=ax, x="sequential_time", y="speedup", hue="threads", data=speedups, palette=thread_colors,legend=True, edgecolor="gray", alpha=0.2, s=12)
    if show_rolling_gmean:
        for th, co in thread_colors.items():
            th_df = speedups[speedups.threads == th]
            ax.plot(th_df['sequential_time'], th_df['rolling_gmean_speedup'], color=co, linewidth=2.2, label=(th if not show_scatter else None))
    
    if display_labels:
        ax.set_ylabel('(rolling gmean) speedup')    
        ax.set_xlabel('sequential time for ' + field + ' [s]')

    ax.grid()
    if xscale != None:
        if xscale == 'log':
            ax.set_xscale('log', base=10)
    if yscale != None:
        if yscale == 'log':
            ax.set_yscale('log', base=2)
