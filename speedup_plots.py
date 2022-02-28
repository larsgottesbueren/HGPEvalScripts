import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.gridspec as grd
import copy
import itertools
import scipy.stats

import commons

def compute_speedups(df, field, seed_aggregator):
    lookup_keys = ["graph"]
    if "k" in df.columns:
        lookup_keys.append("k")
    if "seed" in df.columns:
        if seed_aggregator == None:
            lookup_keys.append("seed")
        elif seed_aggregator == "median":
            df = df.groupby(lookup_keys + ["threads"]).median()[field].reset_index()
        elif seed_aggregator == "mean":
            df = df.groupby(lookup_keys + ["threads"]).mean()[field].reset_index()

    seq_df = df[df["threads"] == 1].copy()
    seq_df.set_index(lookup_keys, inplace=True)

    def seq_runtime(row):
        key = tuple([row[key] for key in lookup_keys])
        return seq_df.loc[key][field]
        
    speedups = df[df["threads"] != 1].copy()
    speedups['sequential_time'] = speedups.apply(seq_runtime, axis=1)
    speedups['speedup'] = speedups.apply(lambda row : row['sequential_time'] / row[field] , axis=1)
    return speedups


def compute_windowed_gmeans(speedups, window_size):
    speedups.sort_values(by=["threads","sequential_time"], inplace=True)
    # take rolling window of size 30, min window size is 1, start new calculation for each thread-count
    # then only take the speedup column, and apply geometric mean to each window
    speedups["rolling_gmean_speedup"] = speedups.groupby('threads')["speedup"].transform(lambda x : x.rolling(window=window_size, min_periods=5).apply(scipy.stats.gmean))

def print_speedups(df, field, min_sequential_time = 0, seed_aggregator="median"):
    speedups = compute_speedups(df, field, seed_aggregator)
    thread_list = list(df.threads.unique())
    if 1 in thread_list:
        thread_list.remove(1)

    # print(thread_list)
    thread_list = [64]
    print("geometric mean speedups")
    for th in thread_list:
        thdf = speedups[(speedups.threads == th) & (speedups.sequential_time >= min_sequential_time)]
        print(scipy.stats.gmean(thdf.speedup))
    #print("max speedups")
    for th in thread_list:
        thdf = speedups[speedups.threads == th]
    #    print(thdf.speedup.max())
    #print("> 64")
    #print(speedups[speedups.speedup > 64])

def scalability_plot(df, field, ax, thread_colors=None, 
                     show_scatter=True, show_rolling_gmean=True, 
                     display_labels=True, display_legend=True,
                     xscale=None, yscale=None, alpha=0.2,
                     seed_aggregator=None, window_size=50):
    
    
    speedups = compute_speedups(df, field, seed_aggregator)
    compute_windowed_gmeans(speedups, window_size)
    
    if thread_colors == None:
        thread_list = list(df.threads.unique())
        if 1 in thread_list:
            thread_list.remove(1)
        print(thread_list)
        thread_colors = commons.construct_new_color_mapping(thread_list)

    if show_scatter:
        sb.scatterplot(ax=ax, x="sequential_time", y="speedup", hue="threads", data=speedups[speedups.threads < 128], palette=thread_colors,legend=display_legend, edgecolor="gray", alpha=alpha, s=8)
    if show_rolling_gmean:
        for th, co in thread_colors.items():
            th_df = speedups[speedups.threads == th]
            ax.plot(th_df['sequential_time'], th_df['rolling_gmean_speedup'], color=co, linewidth=1.8, label=th)
    
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
