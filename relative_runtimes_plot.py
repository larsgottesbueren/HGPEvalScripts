import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
import commons

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('pgf', rcfonts=False)
plt.rc('font', size=13)
plt.rcParams['text.latex.preamble'] = R'\usepackage{pifont}'
plt.rcParams['pgf.preamble'] = R'\usepackage{pifont}'

time_limit = 28800  # adapt later

def plot(plotname, df, baseline_algorithm, colors, field='totalPartitionTime'):
    n_instances = len(commons.infer_instances_from_dataframe(df))
    algos = commons.infer_algorithms_from_dataframe(df)

    keys = ["graph", "k", "epsilon"]
    df = df.groupby(keys + ["algorithm"]).mean()[field].reset_index()   # arithmetic mean over seeds per instance
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

    w = 5.53248027778
    h = 3.406
    fig, ax = plt.subplots()    #figsize=(7,3.5)) # adapt to paper margins

    print(baseline_algorithm)
    print(algos)
    algos.remove(baseline_algorithm)
    
    for algo in algos:
        algo_df = df[df.algorithm == algo]
        n_instances_solved_by_algo = len(algo_df)
        sb.lineplot(y=algo_df["relative_time"], x=range(n_instances_solved_by_algo), label=algo, color=colors[algo], ax=ax)


    ax.set_yscale('log', base=2)
    ax.set_ylabel('relative slowdown to ' + baseline_algorithm)
    ax.set_xlabel('instances')
    ax.grid(axis='y', which='both', ls='dashed')


    step = 500
    custom_ticks = list(range(0, n_instances, step))
    if n_instances % step != 0:
        custom_ticks.append(n_instances)

    plt.xticks(custom_ticks)
    
    fig.savefig(plotname + "_base_" + baseline_algorithm + "_relative_slowdown.pdf", bbox_inches='tight')


def thread_preprocessing(df):
    # remove threads column and put the number of threads in the algorithm name
    return 0


if __name__ == '__main__':
    import sys
    plotname = sys.argv[1]
    baseline_algorithm = sys.argv[2]
    files = sys.argv[3:]
    df = pd.concat(map(commons.read_and_convert, files))
    algos = commons.infer_algorithms_from_dataframe(df)
    plot(plotname, df, baseline_algorithm, colors= commons.construct_new_color_mapping(algos))
