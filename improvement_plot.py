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

invalid_objective_value = 2**32-1

def plot(plotname, df, baseline_algorithm, colors, field='km1'):
    n_instances = len(commons.infer_instances_from_dataframe(df))
    algos = commons.infer_algorithms_from_dataframe(df)

    keys = ["graph", "k", "epsilon"]
    df = df.groupby(keys + ["algorithm"]).mean()[field].reset_index()   # arithmetic mean over seeds per instance
    baseline_df = df[df.algorithm == baseline_algorithm].copy()
    baseline_df.set_index(keys, inplace=True)
    df = df[df.algorithm != baseline_algorithm].copy()

    def improvement_ratio(row):
        if row[field] == invalid_objective_value:   # invalid objective value stands for missing entries
            return -1.0
        baseline_key = tuple([row[key] for key in keys])
        if baseline_key in baseline_df.index:
            my_val = row[field]
            baseline_val = baseline_df.loc[baseline_key][field]
            if my_val <= baseline_val:
            	if baseline_val == 0:
            		return 0.0		# equal
            	else:
                	return 1.0 - float(my_val)/float(baseline_val)
            else:
            	return -1.0 + float(baseline_val)/float(my_val)

            return row[field] / baseline_df.loc[baseline_key][field]
        else:
            return 1.0
    
    df["improvement_ratio"] = df.apply(improvement_ratio, axis='columns')
    df.sort_values(by=["improvement_ratio"], inplace=True)

    w = 5.53248027778
    h = 3.406
    fig, ax = plt.subplots()    #figsize=(7,3.5)) # adapt to paper margins

    print(baseline_algorithm)
    print(algos)
    algos.remove(baseline_algorithm)
    
    for algo in algos:
        algo_df = df[df.algorithm == algo]
        n_instances_solved_by_algo = len(algo_df)
        sb.lineplot(y=algo_df["improvement_ratio"], x=range(n_instances_solved_by_algo), label=algo, color=colors[algo], ax=ax)


    ax.set_ylabel('improvement ratio over ' + baseline_algorithm)
    ax.set_xlabel('instances')
    ax.grid(axis='y', which='both', ls='dashed')


    step = 500
    custom_ticks = list(range(0, n_instances, step))
    if n_instances % step != 0:
        custom_ticks.append(n_instances)

    plt.xticks(custom_ticks)
    
    fig.savefig(plotname + "_base_" + baseline_algorithm + "_improvement_plot.pdf", bbox_inches='tight')


def thread_preprocessing(df):
    # remove threads column and put the number of threads in the algorithm name
    return 0

def fill_missing_entries(df):
    #TODO if an instance is completely missing, add max int value to the objective entries, time limit value
    return


if __name__ == '__main__':
    import sys
    plotname = sys.argv[1]
    baseline_algorithm = sys.argv[2]
    files = sys.argv[3:]
    df = pd.concat(map(commons.read_and_convert, files))
    algos = commons.infer_algorithms_from_dataframe(df)
    plot(plotname, df, baseline_algorithm, colors= commons.construct_new_color_mapping(algos))
