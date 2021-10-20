import pandas as pd
from collections import *
import copy
import color_scheme
from benchmark_instances import *
import sys

from scipy.stats import wilcoxon

time_limit = 28800
algos = [sys.argv[1], sys.argv[2]]
instance_grouper = ["graph", "k", "epsilon"]
objective = "km1"

files = sys.argv[3:]

df = pd.concat(map(pd.read_csv, files))

df = df[(df.algorithm.isin(algos)) & (df.timeout == "no") & (df.totalPartitionTime < time_limit) & (df.imbalance <= df.epsilon)].copy()
grp = df.groupby(instance_grouper + ["algorithm"]).mean()

solved = set(instances)
for algo in algos:
	x = set(df[df.algorithm == algo].groupby(instance_grouper).mean().index)
	solved &= x
	print(algo, "solved", len(x), "remaining", len(solved))

print("build cuts")
cuts = defaultdict(list)
for inst in solved:
	for algo in algos:
		G,k,eps = inst
		key = G,k,eps,algo
		cuts[algo].append(grp.loc[key][objective])


T, pval = wilcoxon(x=cuts[algos[0]], y=cuts[algos[1]])
print(T, pval)
