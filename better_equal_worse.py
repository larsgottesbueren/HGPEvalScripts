import sys
import pandas as pd

max_flow = 2147483647

df = pd.concat(map(pd.read_csv, [sys.argv[1], sys.argv[2]]))
df = df[(df.timeout == "no")]

instances = set()
solutions = dict()
group = df.groupby(["graph", "k", "epsilon", "algorithm"]).mean()["km1"]
for key, val in group.items():
		G,k,eps,algo = key
		instance = (G,k,eps)
		instances.add(instance)
		solutions[key] = val
		if val == max_flow:
			print("Instance", instance, "Algo", algo, "only max flow")

algos = df.algorithm.unique()
algo = algos[0]
other_algo = algos[1]


better = 0
equal = 0
worse = 0
for instance in instances:
	G,k,eps = instance
	mk = (G,k,eps,algo)
	ok = (G,k,eps,other_algo)
	if mk in solutions:
		if not ok in solutions or solutions[mk] < solutions[ok]:
			better += 1
		elif solutions[mk] == solutions[ok]:
			equal += 1
		else:
			worse += 1
	else:
		if ok in solutions:
			worse += 1
		else:
			print("if this occurs, something is wrong")
			equal += 1

print(algo, "was better on", better, "and equal on", equal, "and worse on", worse, "instances than", other_algo)
print("out of", len(instances), "instances")
