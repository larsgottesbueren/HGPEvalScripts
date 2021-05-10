import pandas as pd
import sys


objectives = ["km1", "cut", "initial_km1"]
instance_identifiers = ["graph","k","seed"]


def non_unique_values(entry):
	return any([len(set(entry[field])) != 1 for field in objectives])


df = pd.read_csv(sys.argv[1])
groups = df.groupby(instance_identifiers)
configs_with_differing_objectives = groups.filter(non_unique_values)

if (len(configs_with_differing_objectives) == 0):
	print("all runs on the same instance produced the same objective")
else:
	print("the following instances had runs with differing objectives")
	print(configs_with_differing_objectives[instance_identifiers])
