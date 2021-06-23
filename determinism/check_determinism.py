import pandas as pd
import sys


objectives = ["km1", "cut", "initial_km1"]
instance_identifiers = ["graph","k","seed"]


def non_unique_values(entry):
	return any([len(set(entry[field])) != 1 for field in objectives])

def unique_values(entry):
	return all([len(set(entry[field])) == 1 for field in objectives])	

df = pd.read_csv(sys.argv[1])
groups = df.groupby(instance_identifiers)

# usable_results = groups.filter(unique_values)
# usable_results.to_csv('scalability.csv', index=False)

configs_with_differing_objectives = groups.filter(non_unique_values)
configs_with_differing_objectives = configs_with_differing_objectives.groupby(instance_identifiers).size().reset_index().rename(columns={0:'count'})

if (len(configs_with_differing_objectives) == 0):
	print("all runs on the same instance produced the same objective")
else:
	print("the following instances had runs with differing objectives")
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_colwidth', None)
	print(configs_with_differing_objectives)
