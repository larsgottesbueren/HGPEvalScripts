from collections import *
import pandas as pd

hgstats = pd.read_csv('benchmark_set_stats.csv')
nPins = dict()
for row in hgstats.itertuples():
	nPins[row.graph] = row.pins

general_keys = [
			"algorithm",
			"graph",
			"epsilon",
			"k",
			"seed",
			"phase"
           ]

phase_keys = [
			"minHashSparsifierTime",
			"postMinHashSparsifierTime",
			"communityDetectionTime",
			"coarseningTime",
			"initialPartitionTime",
			"uncoarseningRefinementTime",
			"flowTime"
		]

keylist = general_keys + phase_keys
keys = set(keylist)


phases = ["preprocessing", "coarsening", "initial partition", "local search", "flow refinement"]

phase_map = {
		"preprocessing" : ["minHashSparsifierTime", "postMinHashSparsifierTime", "communityDetectionTime"],
		"coarsening" : ["coarseningTime"],
		"initial partition" : ["initialPartitionTime"],
		"local search" : ["uncoarseningRefinementTime"],
		"flow refinement" : ["flowTime"],
}


print(','.join(general_keys + ["time"]))

for file in ["../results/KaHyPar-MF.results", "../results/KaHyPar-HFC.results", "../results/KaHyPar-HFC-mfstyle-flownetworksizes.results"]:
	replace_algo = "KaHyPar-HFC-mfstyle-flownetworksizes" in file
	with open(file, 'r') as f:
		i = 0
		for l in f:
			i += 1
			if "timeout=yes" in l:
				continue
			tokens = l.split()

			vs = dict()
			for t in tokens:
				if "=" in t:
					k,v = t.split('=')[:2]
					if k in keys:
						vs[k] = v
		
			if replace_algo:
				vs["algorithm"] = "KaHyPar-HFC*"

			for phase in phases:
				for x in general_keys[:-1]:
					print(vs[x], end=',')
				print(phase, end=',')
				try:
					t = sum([float( vs[x] ) for x in phase_map[phase]])
				except KeyError as err:
					print("Error in line", i)
					print(err)
					print(vs)


				if phase == "local search":
					t -= float(vs["flowTime"])

				#t = t * 1000 * 1000 / nPins[vs["graph"]]	# microseconds per pin
				print(t)


			



