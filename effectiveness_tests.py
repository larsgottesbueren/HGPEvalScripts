import commons
import random
import copy
import pandas as pd

max_objective = 2147483647

def create_virtual_instances(idf, algos, num_repetitions=20, instances=None, objective="km1", time_field="totalPartitionTime", random_seed=420, time_limit = 28800):
	df = idf[
				(idf.algorithm.isin(algos)) 
			  & (idf.timeout == 'no') & (idf.failed == 'no')
			  & (idf.imbalance <= idf.epsilon)
			].copy()
	if len(df.algorithm.unique()) > 2:
		raise Exception("not exactly two algorithms selected for effectiveness tests. this can be in the algo list or the data frame")
	
	if instances == None:
		instances = commons.infer_instances_from_dataframe(df)

	seeds = list(df.seed.unique())

	random.seed(random_seed)

	df.set_index(["algorithm","graph","k","epsilon","seed"], inplace=True)
	
	output = []

	for g,k,eps in instances:
		runs = {a : [] for a in algos}
		for a in algos:
			for seed in seeds:
				key = (a,g,k,eps,seed)
				if key in df.index:
					row = df.loc[key]
					runs[a].append((row[objective], row[time_field]))

		a1 = algos[0]
		a2 = algos[1]
		if not runs[a1] and not runs[a2]:
			print("neither algo has entries on", g, k)
			continue

		for rep in range(num_repetitions):
			for a in algos:
				random.shuffle(runs[a])

			if not runs[a1]:
				fast_algo, slow_algo = a2, a1
				slow_time, slow_quality, slow_tl = time_limit + 1, max_objective, "yes"
			elif not runs[a2]:
				fast_algo, slow_algo = a1, a2
				slow_time, slow_quality, slow_tl = time_limit + 1, max_objective, "yes"
			else:
				fast_algo = a1 if runs[a1][0][1] <= runs[a2][0][1] else a2		# take the first run [0], then the time [1]
				slow_algo = a2 if runs[a1][0][1] <= runs[a2][0][1] else a1
				(slow_quality, slow_time), slow_tl = runs[slow_algo][0], "no"

			r = runs[fast_algo]
			fast_time = r[0][1]
			cnt = 1
			n = len(runs[fast_algo])

			while fast_time <= slow_time and cnt < n:		# keep sampling until the fast algo takes as long as the slow one
				fast_time += r[cnt][1]
				cnt += 1

			if fast_time > slow_time:						# reject last sample with a certain probability to achieve expected equal time
				last_time = r[cnt-1][1]
				accept_prob = (slow_time - fast_time + last_time) / last_time
				if random.uniform(0.0, 1.0) > accept_prob:	# < accept_prob --> accept. > accept_prob --> reject
					cnt -= 1 	# reject

			best = 0
			for j in range(1,cnt):
				if r[best][0] > r[j][0]:
					best = j

			output.append((fast_algo, g + "_virt_" + str(rep), k, eps, 0.0, 0, "no", "no", r[best][0], r[best][1]))
			output.append((slow_algo, g + "_virt_" + str(rep), k, eps, 0.0, 0, slow_tl, "no", slow_quality, slow_time))




	columns = ["algorithm","graph","k","epsilon","imbalance","seed","timeout","failed", objective, "totalPartitionTime"]
	output_df = pd.DataFrame(data=output, columns=columns)
	return output_df
