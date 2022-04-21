import pandas as pd
import seaborn as sb
import itertools


def default_color_mapping():
	algos = ["Mt-KaHyPar-D", "Mt-KaHyPar-Q", "hMetis-R", "KaHyPar-CA", "KaHyPar-HFC", "PaToH-D", "PaToH-Q", "Zoltan", "BiPart"]
	mapping = construct_new_color_mapping(algos)

	leftovers = ["Mt-KaHyPar-S", "Mt-KaHyPar-SDet", "Mt-KaHyPar-D-F", "Mt-KaHyPar-Q-F"]
	extra_colors = ["dark purple", "fuchsia", "teal", "squash",]
	for algo, color in zip(leftovers, extra_colors):
		mapping[algo] = sb.xkcd_rgb[color]
	return mapping

def is_algo_in_default_color_mapping(algo):
	return algo in default_color_mapping()

def infer_color_mapping(algos):
	if all([is_algo_in_default_color_mapping(algo) for algo in algos]):
		return default_color_mapping()
	else:
		return construct_new_color_mapping(algos)

def construct_new_color_mapping(algos):
	return dict(zip(algos, sb.color_palette()))

def infer_algorithms_from_dataframe(df):
	return list(df.algorithm.unique())

def infer_instances_from_dataframe(df):
	ks = df.k.unique()
	epss = df.epsilon.unique()
	hgs = df.graph.unique()
	return list(itertools.product(hgs, ks, epss))

def add_threads_to_algorithm_name(df):
	if "threads" in df.columns:
		df["algorithm"] = df["algorithm"] + " " + df["threads"].astype(str)

def add_column_if_missing(df, column, value):
	if not column in df.columns:
		df[column] = [value for i in range(len(df))]

def conversion(df, options={}):
	add_column_if_missing(df, 'failed', 'no')
	add_column_if_missing(df, 'timeout', 'no')
	df.rename(columns={'partitionTime' : 'totalPartitionTime', "num_threads" : "threads"}, inplace=True)
	df["algorithm"].replace(to_replace={'MT-' : 'Mt-', 'Mt-KaHyPar-HD':'Mt-KaHyPar-D-F'}, regex=True, inplace=True)

	if "filter to threads" in options:
		nthreads = int(options["filter to threads"])
		df = df[df.threads == nthreads]

	if "add threads to name" in options:
		add_threads_to_algorithm_name(df)

	return df

def read_and_convert(file, options={}):
	return conversion(pd.read_csv(file), options)
	
def read_file(files, options={}):
	return read_and_convert(files, options)

def read_files(files, options={}):
	return pd.concat(map(read_and_convert, files, [options for i in range(len(files))]))
