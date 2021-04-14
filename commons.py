import pandas as pd
import seaborn as sb
import itertools

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
	if "threads" in df.columns and len(df["threads"].unique()) > 1:
		df["algorithm"] = df["algorithm"] + "-" + df["threads"].astype(str)

def add_column_if_missing(df, column, value):
	if not column in df.columns:
		df[column] = [value for i in range(len(df))]

def conversion(df):
	add_column_if_missing(df, 'failed', 'no')
	add_column_if_missing(df, 'timeout', 'no')
	df.rename(columns={'partitionTime' : 'totalPartitionTime'}, inplace=True)
	add_threads_to_algorithm_name(df)

def read_and_convert(file):
	df = pd.read_csv(file)
	conversion(df)
	return df

def read_files(files):
	return pd.concat(map(read_and_convert, files))
