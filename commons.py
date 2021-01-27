def infer_algorithms_from_dataframe(df):
	return df.algorithm.unique()

def infer_instances_from_dataframe(df):
	ks = df.k.unique()
	epss = df.epsilon.unique()
	hgs = df.graph.unique()
	return list(itertools.product(hgs, ks, epss))


def add_column_if_missing(df, column, value):
	if not column in df.columns:
		df[column] = [value for i in range(len(df))]

def conversion(df):
	add_column_if_missing(df, 'failed', 'no')
	add_column_if_missing(df, 'timeout', 'no')
	df.rename(columns={'partitionTime' : 'totalPartitionTime'}, inplace=True)
