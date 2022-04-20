import performance_profiles
import relative_runtimes_plot
import commons
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import combine_performance_profile_and_relative_slowdown as cpprs

def coarsening_clustering_variants(options, out_dir):
	df = commons.read_and_convert("coarsening_locking_schemes.csv")
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="NoLocking", time_limit=7200)
	fig.savefig(out_dir + "coarsening_clustering_variants.pdf", bbox_inches="tight", pad_inches=0.0)

def attributed_gains(options, out_dir):
	df = commons.read_files(["lp_attributed_gains_impact.csv"])
	fig = plt.figure(figsize=options['half_figsize'])
	performance_profiles.infer_plot(df, fig)
	fig.savefig(out_dir + "attributed_gains_lp.pdf", bbox_inches="tight", pad_inches=0.0)

def gain_tables(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "fm_gains_mt_bench.csv"])
	df = df[df.seed == 0]
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'GainTable'}, inplace=True)
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="GainTable", time_limit=7200)
	fig.savefig(out_dir + "fm_gain_variants.pdf", bbox_inches="tight", pad_inches=0.0)

def lp_and_fm(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "refinement_algos.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'FMandLP'}, inplace=True)
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="FMandLP", time_limit=7200)
	fig.savefig(out_dir + "combine_refinement_algos.pdf", bbox_inches="tight", pad_inches=0.0)

def release_vertices(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "fm_release_nodes.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'Release'}, inplace=True)
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Release", time_limit=7200)
	fig.savefig(out_dir + "release.pdf", bbox_inches="tight", pad_inches=0.0)

def hidden_vs_global_moves(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "fm_apply_moves.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'MoveLocal', 'ApplyToGlobalPartition' : "MoveGlobal"}, inplace=True)
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="MoveLocal", time_limit=7200)
	fig.savefig(out_dir + "hidden_vs_global.pdf", bbox_inches="tight", pad_inches=0.0)

def preprocessing(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "preprocessing.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'WithPreprocessing'}, inplace=True)
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="NoPreprocessing", time_limit=7200)
	fig.savefig(out_dir + "preprocessing.pdf", bbox_inches="tight", pad_inches=0.0)

def run_all(options, out_dir):
	coarsening_clustering_variants(options, out_dir)
	attributed_gains(options, out_dir)
	gain_tables(options, out_dir)
	lp_and_fm(options, out_dir)
	release_vertices(options, out_dir)
	hidden_vs_global_moves(options, out_dir)
	preprocessing(options, out_dir)
