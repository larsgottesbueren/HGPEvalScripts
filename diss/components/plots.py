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
	cpprs.combined_pp_rs(df, fig, baseline="NoLocking", time_limit=7200, runtime_field="coarseningTime")
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
	df = df[df.algorithm != "DeltaGainsInToPQs"]

	relative_runtimes_plot.plot(out_dir + "gain_variants_reltime", df, baseline_algorithm="GainTable", time_limit=7200, 
	                            field="fmTime", figsize=options['half_figsize'],
	                            colors=commons.infer_color_mapping(commons.infer_algorithms_from_dataframe(df)))

	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="GainTable", time_limit=7200, runtime_field="fmTime")
	fig.savefig(out_dir + "fm_gain_variants.pdf", bbox_inches="tight", pad_inches=0.0)

	

def lp_and_fm(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "refinement_algos.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'FMandLP'}, inplace=True)
	df["refinement"] = df["fmTime"] + df["lpTime"]
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="FMandLP", time_limit=7200, runtime_field="refinement")
	fig.savefig(out_dir + "combine_refinement_algos.pdf", bbox_inches="tight", pad_inches=0.0)

def release_vertices(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "fm_release_nodes.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'Release'}, inplace=True)
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="Release", time_limit=7200, runtime_field="fmTime")
	fig.savefig(out_dir + "release.pdf", bbox_inches="tight", pad_inches=0.0)

def hidden_vs_global_moves(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "fm_apply_moves.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'ApplyToLocalPartition'}, inplace=True)
	fig = plt.figure(figsize=options['figsize'])
	cpprs.combined_pp_rs(df, fig, baseline="ApplyToLocalPartition", time_limit=7200, runtime_field="fmTime")
	fig.savefig(out_dir + "hidden_vs_global.pdf", bbox_inches="tight", pad_inches=0.0)

def preprocessing(options, out_dir):
	df = commons.read_files(["mt-kahypar-d-64.csv", "preprocessing.csv"])
	df["algorithm"].replace(to_replace={'Mt-KaHyPar-D' : 'WithPreprocessing'}, inplace=True)
	fig, axes = plt.subplots(ncols=2, figsize=options['figsize'])
	handles, labels = performance_profiles.infer_plot(df, fig, external_subplot=axes[0], display_legend=False)
	performance_profiles.infer_plot(df, fig, external_subplot=axes[1], objective='initial_km1', display_legend=False)
	fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
	axes[0].set_title('final connectivity')
	axes[1].set_title('initial connectivity')
	for ax in axes:
		ax.set_yticks([])
		ax.set_xticks([])
	plt.subplots_adjust(wspace=0.26)
	# cpprs.combined_pp_rs(df, fig, baseline="NoPreprocessing", time_limit=7200)
	fig.savefig(out_dir + "preprocessing.pdf", bbox_inches="tight", pad_inches=0.0)

def fm_deactivate_features(options, out_dir):
	df = commons.read_file('fm_components.csv')

	mapper = {
		'FM' : 'LocExp-NegGain-Loc-NoMQ-Rel-NoAttr',
		'FM-Global-Moves' : 'LocExp-NegGain-Glob-NoMQ-Rel-NoAttr',
		'FM-Global-Moves-No-Release' : 'LocExp-NegGain-Glob-NoMQ-NoRel-NoAttr',
		'FM-Global-Moves-No-Release-No-Localized' : 'Stat-NegGain-Glob-NoMQ-NoRel-NoAttr',
		'FM-Greedy-Localized' : 'LocExp-PosGain-Glob-NoMQ-NoRel-NoAttr',
		'FM-Greedy' : 'Stat-PosGain-Glob-NoMQ-NoRel-NoAttr',
		'Greedy' : 'Stat-PosGain-Glob-NoMQ-NoRel-Attr',
		'FM-Greedy-MQ' : 'Stat-PosGain-Glob-MQ-NoRel-NoAttr',
		'Greedy-MQ' : 'Stat-PosGain-Glob-MQ-NoRel-Attr',
	}
	df["algorithm"].replace(to_replace=mapper, inplace=True)

	algos = [
		'LocExp-NegGain-Loc-NoMQ-Rel-NoAttr',
		'LocExp-NegGain-Glob-NoMQ-Rel-NoAttr',
		'LocExp-NegGain-Glob-NoMQ-NoRel-NoAttr',
		'LocExp-PosGain-Glob-NoMQ-NoRel-NoAttr',

		'Stat-PosGain-Glob-NoMQ-NoRel-NoAttr',		# turn off localization

		'Stat-PosGain-Glob-MQ-NoRel-NoAttr',		# now MQ helps
		'Stat-PosGain-Glob-NoMQ-NoRel-Attr',		# but so does turning on attributed gains
		
		'Stat-PosGain-Glob-MQ-NoRel-Attr',			# combining MQ and attributed gains doesnt do anything
		
		'Stat-NegGain-Glob-NoMQ-NoRel-NoAttr'		# at this point not even negative gains help
	]

	fig = plt.figure(figsize=options['figsize'])
	handles, labels = performance_profiles.infer_plot(df, fig, algos=algos)
	sb.move_legend(fig, ncol=1, loc=(0.475, 0.2), framealpha=1.0)

	fig.savefig(out_dir + 'fm_components.pdf', bbox_inches="tight", pad_inches=0.0)

def run_all(options, out_dir):
	gain_tables(options, out_dir)
	return
	hidden_vs_global_moves(options, out_dir)

	coarsening_clustering_variants(options, out_dir)
	lp_and_fm(options, out_dir)
	preprocessing(options, out_dir)
	fm_deactivate_features(options, out_dir)

	attributed_gains(options, out_dir)
	release_vertices(options, out_dir)
