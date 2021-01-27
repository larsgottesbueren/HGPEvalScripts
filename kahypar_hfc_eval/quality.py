from .. import performance_profiles as pp
import pandas as pd

from benchmark_instances import *
import color_scheme


if __name__ == "__main__":

	files = [
		'KaHyPar-HFC-mfstyle.csv', 'KaHyPar-HFC.csv', 'KaHyPar-MF.csv',
		'km1_patoh_q.csv', 'km1_patoh_d.csv',
		'km1_hmetis_r.csv', 'km1_hmetis_k.csv',
		'km1_zoltan_algd.csv',
		'km1_mondriaan.csv', 
		'km1_hype.csv'
		]
	df = pd.concat(map(pd.read_csv, files))

	algos = color_scheme.algos_ordered_by_solution_quality.copy()

	#pp.performance_profiles(['KaHyPar-HFC*', 'KaHyPar-MF'], instances, df, "mf_vs_hfc-mfstyle")
	#pp.performance_profiles(['KaHyPar-HFC', 'KaHyPar-MF'], instances, df, "mf_vs_hfc")
	#pp.performance_profiles(algos, instances, df, "all")

	for x in ["hMetis-K", "KaHyPar-HFC", "KaHyPar-MF", "PaToH-D"]:
		algos.remove(x)
	#pp.performance_profiles(algos, instances, df, "best_configs")

	algos = color_scheme.algos_ordered_by_solution_quality.copy()
	for x in ["hMetis-K", "KaHyPar-HFC*", "KaHyPar-MF", "PaToH-D"]:
		algos.remove(x)
	#pp.performance_profiles(algos, instances, df, "hfc_vs_best_configs")

	pp.plot("hfc_vs_best_configs", colors=color_scheme.algo_colors, width_scale=1.0)
	pp.plot("best_configs", colors=color_scheme.algo_colors, width_scale=1.0)
	pp.plot("mf_vs_hfc-mfstyle", colors=color_scheme.algo_colors)
	pp.plot("mf_vs_hfc", colors=color_scheme.algo_colors)
	pp.plot("all", colors=color_scheme.algo_colors, width_scale=2.0)

	def performance_profile_per_k():
		algos = color_scheme.algos_ordered_by_solution_quality.copy()
		
		for k in ks:
			inst_with_k = list(filter(lambda x : x[1] == k, instances))
			#performance_profiles(algos, inst_with_k, df, "plot_per_k/" + str(k))
			pp.plot("plot_per_k/" + str(k), colors=color_scheme.algo_colors, display_legend="plot_per_k/legend", title="k=" + str(k))


	def performance_profile_per_instance_class():
		algos = color_scheme.algos_ordered_by_solution_quality.copy()

		for suffix, category in category_map.items():
			inst_with_cat = list(filter(lambda x : suffix in x[0], instances))
			#performance_profiles(algos, inst_with_cat, df, "plot_per_instance_class/" + category)
			pp.plot("plot_per_instance_class/" + category, colors=color_scheme.algo_colors, display_legend="plot_per_instance_class/legend", title=category)
		
		

	performance_profile_per_k()
	performance_profile_per_instance_class()


