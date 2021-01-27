from .. import runtime_plot as rp
import pandas as pd
import color_scheme


if __name__ == '__main__':

	files = [
		'KaHyPar-HFC-mfstyle.csv', 'KaHyPar-HFC.csv', 'KaHyPar-MF.csv',
		'km1_patoh_q.csv', 'km1_patoh_d.csv',
		'km1_hmetis_r.csv', 'km1_hmetis_k.csv',
		'km1_zoltan_algd.csv',
		'km1_mondriaan.csv', 
		'km1_hype.csv'
		]
	df = pd.concat(map(pd.read_csv, files))
	df = df[df.failed == "no"]

	averaged_runtimes = rp.aggregate_dataframe_by_arithmetic_mean_per_instance(df)
	rp.print_gmean_times(averaged_runtimes)
	rp.plot(averaged_runtimes, colors=color_scheme.algo_colors, algo_order=color_scheme.algos_ordered_for_runtime_plot)

