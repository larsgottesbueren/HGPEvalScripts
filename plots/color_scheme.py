import seaborn as sb

color_palette = sb.color_palette()
algo_list_for_color_mapping = ["KaHyPar-MF", "KaHyPar-HFC*", "KaHyPar-HFC", "PaToH-Q", "PaToH-D", "hMetis-R", "hMetis-K", "Zoltan-AlgD", "Mondriaan", "HYPE"]
algo_colors = dict(zip(algo_list_for_color_mapping, color_palette))
algos_ordered_by_solution_quality = ["KaHyPar-HFC*", "KaHyPar-HFC", "KaHyPar-MF", "hMetis-R", "hMetis-K", "PaToH-Q", "Zoltan-AlgD", "Mondriaan", "PaToH-D", "HYPE"]
algos_ordered_for_runtime_plot = ["KaHyPar-MF", "KaHyPar-HFC*", "KaHyPar-HFC", "hMetis-R", "hMetis-K", "Zoltan-AlgD", "PaToH-Q", "PaToH-D", "Mondriaan", "HYPE"]

algo_colors["No-Distance"] = 'black'
algo_colors["No-Iso-DP"] = 'black'
algo_colors["No-MBC"] = 'black'
algo_colors["HFC*-No-Iso-DP"] = 'black'


common_font_size = 12
