import performance_profiles
import relative_runtimes_plot
import matplotlib.gridspec
import commons

def combined_pp_rs(df, fig, baseline, colors=None, algos=None, instances=None, time_limit=7200, width_ratios=[0.5,0.5], ncol=4, runtime_field="totalPartitionTime"):
	if algos == None:
		algos = commons.infer_algorithms_from_dataframe(df)
	if instances == None:
		instances = commons.infer_instances_from_dataframe(df)
	if colors == None:
		colors = commons.infer_color_mapping(algos)

	outer_grid = matplotlib.gridspec.GridSpec(nrows=1, ncols=2, wspace=0.02, hspace=0.0, width_ratios=width_ratios)
	# outer_grid = fig.subplots(nrows=1, ncols=2, wspace=0.02, hspace=0.0, width_ratios=[0.5,0.5])

	## separate plots
	ratios_df = performance_profiles.performance_profiles(algos, instances, df, objective="km1", time_limit=time_limit)
	performance_profiles.plot(ratios_df, fig=fig, external_subplot=outer_grid[0], colors=colors, width_scale=1.0)
	
	runtime_ax = fig.add_subplot(outer_grid[1])
	# runtime_ax = outer_grid[1]
	relative_runtimes_plot.construct_plot(df=df, ax=runtime_ax, baseline_algorithm=baseline, colors=colors, algos=algos, 
	                                      seed_aggregator="mean", field=runtime_field, time_limit=time_limit)
	runtime_ax.yaxis.set_label_position("right")
	runtime_ax.yaxis.tick_right()
	runtime_ax.get_legend().remove()


	performance_profiles.legend_below(fig, ncol=ncol)
