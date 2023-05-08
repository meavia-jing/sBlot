from sbayes.plot import Plot

if __name__ == '__main__':
    plot = Plot()
    # plot.load_config(config_file='experiments/balkan/config_plot_empty_map.json')
    config_file = './config_plot_south_america.json'
    #default_file = "/Users/zhanganjing/Downloads/RA/SbayesPlot/sbayes/config/config_template.yaml"
    plot.load_config(config_file=config_file)
    plot.read_data()

    #experiment = Experiment(config_file=config_file)
    #Data.from_experiment(experiment)
    #data = Data.from_config(config_file)



    plot.combine_files()

    #for name, result in plot.iterate_over_models():
        #plot.get_idw_map(result,name)
        # print('weights',result.weights)
        ## plot map
        #plot.posterior_map(results=result, file_name=f'map_{name}')

        # plot.plot_preferences(result,name)
        #
        # plot.plot_pies(result,os.path.join(name))

    #plot.plot_featuremap()
    #plot.plot_dic(models=plot.results, file_name='dic')



