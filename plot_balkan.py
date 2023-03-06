import warnings
from sbayes.plot import Plot

import os
import numpy as np
from matplotlib import colors

if __name__ == '__main__':
    plot = Plot()
    # plot.load_config(config_file='experiments/balkan/config_plot_empty_map.json')
    plot.load_config(config_file='/Users/zhanganjing/Downloads/RA/SbayesPlot/config_plot_south_america.json')
    plot.read_data()


    for name, result in plot.iterate_over_models():
        #plot.plot_weights(result,name)
        plot.plot_features(result,name)
        #plot.get_idw_map(result, name)
    #     #plot map
         #plot.posterior_map(results=result, file_name=f'map_{name}')

          #plot.plot_preferences(result,name)
    #
         #plot.plot_pies(result,os.path.join(name))


    #plot.plot_dic(models=plot.results, file_name='dic')



