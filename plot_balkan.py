import warnings
from plot import Plot

import os

if __name__ == '__main__':
    plot = Plot()
    # plot.load_config(config_file='experiments/balkan/config_plot_empty_map.json')
    plot.load_config(config_file='config_plot.json')
    plot.read_data()
    for name, result in plot.iterate_over_models():
        ## plot map
        #plot.posterior_map(results=result, file_name=f'map_{name}')

        #plot.plot_pies(result,os.path.join(path,name))
        plot.get_idw_map(result,name)

    plot.plot_dic(models=plot.results, file_name='dic')



