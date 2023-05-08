# sBlot - plotting for sBayes
 This package is design to provide plotting functions for the project sBayes. Here we describe some basic commands to run plotting functions. For more detailed instructions explaining each step in the analysis and the various settings, please consult the user manual (sBlot Documentation.pdf)


# Installation
Same as sbayes project

# Running sBlot
sBot can be used as a python library or through a command line interface. Here we describe how to run sBlot in the command line interface,or run it as a python package. 

To run sBlots from the command line, simply call:
```shell 
python plot.py config_plot.json map
```


To run sBlot as python package, call:
```
plot = Plot()
plot.load_config(config_file='config_plot.json')
plot.read_data()

for name, result in plot.iterate_over_models():      
	plot.posterior_map(results=result, file_name=f'map_{name}')
	plot.get_idw_map(result,name)
```
 





