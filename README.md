# sBlot - plotting for sBayes
 This package provide plotting functions for the project Sbayes



## Configuration file
map >> conten >> type<br>
if the type == "density_map" , plot all the languages.<br>
if the type == "consensus_map", plot languages with afrequency larger than min_posterior_frequency.<br>

graphic >> clusters>> point_size and graphic >> clusters>> line_width<br>
if point_size == "freqency" and line_width is a number ,only the size of the points are drawn according to the language frequency and the line width is fixed.<br>
if line_width == "freqency" and point_size is a number, only the width of line are drawn accoding to the language frequency and the point size is fixed.<br>
if point_size == "freqency" and line_width =="frequency", plot the line width and point size according  to the language frequency.<br>
else, just plot size of point and width of line to a fixed number.

graphic >> base_map>> polygon>>resolution
Basemap is divided into grid. This parameter determines the resolution of Grid. 


## Data description
Data for testing the plotting package:<br>
28 types of language that mainly located in the southern Europe, including Greek, Romance,Turkish, etc.





