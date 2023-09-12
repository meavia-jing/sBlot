#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import numpy as np
import unittest

from sblot.plot import Plot, main, PlotType
from shapely import geometry

from matplotlib import colors

import matplotlib.pyplot as plt

class TestPlot(unittest.TestCase):

    """
    Test cases of plotting functions in ´sblot/plot.py´.
    """
    @classmethod
    def setUpClass(cls) -> None:
        print("start")

    @classmethod
    def tearDown(self) -> None:
        print("end")

    def test_example_linedot_consensus(self):
        main(
            config='plot_test_files/config_plot_linedot_freq_con.json',
            plot_types=[PlotType.map]
        ,
        )
        map_paths = [
            "results/map/posterior_map_K1_0_0.7.pdf",
            "results/map/posterior_map_K3_0_0.7.pdf",
        ]
        for map_path in map_paths:
            assert os.path.exists(map_path)
            assert os.path.getsize(map_path) > 0

    def test_example_linedot_density(self):
        main(
            config='plot_test_files/config_plot_linedot_freq_den.json',
            plot_types=[PlotType.map]
        )
        map_paths = [
            "results/map/posterior_map_K1_0.pdf",
            "results/map/posterior_map_K3_0.pdf",
        ]
        for map_path in map_paths:
            assert os.path.exists(map_path)
            assert os.path.getsize(map_path) > 0

    def test_example_idw(self):
        main(
            config='plot_test_files/config_plot_idw.json',
            plot_types=[PlotType.map]
        )
        map_paths = [
            "results/map/K1_0.pdf",
            "results/map/K3_0.pdf",
        ]
        for map_path in map_paths:
            assert os.path.exists(map_path)
            assert os.path.getsize(map_path) > 0




    def test_weights(self):
        main(
            config="plot_test_files/config_plot_weights.json",
            plot_types=[PlotType.weights_plot]
        )
        map_paths = [
            "results/weights/weights_grid_K1_0.pdf",
            "results/weights/weights_grid_K3_0.pdf",
        ]
        for map_path in map_paths:
            assert os.path.exists(map_path)
            assert os.path.getsize(map_path) > 0


    def test_example_preference(self):
        main(
            config='plot_test_files/config_plot_preference.json',
            plot_types=[PlotType.preference_plot]
        )
        map_paths = [
            "results/preference/prob_grid_K1_0_family_Arawak.pdf",
            "results/preference/prob_grid_K3_0_family_Arawak.pdf",
        ]
        for map_path in map_paths:
            assert os.path.exists(map_path)
            assert os.path.getsize(map_path) > 0
    #
    #
    # def test_example_feature_plot(self):
    #     main(
    #         config='plot_test_files/config_plot_preference.json',
    #         plot_types=[PlotType.feature_plot],
    #         feature_name="F1"
    #     )
    #     map_paths = [
    #         "results/preference/prob_grid_K1_0_family_Arawak.pdf",
    #         "results/preference/prob_grid_K3_0_family_Arawak.pdf",
    #     ]
    #     for map_path in map_paths:
    #         assert os.path.exists(map_path)
    #         assert os.path.getsize(map_path) > 0



    def test_example_dic(self):
        main(
            config='plot_test_files/config_plot_preference.json',
            plot_types=[PlotType.dic_plot]
        )
        map_paths = [
            "results/DIC/dic.pdf",
        ]
        for map_path in map_paths:
            assert os.path.exists(map_path)
            assert os.path.getsize(map_path) > 0
    #
    # def test_example_pie(self):
    #     main(
    #         config='plot_test_files/config_plot_pie.json',
    #         plot_types=[PlotType.pie_plot]
    #     )
    #     map_paths = [
    #         "results/pie/plot_pies_K1_0.pdf",
    #         "results/pie/plot_pies_K3_0.pdf",
    #     ]
    #     for map_path in map_paths:
    #         assert os.path.exists(map_path)
    #         assert os.path.getsize(map_path) > 0
    #
    #
    #
    #

    #
    #
    # def test_compute_bbox(self):
    #     extent = {'x_min': -100000,
    #               "x_max": 1500000,
	# 		      "y_min": 1040000,
    #               'y_max':2000000 }
    #     test_bbox = geometry.box(extent['x_min'], extent['y_min'],
    #                         extent['x_max'], extent['y_max'])
    #     plot = Plot()
    #     bbox = plot.compute_bbox(extent)
    #     self.assertEqual(bbox,test_bbox)
    #
    #
    #
    # def test_get_cluster_colors_seven(self):
    #     cm = plt.get_cmap('gist_rainbow')
    #     custom_colors =  ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e"]
    #     provided = np.array([colors.to_rgba(c) for c in custom_colors])
    #     additional = cm(np.linspace(0, 1, 7 - len(custom_colors), endpoint=False))
    #     test_color = list(np.concatenate((provided, additional), axis=0))
    #
    # #     plot = Plot()
    # #     color = plot.get_cluster_colors(n_clusters=7,custom_colors=custom_colors)
    # #     self.assertTrue(np.isin(color,test_color).all())
    #
    #
    # def test_get_cluster_colors_six(self):
    #     custom_colors =  ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e"]
    #     test_color = np.array([colors.to_rgba(c) for c in custom_colors])
    #     plot = Plot()
    #     color = plot.get_cluster_colors(n_clusters=6,custom_colors=custom_colors )
    #     self.assertIn(color,test_color)















