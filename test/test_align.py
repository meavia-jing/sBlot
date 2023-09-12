#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os.path

import numpy as np
import unittest
import re
from collections import Counter

import pandas as pd

from sblot.plot import Plot, main, PlotType
from fnmatch import fnmatch
import pandas

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

    # def setUp(self):
    #     self.sequence = [1, 2]
    #     self.index = 0



    def compare_files(self,cluster_compare,hasHeader=False):
        ## for loop different pair of backup and aligned files
        for item in cluster_compare:
            print("item",item)
            ## stats file with header
            if hasHeader:
                file1 = pd.read_csv(item[0],delimiter=r"\s+")
                file2 = pd.read_csv(item[1],delimiter=r"\s+")
            ## cluster file without header
            else:
                file1 = pd.read_csv(item[0],header=None,delimiter=r"\s+")
                file2 = pd.read_csv(item[1], header=None, delimiter=r"\s+")

            columns1 = file1.to_numpy()
            columns2 = file2.to_numpy()

            print("columns1", columns1.shape)
            print("columns2", columns2.shape)


            count1 = Counter([tuple(sublist) for sublist in columns1])
            count2 = Counter([tuple(sublist) for sublist in columns2])

            assert count1 == count2

    def get_aligned_backup_file(self,align_path, backup_path):
        align_backup = []
        for i in align_path:
            for j in backup_path:
                if os.path.basename(i) == os.path.basename(j):
                    align_backup.append([i,j])
        return align_backup

    def test_align_files(self):
        ## running combine_files function in plot file(
        plot = Plot()
        config_file = 'plot_test_files/config_plot_align.json'
        plot.load_config(config_file=config_file)
        plot.read_data()
        #plot.combine_files()

        ### get cluster txt file and stats txt file in backup folder
        path = os.path.join(plot.input_main_paths, "backup")

        clustertxtfile = []
        statstxtfile = []
        for filename in os.listdir(path):
            txtdatapath = os.path.join(path, filename)
            #assert os.path.exists(txtdatapath)
            if os.path.exists(txtdatapath) and (fnmatch(filename, '*.txt')) and "stats" in filename:
                statstxtfile.append(txtdatapath)

            elif os.path.exists(txtdatapath) and (fnmatch(filename, '*.txt')) and "area" in filename:
                clustertxtfile.append(txtdatapath)

        ## get the raw file and backup file with same name for stats
        stats_compare = self.get_aligned_backup_file(statstxtfile,plot.all_stats_paths)
        ## get the raw file and backup file with same name for clusterfile
        cluster_compare = self.get_aligned_backup_file(clustertxtfile,plot.all_cluster_paths)

        # compare raw file and backup file

        #self.compare_files(cluster_compare,hasHeader=False)
        self.compare_files(stats_compare, hasHeader=True)






































