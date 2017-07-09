# -*- coding: utf-8 -*-

############################################################################
#  This file is part of the 4D Light Field Benchmark.                      #
#                                                                          #
#  This work is licensed under the Creative Commons                        #
#  Attribution-NonCommercial-ShareAlike 4.0 International License.         #
#  To view a copy of this license,                                         #
#  visit http://creativecommons.org/licenses/by-nc-sa/4.0/.                #
#                                                                          #
#  Authors: Katrin Honauer & Ole Johannsen                                 #
#  Contact: contact@lightfield-analysis.net                                #
#  Website: www.lightfield-analysis.net                                    #
#                                                                          #
#  The 4D Light Field Benchmark was jointly created by the University of   #
#  Konstanz and the HCI at Heidelberg University. If you use any part of   #
#  the benchmark, please cite our paper "A dataset and evaluation          #
#  methodology for depth estimation on 4D light fields". Thanks!           #
#                                                                          #
#  @inproceedings{honauer2016benchmark,                                    #
#    title={A dataset and evaluation methodology for depth estimation on   #
#           4D light fields},                                              #
#    author={Honauer, Katrin and Johannsen, Ole and Kondermann, Daniel     #
#            and Goldluecke, Bastian},                                     #
#    booktitle={Asian Conference on Computer Vision},                      #
#    year={2016},                                                          #
#    organization={Springer}                                               #
#    }                                                                     #
#                                                                          #
############################################################################


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from toolkit.utils import plotting, misc


def plot_pairwise_comparisons(algorithms, scenes):
    for idx_a1, algo1 in enumerate(algorithms):
        for idx_a2 in range(idx_a1 + 1, len(algorithms)):
            algo2 = algorithms[idx_a2]
            plot_pairwise_comparison(algo1, algo2, scenes)


def plot_pairwise_comparison(algo1, algo2, scenes, n_scenes_per_row=4, subdir="pairwise_diffs"):
    rows, cols = int(np.ceil(len(scenes) / float(n_scenes_per_row))), n_scenes_per_row
    fig = plt.figure(figsize=(4*cols, 3*rows))

    for idx_s, scene in enumerate(scenes):
        algo_result_1 = misc.get_algo_result(algo1, scene)
        algo_result_2 = misc.get_algo_result(algo2, scene)
        gt = scene.get_gt()

        plt.subplot(rows, cols, idx_s+1)
        cb = plt.imshow(np.abs(algo_result_1 - gt) - np.abs(algo_result_2 - gt),
                        interpolation="none", cmap=cm.seismic, vmin=-.1, vmax=.1)
        plt.colorbar(cb, shrink=0.7)
        plt.title(scene.get_display_name())

    # title
    a1 = algo1.get_display_name()
    a2 = algo2.get_display_name()
    plt.suptitle("|%s - GT| - |%s - GT|\nblue: %s is better, red: %s is better" % (a1, a2, a1, a2))

    fig_name = "pairwise_diffs_%s_%s" % (algo1.get_name(), algo2.get_name())
    fig_path = plotting.get_path_to_figure(fig_name, subdir=subdir)
    plotting.save_tight_figure(fig, fig_path, hide_frames=True, padding_top=0.85,
                               hspace=0.15, wspace=0.15)
