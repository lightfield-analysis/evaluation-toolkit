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
from matplotlib import cm
import numpy as np

from toolkit import settings
from toolkit.utils import misc, plotting


def plot(algorithms, scenes, meta_algo, subdir="meta_algo_comparisons",
         fig_name=None, with_gt_row=False, fs=12):

    # prepare figure
    rows, cols = len(algorithms) + int(with_gt_row), len(scenes)*3+1
    fig = plt.figure(figsize=(cols * 1.3, rows * 1.5))
    grid, cb_height, cb_width = plotting.get_grid_with_colorbar(rows, cols, scenes[0])
    colorbar_args = {"height": cb_height*0.8, "width": cb_width, "colorbar_bins": 4, "fontsize": fs}

    for idx_s, scene in enumerate(scenes):
        gt = scene.get_gt()
        meta_algo_result = misc.get_algo_result(meta_algo, scene)
        add_label = idx_s == 0  # is first column
        add_colorbar = idx_s == len(scenes)-1  # is last column

        # plot one row per algorithm
        for idx_a, algorithm in enumerate(algorithms):
            algo_result = misc.get_algo_result(algorithm, scene)
            add_title = idx_a == 0  # is top row

            idx = idx_a * cols + 3 * idx_s

            # disparity map
            plt.subplot(grid[idx])
            plt.imshow(algo_result, **settings.disp_map_args(scene))
            if add_title:
                plt.title("DispMap", fontsize=fs)
            if add_label:
                plt.ylabel(algorithm.get_display_name(), fontsize=fs)

            # error map: gt - algo
            plt.subplot(grid[idx + 1])
            cb1 = plt.imshow(gt-algo_result, **settings.diff_map_args(vmin=-.1, vmax=.1))
            if add_title:
                plt.title("GT-Algo", fontsize=fs)

            # error map: |meta-gt| - |algo-gt|
            plt.subplot(grid[idx + 2])
            median_diff = np.abs(meta_algo_result - gt) - np.abs(algo_result - gt)
            cb2 = plt.imshow(median_diff, interpolation="none", cmap=cm.RdYlGn, vmin=-.05, vmax=.05)
            if add_title:
                plt.title(meta_algo.get_display_name().replace("PerPix", ""), fontsize=fs)

            if add_colorbar:
                if idx_a % 2 == 0:
                    plotting.add_colorbar(grid[idx + 3], cb1, **colorbar_args)
                else:
                    plotting.add_colorbar(grid[idx + 3], cb2, **colorbar_args)

        if with_gt_row:
            idx = len(algorithms) * cols + 3 * idx_s

            plt.subplot(grid[idx])
            plt.imshow(gt, **settings.disp_map_args(scene))
            plt.xlabel("GT", fontsize=fs)

            if add_label:
                plt.ylabel("Reference")

            plt.subplot(grid[idx + 1])
            cb1 = plt.imshow(np.abs(gt - meta_algo_result), **settings.abs_diff_map_args())
            plt.xlabel("|GT-%s|" % meta_algo.get_display_name(), fontsize=fs-2)

            if add_colorbar:
                plotting.add_colorbar(grid[idx + 3], cb1, **colorbar_args)

    if fig_name is None:
        scene_names = "_".join(s.get_name() for s in scenes)
        fig_name = "%s_comparison_%s" % (meta_algo.get_name(), scene_names)
    fig_path = plotting.get_path_to_figure(fig_name, subdir=subdir)
    plotting.save_tight_figure(fig, fig_path, hide_frames=True, hspace=0.02, wspace=0.0)
