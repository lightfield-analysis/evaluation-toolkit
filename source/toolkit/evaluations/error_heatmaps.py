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
import numpy as np

from toolkit import settings
from toolkit.utils import plotting, misc


def plot(algorithms, scenes, thresh=settings.BAD_PIX_THRESH,
         subdir="error_heatmaps", fs=18, max_per_row=4):

    # prepare figure
    n_scenes = len(scenes)
    cols = min(n_scenes, max_per_row) + 1  # + 1 for colorbars
    rows = int(np.ceil(n_scenes / float(cols - 1)))
    fig = plt.figure(figsize=(2.7*cols, 3*rows))
    grid, cbh, cbw = plotting.get_grid_with_colorbar(rows, cols, scenes[0], hscale=1, wscale=9)
    colorbar_args = {"height": cbh, "width": cbw, "colorbar_bins": 5, "fontsize": 10, "scale": 0.8}

    # plot heatmaps
    idx_scene = 0
    for idx in range(rows*cols):

        if (idx + 1) % cols:
            # plot error heatmap for scene
            scene = scenes[idx_scene]
            idx_scene += 1

            plt.subplot(grid[idx])
            bad_count = get_bad_count(scene, algorithms, thresh, percentage=True)
            cm = plt.imshow(bad_count, vmin=0, vmax=100, cmap="inferno")
            plt.ylabel(scene.get_display_name(), fontsize=fs, labelpad=2.5)
        else:
            # plot colorbar
            plotting.add_colorbar(grid[idx], cm, **colorbar_args)

        if idx_scene >= n_scenes:
            if idx % cols or n_scenes == 1:
                plotting.add_colorbar(grid[idx+1], cm, **colorbar_args)
            break

    plt.suptitle("Per Pixel: Percentage of %d Algorithms with abs(gt-algo) > %0.2f" %
                 (len(algorithms), thresh), fontsize=fs)

    fig_name = ("error_heatmaps_%.3f" % thresh).replace(".", "")
    fig_path = plotting.get_path_to_figure(fig_name, subdir=subdir)
    plotting.save_tight_figure(fig, fig_path, hide_frames=True, hspace=0.02)


def get_bad_count(scene, algorithms, thresh, percentage=False):
    bad_count = np.zeros(scene.get_shape())
    gt = scene.get_gt()

    for algorithm in algorithms:
        algo_result = misc.get_algo_result(algorithm, scene)
        abs_diffs = np.abs(gt - algo_result)

        with np.errstate(invalid="ignore"):
            bad = abs_diffs > thresh
            bad += misc.get_mask_invalid(abs_diffs)

        bad_count += bad

    if percentage:
        bad_count = misc.percentage(len(algorithms), bad_count)

    return bad_count
