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
from matplotlib import gridspec

import settings
from utils import plotting, misc


def plot(algorithms, scenes, thresh=settings.BAD_PIX_THRESH, subdir="heatmaps"):
    # prepare figure
    n_scenes = len(scenes)
    rows, cols = int(np.ceil(n_scenes / 4.0)), 5
    fig = plt.figure(figsize=(2.7*cols, 3*rows))
    wscale = 9
    grid = gridspec.GridSpec(rows, cols, height_ratios=[1] * rows, width_ratios=[wscale] * 4 + [1])
    colorbar_height, w = scenes[0].get_shape()
    colorbar_width = w / float(wscale)
    colorbar_args = {"height": colorbar_height, "width": colorbar_width,
                     "colorbar_bins": 5, "fontsize": 10, "scale": 0.8}

    idx_scene = 0

    # plot heatmap per scene
    for idx in range(rows*cols):
        if (idx + 1) % 5 != 0:
            scene = scenes[idx_scene]
            idx_scene += 1
            plt.subplot(grid[idx])
            cm = plt.imshow(get_bad_count(scene, algorithms, thresh, percentage=True), vmin=0, vmax=100, cmap="inferno")
            plt.ylabel(scene.get_display_name(), fontsize=18, labelpad=2.5)
        else:
            plotting.add_colorbar(grid[idx], cm, **colorbar_args)

    plt.suptitle("Per Pixel: Percentage of Algorithms with abs(gt-algo) > %0.2f" % thresh, fontsize=18)

    fig_path = plotting.get_path_to_figure("error_heatmaps", subdir=subdir)
    plotting.save_tight_figure(fig, fig_path, hide_frames=True, remove_ticks=True, hspace=0.02)


def get_bad_count(scene, algorithms, thresh, percentage=False):
    bad_count = np.zeros((scene.get_height(), scene.get_width()))
    gt = scene.get_gt()

    for idx_a, algorithm in enumerate(algorithms):
        algo_result = misc.get_algo_result(scene, algorithm)
        abs_diffs = np.abs(gt - algo_result)

        with np.errstate(invalid="ignore"):
            bad = abs_diffs > thresh
            bad += misc.get_mask_invalid(abs_diffs)

        bad_count += bad

    if percentage:
        bad_count = misc.percentage(len(algorithms), bad_count)

    return bad_count


