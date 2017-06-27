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

import settings
from utils import plotting


def plot(scores_algos_metric, metric_names, algorithms, fig_name, max_per_metric=None, add_legend=True):
    n_axes = len(metric_names)
    if max_per_metric is None:
        max_per_metric = np.max(scores_algos_metric, axis=0) * 1.2

    fig = plt.figure(figsize=(10, 9))
    n_circles = 4 + 1

    rect = [0.1, 0.1, 0.8, 0.8]  # left, bottom, width, height
    axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) for i in range(n_axes)]
    angles = np.arange(90, 90+360, 360.0/n_axes) % 360

    # add metric labels
    axes[0].set_thetagrids(angles, labels=metric_names, fontsize=18)

    # make default labeling invisible
    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid("off")
        ax.xaxis.set_visible(False)

    # compute metric values at axis intersections
    axes_values = []
    for vmax in max_per_metric:
        steps = list(np.linspace(0, vmax, n_circles))
        steps = steps[:-1]
        axes_values.append(steps)

    # add metric values at axis intersections
    for ax, angle, labels in zip(axes, angles, axes_values):
        ax.set_rgrids(range(1, n_circles+1), angle=angle, labels=['%0.1f' % e for e in labels])
        ax.spines["polar"].set_visible(False)
        ax.set_ylim(0, n_circles)

    # add first angle at the end to close the line-loop
    angle = np.deg2rad(np.r_[angles, angles[0]])

    # plot one line per algorithm, passing through all metrics
    for idx_a, algorithm in enumerate(algorithms):
        metric_scores_cur_algo = scores_algos_metric[idx_a, :]

        for idx_m in range(len(metric_names)):
            vmax = float(max_per_metric[idx_m])
            steps = axes_values[idx_m]
            step = steps[1] - steps[0]
            scale_factor = n_circles / (vmax + step)
            adjusted_value = (metric_scores_cur_algo[idx_m] + step) * scale_factor
            metric_scores_cur_algo[idx_m] = adjusted_value

        # add first score at end to close the line-loop
        metric_scores_cur_algo = np.r_[metric_scores_cur_algo, metric_scores_cur_algo[0]]

        ax.plot(angle, metric_scores_cur_algo, "-", lw=4, alpha=0.7,
                color=algorithm.get_color(), label=algorithm.get_display_name(),
                markersize=7, markeredgecolor=algorithm.get_color())

    if add_legend:
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.), title="Algorithms:",
                           frameon=False, prop={'size': 16}, labelspacing=0.2)
        plt.setp(legend.get_title(), fontsize=18)
        plt.setp(legend.get_title(), style='italic')

    plotting.save_fig(fig, fig_name)



