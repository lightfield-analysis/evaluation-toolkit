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


import abc

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from toolkit import settings
from toolkit.scenes import BaseScene
from toolkit.utils import log, misc, plotting


class BaseStratified(BaseScene):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, category=settings.STRATIFIED, **kwargs):
        super(BaseStratified, self).__init__(name, category=category, **kwargs)

    def set_scale_for_algo_overview(self):
        self.set_high_gt_scale()

    def plot_algo_overview(self, algorithms, with_metric_vis=True, subdir="algo_overview", fs=14):
        self.set_scale_for_algo_overview()
        metrics = self.get_scene_specific_metrics()
        n_metrics = len(metrics)

        if not with_metric_vis:
            rows, cols = 2 + n_metrics, len(algorithms) + 2
            fig = plt.figure(figsize=(2.6*len(algorithms), 4.9))
            offset = 0
        else:
            rows, cols = 2 + 2*n_metrics, len(algorithms) + 2
            fig = plt.figure(figsize=(2.6*len(algorithms), rows+3))
            offset = n_metrics

        labelpad = -15
        hscale, wscale = 7, 5
        width_ratios = [wscale] * (len(algorithms) + 1) + [1]
        height_ratios = [hscale] * (rows - n_metrics) + [1] * n_metrics
        gs = gridspec.GridSpec(rows, cols, height_ratios=height_ratios, width_ratios=width_ratios)

        gt = self.get_gt()
        dummy = np.ones((self.get_height() / hscale, self.get_width()))
        cb_height, w = np.shape(gt)
        cb_width = w / float(wscale)

        # first column (gt, center view, ...)
        plt.subplot(gs[0])
        plt.imshow(gt, **settings.disp_map_args(self))
        plt.title("Ground Truth", fontsize=fs)
        plt.ylabel("Disparity Map", fontsize=fs)

        plt.subplot(gs[cols])
        plt.imshow(self.get_center_view())
        plt.ylabel("diff: GT - Algo", fontsize=fs)

        for idx_m, metric in enumerate(metrics):
            plt.subplot(gs[(2+idx_m+offset)*cols])
            plt.xlabel(metric.get_short_name(), labelpad=labelpad, fontsize=fs)
            plt.imshow(dummy, cmap="gray_r")

        # algorithm columns
        for idx_a, algorithm in enumerate(algorithms):
            log.info("Processing algorithm: %s" % algorithm)
            algo_result = misc.get_algo_result(algorithm, self)

            # algorithm disparities
            plt.subplot(gs[idx_a+1])
            plt.title(algorithm.get_display_name(), fontsize=fs)
            cm1 = plt.imshow(algo_result, **settings.disp_map_args(self))

            # algorithm diff map
            plt.subplot(gs[cols+idx_a+1])
            cm2 = plt.imshow(gt - algo_result, **settings.diff_map_args())

            # add colorbar if last column
            if idx_a == (len(algorithms) - 1):
                plotting.add_colorbar(gs[idx_a + 2], cm1, cb_height, cb_width,
                                      colorbar_bins=5, fontsize=fs-4)
                plotting.add_colorbar(gs[cols + idx_a + 2], cm2, cb_height, cb_width,
                                      colorbar_bins=5, fontsize=fs-4)

            # score and background color for metrics
            for idx_m, metric in enumerate(metrics):

                if with_metric_vis:
                    plt.subplot(gs[(2+idx_m)*cols+idx_a+1])
                    score, vis = metric.get_score(algo_result, gt, self, with_visualization=True)
                    cm3 = plt.imshow(vis, **settings.metric_args(metric))

                    if idx_a == 0:
                        plt.ylabel(metric.get_short_name(), fontsize=fs)
                    elif idx_a == (len(algorithms) - 1):
                        plotting.add_colorbar(gs[(2+idx_m)*cols+idx_a+2], cm3, cb_height, cb_width,
                                              colorbar_bins=metric.colorbar_bins, fontsize=fs-4)

                else:
                    score = metric.get_score(algo_result, gt, self)

                plt.subplot(gs[(2+idx_m+offset)*cols+idx_a+1])
                plt.imshow(dummy*score,
                           **settings.score_color_args(vmin=metric.vmin, vmax=metric.vmax))
                plt.xlabel(metric.format_score(score), labelpad=labelpad, fontsize=fs)

        fig_name = "algo_overview_" + self.get_name() + with_metric_vis * "_vis"
        fig_path = plotting.get_path_to_figure(fig_name, subdir=subdir)
        plotting.save_tight_figure(fig, fig_path, wspace=0.04, hide_frames=True)
