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
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

from toolkit import settings
from toolkit.metrics import MSE, BadPix, BumpinessPlanes, BumpinessContinSurf, \
    Discontinuities, FineFattening, FineThinning
from toolkit.scenes import BaseScene
from toolkit.utils import log, misc, plotting


class PhotorealisticScene(BaseScene):

    def get_scene_specific_metrics(self):
        return [m for m in misc.get_region_metrics() if
                m.mask_exists(self, settings.LOWRES) or m.mask_exists(self, settings.HIGHRES)]

    def plot_algo_overview(self, algorithms, subdir="algo_overview", fs=6):
        accv_metrics = [MSE(), BadPix(0.07), BumpinessPlanes(), BumpinessContinSurf(),
                        Discontinuities(), FineFattening(), FineThinning()]
        metrics_low_res = [m for m in self.get_applicable_metrics_low_res() if m in accv_metrics]
        metrics_high_res = [m for m in self.get_applicable_metrics_high_res() if m in accv_metrics]

        # prepare figure
        rows = len(metrics_low_res + metrics_high_res) + 1
        cols = len(algorithms) + 1
        fig = plt.figure(figsize=(cols, rows*1.1))
        grids = self._get_grids(fig, rows, cols, axes_pad=-0.2)

        # center view on top left grid cell
        self.set_high_gt_scale()
        plt.sca(grids[0][0])
        plt.imshow(self.get_center_view())
        plt.title("Center View", fontsize=fs)
        plt.ylabel("Disparity Map", fontsize=fs)

        # mask visualizations + algorithm disparity maps + metric visualizations
        log.info("Computing scores and visualizations for low resolution metrics.")
        self.set_low_gt_scale()
        self.plot_metric_rows(grids, algorithms, metrics_low_res, offset=0, fontsize=fs)

        log.info("Computing scores and visualizations for high resolution metrics.")
        self.set_high_gt_scale()
        self.plot_metric_rows(grids, algorithms, metrics_high_res,
                              offset=len(metrics_low_res), fontsize=fs)

        # finalize figure
        for grid in grids:
            plotting.remove_ticks_from_axes(grid.axes_all)
            plotting.remove_frames_from_axes(grid.axes_all)
        plt.suptitle(self.get_display_name(), fontsize=fs+2)

        fig_path = plotting.get_path_to_figure("algo_overview_%s" % self.get_name(), subdir=subdir)
        plotting.save_fig(fig, fig_path, pad_inches=0.1)

    @staticmethod
    def _get_grids(fig, rows, cols, axes_pad=0):
        grids = []
        for row in range(rows):
            grid_id = int("%d%d%d" % (rows, 1, row + 1))
            grid = ImageGrid(fig, grid_id,
                             nrows_ncols=(1, cols),
                             axes_pad=(0.05, axes_pad),
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="10%",
                             cbar_pad="5%")
            grids.append(grid)
        return grids

    def plot_metric_rows(self, grids, algorithms, metrics, offset, fontsize):
        gt = self.get_gt()
        center_view = self.get_center_view()

        for idx_a, algorithm in enumerate(algorithms):
            log.info("Algorithm: %s" % algorithm)
            algo_result = misc.get_algo_result(algorithm, self)

            # add algorithm disparity map
            plt.sca(grids[0][idx_a + 1])
            cm = plt.imshow(algo_result, **settings.disp_map_args(self))
            plt.title(algorithm.get_display_name(), fontsize=fontsize)

            # add colorbar to last disparity map in row
            if idx_a == (len(algorithms) - 1):
                plotting.create_colorbar(cm, cax=grids[0].cbar_axes[0],
                                         colorbar_bins=7, fontsize=fontsize)

            # add algorithm metric visualizations
            for idx_m, metric in enumerate(metrics):
                log.info(metric.get_display_name())
                mask = metric.get_evaluation_mask(self)

                plt.sca(grids[idx_m + offset + 1][idx_a + 1])
                cm = self.plot_algo_vis_for_metric(metric, algo_result, gt, mask,
                                                   self.hidden_gt(), fontsize)

                # add colorbar to last metric visualization in row
                if idx_a == len(algorithms) - 1:
                    plotting.create_colorbar(cm, cax=grids[idx_m + offset + 1].cbar_axes[0],
                                             colorbar_bins=metric.colorbar_bins, fontsize=fontsize)

                # add mask visualizations as 1st column
                if idx_a == 0:
                    plt.sca(grids[idx_m + offset + 1][0])
                    plotting.plot_img_with_transparent_mask(center_view, mask,
                                                            alpha=0.7, color=settings.MASK_COLOR)
                    plt.ylabel(metric.get_short_name(), fontsize=fontsize)
                    plt.title("Region Mask", fontsize=fontsize)

    def plot_algo_vis_for_metric(self, metric, algo_result, gt, mask, hide_gt=False, fontsize=10):
        score, vis = metric.get_score(algo_result, gt, self, with_visualization=True)

        if hide_gt and metric.pixelize_results():
            vis = plotting.pixelize(vis)

        # plot algorithm disparity map as background
        plt.imshow(algo_result, **settings.disp_map_args(self, cmap="gray"))

        # plot masked metric visualization on top
        cm = plt.imshow(np.ma.masked_array(vis, mask=~mask), **settings.metric_args(metric))
        plt.title(metric.format_score(score), fontsize=fontsize)

        return cm
