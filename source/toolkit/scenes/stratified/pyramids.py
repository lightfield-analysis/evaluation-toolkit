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

from toolkit.metrics import PyramidsParallelBumpiness, PyramidsSlantedBumpiness
from toolkit.scenes import BaseStratified
from toolkit.utils import plotting, misc


class Pyramids(BaseStratified):

    mn_plane = "mask_plane"
    mn_sphere_out = "mask_sphere_out"
    mn_sphere_in = "mask_sphere_in"
    mn_pyramids = "mask_pyramids"

    def __init__(self, name="pyramids", general_metrics_high_res=False, **kwargs):
        super(Pyramids, self).__init__(name, general_metrics_high_res=general_metrics_high_res,
                                       **kwargs)

    @staticmethod
    def get_scene_specific_metrics():
        return [PyramidsSlantedBumpiness(), PyramidsParallelBumpiness()]

    def set_scale_for_algo_overview(self):
        self.set_low_gt_scale()

    def plot_algo_disp_vs_gt_disp(self, algorithms, subdir="stratified"):
        self.set_low_gt_scale()

        # prepare data
        gt = self.get_gt()
        m_eval = self.get_boundary_mask()
        mask_names = ["Sphere In", "Sphere Out"]
        masks = [self.get_sphere_in()*m_eval, self.get_sphere_out()*m_eval]

        factor = 1000.0
        gt_rounded = np.asarray(gt * factor, dtype=np.int)
        disp_values = np.unique(gt_rounded)
        n_values = np.size(disp_values)

        # prepare figure
        fig = plt.figure(figsize=(14, 6))
        rows, cols = 1, 2
        fontsize = 14
        legend_lines = []
        legend_labels = []

        for algorithm in algorithms:
            algo_result = misc.get_algo_result(algorithm, self)

            # go through ground truth disparity values
            for idx_d in range(n_values):
                current_disp = disp_values[idx_d]
                m_disp = (gt_rounded == current_disp)

                # find median disparity of algorithm result at image regions
                # of given ground truth disparity value
                for idx_m, (mask, mask_name) in enumerate(zip(masks, mask_names)):
                    algo_disps = algo_result[m_disp * mask]

                    if np.size(algo_disps) > 0:
                        median = np.median(algo_disps)
                        plt.subplot(rows, cols, idx_m+1)
                        s = plt.scatter(current_disp/factor, median,
                                        marker="o", c=algorithm.get_color(), alpha=0.8, s=5, lw=0)

            legend_lines.append(s)
            legend_labels.append(algorithm.get_display_name())

        # finalize figure attributes
        for idx_m, (mask, mask_name) in enumerate(zip(masks, mask_names)):
            plt.subplot(rows, cols, idx_m+1)
            vmin = np.min(gt_rounded[mask]) / factor
            vmax = np.max(gt_rounded[mask]) / factor
            plt.xlim([vmin, vmax])
            plt.ylim([vmin, vmax])
            plt.xlabel("Ground truth disparities", fontsize=fontsize)
            plt.ylabel("Algorithm disparities", fontsize=fontsize)
            plt.title(mask_name, fontsize=fontsize)
            plotting.hide_upper_right()

        legend = plt.legend(legend_lines, legend_labels, frameon=False, ncol=1, scatterpoints=1,
                            title="Algorithms:", bbox_to_anchor=(1.25, .85), borderaxespad=0.0)
        for idx in range(len(legend.legendHandles)):
            legend.legendHandles[idx]._sizes = [22]
        plt.suptitle("Ground Truth Disparities vs. Algorithm Disparities", fontsize=fontsize)

        fig_path = plotting.get_path_to_figure("pyramids_disp_disp", subdir=subdir)
        plotting.save_tight_figure(fig, fig_path, remove_ticks=False,
                                   hspace=0.2, wspace=0.3, padding_top=0.88)

    # -------------------------
    # evaluation masks
    # -------------------------

    def get_plane_mask(self):
        return self.get_mask(self.mn_plane)

    def get_objects(self):
        return self.get_spheres() + self.get_pyramids()

    def get_spheres(self):
        return self.get_sphere_in() + self.get_sphere_out()

    def get_sphere_in(self):
        return self.get_mask(self.mn_sphere_in)

    def get_sphere_out(self):
        return self.get_mask(self.mn_sphere_out)

    def get_pyramids(self):
        return self.get_mask(self.mn_pyramids)
