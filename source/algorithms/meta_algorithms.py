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

import numpy as np

from algorithms import Algorithm
from utils import misc


class MetaAlgorithm(Algorithm):

    def __init__(self, file_name, display_name, is_meta=True, **kwargs):
        super(MetaAlgorithm, self).__init__(file_name=file_name, display_name=display_name, is_meta=is_meta, **kwargs)

    @staticmethod
    def prepare_meta_algorithms(meta_algorithms, algorithms, scenes):
        for meta_algorithm in meta_algorithms:
            meta_algorithm.compute_meta_results(scenes, algorithms)

    @staticmethod
    def get_meta_algorithms():
        return [PerPixBest(), PerPixMean(), PerPixMedianDiff(), PerPixMedianDisp()]


class PerPixMean(MetaAlgorithm):

    def __init__(self, file_name="per_pix_mean", display_name="PerPixMean", color=(0.6, 0.6, 0.6), **kwargs):
        super(PerPixMean, self).__init__(file_name=file_name, display_name=display_name, color=color, **kwargs)

    def compute_meta_results(self, scenes, algorithms):
        for idx_s, scene in enumerate(scenes):
            # average runtime
            runtimes = misc.get_runtimes(scene, algorithms)
            misc.save_runtime(np.mean(runtimes), scene, self)

            # average disparity estimate per pixel
            algo_results = misc.get_algo_results(scene, algorithms)
            misc.save_algo_result(np.ma.average(algo_results, axis=2), scene, self)


class PerPixMedianDisp(MetaAlgorithm):

    def __init__(self, file_name="per_pix_median_disp", display_name="PerPixMedianDisp", color=(0.4, 0.4, 0.4),  **kwargs):
        super(PerPixMedianDisp, self).__init__(file_name=file_name, display_name=display_name, color=color, **kwargs)

    def compute_meta_results(self, scenes, algorithms):
        for idx_s, scene in enumerate(scenes):
            # median runtime
            runtimes = misc.get_runtimes(scene, algorithms)
            misc.save_runtime(np.median(runtimes), scene, self)

            # median disparity estimate per pixel
            algo_results = misc.get_algo_results(scene, algorithms)
            misc.save_algo_result(np.ma.median(algo_results, axis=2), scene, self)


class PerPixMedianDiff(MetaAlgorithm):

    def __init__(self, file_name="per_pix_median_diff", display_name="PerPixMedian", color=(0.2, 0.2, 0.2), **kwargs):
        super(PerPixMedianDiff, self).__init__(file_name=file_name, display_name=display_name, color=color, **kwargs)

    def compute_meta_results(self, scenes, algorithms):
        for idx_s, scene in enumerate(scenes):
            h, w = scene.get_height(), scene.get_width()

            # best runtime
            runtimes = misc.get_runtimes(scene, algorithms)
            misc.save_runtime(np.median(runtimes), scene, self)

            # per pixel: disparity estimate of the algorithm with the median absolute error
            algo_results = misc.get_algo_results(scene, algorithms)
            gt_stacked = misc.get_stacked_gt(scene, len(algorithms))

            abs_diffs = np.ma.masked_array(np.abs(gt_stacked - algo_results), mask=misc.get_mask_invalid(algo_results))
            idx_sorted_diffs = np.argsort(abs_diffs)
            xx, yy = np.meshgrid(np.arange(w, dtype=np.int), np.arange(h, dtype=np.int))

            n_algos = len(algorithms)
            odd_algo_count = n_algos % 2

            if odd_algo_count:
                abs_diffs = abs_diffs[yy, xx, idx_sorted_diffs[:, :, n_algos / 2]]
            else:
                abs_diffs_1 = abs_diffs[yy, xx, idx_sorted_diffs[:, :, n_algos / 2]]
                abs_diffs_2 = abs_diffs[yy, xx, idx_sorted_diffs[:, :, n_algos / 2 - 1]]
                abs_diffs = 0.5 * (abs_diffs_1 + abs_diffs_2)

            abs_diff_median = gt_stacked[:, :, 0] - abs_diffs
            misc.save_algo_result(abs_diff_median, scene, self)


class PerPixBest(MetaAlgorithm):
    def __init__(self, file_name="per_pix_best", display_name="PerPixBest", color=(0., 0., 0.), **kwargs):
        super(PerPixBest, self).__init__(file_name=file_name, display_name=display_name, color=color, **kwargs)

    def compute_meta_results(self, scenes, algorithms):
        for idx_s, scene in enumerate(scenes):
            h, w = scene.get_height(), scene.get_width()

            # best runtime
            runtimes = misc.get_runtimes(scene, algorithms)
            misc.save_runtime(np.min(runtimes), scene, self)

            # best disparity estimate per pixel
            algo_results = misc.get_algo_results(scene, algorithms)
            gt_stacked = misc.get_stacked_gt(scene, len(algorithms))

            abs_diffs = np.ma.masked_array(np.abs(gt_stacked - algo_results), mask=misc.get_mask_invalid(algo_results))
            idx_best_algo_per_pix = np.asarray(np.ma.argmin(abs_diffs, axis=2), dtype=np.int)
            xx, yy = np.meshgrid(np.arange(w, dtype=np.int), np.arange(h, dtype=np.int))
            best_disp_per_pixel = algo_results[yy, xx, idx_best_algo_per_pix]

            misc.save_algo_result(best_disp_per_pixel, scene, self)