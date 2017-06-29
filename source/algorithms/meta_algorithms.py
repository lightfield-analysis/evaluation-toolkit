from utils import misc
from algorithms import Algorithm

import numpy as np


class PerPixMean(Algorithm):

    def __init__(self, file_name="per_pix_mean", display_name="PerPixMean", is_meta=True, **kwargs):
        super(PerPixMean, self).__init__(file_name=file_name, display_name=display_name, is_meta=is_meta, **kwargs)

    def compute_meta_results(self, scenes, algorithms):
        for idx_s, scene in enumerate(scenes):
            # average runtime
            runtimes = misc.get_runtimes(scene, algorithms)
            misc.save_runtime(np.mean(runtimes), scene, self)

            # average disparity estimate per pixel
            algo_results = misc.get_algo_results(scene, algorithms)
            misc.save_algo_result(np.ma.average(algo_results, axis=2), scene, self)


class PerPixMedianDisp(Algorithm):

    def __init__(self, file_name="per_pix_median_disp", display_name="PerPixMedianDisp", is_meta=True, **kwargs):
        super(PerPixMedianDisp, self).__init__(file_name=file_name, display_name=display_name, is_meta=is_meta, **kwargs)

    def compute_meta_results(self, scenes, algorithms):
        for idx_s, scene in enumerate(scenes):
            # median runtime
            runtimes = misc.get_runtimes(scene, algorithms)
            misc.save_runtime(np.median(runtimes), scene, self)

            # median disparity estimate per pixel
            algo_results = misc.get_algo_results(scene, algorithms)
            misc.save_algo_result(np.ma.median(algo_results, axis=2), scene, self)


class PerPixMedianDiff(Algorithm):

    def __init__(self, file_name="per_pix_median_diff", display_name="PerPixMedian", is_meta=True, **kwargs):
        super(PerPixMedianDiff, self).__init__(file_name=file_name, display_name=display_name, is_meta=is_meta, **kwargs)

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


class PerPixBest(Algorithm):
    def __init__(self, file_name="per_pix_best", display_name="PerPixBest", is_meta=True, **kwargs):
        super(PerPixBest, self).__init__(file_name=file_name, display_name=display_name, is_meta=is_meta, **kwargs)

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