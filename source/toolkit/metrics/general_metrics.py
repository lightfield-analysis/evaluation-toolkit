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
import os.path as op

import numpy as np

from toolkit import settings
from toolkit.utils import misc, plotting


class BaseMetric(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, vmin=0, vmax=1, colorbar_bins=1, cmap=settings.CMAP_ABS_ERROR,
                 eval_on_high_res=False):

        self.name = name
        self.mask_name = None
        self.category = settings.GENERAL_METRIC

        # default: evaluate on low/original scene resolution
        self.eval_on_high_res = eval_on_high_res

        # plotting properties
        self.vmin = vmin
        self.vmax = vmax

        self.colorbar_bins = colorbar_bins
        self.cmin = vmin
        self.cmax = vmax
        self.cmap = cmap

    def __hash__(self):
        return hash((self.get_id()))

    def __eq__(self, other):
        return self.get_id() == other.get_id()

    def __str__(self):
        return self.get_id()

    def __repr__(self):
        return self.get_id()

    @abc.abstractmethod
    def get_id(self):
        return

    # used for most figures and on website
    def get_display_name(self):
        return self.name

    # some figures and website features require a special, shorter name
    def get_short_name(self):
        return self.get_display_name()

    # used on website to explain the metric
    def get_description(self):
        return ""

    # displayed on website as overlay on metric visualization
    def get_legend(self):
        return ""

    def get_category(self):
        return self.category

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        # default mask: everything except for image boundary
        return scene.get_boundary_mask(ignore_boundary)

    def evaluate_on_high_resolution(self):
        return self.eval_on_high_res

    def evaluate_on_low_resolution(self):
        return not self.eval_on_high_res

    @staticmethod
    def format_score(score):
        return "%0.2f" % score

    def is_general(self):
        return self.category == settings.GENERAL_METRIC

    def mask_exists(self, scene, resolution):
        # general metrics don't require a mask file with a specific region
        if self.mask_name is None:
            return True
        fname = op.join(scene.data_path, "%s_%s.png" % (self.mask_name, resolution))
        return op.isfile(fname)

    def pixelize_results(self):
        return self.get_id().startswith(("mse", "badpix", "q"))


class BadPix(BaseMetric):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="BadPix", **kwargs):
        super(BadPix, self).__init__(name=name, **kwargs)
        self.thresh = thresh
        self.cmin = 0
        self.cmax = 1

    def get_id(self):
        return ("badpix_%0.3f" % self.thresh).replace(".", "")

    def get_display_name(self):
        if self.name == "BadPix":
            return "BadPix(%0.2f)" % self.thresh

        return self.name

    def get_short_name(self):
        return self.name

    def get_description(self):
        return "The percentage of pixels at the given mask " \
               "with abs(gt - algo) > %0.2f." % self.thresh

    def get_legend(self):
        return "green = good, red = bad"

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        m_bad_pix = self.get_bad_pix(algo_result - gt)
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(m_bad_pix)
        n_bad = np.sum(m_bad_pix[mask])
        score = misc.percentage(np.sum(mask), n_bad)

        if not with_visualization:
            return score

        m_bad_pix = plotting.adjust_binary_vis(m_bad_pix)
        vis = np.ma.masked_array(m_bad_pix, mask=~mask)
        return score, vis

    def get_bad_pix(self, diffs):
        with np.errstate(invalid="ignore"):
            m_bad_pix = np.abs(diffs) > self.thresh
        return m_bad_pix

    def get_score_from_diffs(self, diffs):
        if np.size(diffs) == 0:
            return np.nan

        m_bad_pix = self.get_bad_pix(diffs)
        return misc.percentage(np.size(diffs), np.sum(m_bad_pix))

    @staticmethod
    def format_score(score):
        return "%0.2f%%" % score


class MSE(BaseMetric):
    def __init__(self, factor=100, name="MSE", vmin=settings.DMIN, vmax=settings.DMAX,
                 cmap=settings.CMAP_ERROR, colorbar_bins=4, **kwargs):
        super(MSE, self).__init__(name=name, vmin=vmin, vmax=vmax,
                                  cmap=cmap, colorbar_bins=colorbar_bins, **kwargs)
        self.factor = factor

    def get_id(self):
        return "mse_%d" % self.factor

    def get_description(self):
        return "The mean squared error over all pixels " \
               "at the given mask, multiplied with %d." % self.factor

    def get_legend(self):
        return "white = correct, red = too far, blue = too close"

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        mask = self.get_evaluation_mask(scene) * \
               misc.get_mask_valid(algo_result) * \
               misc.get_mask_valid(gt)
        score = self.get_masked_score(algo_result, gt, mask)

        if not with_visualization:
            return score

        vis = np.ma.masked_array(gt - algo_result, mask=~mask)
        return score, vis

    def get_masked_score(self, algo_result, gt, mask):
        with np.errstate(invalid="ignore"):
            diff = np.square(gt - algo_result)
        return np.average(diff[mask]) * self.factor


class Quantile(BaseMetric):
    def __init__(self, percentage, factor=100, name="Quantile", vmin=0, vmax=0.5,
                 cmap=settings.CMAP_QUANTILE, colorbar_bins=5, **kwargs):
        super(Quantile, self).__init__(name=name, vmin=vmin, vmax=vmax,
                                       cmap=cmap, colorbar_bins=colorbar_bins, **kwargs)
        self.percentage = percentage
        self.factor = factor
        self.cmin = 0
        self.cmax = vmax

    def get_id(self):
        return "q_%d_%d" % (self.percentage, self.factor)

    def get_display_name(self):
        return "Q%d" % self.percentage

    def get_description(self):
        return "The %dth percentile of the disparity errors: " \
               "The maximum absolute disparity error of the best %d%% of pixels " \
               "for each algorithm, multiplied by 100." % (self.percentage, self.percentage)

    def get_legend(self):
        return "gray = errors above %dth percentile, " \
               "white/yellow = good, red = relatively bad" % self.percentage

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        diffs = np.abs(algo_result - gt) * self.factor
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(diffs) * misc.get_mask_valid(algo_result)
        sorted_diffs = np.sort(diffs[mask])
        idx = np.size(sorted_diffs) * self.percentage / 100.
        score = sorted_diffs[int(idx)]

        if not with_visualization:
            return score

        with np.errstate(invalid="ignore"):
            m_bad_pix = np.abs(diffs) > score
        vis = np.abs(diffs)
        vis[m_bad_pix] = -1
        vis = np.ma.masked_array(vis, mask=~mask)
        return score, vis

    @staticmethod
    def format_score(score):
        return "%0.2f%%" % score


class Runtime(BaseMetric):
    def __init__(self, log=False, name="Runtime", **kwargs):
        super(Runtime, self).__init__(name=name, **kwargs)
        self.log = log

    def get_id(self):
        if self.log:
            return "runtime_log"
        return "runtime"

    def get_description(self):
        if self.log:
            return "Decadic logarithm of the runtime in seconds as reported by the authors " \
                   "(hence runtime scores below 1 second are negative)."

        return "The runtime in seconds as reported by the authors."

    def get_display_name(self):
        display_name = self.name
        if self.log:
            display_name += " (log10)"
        return display_name

    def get_short_name(self):
        short_name = "Time"
        if self.log:
            short_name += " (log10)"
        return short_name

    def get_score(self, scene, algorithm):
        runtime = misc.get_runtime(algorithm, scene)
        if self.log:
            runtime = np.log10(runtime)
        return runtime

    def get_score_from_dir(self, scene, algo_dir):
        runtime = misc.get_runtime_from_dir(algo_dir, scene)
        if self.log:
            runtime = np.log10(runtime)
        return runtime
