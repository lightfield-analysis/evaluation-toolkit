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

import settings
from utils import file_io, misc, plotting


class BaseMetric(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, vmin=0, vmax=1, colorbar_bins=1, cmap=settings.abs_error_cmap):
        self.name = name

        # plotting properties
        self.vmin = vmin
        self.vmax = vmax

        self.colorbar_bins = colorbar_bins
        self.cmin = vmin
        self.cmax = vmax
        self.cmap = cmap

    # used as identifier for evaluation results on website and for reading/writing temporary results
    @abc.abstractmethod
    def get_identifier(self):
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

    @staticmethod
    def evaluate_on_high_res():
        # default: evaluate on low resolution
        return False

    @staticmethod
    def format_score(score):
        return "%0.2f" % score

    def is_general(self):
        return self.category == settings.GENERAL

    # region methods
    def is_applicable_for_low_res_scene(self, scene):
        return not self.evaluate_on_high_res() and self.mask_exists(scene, settings.LOWRES)

    def is_applicable_for_high_res_scene(self, scene):
        return self.evaluate_on_high_res() and self.mask_exists(scene, settings.HIGHRES)

    def mask_exists(self, scene, resolution):
        fname = op.join(scene.data_path, "%s_%s.png" % (self.mask_name, resolution))
        return op.isfile(fname)

    def pixelize_results(self):
        return self.get_identifier().startswith(("mse", "badpix"))


class BadPix(BaseMetric):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="BadPix", **kwargs):
        super(BadPix, self).__init__(name=name, **kwargs)
        self.thresh = thresh
        self.category = settings.GENERAL
        self.cmin = 0
        self.cmax = 1

    def get_identifier(self):
        return ("badpix_%0.3f" % self.thresh).replace(".", "")

    def get_display_name(self):
        if self.name == "BadPix":
            return "BadPix(%0.2f)" % self.thresh
        return self.name

    def get_short_name(self):
        return self.name

    def get_description(self):
        return "The percentage of pixels at the given mask with abs(gt - algo) > %0.2f." % self.thresh

    def get_legend(self):
        return "green = good, red = bad"

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        m_bad_pix = self.get_bad_pix(algo_result - gt)
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(m_bad_pix)
        n_bad = np.sum(m_bad_pix[mask])
        score = misc.percentage(np.sum(mask), n_bad)

        if not with_visualization:
            return score
        else:
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

    def format_score(self, score):
        return "%0.2f%%" % score


class MSE(BaseMetric):
    def __init__(self, factor=100, name="MSE",
                 vmin=settings.DMIN, vmax=settings.DMAX, cmap=settings.error_cmap, colorbar_bins=4):
        super(MSE, self).__init__(name=name, vmin=vmin, vmax=vmax,
                                  cmap=cmap, colorbar_bins=colorbar_bins)
        self.factor = factor
        self.category = settings.GENERAL

    def get_identifier(self):
        return "mse_%d" % self.factor

    def get_description(self):
        return "The mean squared error over all pixels at the given mask, multiplied with %d." % self.factor

    def get_legend(self):
        return "white = correct, red = too far, blue = too close"

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(algo_result) * misc.get_mask_valid(gt)
        score = self.get_masked_score(algo_result, gt, mask)

        if not with_visualization:
            return score
        else:
            vis = np.ma.masked_array(gt - algo_result, mask=~mask)
            return score, vis

    def get_masked_score(self, algo_result, gt, mask):
        with np.errstate(invalid="ignore"):
            diff = np.square(gt - algo_result)
        return np.average(diff[mask]) * self.factor

    @staticmethod
    def format_score(score):
        return "%0.2f" % score


class Runtime(BaseMetric):
    def __init__(self, log=False, name="Runtime"):
        super(Runtime, self).__init__(name=name)
        self.log = log
        self.category = settings.GENERAL

    def get_identifier(self):
        if self.log:
            return "runtime_log"
        else:
            return "runtime"

    @staticmethod
    def get_description():
        return "The runtime in seconds as reported by the authors."

    def get_display_name(self):
        display_name = self.name
        if self.log:
            display_name += " (log10)"
        return display_name

    def get_short_name(self):
        display_name = "Time"
        if self.log:
            display_name += " (log10)"
        return display_name

    def get_score(self, scene, algo_name):
        runtime = misc.get_runtime(scene, algo_name)
        if self.log:
            runtime = np.log10(runtime)
        return runtime
