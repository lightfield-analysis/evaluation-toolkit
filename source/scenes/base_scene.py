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


import ConfigParser
import abc
import os.path as op

import numpy as np

import settings
from metrics import Runtime, MSE, BadPix, Quantile
from utils import misc, file_io


class BaseScene(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, img_name, gt_scale=1, data_path=None, boundary_offset=15):
        self.img_name = img_name
        self.gt_scale = gt_scale
        self.boundary_offset = boundary_offset  # how many pixels to ignore on each side during evaluation on gt_scale=1

        if data_path is None:
            data_path = settings.DATA_PATH
        self.data_path = op.join(*[data_path, self.get_type(), self.get_name()])

        # set scene params from file
        with open(op.join(self.data_path, "parameters.cfg"), "r") as f:
            parser = ConfigParser.ConfigParser()
            parser.readfp(f)

            section = "intrinsics"
            self.original_width = int(parser.get(section, 'image_resolution_x_px'))
            self.original_height = int(parser.get(section, 'image_resolution_y_px'))
            self.focal_length_mm = float(parser.get(section, 'focal_length_mm'))
            self.sensor_size_mm = float(parser.get(section, 'sensor_size_mm'))

            section = "extrinsics"
            self.num_cams_x = int(parser.get(section, 'num_cams_x'))
            self.num_cams_y = int(parser.get(section, 'num_cams_y'))
            self.baseline_mm = float(parser.get(section, 'baseline_mm'))
            self.focus_distance_m = float(parser.get(section, 'focus_distance_m'))

            section = "meta"
            self.disp_min = float(parser.get(section, 'disp_min'))
            self.disp_max = float(parser.get(section, 'disp_max'))
            self.highres_scale = float(parser.get(section, 'depth_map_scale'))

    # ----------------------------------------------------------
    # getter for simple scene attributes
    # ----------------------------------------------------------

    def get_name(self):
        """This name corresponds to the file name."""
        return self.img_name

    def get_display_name(self):
        """You may choose a different name to be displayed on figures etc."""
        return self.get_name().title()

    def get_prefix(self):
        return "%s_%s" % (self.get_type(), self.get_name())

    def get_width(self):
        return int(self.original_width * self.gt_scale)

    def get_height(self):
        return int(self.original_height * self.gt_scale)

    def get_boundary_offset(self):
        return int(self.boundary_offset * self.gt_scale)

    def get_shape(self):
        return self.get_height(), self.get_width()

    def get_center_cam(self):
        return int(self.num_cams_x * self.num_cams_y / 2.0)

    def get_data_path(self):
        return self.data_path

    # ----------------------------------------------------------
    # getter for scene data with appropriate scale
    # ----------------------------------------------------------

    def get_center_view(self):
        fname = "input_Cam%03d.png" % self.get_center_cam()
        center_view = file_io.read_file(op.join(self.data_path, fname))
        if self.gt_scale != 1.0:
            center_view = misc.resize_to_shape(center_view, self.get_height(), self.get_width(), order=0)
        return center_view

    def get_gt(self):
        return self._get_data("gt_disp", "pfm")

    def get_mask(self, mask_name, binary=True):
        mask = self._get_data(mask_name, "png")
        if binary:
            mask = np.asarray(mask, dtype=np.bool)
        return mask

    def _get_data(self, descr, file_type, order=0):
        if self.gt_scale == 1:
            # original lowres version if gt_scale is 1
            fname = "%s_%s.%s" % (descr, settings.LOWRES, file_type)
            data = file_io.read_file(op.join(self.data_path, fname))
        else:
            # original highres version if gt_scale matches highres_scale
            fname = "%s_%s.%s" % (descr, settings.HIGHRES, file_type)
            data = file_io.read_file(op.join(self.data_path, fname))

            # otherwise scale highres version to required shape
            if self.gt_scale != self.highres_scale:
                data = misc.resize_to_shape(data, self.get_height(), self.get_width(), order=order)

        return data

    def get_boundary_mask(self, ignore_boundary=True):
        if ignore_boundary:
            mask = np.full((self.get_height(), self.get_width()), fill_value=0, dtype=np.bool)
            f_offset = self.get_boundary_offset()
            mask[f_offset:self.get_height()-f_offset, f_offset:self.get_width()-f_offset] = True
        else:
            mask = np.full((self.get_height(), self.get_width()), fill_value=1, dtype=np.bool)
        return mask

    # ----------------------------------------------------------
    # scene type utilities
    # ----------------------------------------------------------

    def hidden_gt(self):
        return False

    def is_test(self):
        return False

    @staticmethod
    def get_benchmark_scenes(gt_scale=1.0, data_path=None):
        return BaseScene.get_test_scenes(gt_scale, data_path) + \
               BaseScene.get_training_scenes(gt_scale, data_path) + \
               BaseScene.get_stratified_scenes(gt_scale, data_path)

    @staticmethod
    def get_stratified_scenes(gt_scale=1.0, data_path=None):
        from scenes import BaseStratified
        return BaseScene.set_scales(BaseStratified.get_scenes(data_path), gt_scale)

    @staticmethod
    def get_training_scenes(gt_scale=1.0, data_path=None):
        from scenes import BaseTraining
        return BaseScene.set_scales(BaseTraining.get_scenes(data_path), gt_scale)

    @staticmethod
    def get_test_scenes(gt_scale=1.0, data_path=None):
        from scenes import BaseTest
        return BaseScene.set_scales(BaseTest.get_scenes(data_path), gt_scale)

    @staticmethod
    def get_additional_scenes(gt_scale=1.0, data_path=None):
        from scenes import BaseAdditional
        return BaseScene.set_scales(BaseAdditional.get_scenes(data_path), gt_scale)

    # ----------------------------------------------------------
    # scene specific metric utilities
    # ----------------------------------------------------------

    @staticmethod
    def get_general_metrics():
        return [MSE(), BadPix(), Quantile(25)]

    @staticmethod
    def get_all_metrics_wo_runtime():
        from scenes import BaseStratified, BasePhotorealistic
        all_without_runtime = BaseScene.get_general_metrics() + \
                              BaseStratified.get_stratified_metrics() + \
                              BasePhotorealistic.get_region_metrics()
        return all_without_runtime

    @staticmethod
    def get_all_metrics():
        return BaseScene.get_all_metrics_wo_runtime() + [Runtime(log=10)]

    @staticmethod
    def set_scales(scenes, gt_scale):
        for scene in scenes:
            scene.gt_scale = gt_scale
        return scenes

    def set_high_gt_scale(self):
        self.gt_scale = 10.0

    def set_low_gt_scale(self):
        self.gt_scale = 1.0

    # ----------------------------------------------------------
    # scores
    # ----------------------------------------------------------

    def get_scores(self, algo_names, metric):
        scores = np.full(len(algo_names), fill_value=np.nan)
        gt = self.get_gt()

        for idx_a, algo_name in enumerate(algo_names):
            algo_result = misc.get_algo_result(self, algo_name)
            if "runtime" in metric.get_identifier():
                scores[idx_a] = metric.get_score(self, algo_name)
            else:
                scores[idx_a] = metric.get_score(algo_result, gt, self)

        return scores

    @staticmethod
    def get_average_scores(algo_names, metric, scenes):
        if len(scenes) == 0:
            return np.full((1, len(algo_names)), fill_value=np.nan)
        scores = np.full((len(scenes), len(algo_names)), fill_value=np.nan)

        for idx_s, scene in enumerate(scenes):
            scores[idx_s, :] = scene.get_scores(algo_names, metric)

        avg_scores = np.ma.average(np.ma.masked_array(scores, mask=misc.get_mask_invalid(scores)), axis=0)
        return avg_scores

