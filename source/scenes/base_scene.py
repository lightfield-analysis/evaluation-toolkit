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
from scipy import signal as ssig

import settings
from utils import misc, file_io


class BaseScene(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, gt_scale=1, data_path=None, category="other",
                 boundary_offset=15, display_name=None):

        self.name = name  # corresponds to the file name
        if display_name is None:  # is used for figures etc.
            self.display_name = name.title()

        self.category = category
        self.gt_scale = gt_scale
        self.boundary_offset = boundary_offset  # how many pixels to ignore on each side during evaluation on gt_scale=1

        if data_path is None:
            data_path = settings.DATA_PATH
        self.data_path = op.join(*[data_path, self.get_category(), self.get_name()])

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

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.get_name()

    # ----------------------------------------------------------
    # getter for simple scene attributes
    # ----------------------------------------------------------

    def get_category(self):
        return self.category

    def get_name(self):
        """This name corresponds to the file name."""
        return self.name

    def get_display_name(self):
        """You may choose a different name to be displayed on figures etc."""
        return self.display_name

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

    def hidden_gt(self):
        return self.is_test()

    def is_test(self):
        return self.category == settings.TEST_SCENE

    def is_stratified(self):
        return self.category == settings.STRATIFIED_SCENE

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

    def disp2depth(self, disp_map):
        q = self.baseline_mm * self.focal_length_mm * max(self.original_width, self.original_height)
        depth_map = 1.0 / ((1000.0 * self.sensor_size_mm) * disp_map / q + (1.0 / self.focus_distance_m))
        return depth_map

    def get_depth_normals(self, depth_map):
        h, w = np.shape(depth_map)
        zz = depth_map
        xx, yy = np.meshgrid(range(0, h), range(0, w))
        xx = (xx / (h - 1.0) * 0.5) * self.sensor_size_mm * zz / self.focal_length_mm
        yy = (yy / (w - 1.0) * 0.5) * self.sensor_size_mm * zz / self.focal_length_mm

        kernel = np.asarray([[3., 10., 3.], [0., 0., 0.], [-3., -10., -3.]])
        kernel /= 64.

        dxdx = ssig.convolve2d(xx, kernel, mode="same", boundary="wrap")
        dydx = ssig.convolve2d(yy, kernel, mode="same", boundary="wrap")
        dzdx = ssig.convolve2d(zz, kernel, mode="same", boundary="wrap")

        dxdy = ssig.convolve2d(xx, np.transpose(kernel), mode="same", boundary="wrap")
        dydy = ssig.convolve2d(yy, np.transpose(kernel), mode="same", boundary="wrap")
        dzdy = ssig.convolve2d(zz, np.transpose(kernel), mode="same", boundary="wrap")

        normal_map = np.full((h, w, 3), fill_value=np.nan)

        normal_map[:, :, 0] = (dzdx * dxdy - dxdx * dzdy)
        normal_map[:, :, 1] = - (dydx * dzdy - dzdx * dydy)
        normal_map[:, :, 2] = - (dxdx * dydy - dydx * dxdy)

        magnitude = np.sqrt(np.sum(np.square(normal_map), axis=2))
        normal_map = normal_map / np.dstack((magnitude, magnitude, magnitude))

        return normal_map

    def get_normal_vis_from_disp_map(self, disp_map):
        return (self.get_depth_normals(self.disp2depth(disp_map)) + 1.) * .5

    # ----------------------------------------------------------
    # setter
    # ----------------------------------------------------------

    def set_high_gt_scale(self):
        self.gt_scale = 10.0

    def set_low_gt_scale(self):
        self.gt_scale = 1.0

    # ----------------------------------------------------------
    # scores
    # ----------------------------------------------------------

    def get_scores(self, algorithms, metric):
        scores = np.full(len(algorithms), fill_value=np.nan)
        gt = self.get_gt()

        for idx_a, algorithm in enumerate(algorithms):
            algo_result = misc.get_algo_result(self, algorithm)
            if "runtime" in metric.get_identifier():
                scores[idx_a] = metric.get_score(self, algorithm)
            else:
                scores[idx_a] = metric.get_score(algo_result, gt, self)

        return scores

    @staticmethod
    def get_average_scores(algorithms, metric, scenes):
        if len(scenes) == 0:
            return np.full((1, len(algorithms)), fill_value=np.nan)
        scores = np.full((len(scenes), len(algorithms)), fill_value=np.nan)

        for idx_s, scene in enumerate(scenes):
            scores[idx_s, :] = scene.get_scores(algorithms, metric)

        avg_scores = np.ma.average(np.ma.masked_array(scores, mask=misc.get_mask_invalid(scores)), axis=0)
        return avg_scores

