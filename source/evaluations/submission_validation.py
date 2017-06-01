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


import os.path as op

import numpy as np

from utils.logger import log
from utils import file_io, misc
import settings


def validate_extracted_submission(submission_dir, data_path=None):
    log.info("Validating extracted submission: %s." % submission_dir)

    scene_names = [s.get_name() for s in misc.get_benchmark_scenes(data_path=data_path)]
    exp_height, exp_width = settings.HEIGHT, settings.WIDTH
    errors = []

    # check disparity maps
    disp_maps_dir = op.normpath(op.join(submission_dir, settings.DISP_MAP_DIR))
    if not op.isdir(disp_maps_dir):
        errors.append('Could not find disparity map directory: "%s".' % settings.DISP_MAP_DIR)
    else:
        log.info("Validating disparity map files.")
        for scene_name in scene_names:
            fname_disp_map = op.join(disp_maps_dir, "%s.pfm" % scene_name)
            relative_fname_disp_map = op.join(settings.DISP_MAP_DIR, "%s.pfm" % scene_name)
            if not op.isfile(fname_disp_map):
                errors.append('Frame %s: Could not find disparity file: "%s".' % (scene_name, relative_fname_disp_map))
            else:
                try:
                    disp_map = file_io.read_pfm(fname_disp_map)
                    height, width = np.shape(disp_map)
                    if height != exp_height or width != exp_width:
                        errors.append("Frame %s, File %s: Resolution mismatch. Expected (%d, %d), got (%d, %d)."
                                      % (scene_name, relative_fname_disp_map, exp_height, exp_width, height, width))
                except file_io.PFMExeption as e:
                    errors.append("Frame %s, File %s, PFM Error: %s." % (scene_name, relative_fname_disp_map, e))

    # check runtimes
    runtimes_dir = op.normpath(op.join(submission_dir, "runtimes"))
    if not op.isdir(runtimes_dir):
        errors.append('Could not find runtimes directory: "%s".' % settings.RUNTIME_DIR)
    else:
        log.info("Validating runtime files.")
        for scene_name in scene_names:
            fname_runtime = op.join(runtimes_dir, "%s.txt" % scene_name)
            relative_fname_runtime = op.join(settings.RUNTIME_DIR, "%s.txt" % scene_name)
            if not op.isfile(fname_runtime):
                errors.append('Frame %s: Could not find runtime file: "%s".' % (scene_name, relative_fname_runtime))
            else:
                try:
                    file_io.read_runtime(fname_runtime)
                except IOError as error:
                    errors.append("Frame %s, File %s, Error: %s." % (scene_name, relative_fname_runtime, error))

    success = not errors
    error_json = {"messages": errors}

    if success:
        log.info("Validated submission successfully :)")
    return success, error_json


