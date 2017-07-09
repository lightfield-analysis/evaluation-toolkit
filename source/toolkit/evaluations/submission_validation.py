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

from toolkit import settings
from toolkit.utils import log, misc, file_io


def validate_extracted_submission(submission_dir, data_path=None):
    log.info("Validating extracted submission: %s." % submission_dir)

    scene_names = [s.get_name() for s in misc.get_benchmark_scenes(data_path=data_path)]
    exp_height, exp_width = settings.HEIGHT, settings.WIDTH
    errors = []

    # check disparity maps
    disp_maps_dir = op.normpath(op.join(submission_dir, settings.DIR_NAME_DISP_MAPS))

    if not op.isdir(disp_maps_dir):
        errors.append('Could not find disparity map directory: "%s".' % settings.DIR_NAME_DISP_MAPS)
    else:
        log.info("Validating disparity map files.")

        for scene_name in scene_names:
            path_disp_maps = op.join(disp_maps_dir, "%s.pfm" % scene_name)
            relative_path_disp_maps = op.join(settings.DIR_NAME_DISP_MAPS, "%s.pfm" % scene_name)

            if not op.isfile(path_disp_maps):
                errors.append('Frame %s: Could not find disparity file: "%s".' %
                              (scene_name, relative_path_disp_maps))
            else:
                try:
                    disp_map = file_io.read_pfm(path_disp_maps)
                    height, width = np.shape(disp_map)

                    if height != exp_height or width != exp_width:
                        errors.append("Frame %s, File %s: Resolution mismatch. "
                                      "Expected (%d, %d), got (%d, %d)." %
                                      (scene_name, relative_path_disp_maps,
                                       exp_height, exp_width, height, width))

                except file_io.PFMExeption as e:
                    errors.append("Frame %s, File %s, PFM Error: %s." %
                                  (scene_name, relative_path_disp_maps, e))

    # check runtimes
    runtimes_dir = op.normpath(op.join(submission_dir, "runtimes"))

    if not op.isdir(runtimes_dir):
        errors.append('Could not find runtimes directory: "%s".' % settings.DIR_NAME_RUNTIMES)
    else:
        log.info("Validating runtime files.")

        for scene_name in scene_names:
            path_runtimes = op.join(runtimes_dir, "%s.txt" % scene_name)
            relative_path_runtimes = op.join(settings.DIR_NAME_RUNTIMES, "%s.txt" % scene_name)

            if not op.isfile(path_runtimes):
                errors.append('Frame %s: Could not find runtime file: "%s".' %
                              (scene_name, relative_path_runtimes))
            else:
                try:
                    file_io.read_runtime(path_runtimes)
                except IOError as error:
                    errors.append("Frame %s, File %s, Error: %s." %
                                  (scene_name, relative_path_runtimes, error))

    success = not errors
    error_json = {"messages": errors}

    if success:
        log.info("Validated submission successfully :)")
    return success, error_json
