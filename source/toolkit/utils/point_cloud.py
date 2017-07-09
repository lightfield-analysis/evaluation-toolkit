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

from toolkit.utils import misc, file_io


def save(points, fpath):
    file_io.check_dir_for_fname(fpath)

    # remove all points with invalid entries
    mask_valid_points = np.sum(misc.get_mask_invalid(points), axis=1) == 0
    valid_points = points[mask_valid_points]

    points = ["%010f %010f %010f %d %d %d\n" % tuple(point) for point in valid_points]

    with open(fpath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % len(points))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        f.writelines(points)


def convert(scene, disp_map, color_map=None):
    height, width = np.shape(disp_map)
    assert (height == scene.height and width == scene.width), \
        "Expected the shape of the disparity map to be %dx%d but got %dx%d." \
        % (scene.height, scene.width, height, width)

    # compute x,y,z coordinates
    points = np.zeros((np.size(disp_map), 6))
    xx, yy = np.meshgrid(range(0, height), range(0, width))

    max_res = max(width, height)
    focus_dist_mm = scene.focus_dist_m * 1000
    b = scene.baseline_mm * scene.focal_length_mm * max_res

    zz = (b * focus_dist_mm) / (disp_map[:, :] * focus_dist_mm * scene.sensor_mm + b)
    xx = (xx / (height - 1.0) - 0.5) * scene.sensor_mm * zz / scene.focal_length_mm
    yy = (yy / (width - 1.0) - 0.5) * scene.sensor_mm * zz / scene.focal_length_mm

    # add coordinates
    points[:, 0] = xx.flatten()
    points[:, 1] = -yy.flatten()
    points[:, 2] = -zz.flatten()

    if color_map is not None:
        points[:, 3] = color_map[:, :, 0].flatten()
        points[:, 4] = color_map[:, :, 1].flatten()
        points[:, 5] = color_map[:, :, 2].flatten()

    return points
