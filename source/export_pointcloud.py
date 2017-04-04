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


import optparse
import os.path as op


def parse_pointcloud_options():
    parser = optparse.OptionParser()
    parser.add_option("-s", type="string", dest="scene",
                      help="scene name (e.g. 'cotton')")
    parser.add_option("-a", type="string", dest="algo_name",
                      help="algorithm name (e.g. 'epi1'), default: ground truth")

    options, args = parser.parse_args()

    # delay imports to speed up usage response
    from utils import misc

    # scene
    scene_dict = misc.get_scene_dict()
    scene_options = ", ".join(sorted(scene_dict.keys()))
    if options.scene is None:
        parser.error("Scene name is required. Options: %s" % scene_options)
    else:
        scene = scene_dict.get(options.scene, None)
        if scene is None:
            parser.error("Could not match scene name: %s. Options: %s" % (options.scene, scene_options))

    # algorithm name
    if options.algo_name is not None:
        available_algo_names = misc.get_available_algo_names()
        if options.algo_name not in available_algo_names:
            parser.error("Could not match algorithm name. Options: %s" % (", ".join(sorted(available_algo_names))))

    return scene, options.algo_name


if __name__ == "__main__":
    scene, algo_name = parse_pointcloud_options()

    # delay imports to speed up usage response
    from utils.logger import log
    from utils import point_cloud, misc
    import settings

    if algo_name is None:
        algo_name = "gt"
        disp_map = scene.get_gt()
    else:
        disp_map = misc.get_algo_result(scene, algo_name)

    log.info("Creating point cloud for scene '%s' with '%s' disparity map." % (scene.get_name(), algo_name))
    pc = point_cloud.convert(scene, disp_map)

    fpath = op.join(*[settings.EVAL_PATH, "point_clouds", "%s_%s.ply" % (scene.get_name(), algo_name)])
    log.info("Saving point cloud to: %s" % fpath)
    point_cloud.save(pc, fpath)
