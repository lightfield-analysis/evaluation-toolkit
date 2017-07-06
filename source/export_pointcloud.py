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

from utils.option_parser import *


if __name__ == "__main__":
    parser = OptionParser([SceneOps(), AlgorithmOps(with_gt=True), MetaAlgorithmOps(default=[])])
    scenes, algorithms, meta_algorithms, load_meta_algorithm_files = parser.parse_args()

    # delay imports to speed up usage response
    from algorithms import MetaAlgorithm
    import settings
    from utils.logger import log
    from utils import point_cloud, misc

    if not load_meta_algorithm_files and meta_algorithms:
        MetaAlgorithm.prepare_meta_algorithms(meta_algorithms, algorithms, scenes)

    algorithms += meta_algorithms
    algorithm_names = [algorithm.get_name() for algorithm in algorithms]

    for scene in scenes:
        for algorithm in algorithms:
            if algorithm.get_name() == "gt":
                disp_map = scene.get_gt()
            else:
                disp_map = misc.get_algo_result(scene, algorithm)

            log.info("Creating point cloud for scene '%s' with '%s' disparity map." % (scene.get_name(), algorithm.get_name()))
            pc = point_cloud.convert(scene, disp_map)

            fpath = op.join(*[settings.EVAL_PATH, "point_clouds", "%s_%s.ply" % (scene.get_name(), algorithm.get_name())])
            log.info("Saving point cloud to: %s" % fpath)
            point_cloud.save(pc, fpath)
