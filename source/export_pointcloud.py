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

from toolkit.utils.option_parser import OptionParser, SceneOps, AlgorithmOps, MetaAlgorithmOps


def main():
    parser = OptionParser([SceneOps(), AlgorithmOps(with_gt=True), MetaAlgorithmOps(default=[])])
    scenes, algorithms, meta_algorithms, compute_meta_algos = parser.parse_args()

    # delay imports to speed up usage response
    from toolkit import settings
    from toolkit.algorithms import MetaAlgorithm
    from toolkit.utils import log, misc, point_cloud

    if compute_meta_algos and meta_algorithms:
        MetaAlgorithm.prepare_meta_algorithms(meta_algorithms, algorithms, scenes)

    algorithms += meta_algorithms

    for scene in scenes:
        center_view = scene.get_center_view()

        for algorithm in algorithms:
            if algorithm.get_name() == "gt":
                disp_map = scene.get_gt()
            else:
                disp_map = misc.get_algo_result(algorithm, scene)

            log.info("Creating point cloud for scene '%s' with '%s' disparity map." %
                     (scene.get_name(), algorithm.get_name()))

            pc = point_cloud.convert(scene, disp_map, center_view)

            file_name = "%s_%s.ply" % (scene.get_name(), algorithm.get_name())
            file_path = op.join(settings.EVAL_PATH, "point_clouds", file_name)
            log.info("Saving point cloud to: %s" % file_path)
            point_cloud.save(pc, file_path)


if __name__ == "__main__":
    main()
