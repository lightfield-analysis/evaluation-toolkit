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

from toolkit.utils.option_parser import OptionParser, SceneOps, AlgorithmOps, MetricOps, \
    VisualizationOps, MetaAlgorithmOps, OverwriteOps


def main():
    parser = OptionParser([SceneOps(), AlgorithmOps(), MetricOps(),
                           VisualizationOps(), OverwriteOps(), MetaAlgorithmOps(default=[])])
    scenes, algorithms, metrics, with_vis, add_to_existing, meta_algorithms, compute_meta_algos = parser.parse_args()

    # delay import to speed up usage response
    from toolkit import settings
    from toolkit.algorithms import MetaAlgorithm
    from toolkit.evaluations import submission_evaluation
    from toolkit.utils import misc

    if compute_meta_algos and meta_algorithms:
        MetaAlgorithm.prepare_meta_algorithms(meta_algorithms, algorithms, scenes)

    algorithms += meta_algorithms

    for algorithm in algorithms:
        evaluation_output_path = op.join(settings.ALGO_EVAL_PATH, algorithm.get_name())
        algorithm_input_path = misc.get_path_to_algo_data(algorithm)
        submission_evaluation.evaluate(scenes=scenes,
                                       metrics=metrics,
                                       visualize=with_vis,
                                       evaluation_output_path=evaluation_output_path,
                                       algorithm_input_path=algorithm_input_path,
                                       add_to_existing_results=add_to_existing)


if __name__ == "__main__":
    main()
