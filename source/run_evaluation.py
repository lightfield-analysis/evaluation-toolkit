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
    parser = OptionParser([SceneOps(), AlgorithmOps(), MetricOps(), VisualizationOps(), MetaAlgorithmOps(default=[])])
    scenes, algorithms, metrics, visualize, meta_algorithms, load_meta_algorithm_files = parser.parse_args()

    # delay import to speed up usage response
    from algorithms import MetaAlgorithm
    from evaluations import submission_evaluation
    import settings
    from utils import misc

    if not load_meta_algorithm_files and meta_algorithms:
        MetaAlgorithm.prepare_meta_algorithms(meta_algorithms, algorithms, scenes)
        algorithms += meta_algorithms

    for algorithm in algorithms:
        submission_evaluation.evaluate(scenes=scenes,
                                       metrics=metrics,
                                       visualize=visualize,
                                       ground_truth_path=settings.DATA_PATH,
                                       evaluation_output_path=op.join(settings.ALGO_EVAL_PATH, algorithm.get_name()),
                                       algorithm_input_path=misc.get_path_to_algo_data(algorithm))