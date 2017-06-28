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
    parser = OptionParser([FigureOpsCVPR17()])
    scene_overview, bad_pix_series, normals_overview, high_accuracy_overview = parser.parse_args()

    # delay imports to speed up usage response
    from utils import misc, file_io
    from utils.logger import log
    from evaluations import cvprw_2017_figures
    import settings
    from algorithms import Algorithm

    # prepare scenes
    benchmark_scenes = sorted(misc.get_stratified_scenes()) + \
                       sorted(misc.get_training_scenes()) + \
                       sorted(misc.get_test_scenes())

    # prepare algorithms
    fnames_baseline_algos = ["epi1", "epi2", "lf", "mv", "lf_occ26"]
    fnames_challenge_participants = ["ober", "omg_occ", "ps_rf25", "rm3de", "sc_gc", "spo_lf4cv", "zctv1"]
    fnames_other_submissions = ["obercross", "ofsy_330dnr2"]

    meta_data = file_io.read_file(op.join(settings.ALGO_PATH, "meta_data.json"))
    baseline_algorithms = sorted(Algorithm.initialize_algorithms(meta_data, fnames_baseline_algos, is_baseline=True))
    challenge_algorithms = sorted(Algorithm.initialize_algorithms(meta_data, fnames_challenge_participants))
    other_algorithms = sorted(Algorithm.initialize_algorithms(meta_data, fnames_other_submissions))
    for a in other_algorithms:
        a.display_name = "'" + a.display_name

    all_benchmark_algorithms = sorted(baseline_algorithms) + sorted(other_algorithms) + sorted(challenge_algorithms)
    all_benchmark_algorithms = Algorithm.set_colors(all_benchmark_algorithms)

    if scene_overview:
        log.info("Creating scene overview figure.")
        cvprw_2017_figures.plot_scene_overview(benchmark_scenes)

    if bad_pix_series:
        log.info("Creating figures with BadPix series.")
        cvprw_2017_figures.plot_bad_pix_series(all_benchmark_algorithms, with_cached_scores=False)

    if normals_overview:
        log.info("Creating surface normal figure(s).")
        from scenes import Cotton
        cvprw_2017_figures.plot_normals_overview(all_benchmark_algorithms, [Cotton()])

    if  high_accuracy_overview:
        log.info("Creating high accuracy figure.")
        from scenes import Cotton, Boxes
        high_accuracy_algorithms = []
        for fname in ["ofsy_330dnr2", "zctv1", "obercross", "ober", "sc_gc", "spo_lf4cv", "rm3de", "ps_rf25"]:
            high_accuracy_algorithms.append([a for a in all_benchmark_algorithms if a.get_name() == fname][0])
        cvprw_2017_figures.plot_high_accuracy(high_accuracy_algorithms, [Cotton(), Boxes()])