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

SUBDIR = "paper_cvprw_2017"

if __name__ == "__main__":
    parser = OptionParser([FigureOpsCVPR17()])
    figure_options = parser.parse_args()

    # delay imports to speed up usage response
    from utils import misc, file_io
    from utils.logger import log
    from evaluations import paper_cvprw_2017
    import settings
    from algorithms import Algorithm

    # prepare scenes
    benchmark_scenes = misc.get_stratified_scenes() + misc.get_training_scenes() + misc.get_test_scenes()

    # prepare algorithms
    fnames_baseline_algos = ["epi1", "epi2", "lf", "mv", "lf_occ26"]
    fnames_challenge_participants = ["ober", "omg_occ", "ps_rf25", "rm3de", "sc_gc", "spo_lf4cv", "zctv1"]
    fnames_other_submissions = ["obercross", "ofsy_330dnr2"]

    meta_data = file_io.read_file(op.join(settings.ALGO_PATH, "meta_data.json"))
    baseline_algorithms = sorted(Algorithm.initialize_algorithms(meta_data, fnames_baseline_algos, is_baseline=True))
    challenge_algorithms = sorted(Algorithm.initialize_algorithms(meta_data, fnames_challenge_participants))
    other_algorithms = sorted(Algorithm.initialize_algorithms(meta_data, fnames_other_submissions))

    # tag non-challenge algorithms with '
    for a in other_algorithms:
        a.display_name = "'" + a.display_name

    all_benchmark_algorithms = baseline_algorithms + other_algorithms + challenge_algorithms
    all_benchmark_algorithms = Algorithm.set_colors(all_benchmark_algorithms)

    # create figures

    if "scenes" in figure_options:
        log.info("Creating scene overview figure.")
        paper_cvprw_2017.plot_benchmark_scene_overview(benchmark_scenes, subdir=SUBDIR)

    if "difficulty" in figure_options:
        log.info("Creating scene difficulty figure.")
        if settings.USE_TEST_SCENE_GT:
            scenes = benchmark_scenes
        else:
            scenes = misc.get_stratified_scenes() + misc.get_training_scenes()
        paper_cvprw_2017.plot_scene_difficulty(scenes, subdir=SUBDIR)

    if "normalsdemo" in figure_options:
        log.info("Creating normals demo figure with Sideboard scene.")
        from scenes import PhotorealisticScene
        paper_cvprw_2017.plot_normals_explanation(PhotorealisticScene("sideboard"), Algorithm("epi1"), subdir=SUBDIR)

    if "radar" in figure_options:
        log.info("Creating radar charts.")
        paper_cvprw_2017.plot_radar_charts(all_benchmark_algorithms, subdir=SUBDIR)

    if "badpix" in figure_options:
        log.info("Creating figures with BadPix series.")
        paper_cvprw_2017.plot_bad_pix_series(all_benchmark_algorithms, with_cached_scores=False, subdir=SUBDIR)

    if "median" in figure_options:
        log.info("Creating median comparison figures.")
        paper_cvprw_2017.plot_median_comparisons(misc.get_stratified_scenes(), all_benchmark_algorithms, subdir=SUBDIR)
        paper_cvprw_2017.plot_median_comparisons(misc.get_training_scenes(), all_benchmark_algorithms, subdir=SUBDIR)

    if "normals" in figure_options:
        log.info("Creating surface normal figure with Cotton scene.")
        from scenes import PhotorealisticScene
        paper_cvprw_2017.plot_normals_overview(all_benchmark_algorithms, [PhotorealisticScene("cotton")], subdir=SUBDIR)

    if settings.USE_TEST_SCENE_GT and "discont" in figure_options:
        log.info("Creating discontinuity figure with Bicycle scene.")
        from scenes import PhotorealisticScene
        paper_cvprw_2017.plot_discont_overview(all_benchmark_algorithms, PhotorealisticScene("bicycle"), subdir=SUBDIR)

    if "accuracy" in figure_options:
        log.info("Creating high accuracy figure.")
        from scenes import PhotorealisticScene, PhotorealisticScene
        high_accuracy_algorithms = []
        for fname in ["ofsy_330dnr2", "zctv1", "obercross", "ober", "sc_gc", "spo_lf4cv", "rm3de", "ps_rf25"]:
            high_accuracy_algorithms.append([a for a in all_benchmark_algorithms if a.get_name() == fname][0])
        scenes = [PhotorealisticScene("cotton"), PhotorealisticScene("boxes")]
        paper_cvprw_2017.plot_high_accuracy(high_accuracy_algorithms, scenes, subdir=SUBDIR)