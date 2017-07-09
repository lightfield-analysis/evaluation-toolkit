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


from toolkit.utils.option_parser import OptionParser, FigureOpsCVPR17


SUBDIR = "paper_cvprw_2017"
USE_TEST_SCENE_GT = True


def main():
    figure_options = OptionParser([FigureOpsCVPR17()]).parse_args()

    # delay imports to speed up usage response
    from toolkit.algorithms import Algorithm, PerPixBest
    from toolkit.evaluations import paper_cvprw_2017 as cvprw
    from toolkit.utils import log, misc
    from toolkit.scenes import PhotorealisticScene

    # prepare scenes
    if USE_TEST_SCENE_GT:
        benchmark_scenes = misc.get_benchmark_scenes()
    else:
        benchmark_scenes = misc.get_stratified_scenes() + misc.get_training_scenes()

    # prepare algorithms
    fnames_baseline_algos = ["epi1", "epi2", "lf", "mv", "lf_occ26"]
    fnames_challenge_algos = ["ober", "omg_occ", "ps_rf25", "rm3de", "sc_gc", "spo_lf4cv", "zctv1"]
    fnames_other_submissions = ["obercross", "ofsy_330dnr2"]

    baseline_algorithms = Algorithm.initialize_algorithms(fnames_baseline_algos, is_baseline=True)
    challenge_algorithms = Algorithm.initialize_algorithms(fnames_challenge_algos)
    other_algorithms = Algorithm.initialize_algorithms(fnames_other_submissions)

    for algorithm in other_algorithms:
        algorithm.display_name = "'" + algorithm.display_name

    algorithms = sorted(baseline_algorithms) + sorted(other_algorithms) + sorted(challenge_algorithms)
    algorithms = Algorithm.set_colors(algorithms)

    # create figures

    if "normalsdemo" in figure_options:
        log.info("Creating normals demo figure with Sideboard scene.")
        scene = PhotorealisticScene("sideboard")
        cvprw.plot_normals_explanation(Algorithm("epi1"), scene, subdir=SUBDIR)

    if "radar" in figure_options:
        log.info("Creating radar charts.")
        cvprw.plot_radar_charts(algorithms, subdir=SUBDIR)

    if "badpix" in figure_options:
        log.info("Creating figures with BadPix series.")
        per_pix_best = PerPixBest()
        per_pix_best.compute_meta_results(algorithms, benchmark_scenes)
        cvprw.plot_bad_pix_series(algorithms+[per_pix_best], USE_TEST_SCENE_GT, subdir=SUBDIR)

    if "median" in figure_options:
        log.info("Creating median comparison figures.")
        cvprw.plot_median_diffs(algorithms, misc.get_stratified_scenes(), subdir=SUBDIR)
        cvprw.plot_median_diffs(algorithms, misc.get_training_scenes(), subdir=SUBDIR)

    if "normals" in figure_options:
        log.info("Creating surface normal figure with Cotton scene.")
        cvprw.plot_normal_maps(algorithms, PhotorealisticScene("cotton"), subdir=SUBDIR)

    if USE_TEST_SCENE_GT and "discont" in figure_options:
        log.info("Creating discontinuity figure with Bicycle scene.")
        cvprw.plot_discont_overview(algorithms, PhotorealisticScene("bicycle"), subdir=SUBDIR)

    if "accuracy" in figure_options:
        log.info("Creating high accuracy figure.")
        selection = ["ofsy_330dnr2", "zctv1", "obercross", "ober",
                     "sc_gc", "spo_lf4cv", "rm3de", "ps_rf25"]
        high_accuracy_algorithms = []
        # algorithms should be exactly in the order of 'selection'
        for algo_name in selection:
            high_accuracy_algorithms.append([a for a in algorithms if a.get_name() == algo_name][0])
        scenes = [PhotorealisticScene("cotton"), PhotorealisticScene("boxes")]
        cvprw.plot_high_accuracy(high_accuracy_algorithms, scenes, subdir=SUBDIR)

    if "scenes" in figure_options:
        log.info("Creating scene overview figure.")
        cvprw.plot_benchmark_scene_overview(misc.get_benchmark_scenes(), subdir=SUBDIR)

    if "difficulty" in figure_options:
        log.info("Creating scene difficulty figure.")
        cvprw.plot_scene_difficulty(benchmark_scenes, subdir=SUBDIR)

if __name__ == "__main__":
    main()
