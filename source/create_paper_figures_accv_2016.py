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


from toolkit.utils.option_parser import OptionParser, AlgorithmOps, FigureOpsACCV16


SUBDIR = "paper_accv_2016"


def main():
    accv_algo_names = ["epi1", "epi2", "lf_occ", "lf", "mv"]
    parser = OptionParser([AlgorithmOps(default=accv_algo_names), FigureOpsACCV16()])

    algorithms, figure_options = parser.parse_args()

    # delay imports to speed up usage response
    from toolkit.evaluations import paper_accv_2016, error_heatmaps
    from toolkit.scenes import Backgammon, Dots, Pyramids, Stripes
    from toolkit.utils import log, misc

    if "heatmaps" in figure_options:
        log.info("Creating error heatmaps.")
        scenes = misc.get_stratified_scenes() + misc.get_training_scenes()
        error_heatmaps.plot(algorithms, scenes, subdir=SUBDIR)

    if "radar" in figure_options:
        log.info("Creating radar charts for stratified and training scenes.")
        paper_accv_2016.plot_radar_charts(algorithms, subdir=SUBDIR)

    if "backgammon" in figure_options:
        log.info("Creating special chart for backgammon scene.")
        Backgammon().plot_fattening_thinning(algorithms, subdir=SUBDIR)

    if "pyramids" in figure_options:
        log.info("Creating special chart for pyramids scene.")
        Pyramids().plot_algo_disp_vs_gt_disp(algorithms, subdir=SUBDIR)

    if "dots" in figure_options:
        log.info("Creating special chart for dots scene.")
        Dots().plot_error_vs_noise(algorithms, subdir=SUBDIR)

    if "stripes" in figure_options:
        log.info("Creating special chart for stripes scene.")
        Stripes().visualize_masks(subdir=SUBDIR)

    if "stratified" in figure_options:
        for scene in misc.get_stratified_scenes():
            log.info("Creating metric visualizations for scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algorithms, with_metric_vis=True, subdir=SUBDIR)

    if "training" in figure_options:
        for scene in misc.get_training_scenes():
            log.info("Creating metric visualizations for scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algorithms, subdir=SUBDIR)


if __name__ == "__main__":
    main()
