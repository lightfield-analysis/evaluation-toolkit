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

from utils.option_parser import *

SUBDIR = "paper_accv_2016"

if __name__ == "__main__":
    default_accv_algorithms = ["epi1", "epi2", "lf_occ", "lf", "mv"]
    parser = OptionParser([AlgorithmOps(default=default_accv_algorithms), FigureOpsACCV16()])

    algorithms, heatmaps, radar_charts, stratified, training, stratified_charts = parser.parse_args()

    # delay imports to speed up usage response
    from utils.logger import log
    from utils import misc
    from scenes import Backgammon, Pyramids, Dots
    log.info("Creating figures with algorithms: %s" % algorithms)

    if heatmaps:
        log.info("Creating heatmaps figure.")
        from evaluations import error_heatmaps
        scenes = misc.get_stratified_scenes() + misc.get_training_scenes()
        error_heatmaps.plot(algorithms, scenes, subdir=SUBDIR)

    if radar_charts:
        log.info("Creating radar charts for stratified and training scenes.")
        from evaluations import paper_accv_2016
        paper_accv_2016.plot_radar_charts(algorithms, subdir=SUBDIR)

    if stratified:
        for scene in misc.get_stratified_scenes():
            log.info("Processing scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algorithms, with_metric_vis=True, subdir=SUBDIR)

    if training:
        for scene in misc.get_training_scenes():
            log.info("Processing scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algorithms, subdir=SUBDIR)

    if stratified_charts:
        log.info("Creating special charts for stratified scenes.")
        Backgammon().plot_fattening_thinning(algorithms, subdir=SUBDIR)
        Pyramids().plot_algo_disp_vs_gt_disp(algorithms, subdir=SUBDIR)
        Dots().plot_error_vs_noise(algorithms, subdir=SUBDIR)



