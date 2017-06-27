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

if __name__ == "__main__":
    accv_algorithms = ["epi1", "epi2", "lf_occ", "lf", "mv"]
    additional_help_text = 'use: "-a %s" for original ACCV algorithms' % " ".join(accv_algorithms)
    parser = OptionParser([AlgorithmOps(additional_help_text=additional_help_text), FigureOpsACCV()])

    algorithms, heatmaps, radar_charts, stratified, training, stratified_charts = parser.parse_args()

    # delay imports to speed up usage response
    from utils.logger import log
    from utils import misc
    from scenes import BaseStratified, PhotorealisticScene, Backgammon, Pyramids, Dots
    log.info("Creating figures with algorithms: %s" % algorithms)

    if heatmaps:
        from evaluations import performance_heatmaps
        log.info("Creating heatmaps figure.")
        scenes = misc.get_stratified_scenes() + misc.get_training_scenes()
        performance_heatmaps.plot(algorithms, scenes)

    if radar_charts:
        log.info("Creating radar charts for stratified and training scenes.")
        BaseStratified.plot_radar_chart(algorithms)
        max_per_metric = [16, 40, 4, 4, 12, 80, 80, 6]
        PhotorealisticScene.plot_radar_chart(algorithms, misc.get_training_scenes(), max_per_metric)

    if stratified:
        for scene in misc.get_stratified_scenes():
            log.info("Processing scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algorithms, with_metric_vis=True)

    if training:
        for scene in misc.get_training_scenes():
            log.info("Processing scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algorithms)

    if stratified_charts:
        log.info("Creating special charts for stratified scenes.")
        Backgammon().plot_fattening_thinning(algorithms)
        Pyramids().plot_algo_disp_vs_gt_disp(algorithms)
        Dots().plot_error_vs_noise(algorithms)



