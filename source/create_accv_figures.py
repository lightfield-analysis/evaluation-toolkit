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

import optparse


def parse_figure_options():
    parser = optparse.OptionParser()

    def get_comma_separated_args(option, opt, value, parser):
        values = [v for v in value.split(',')]
        setattr(parser.values, option.dest, values)

    parser.add_option("-a", type="string", action="callback", callback=get_comma_separated_args, dest="algo_names",
                      help="comma separated list of algorithm names, e.g. 'epi1,lf' default: ACCV baseline algorithms")

    # flags for different figures
    parser.add_option("--heatmaps", action="store_true", dest="heat", default=False,
                      help="Create figure with error heatmap per scene over all algorithms.")

    parser.add_option("--radar", action="store_true", dest="radar", default=False,
                      help="Create radar charts for stratified and training scenes.")

    parser.add_option("--stratified", action="store_true", dest="strat", default=False,
                      help="Create metric visualization figure for each stratified scene with all algorithms.")

    parser.add_option("--training", action="store_true", dest="train", default=False,
                      help="Create metric visualization figure for each training scene with all algorithms.")

    parser.add_option("--charts", action="store_true", dest="strat_charts", default=False,
                      help="Create charts along image dimensions of stratified scenes.")

    options, args = parser.parse_args()

    if not (options.heat or options.radar or options.strat or options.strat_charts or options.train):
        parser.print_help()
        parser.error("No figure option was selected.")

    # delay imports to speed up usage response
    import settings
    from utils import misc

    # keep only those algorithms that match with directory name in ALGO_RESULTS directory
    if options.algo_names is None:
        options.algo_names = settings.get_algo_names_accv_paper()
    algo_names = [a for a in options.algo_names if a in misc.get_available_algo_names()]

    return algo_names, options.heat, options.radar, options.strat, options.strat_charts, options.train


if __name__ == "__main__":
    algo_names, heatmaps, radar_charts, stratified, stratified_charts, training = parse_figure_options()

    # delay imports to speed up usage response
    from utils.logger import log
    from utils import misc
    import settings
    from scenes import BaseStratified, PhotorealisticScene, Backgammon, Pyramids, Dots
    log.info("Creating figures with algorithms: %s" % ", ".join(algo_names))

    if heatmaps:
        from evaluations import performance_heatmaps
        log.info("Creating heatmaps figure.")
        scenes = misc.get_stratified_scenes() + misc.get_training_scenes()
        performance_heatmaps.plot(algo_names, scenes)

    if radar_charts:
        log.info("Creating radar charts for stratified and training scenes.")
        BaseStratified.plot_radar_chart(algo_names)

        max_per_metric = [16, 40, 4, 4, 12, 80, 80, 6]
        PhotorealisticScene.plot_radar_chart(algo_names, misc.get_training_scenes(), max_per_metric)

    if stratified:
        for scene in misc.get_stratified_scenes():
            log.info("Processing scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algo_names, with_metric_vis=True)

    if training:
        for scene in misc.get_training_scenes():
            log.info("Processing scene: %s." % scene.get_display_name())
            scene.plot_algo_overview(algo_names)

    if stratified_charts:
        log.info("Creating special charts for stratified scenes.")
        Backgammon().plot_fattening_thinning(algo_names)
        Pyramids().plot_algo_disp_vs_gt_disp(algo_names)
        Dots().plot_error_vs_noise(algo_names)



