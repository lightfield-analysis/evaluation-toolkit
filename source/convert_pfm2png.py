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


from toolkit.utils.option_parser import OptionParser, ConverterOps


MIN = 0.
MAX = 255.


def main():
    parser = OptionParser([ConverterOps(input_help="path to pfm disparity map",
                                        output_help="path to png disparity map")])

    pfm_path, config_path, png_path = parser.parse_args()

    from toolkit.scenes import PhotorealisticScene
    from toolkit.utils import log, file_io
    import numpy as np

    scene = PhotorealisticScene("demo", path_to_config=config_path)

    disp_map = file_io.read_file(pfm_path)
    log.info("Input range: [%0.1f, %0.1f]" % (np.min(disp_map), np.max(disp_map)))

    # scale from [disp_min, disp_max] to [MIN, MAX]
    disp_map = (MAX - MIN) * (disp_map - scene.disp_min) / (scene.disp_max - scene.disp_min) + MIN
    log.info("Output range: [%0.1f, %0.1f]" % (np.min(disp_map), np.max(disp_map)))

    file_io.write_file(disp_map, png_path)


if __name__ == "__main__":
    main()
