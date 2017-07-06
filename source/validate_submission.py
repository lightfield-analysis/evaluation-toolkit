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


import argparse
import os
import os.path as op
import shutil

from utils import log


def run_validation(path_to_submission):
    is_unpacked = op.isdir(path_to_submission)

    try:
        if is_unpacked:
            path_to_unpacked_submission = path_to_submission
        else:
            # unpack zip archive
            from utils.file_io import unzip
            tmp_dir = op.normpath(op.join(os.getcwd(), "../tmp"))
            try:
                log.info("Extracting archive.")
                path_to_unpacked_submission = op.join(tmp_dir, op.splitext(op.basename(path_to_submission))[0])
                unzip(path_to_submission, path_to_unpacked_submission)
            except IOError as e:
                log.error('Zip Error: %s.\nTerminated submission validation.' % e)

        # validate submission
        success, error_json = submission_validation.validate_extracted_submission(path_to_unpacked_submission)

        # report results
        print_validation_results(success, error_json)

    finally:
        # clean up
        if not is_unpacked and op.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)


def print_validation_results(success, error_json):
    if success:
        log.info("Yeah :) Congratulations, your submission archive is valid. Go ahead and submit it online!")
    else:
        error_messages = error_json["messages"]
        log.info('Validation found %d error(s).' % len(error_messages))
        log.info('A detailed format description can be found in the SUBMISSION_INSTRUCTIONS. '
                 '\nThe identified problems are: ')
        for error in error_messages:
            log.error(error)


def parse_submission_validation_options():
    parser = argparse.ArgumentParser()
    action = parser.add_argument(type=str, dest="fname_submission",
                                 help='path to the submission (zip archive or extracted submission)')

    namespace = parser.parse_args()
    fname_submission = getattr(namespace, action.dest)

    # check path to submission
    fname_submission = op.abspath(fname_submission)
    if fname_submission.endswith(".zip"):
        if not op.isfile(fname_submission):
            parser.error("Could not find file: %s" % fname_submission)
    else:
        if not op.isdir(fname_submission):
            parser.error("Could not find directory: %s" % fname_submission)

    return fname_submission


if __name__ == "__main__":
    path_to_submission = parse_submission_validation_options()

    # delay imports to speed up usage response
    from evaluations import submission_validation
    run_validation(path_to_submission)
