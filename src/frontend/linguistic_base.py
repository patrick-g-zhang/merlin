import logging
import sys
from multiprocessing.pool import ThreadPool as Pool
import pdb

## a generic class of linguistic feature extraction
##
class LinguisticBase(object):
    def __init__(self, dimension=0):
        self.dimension = dimension  ##the feature dimensionality of output (should that read 'input' ?)

        ## the number of utterances to be normalised
        self.utterance_num = 0

    ## the ori_file_list contains the file paths of the raw linguistic data
    ## the output_file_list contains the file paths of the normalised linguistic data

    def perform_normalisation(self, ori_file_list, output_file_list, label_type="state_align", state_number=5, dur_file_list=None):

        logger = logging.getLogger("perform_normalisation")
        logger.info('perform linguistic feature extraction')
        self.utterance_num = len(ori_file_list)
        if self.utterance_num != len(output_file_list):
            logger.error('the number of input and output linguistic files should be the same!\n')
            sys.exit(1)

        def _perform_normalisation(i):
            # for i in range(self.utterance_num):
            if not dur_file_list:
                self.extract_linguistic_features(ori_file_list[i], output_file_list[i], label_type, state_number)
            else:
                self.extract_linguistic_features(ori_file_list[i], output_file_list[i], label_type, dur_file_list[i])

        # remove the paralleled mechanism for debug
        pool = Pool()
        pool.map(_perform_normalisation, range(self.utterance_num))
        pool.close()
        pool.join()

    ## the exact function to do the work
    ## need to be implemented in the specific class
    ## the function will write the linguistic features directly to the output file
    def extract_linguistic_features(self, in_file_name, out_file_name, label_type, dur_file_name=None):
        pass
