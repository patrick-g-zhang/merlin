import numpy
import logging
import pdb
from multiprocessing.pool import ThreadPool as Pool
class   AcousticBase(object):
    def __init__(self, delta_win = [-0.5, 0.0, 0.5], acc_win = [1.0, -2.0, 1.0]):

        ### whether dynamic features are needed for each data stream
        self.compute_dynamic = {}
        self.file_number = 0
        self.data_stream_number = 0
        self.data_stream_list = []

        self.out_dimension = 0
        self.record_vuv    = False

        self.delta_win = delta_win
        self.acc_win   = acc_win

        self.logger = logging.getLogger("acoustic_data")

    '''
    in_file_list_dict: if there are multiple acoustic features,
                       each feature has a key in the dict() and correspond to a list of file paths
    out_file_list_dict: merge all the input files

    three types of data:
        CMP     : the one used for HTS training
        DIY     : raw data without header, such the data to compose CMP files
        CMP_DIY : mix of CMP and DIY data
    '''
    def prepare_nn_data(self, in_file_list_dict, out_file_list, in_dimension_dict, out_dimension_dict):
        self.file_number = len(out_file_list)

        for data_stream_name in list(in_file_list_dict.keys()):

            try:
                assert len(in_file_list_dict[data_stream_name]) == self.file_number
            except AssertionError:
                self.logger.critical('file number of stream %s is different from others: %d %d' \
                                     %(data_stream_name, len(in_file_list_dict[data_stream_name]), self.file_number))
                raise

            try:
                assert data_stream_name in in_dimension_dict
            except AssertionError:
                self.logger.critical('data stream %s is missing in  the input dimension dict!' %(data_stream_name))
                raise

            try:
                assert data_stream_name in out_dimension_dict
            except AssertionError:
                self.logger.critical('data stream %s is missing in  the output dimension dict!' %(data_stream_name))
                raise

            ## we assume static+delta+delta-delta
            if out_dimension_dict[data_stream_name] == 3 * in_dimension_dict[data_stream_name]:
                self.compute_dynamic[data_stream_name] = True
            elif out_dimension_dict[data_stream_name] == in_dimension_dict[data_stream_name]:
                self.compute_dynamic[data_stream_name] = False
            else:
                self.logger.critical('output dimension of stream %s should be equal to or three times of input dimension: %d %d'
                                     %(data_stream_name, out_dimension_dict[data_stream_name], in_dimension_dict[data_stream_name]))
                raise

            self.data_stream_list.append(data_stream_name)

        self.data_stream_number = len(self.data_stream_list)

        if 'vuv' in out_dimension_dict:
            self.record_vuv = True

            if not ('lf0' in in_dimension_dict or 'F0' in in_dimension_dict):
                self.logger.critical("if voiced and unvoiced information are to be recorded, the 'lf0' information must be provided")
                raise

        for data_stream_name in list(out_dimension_dict.keys()):
            self.out_dimension += out_dimension_dict[data_stream_name]

        ### merge the data: like the cmp file
        # pdb.set_trace()
        self.prepare_data(in_file_list_dict, out_file_list, in_dimension_dict, out_dimension_dict)

    ### the real function to do the work
    ### need to be implemented for a specific format
    def prepare_data(self, in_file_list_dict, out_file_list, in_dimension_dict, out_dimension_dict):
        pass

    ### interpolate F0, if F0 has already been interpolated, nothing will be changed after passing this function
    def interpolate_f0(self, data):

        data = numpy.reshape(data, (data.size, 1))

        vuv_vector = numpy.zeros((data.size, 1))
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0

        ip_data = data

        frame_number = data.size
        last_value = 0.0
        for i in range(frame_number):
            if data[i] <= 0.0:
                j = i+1
                for j in range(i+1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number-1:
                    if last_value > 0.0:
                        step = (data[j] - data[i-1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i-1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]

        return  ip_data, vuv_vector

#        delta_win = [-0.5, 0.0, 0.5]
#        acc_win   = [1.0, -2.0, 1.0]
    def compute_dynamic_vector(self, vector, dynamic_win, frame_number):

        vector = numpy.reshape(vector, (frame_number, 1))

        win_length = len(dynamic_win)
        win_width = int(win_length/2)
        temp_vector = numpy.zeros((frame_number + 2 * win_width, 1))
        delta_vector = numpy.zeros((frame_number, 1))

        temp_vector[win_width:frame_number+win_width] = vector
        for w in range(win_width):
            temp_vector[w, 0] = vector[0, 0]
            temp_vector[frame_number+win_width+w, 0] = vector[frame_number-1, 0]

        for i in range(frame_number):
            for w in range(win_length):
                delta_vector[i] += temp_vector[i+w, 0] * dynamic_win[w]

        return  delta_vector

    ### compute dynamic features for a data matrix
    def compute_dynamic_matrix(self, data_matrix, dynamic_win, frame_number, dimension):
        dynamic_matrix = numpy.zeros((frame_number, dimension))

        ###compute dynamic feature dimension by dimension
        for dim in range(dimension):
            dynamic_matrix[:, dim:dim+1] = self.compute_dynamic_vector(data_matrix[:, dim], dynamic_win, frame_number)

        return  dynamic_matrix
