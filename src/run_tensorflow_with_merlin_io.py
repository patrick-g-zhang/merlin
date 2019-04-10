import os
import sys
from tensorflow_lib import data_utils
import time
import tensorflow as tf
import pdb
import numpy as np
import logging
import configuration
from tensorflow_lib.train import TrainTensorflowModels


# from frontend.mlpg import get_static_features
class TensorflowClass(object):

    def __init__(self, cfg):

        ###################################################
        ########## User configurable variables ############
        ###################################################
        # pdb.set_trace()
        inp_feat_dir = cfg.inp_feat_dir
        out_feat_dir = cfg.out_feat_dir
        pred_feat_dir = cfg.pred_feat_dir
        # aux_feat_dir = "/home/gyzhang/merlin/egs/mixcc/s1/experiments/mixcc/acoustic_model/data/lid"
        inp_file_ext = cfg.inp_file_ext
        out_file_ext = cfg.out_file_ext
        # aux_file_ext = ".lid"
        ### Input-Output ###

        self.inp_dim = cfg.inp_dim
        self.out_dim = cfg.out_dim

        self.inp_norm = cfg.inp_norm
        self.out_norm = cfg.out_norm

        self.inp_stats_file = cfg.inp_stats_file
        self.out_stats_file = cfg.out_stats_file

        self.inp_scaler = None
        self.out_scaler = None

        #### define model params ####

        self.hidden_layer_type = cfg.hidden_layer_type
        self.hidden_layer_size = cfg.hidden_layer_size

        self.sequential_training = cfg.sequential_training
        self.encoder_decoder = cfg.encoder_decoder

        self.attention = cfg.attention
        self.cbhg = cfg.cbhg
        self.batch_size = cfg.batch_size
        self.shuffle_data = cfg.shuffle_data

        self.output_layer_type = cfg.output_layer_type
        self.loss_function = cfg.loss_function
        self.optimizer = cfg.optimizer

        self.rnn_params = cfg.rnn_params
        self.dropout_rate = cfg.dropout_rate
        self.num_of_epochs = cfg.num_of_epochs

        self.learning_rate = cfg.learning_rate
        ### Define the work directory###
        self.model_dir = cfg.model_dir

        ### define train, valid, test ###

        train_file_number = cfg.train_file_number
        valid_file_number = cfg.valid_file_number
        test_file_number = cfg.test_file_number

        file_id_scp = cfg.file_id_scp
        test_id_scp = cfg.test_id_scp

        #### main processess ####

        self.NORMDATA = cfg.NORMDATA
        self.TRAINMODEL = cfg.TRAINMODEL
        self.TESTMODEL = cfg.TESTMODEL

        #### vocoder ####
        self.delta_win = cfg.delta_win
        self.acc_win = cfg.acc_win
        self.windows = [(0, 0, np.array([1.])), (1, 1, np.array([-0.5, 0., 0.5])), (1, 1, np.array([1., -2., 1.]))]
        #### Generate only test list ####
        self.GenTestList = cfg.GenTestList

        ###################################################
        ####### End of user-defined conf variables ########
        ###################################################

        #### Create train, valid and test file lists ####
        file_id_list = data_utils.read_file_list(file_id_scp)

        train_id_list = file_id_list[0: train_file_number]
        valid_id_list = file_id_list[train_file_number: train_file_number + valid_file_number]
        test_id_list = file_id_list[
                       train_file_number + valid_file_number: train_file_number + valid_file_number + test_file_number]

        valid_test_id_list = file_id_list[
                             cfg.train_file_number - 20: train_file_number + valid_file_number + test_file_number]

        self.inp_train_file_list = data_utils.prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
        self.out_train_file_list = data_utils.prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)
        # self.inp_train_aux_file_list = data_utils.prepare_file_path_list(train_id_list, aux_feat_dir, aux_file_ext)
        self.inp_valid_file_list = data_utils.prepare_file_path_list(valid_id_list, inp_feat_dir, inp_file_ext)
        self.out_valid_file_list = data_utils.prepare_file_path_list(valid_id_list, out_feat_dir, out_file_ext)

        self.inp_test_file_list = data_utils.prepare_file_path_list(valid_test_id_list, inp_feat_dir, inp_file_ext)
        self.out_test_file_list = data_utils.prepare_file_path_list(valid_test_id_list, out_feat_dir, out_file_ext)

        self.gen_test_file_list = data_utils.prepare_file_path_list(valid_test_id_list, pred_feat_dir, out_file_ext)

        if self.GenTestList:
            test_id_list = data_utils.read_file_list(test_id_scp)
            self.inp_test_file_list = data_utils.prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
            self.gen_test_file_list = data_utils.prepare_file_path_list(test_id_list, pred_feat_dir, out_file_ext)

        if not self.encoder_decoder:
            self.tensorflow_models = TrainTensorflowModels(self.inp_dim, self.hidden_layer_size, self.out_dim,
                                                           self.hidden_layer_type, self.model_dir,
                                                           output_type=self.output_layer_type,
                                                           dropout_rate=self.dropout_rate,
                                                           loss_function=self.loss_function, optimizer=self.optimizer,
                                                           learning_rate=self.learning_rate)
        else:
            self.encoder_decoder_models = Train_Encoder_Decoder_Models(self.inp_dim, self.hidden_layer_size,
                                                                       self.out_dim, self.hidden_layer_type,
                                                                       output_type=self.output_layer_type, \
                                                                       dropout_rate=self.dropout_rate,
                                                                       loss_function=self.loss_function,
                                                                       optimizer=self.optimizer, \
                                                                       attention=self.attention, cbhg=self.cbhg)

    def normlize_data(self):
        ### normalize train data ###
        if os.path.isfile(self.inp_stats_file) and os.path.isfile(self.out_stats_file):
            self.inp_scaler = data_utils.load_norm_stats(self.inp_stats_file, self.inp_dim, method=self.inp_norm)
            self.out_scaler = data_utils.load_norm_stats(self.out_stats_file, self.out_dim, method=self.out_norm)
        else:
            print('preparing train_x, train_y from input and output feature files...')
            train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list,
                                                                               self.out_train_file_list, \
                                                                               self.inp_dim, self.out_dim,
                                                                               sequential_training=True if self.sequential_training or self.encoder_decoder else False)

            print('computing norm stats for train_x...')
            inp_scaler = data_utils.compute_norm_stats(train_x, self.inp_stats_file, method=self.inp_norm)

            print('computing norm stats for train_y...')
            out_scaler = data_utils.compute_norm_stats(train_y, self.out_stats_file, method=self.out_norm)

    def train_tensorflow_model(self):
        logging.getLogger("train model")
        print('preparing train_x, train_y from input and output feature files...')
        # pdb.set_trace()
        num_list = range(100)
        # num_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,354]
        # num_list = [0]
        self.inp_train_file_list = list(self.inp_train_file_list[i] for i in num_list)
        self.out_train_file_list = list(self.out_train_file_list[i] for i in num_list)
        train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list,
                                                                           self.out_train_file_list,
                                                                           self.inp_dim, self.out_dim,
                                                                           sequential_training=True if self.sequential_training or self.encoder_decoder else False)


        # train_y_staic = get_static_features(train_y, self.windows, stream_sizes=[180, 3, 1, 3],
        # has_dynamic_features=[True, True, False, True],
        # streams=[True, True, True, True])
        valid_x, valid_y, valid_flen = data_utils.read_data_from_file_list(self.inp_valid_file_list,
                                                                           self.out_valid_file_list,
                                                                           self.inp_dim, self.out_dim,
                                                                           sequential_training=True if self.sequential_training or self.encoder_decoder else False)
        # train_x = train_x[0:100000,:]
        # train_y = train_y[0:100000,:]
        print("shape of training set {}".format(train_x.shape))
        #### define the model ####
        if self.sequential_training:
            self.tensorflow_models.define_sequence_model()
        elif self.encoder_decoder:
            utt_length = train_flen["utt2framenum"].values()
            super(Train_Encoder_Decoder_Models, self.encoder_decoder_models).__setattr__("max_step", max(utt_length))
            self.encoder_decoder_models.define_encoder_decoder()
        else:
            self.tensorflow_models.define_feedforward_model()

        #### train the model ####
        print('training...')
        if self.sequential_training:
            # pdb.set_trace()
            self.tensorflow_models.train_sequence_model(train_x, train_y, valid_x, valid_y,
                                                        batch_size=self.batch_size,
                                                        num_of_epochs=self.num_of_epochs,
                                                        shuffle_data=self.shuffle_data)

        elif self.encoder_decoder:
            self.encoder_decoder_models.train_encoder_decoder_model(train_x, train_y, batch_size=self.batch_size,
                                                                    num_of_epochs=self.num_of_epochs, shuffle_data=True,
                                                                    utt_length=utt_length)
        else:
            self.tensorflow_models.train_feedforward_model(train_x, train_y, valid_x, valid_y,
                                                           batch_size=self.batch_size, num_of_epochs=self.num_of_epochs,
                                                           shuffle_data=self.shuffle_data)

    def test_tensorflow_model(self):

        #### load the data ####
        print('preparing test_x from input feature files...')
        test_x, test_flen = data_utils.read_test_data_from_file_list(self.inp_test_file_list, self.inp_dim)

        #### normalize the data ####
        data_utils.norm_data(test_x, self.inp_scaler)
        #### compute predictions ####
        if self.encoder_decoder:
            self.encoder_decoder_models.predict(test_x, self.out_scaler, self.gen_test_file_list)
        else:
            self.tensorflow_models.predict(test_x, self.out_scaler, self.gen_test_file_list, self.sequential_training)

    def main_function(self):
        ### Implement each module ###
        if self.NORMDATA:
            self.normlize_data()

        if self.TRAINMODEL:
            self.train_tensorflow_model()

        if self.TESTMODEL:
            self.test_tensorflow_model()


if __name__ == "__main__":
    # create a configuration instance
    # and get a short name for this instance
    cfg = configuration.cfg

    config_file = "/home/gyzhang/merlin/egs/slt_arctic/s1/conf/acoustic_slt_arctic_full.conf"

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    logger = logging.getLogger("main")
    print("--- Job started ---")
    # start_time = time.time()
    tensorflow_instance = TensorflowClass(cfg)
    # main function
    tensorflow_instance.train_tensorflow_model()
    # (m, s) = divmod(int(time.time() - start_time), 60)
    # print("--- Job completion time: %d min. %d sec ---" % (m, s))