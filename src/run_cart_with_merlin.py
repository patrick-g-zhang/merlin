import os
import sys
from sklearn import tree
import time
import tensorflow as tf
import pickle
import pdb
import numpy as np
import logging
import configuration
from tensorflow_lib import data_utils
from io_funcs.binary_io import BinaryIOCollection
from frontend.mean_variance_norm import MeanVarianceNorm

# from frontend.mlpg import get_static_features
class CARTClass(object):

    def __init__(self, cfg):

        ###################################################
        ########## User configurable variables ############
        ###################################################
        # pdb.set_trace()
        inp_feat_dir = "/home/gyzhang/merlin/egs/blz/s1/experiments/blz2/acoustic_model/inter_module/nn_no_silence_lab_norm_329"
        out_feat_dir = "/home/gyzhang/merlin/egs/blz/s1/experiments/blz2/acoustic_model/inter_module/nn_norm_mgc_lf0_vuv_bap_187"
        pred_feat_dir = cfg.pred_feat_dir
        # aux_feat_dir = "/home/gyzhang/merlin/egs/mixcc/s1/experiments/mixcc/acoustic_model/data/lid"
        inp_file_ext = cfg.inp_file_ext
        out_file_ext = cfg.out_file_ext
        # aux_file_ext = ".lid"
        ### Input-Output ###
        self.inp_dim = 329
        self.out_dim = 187

        self.inp_norm = 'MINMAX'
        self.out_norm = 'MEAN'

        # self.inp_stats_file = "label_norm_HTS_329.dat"
        # self.out_stats_file = cfg.out_stats_file

        # self.inp_scaler = None
        # self.out_scaler = None

        #### define model params ####

        # self.hidden_layer_type = cfg.hidden_layer_type
        # self.hidden_layer_size = cfg.hidden_layer_size
        #
        self.sequential_training = cfg.sequential_training
        self.encoder_decoder = cfg.encoder_decoder
        #
        # self.attention = cfg.attention
        # self.cbhg = cfg.cbhg
        # self.batch_size = cfg.batch_size
        # self.shuffle_data = cfg.shuffle_data

        # self.output_layer_type = cfg.output_layer_type
        # self.loss_function = cfg.loss_function
        # self.optimizer = cfg.optimizer

        # self.rnn_params = cfg.rnn_params
        # self.dropout_rate = cfg.dropout_rate
        # self.num_of_epochs = cfg.num_of_epochs
        #
        # self.learning_rate = cfg.learning_rate
        ## Define the work directory###
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


    def train_cart_model(self):
        logging.getLogger("train cart model")
        print('preparing train_x, train_y from input and output feature files...')
        pdb.set_trace()

        train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list,
                                                                           self.out_train_file_list,
                                                                           self.inp_dim, self.out_dim,
                                                                           sequential_training=True if self.sequential_training or self.encoder_decoder else False)

        # train_y_staic = get_static_features(train_y, self.windows, stream_sizes=[180, 3, 1, 3],
        # has_dynamic_features=[True, True, False, True],
        # streams=[True, True, True, True])
        # valid_x, valid_y, valid_flen = data_utils.read_data_from_file_list(self.inp_valid_file_list,
        #                                                                    self.out_valid_file_list,
        #                                                                    self.inp_dim, self.out_dim,
        #                                                                    sequential_training=True if self.sequential_training or self.encoder_decoder else False)
        #### define the model ####
        clf = tree.DecisionTreeRegressor(max_depth=70)

        #### train the model ####
        print('training...')
        pdb.set_trace()
        filename = 'finalized_model.sav'
        clf = clf.fit(train_x,train_y)
        pickle.dump(clf,open(filename,'wb'),protocol=4)
        pass
    def test_cart_model(self):
        valid_x, valid_y, valid_flen = data_utils.read_data_from_file_list(self.inp_valid_file_list,
                                                                           self.out_valid_file_list,
                                                                           self.inp_dim, self.out_dim,
                                                                           sequential_training=False)
        # train_x, train_y, train_flen = data_utils.read_data_from_file_list(self.inp_train_file_list,
        #                                                                    self.out_train_file_list,
        #                                                                    self.inp_dim, self.out_dim,
        #                                                                    sequential_training=False)

        filename = 'finalized_model.sav'
        clf = pickle.load(open(filename, 'rb'))
        pdb.set_trace()
        print(clf.score(valid_x,valid_y))

    def predict(self, test_id_list, gen_test_file_list):
        """
            predict the results with given model
        """

        io_funcs = BinaryIOCollection()
        # test_id_list.sort()
        # gen_test_file_list.sort()
        test_file_number = len(test_id_list)

        print("generating features on held-out test data...")
        filename = 'finalized_model.sav'
        clf = pickle.load(open(filename, 'rb'))
        print("loading the model parameters...")
        for utt_index in range(test_file_number):
            gen_test_file_name = gen_test_file_list[utt_index]
            temp_test_x        = io_funcs.load_binary_file(test_id_list[utt_index],self.inp_dim)
            pdb.set_trace()
            y_predict = clf.predict(temp_test_x)
            io_funcs.array_to_binary_file(y_predict, gen_test_file_name)

    # def test_cart_model(self):
    #
    #     #### load the data ####
    #     print('preparing test_x from input feature files...')
    #     test_x, test_flen = data_utils.read_test_data_from_file_list(self.inp_test_file_list, self.inp_dim)
    #
    #     #### normalize the data ####
    #     data_utils.norm_data(test_x, self.inp_scaler)
    #     #### compute predictions ####
    #     if self.encoder_decoder:
    #         self.encoder_decoder_models.predict(test_x, self.out_scaler, self.gen_test_file_list)
    #     else:
    #         self.tensorflow_models.predict(test_x, self.out_scaler, self.gen_test_file_list, self.sequential_training)
    #
    # def main_function(self):
    #     ### Implement each module ###
    #
    #     if self.TRAINMODEL:
    #         self.train()
    #
    #     if self.TESTMODEL:
    #         self.test_tensorflow_model()


if __name__ == "__main__":
    # create a configuration instance
    # and get a short name for this instance
    cfg = configuration.cfg

    config_file = "/home/gyzhang/merlin/egs/blz/s1/conf/acoustic_blz.conf"

    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    # logger = logging.getLogger("main")
    # print("--- Job started ---")
    # start_time = time.time()
    cart_instance = CARTClass(cfg)
    # main function
    # cart_instance.test_cart_model()
    test_id_list = ["/home/gyzhang/merlin/egs/blz/s1/experiments/blz2/acoustic_model/inter_module/nn_no_silence_lab_norm_329/100442.lab"]
    gen_test_file_list=["/home/gyzhang/merlin/egs/blz/s1/experiments/blz2/acoustic_model/gen/cart/100442.cmp"]
    pdb.set_trace()
    # cart_instance.train_cart_model()
    # cart_instance.predict(test_id_list,gen_test_file_list)
    cart_instance.test_cart_model()
    # (m, s) = divmod(int(time.time() - start_time), 60)
    # print("--- Job completion time: %d min. %d sec ---" % (m, s))
    # mvn = MeanVarianceNorm(feature_dimension=187)

    # mean_var_file = "/home/gyzhang/merlin/egs/blz/s1/experiments/blz2/acoustic_model/inter_module/norm_info__mgc_lf0_vuv_bap_187_MVN.dat"
    # mvn.load_mean_std_values(mean_var_file)
    # cmp_file_list = ["/home/gyzhang/merlin/egs/blz/s1/experiments/blz2/acoustic_model/gen/cart/100442.cmp"]
    # pdb.set_trace()
    # mvn.feature_denormalisation(cmp_file_list, cmp_file_list, mvn.mean_vector, mvn.std_vector)