import pickle
import gzip
import os, sys, errno
import time
import math

import subprocess

import numpy
#import gnumpy as gnp
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano
import pdb
# from utils.providers import ListDataProvider

from frontend.label_normalisation import HTSLabelNormalisation
from frontend.silence_remover import SilenceRemover
from frontend.silence_remover import trim_silence
from frontend.min_max_norm import MinMaxNormalisation
from frontend.acoustic_composition import AcousticComposition
from frontend.parameter_generation import ParameterGeneration
from frontend.mean_variance_norm import MeanVarianceNorm

# the new class for label composition and normalisation
from frontend.label_composer import LabelComposer
from frontend.label_modifier import HTSLabelModification
from frontend.merge_features import MergeFeat

import configuration
# from models.deep_rnn import DeepRecurrentNetwork

from utils.compute_distortion import DistortionComputation, IndividualDistortionComp
from utils.generate import generate_wav
from utils.learn_rates import ExpDecreaseLearningRate

from io_funcs.binary_io import  BinaryIOCollection

# our custom logging class that can also plot
from logplot.logging_plotting import LoggerPlotter, MultipleSeriesPlot, SingleWeightMatrixPlot
import logging # as logging
import logging.config
import io
from utils.file_paths import FilePaths
from utils.utils import read_file_list, prepare_file_path_list


def extract_file_id_list(file_list):
    file_id_list = []
    for file_name in file_list:
        file_id = os.path.basename(os.path.splitext(file_name)[0])
        file_id_list.append(file_id)

    return  file_id_list

def make_output_file_list(out_dir, in_file_lists):
    out_file_lists = []

    for in_file_name in in_file_lists:
        file_id = os.path.basename(in_file_name)
        out_file_name = out_dir + '/' + file_id
        out_file_lists.append(out_file_name)

    return  out_file_lists


def load_covariance(var_file_dict, out_dimension_dict):
    var = {}
    io_funcs = BinaryIOCollection()
    for feature_name in list(var_file_dict.keys()):
        var_values, dimension = io_funcs.load_binary_file_frame(var_file_dict[feature_name], 1)

        var_values = numpy.reshape(var_values, (out_dimension_dict[feature_name], 1))

        var[feature_name] = var_values

    return  var

def main_function(cfg):
    file_paths = FilePaths(cfg)
    # get a logger for this main function
    logger = logging.getLogger("main")

    # get another logger to handle plotting duties
    plotlogger = logging.getLogger("plotting")

    # later, we might do this via a handler that is created, attached and configured
    # using the standard config mechanism of the logging module
    # but for now we need to do it manually
    plotlogger.set_plot_path(cfg.plot_dir)

    # create plot dir if set to True
    if not os.path.exists(cfg.plot_dir) and cfg.plot:
        os.makedirs(cfg.plot_dir)

    #### parameter setting########
    hidden_layer_size = cfg.hyper_params['hidden_layer_size']
    ####prepare environment
    try:
        file_id_list = read_file_list(cfg.file_id_scp)
        logger.debug('Loaded file id list from %s' % cfg.file_id_scp)
    except IOError:
        # this means that open(...) threw an error
        logger.critical('Could not load file id list from %s' % cfg.file_id_scp)
        raise

    ###total file number including training, development, and testing
    # pdb.set_trace()
    total_file_number = len(file_id_list)
    assert cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number == total_file_number, 'check train, valid, test file number'

    data_dir = cfg.data_dir

    inter_data_dir = cfg.inter_data_dir
    nn_cmp_dir       = file_paths.nn_cmp_dir
    nn_cmp_norm_dir   = file_paths.nn_cmp_norm_dir
    model_dir = file_paths.model_dir
    gen_dir   = file_paths.gen_dir

    in_file_list_dict = {}

    for feature_name in list(cfg.in_dir_dict.keys()):
        in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)

    nn_cmp_file_list         = file_paths.get_nn_cmp_file_list()
    nn_cmp_norm_file_list    = file_paths.get_nn_cmp_norm_file_list()

    ###normalisation information
    norm_info_file = file_paths.norm_info_file

    ### normalise input full context label
    # currently supporting two different forms of lingustic features
    # later, we should generalise this

    assert cfg.label_style == 'HTS', 'Only HTS-style labels are now supported as input to Merlin'


    label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
    add_feat_dim = sum(cfg.additional_features.values())
    lab_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim
    if cfg.VoiceConversion:
        lab_dim = cfg.cmp_dim
    logger.info('Input label dimension is %d' % lab_dim)
    suffix=str(lab_dim)


    if cfg.process_labels_in_work_dir:
        inter_data_dir = cfg.work_dir

    # the number can be removed
    file_paths.set_label_dir(label_normaliser.dimension, suffix, lab_dim)
    file_paths.set_label_file_list()

    binary_label_dir      = file_paths.binary_label_dir
    nn_label_dir          = file_paths.nn_label_dir
    nn_label_norm_dir     = file_paths.nn_label_norm_dir

    in_label_align_file_list = file_paths.in_label_align_file_list
    binary_label_file_list   = file_paths.binary_label_file_list
    nn_label_file_list       = file_paths.nn_label_file_list
    nn_label_norm_file_list  = file_paths.nn_label_norm_file_list

    min_max_normaliser = None

    label_norm_file = file_paths.label_norm_file

    test_id_list = file_paths.test_id_list
    if cfg.NORMLAB:
        # simple HTS labels
    
        logger.info('preparing label data (input) using standard HTS style labels')
        label_normaliser.perform_normalisation(in_label_align_file_list, binary_label_file_list, label_type=cfg.label_type, state_number=cfg.state_number)

        if cfg.additional_features:
            out_feat_file_list = file_paths.out_feat_file_list
            in_dim = label_normaliser.dimension

            for new_feature, new_feature_dim in cfg.additional_features.items():
                new_feat_dir  = os.path.join(data_dir, new_feature)
                new_feat_file_list = prepare_file_path_list(file_id_list, new_feat_dir, '.'+new_feature)

                merger = MergeFeat(lab_dim = in_dim, feat_dim = new_feature_dim)
                merger.merge_data(binary_label_file_list, new_feat_file_list, out_feat_file_list)
                in_dim += new_feature_dim

                binary_label_file_list = out_feat_file_list

        remover = SilenceRemover(n_cmp = lab_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats, state_number=cfg.state_number)
        # pdb.set_trace()
        remover.remove_silence(binary_label_file_list, in_label_align_file_list, nn_label_file_list)

        min_max_normaliser = MinMaxNormalisation(feature_dimension = lab_dim, min_value = 0.01, max_value = 0.99)

        ###use only training data to find min-max information, then apply on the whole dataset
        if cfg.GenTestList:
            min_max_normaliser.load_min_max_values(label_norm_file)
        else:
            min_max_normaliser.find_min_max_values(nn_label_file_list[0:cfg.train_file_number])

        ### enforce silence such that the normalization runs without removing silence: only for final synthesis
        if cfg.GenTestList and cfg.enforce_silence:
            min_max_normaliser.normalise_data(binary_label_file_list, nn_label_norm_file_list)
        else:
            min_max_normaliser.normalise_data(nn_label_file_list, nn_label_norm_file_list)



    if min_max_normaliser != None and not cfg.GenTestList:
        ### save label normalisation information for unseen testing labels
        label_min_vector = min_max_normaliser.min_vector
        label_max_vector = min_max_normaliser.max_vector
        label_norm_info = numpy.concatenate((label_min_vector, label_max_vector), axis=0)

        label_norm_info = numpy.array(label_norm_info, 'float32')
        fid = open(label_norm_file, 'wb')
        label_norm_info.tofile(fid)
        fid.close()
        logger.info('saved %s vectors to %s' %(label_min_vector.size, label_norm_file))

    ### make output duration data
    # pdb.set_trace()
    if cfg.MAKEDUR:
        logger.info('creating duration (output) features')
        label_normaliser.prepare_dur_data(in_label_align_file_list, file_paths.dur_file_list, cfg.label_type, cfg.dur_feature_type,state_number=cfg.state_number)

    ### make output acoustic data
    # pdb.set_trace()
    if cfg.MAKECMP:
        logger.info('creating acoustic (output) features')
        delta_win = cfg.delta_win #[-0.5, 0.0, 0.5]
        acc_win = cfg.acc_win     #[1.0, -2.0, 1.0]

        if cfg.GenTestList:
            for feature_name in list(cfg.in_dir_dict.keys()):
                in_file_list_dict[feature_name] = prepare_file_path_list(test_id_list, cfg.in_dir_dict[feature_name], cfg.file_extension_dict[feature_name], False)
            nn_cmp_file_list      = prepare_file_path_list(test_id_list, nn_cmp_dir, cfg.cmp_ext)
            nn_cmp_norm_file_list = prepare_file_path_list(test_id_list, nn_cmp_norm_dir, cfg.cmp_ext)
        
        acoustic_worker = AcousticComposition(delta_win = delta_win, acc_win = acc_win)

        if 'dur' in list(cfg.in_dir_dict.keys()) and cfg.AcousticModel:
            lf0_file_list = file_paths.get_lf0_file_list()
            acoustic_worker.make_equal_frames(dur_file_list, lf0_file_list, cfg.in_dimension_dict)

        # generate inter_moudle nn_mgc_lf0_vuv_bap_187 folder which are *.cmp files
        acoustic_worker.prepare_nn_data(in_file_list_dict, nn_cmp_file_list, cfg.in_dimension_dict, cfg.out_dimension_dict)

        if cfg.remove_silence_using_binary_labels:
            ## do this to get lab_dim:
            label_composer = LabelComposer()
            label_composer.load_label_configuration(cfg.label_config_file)
            lab_dim=label_composer.compute_label_dimension()

            silence_feature = 0 ## use first feature in label -- hardcoded for now
            logger.info('Silence removal from CMP using binary label file')

            ## overwrite the untrimmed audio with the trimmed version:
            trim_silence(nn_cmp_file_list, nn_cmp_file_list, cfg.cmp_dim,
                                binary_label_file_list, lab_dim, silence_feature)

        # we only use this function. we remove silence part with respect to label files and save to itself.
        elif cfg.remove_silence_using_hts_labels: 
            ## back off to previous method using HTS labels:
            remover = SilenceRemover(n_cmp = cfg.cmp_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features, subphone_feats = cfg.subphone_feats)
            remover.remove_silence(nn_cmp_file_list, in_label_align_file_list, nn_cmp_file_list) # save to itself

    ### save acoustic normalisation information for normalising the features back
    var_dir  = file_paths.var_dir
    var_file_dict = file_paths.get_var_dic()

    ### normalise output acoustic data
    if cfg.NORMCMP:
        logger.info('normalising acoustic (output) features using method %s' % cfg.output_feature_normalisation)
        cmp_norm_info = None
        if cfg.output_feature_normalisation == 'MVN':
            normaliser = MeanVarianceNorm(feature_dimension=cfg.cmp_dim)
            if cfg.GenTestList:
                # load mean std values
                global_mean_vector, global_std_vector = normaliser.load_mean_std_values(norm_info_file)
            else:
                ###calculate mean and std vectors on the training data, and apply on the whole dataset
                global_mean_vector = normaliser.compute_mean(nn_cmp_file_list[0:cfg.train_file_number], 0, cfg.cmp_dim)
                global_std_vector = normaliser.compute_std(nn_cmp_file_list[0:cfg.train_file_number], global_mean_vector, 0, cfg.cmp_dim)
                # for hmpd vocoder we don't need to normalize the 
                # pdd values
                if cfg.vocoder_type == 'hmpd':
                    stream_start_index = {}
                    dimension_index = 0
                    recorded_vuv = False
                    vuv_dimension = None
                    for feature_name in cfg.out_dimension_dict.keys():
                        if feature_name != 'vuv':
                            stream_start_index[feature_name] = dimension_index
                        else:
                            vuv_dimension = dimension_index
                            recorded_vuv = True
                        
                        dimension_index += cfg.out_dimension_dict[feature_name]
                    logger.info('hmpd pdd values are not normalized since they are in 0 to 1')
                    global_mean_vector[:,stream_start_index['pdd']: stream_start_index['pdd'] + cfg.out_dimension_dict['pdd']] = 0
                    global_std_vector[:,stream_start_index['pdd']: stream_start_index['pdd'] + cfg.out_dimension_dict['pdd']] = 1
            normaliser.feature_normalisation(nn_cmp_file_list, nn_cmp_norm_file_list)
            cmp_norm_info = numpy.concatenate((global_mean_vector, global_std_vector), axis=0)

        elif cfg.output_feature_normalisation == 'MINMAX':
            min_max_normaliser = MinMaxNormalisation(feature_dimension = cfg.cmp_dim, min_value = 0.01, max_value = 0.99)
            if cfg.GenTestList:
                min_max_normaliser.load_min_max_values(norm_info_file)
            else:
                min_max_normaliser.find_min_max_values(nn_cmp_file_list[0:cfg.train_file_number])
            min_max_normaliser.normalise_data(nn_cmp_file_list, nn_cmp_norm_file_list)

            cmp_min_vector = min_max_normaliser.min_vector
            cmp_max_vector = min_max_normaliser.max_vector
            cmp_norm_info = numpy.concatenate((cmp_min_vector, cmp_max_vector), axis=0)

        else:
            logger.critical('Normalisation type %s is not supported!\n' %(cfg.output_feature_normalisation))
            raise

        if not cfg.GenTestList:
            cmp_norm_info = numpy.array(cmp_norm_info, 'float32')
            fid = open(norm_info_file, 'wb')
            cmp_norm_info.tofile(fid)
            fid.close()
            logger.info('saved %s vectors to %s' %(cfg.output_feature_normalisation, norm_info_file))
        if cfg.output_feature_normalisation == 'MVN':
            feature_index = 0
            # pdb.set_trace()
            for feature_name in list(cfg.out_dimension_dict.keys()):
                feature_std_vector = numpy.array(global_std_vector[:,feature_index:feature_index+cfg.out_dimension_dict[feature_name]], 'float32')

                fid = open(var_file_dict[feature_name], 'w')
                feature_var_vector = feature_std_vector**2
                feature_var_vector.tofile(fid)
                fid.close()

                logger.info('saved %s variance vector to %s' %(feature_name, var_file_dict[feature_name]))

                feature_index += cfg.out_dimension_dict[feature_name]

    train_x_file_list, train_y_file_list = file_paths.get_train_list_x_y()
    valid_x_file_list, valid_y_file_list = file_paths.get_valid_list_x_y()
    test_x_file_list, test_y_file_list = file_paths.get_test_list_x_y()

    # we need to know the label dimension before training the DNN
    # computing that requires us to look at the labels
    #
    label_normaliser = HTSLabelNormalisation(question_file_name=cfg.question_file_name, add_frame_features=cfg.add_frame_features, subphone_feats=cfg.subphone_feats)
    add_feat_dim = sum(cfg.additional_features.values())
    lab_dim = label_normaliser.dimension + add_feat_dim + cfg.appended_input_dim
    if cfg.VoiceConversion:
        lab_dim = cfg.cmp_dim

    logger.info('label dimension is %d' % lab_dim)

    combined_model_arch = str(len(hidden_layer_size))
    for hid_size in hidden_layer_size:
        combined_model_arch += '_' + str(hid_size)

    nnets_file_name = file_paths.get_nnets_file_name()
    temp_dir_name = file_paths.get_temp_nn_dir_name()

    gen_dir = os.path.join(gen_dir, temp_dir_name)

    if cfg.switch_to_keras or cfg.switch_to_tensorflow:
        ### set configuration variables ###
        cfg.inp_dim = lab_dim
        cfg.out_dim = cfg.cmp_dim

        cfg.inp_feat_dir  = nn_label_norm_dir
        cfg.out_feat_dir  = nn_cmp_norm_dir
        cfg.pred_feat_dir = gen_dir

        if cfg.GenTestList and cfg.test_synth_dir!="None":
            cfg.inp_feat_dir  = cfg.test_synth_dir
            cfg.pred_feat_dir = cfg.test_synth_dir


    if cfg.switch_to_tensorflow:
        ### call Tensorflowclass and use an instance ###
        from run_tensorflow_with_merlin_io import TensorflowClass
        tf_instance = TensorflowClass(cfg)

    ### DNN model training
    if cfg.TRAINDNN:

        var_dict = load_covariance(var_file_dict, cfg.out_dimension_dict)

        logger.info('training DNN')

        fid = open(norm_info_file, 'rb')
        cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
        fid.close()
        cmp_min_max = cmp_min_max.reshape((2, -1))
        cmp_mean_vector = cmp_min_max[0, ]
        cmp_std_vector  = cmp_min_max[1, ]


        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                # not an error - just means directory already exists
                pass
            else:
                logger.critical('Failed to create model directory %s' % model_dir)
                logger.critical(' OS error was: %s' % e.strerror)
                raise

        try:
            tf_instance.train_tensorflow_model()
        except KeyboardInterrupt:
            logger.critical('train_DNN interrupted via keyboard')
            # Could 'raise' the exception further, but that causes a deep traceback to be printed
            # which we don't care about for a keyboard interrupt. So, just bail out immediately
            sys.exit(1)
        except:
            logger.critical('train_DNN threw an exception')
            raise

    if cfg.GENBNFEA:
        # Please only tune on this step when you want to generate bottleneck features from DNN
        gen_dir = file_paths.bottleneck_features

        bottleneck_size = min(hidden_layer_size)
        bottleneck_index = 0
        for i in range(len(hidden_layer_size)):
            if hidden_layer_size[i] == bottleneck_size:
                bottleneck_index = i

        logger.info('generating bottleneck features from DNN')

        try:
            os.makedirs(gen_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                # not an error - just means directory already exists
                pass
            else:
                logger.critical('Failed to create generation directory %s' % gen_dir)
                logger.critical(' OS error was: %s' % e.strerror)
                raise

        gen_file_id_list = file_id_list[cfg.train_file_number-20:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        pdb.set_trace()
        test_x_file_list = nn_label_norm_file_list[0:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

        gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)

    ### generate parameters from DNN
    ### modified generate all the labels I want to see the error in training set 
    # gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
    # we choose 20 utterances of training set to regenerate to compute training MCD etc.
    pdb.set_trace()
    gen_file_id_list = file_id_list[cfg.train_file_number-20:cfg.train_file_number + cfg.valid_file_number + cfg.test_file_number]
    test_x_file_list  = nn_label_norm_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
    if cfg.GenTestList:
        gen_file_id_list = test_id_list
        test_x_file_list = nn_label_norm_file_list
        if cfg.test_synth_dir!="None":
            gen_dir = cfg.test_synth_dir

    # pdb.set_trace()
    if cfg.DNNGEN:
        logger.info('generating from DNN')
        if not os.path.exists(gen_dir):
            os.mkdir(gen_dir)
        gen_file_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.cmp_ext)
        if cfg.switch_to_tensorflow:
            tf_instance.test_tensorflow_model()
        logger.debug('denormalising generated output using method %s' % cfg.output_feature_normalisation)
        fid = open(norm_info_file, 'rb')
        cmp_min_max = numpy.fromfile(fid, dtype=numpy.float32)
        fid.close()
        cmp_min_max = cmp_min_max.reshape((2, -1))
        cmp_min_vector = cmp_min_max[0, ]
        cmp_max_vector = cmp_min_max[1, ]

        if cfg.output_feature_normalisation == 'MVN':
            denormaliser = MeanVarianceNorm(feature_dimension = cfg.cmp_dim)
            denormaliser.feature_denormalisation(gen_file_list, gen_file_list, cmp_min_vector, cmp_max_vector)

        if cfg.output_feature_normalisation == 'MINMAX':
            denormaliser = MinMaxNormalisation(cfg. cmp_dim, min_value = 0.01, max_value = 0.99, min_vector = cmp_min_vector, max_vector = cmp_max_vector)
            denormaliser.denormalise_data(gen_file_list, gen_file_list)

        if cfg.AcousticModel:
            ##perform MLPG to smooth parameter trajectory
            ## lf0 is included, the output features must have vuv.
        #    pdb.set_trace()
            generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features, enforce_silence = cfg.enforce_silence)
            generator.acoustic_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict, var_file_dict, do_MLPG=cfg.do_MLPG, cfg=cfg)

        if cfg.DurationModel:
            ### Perform duration normalization(min. state dur set to 1) ###
            gen_dur_list   = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.dur_ext)
            gen_label_list = prepare_file_path_list(gen_file_id_list, gen_dir, cfg.lab_ext)
            in_gen_label_align_file_list = prepare_file_path_list(gen_file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)

            generator = ParameterGeneration(gen_wav_features = cfg.gen_wav_features)
            generator.duration_decomposition(gen_file_list, cfg.cmp_dim, cfg.out_dimension_dict, cfg.file_extension_dict)
            label_modifier = HTSLabelModification(silence_pattern = cfg.silence_pattern, label_type = cfg.label_type, state_number=cfg.state_number)
            label_modifier.modify_duration_labels(in_gen_label_align_file_list, gen_dur_list, gen_label_list)


    ### generate wav
    if cfg.GENWAV:
        # pdb.set_trace()
        logger.info('reconstructing waveform(s)')
        # wav_dir = os.path.join(gen_dir,'wav')
      #  pdb.set_trace()
        if not os.path.exists(gen_dir):
            os.mkdir(gen_dir)
        generate_wav(gen_dir, gen_file_id_list, cfg)     # generated speech
    #    generate_wav(nn_cmp_dir, gen_file_id_list, cfg)  # reference copy synthesis speech

    ### setting back to original conditions before calculating objective scores ###
    if cfg.GenTestList:
        in_label_align_file_list = prepare_file_path_list(file_id_list, cfg.in_label_align_dir, cfg.lab_ext, False)
        binary_label_file_list   = prepare_file_path_list(file_id_list, binary_label_dir, cfg.lab_ext)
        gen_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

    ### evaluation: RMSE and CORR for duration
    if cfg.CALMCD and cfg.DurationModel:
        logger.info('calculating MCD')

        ref_data_dir = os.path.join(inter_data_dir, 'ref_data')

        ref_dur_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.dur_ext)

        in_gen_label_align_file_list = in_label_align_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        calculator = IndividualDistortionComp()

        train_file_id_list = file_id_list[cfg.train_file_number-20:cfg.train_file_number]
        valid_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
        test_file_id_list  = file_id_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

        if cfg.remove_silence_using_binary_labels:
            untrimmed_reference_data = in_file_list_dict['dur'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
            trim_silence(untrimmed_reference_data, ref_dur_list, cfg.dur_dim, \
                                untrimmed_test_labels, lab_dim, silence_feature)
        else:
            remover = SilenceRemover(n_cmp = cfg.dur_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type, remove_frame_features = cfg.add_frame_features)
            remover.remove_silence(in_file_list_dict['dur'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_dur_list)

        valid_dur_rmse, valid_dur_corr = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.dur_ext, cfg.dur_dim)
        test_dur_rmse, test_dur_corr = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.dur_ext, cfg.dur_dim)

        logger.info('Develop: DNN -- RMSE: %.3f frames/phoneme; CORR: %.3f; ' \
                    %(valid_dur_rmse, valid_dur_corr))
        logger.info('Test: DNN -- RMSE: %.3f frames/phoneme; CORR: %.3f; ' \
                    %(test_dur_rmse, test_dur_corr))

    ### evaluation: calculate distortion
    if cfg.CALMCD and cfg.AcousticModel:
        logger.info('calculating MCD')

        ref_data_dir = os.path.join(inter_data_dir, 'ref_data')
        ref_lf0_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.lf0_ext)
        # for straight or world vocoders
        ref_mgc_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.mgc_ext)
        ref_bap_list = prepare_file_path_list(gen_file_id_list, ref_data_dir, cfg.bap_ext)
        # in_gen_label_align_file_list = in_label_align_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        in_gen_label_align_file_list = in_label_align_file_list[cfg.train_file_number-20:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
        calculator = IndividualDistortionComp()

        spectral_distortion = 0.0
        bap_mse             = 0.0
        f0_mse              = 0.0
        vuv_error           = 0.0
        train_file_id_list = file_id_list[cfg.train_file_number-20:cfg.train_file_number]
        valid_file_id_list = file_id_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number]
        test_file_id_list  = file_id_list[cfg.train_file_number+cfg.valid_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]

        if cfg.remove_silence_using_binary_labels:
            ## get lab_dim:
            label_composer = LabelComposer()
            label_composer.load_label_configuration(cfg.label_config_file)
            lab_dim=label_composer.compute_label_dimension()

            ## use first feature in label -- hardcoded for now
            silence_feature = 0

            ## Use these to trim silence:
            untrimmed_test_labels = binary_label_file_list[cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]


        if 'mgc' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_mgc_list, cfg.mgc_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            elif cfg.remove_silence_using_hts_labels:
                remover = SilenceRemover(n_cmp = cfg.mgc_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                # remover.remove_silence(in_file_list_dict['mgc'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_mgc_list)
                remover.remove_silence(in_file_list_dict['mgc'][cfg.train_file_number-20:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_mgc_list)
            else:
                ref_data_dir = os.path.join(data_dir, 'mgc')
            train_spectral_distortion = calculator.compute_distortion(train_file_id_list, ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            valid_spectral_distortion = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            test_spectral_distortion  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.mgc_ext, cfg.mgc_dim)
            train_spectral_distortion *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
            valid_spectral_distortion *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD
            test_spectral_distortion  *= (10 /numpy.log(10)) * numpy.sqrt(2.0)    ##MCD


        if 'bap' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_bap_list, cfg.bap_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            elif cfg.remove_silence_using_hts_labels:
                remover = SilenceRemover(n_cmp = cfg.bap_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['bap'][cfg.train_file_number-20:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_bap_list)
                # remover.remove_silence(in_file_list_dict['bap'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_bap_list)
            else:
                ref_data_dir = os.path.join(data_dir, 'bap')
            train_bap_mse = calculator.compute_distortion(train_file_id_list, ref_data_dir, gen_dir, cfg.bap_ext,
                                                          cfg.bap_dim)
            valid_bap_mse = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            test_bap_mse  = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.bap_ext, cfg.bap_dim)
            train_bap_mse = train_bap_mse / 10.0
            valid_bap_mse = valid_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC
            test_bap_mse  = test_bap_mse / 10.0    ##Cassia's bap is computed from 10*log|S(w)|. if use HTS/SPTK style, do the same as MGC

        if 'lf0' in cfg.in_dimension_dict:
            if cfg.remove_silence_using_binary_labels:
                untrimmed_reference_data = in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number]
                trim_silence(untrimmed_reference_data, ref_lf0_list, cfg.lf0_dim, \
                                    untrimmed_test_labels, lab_dim, silence_feature)
            elif cfg.remove_silence_using_hts_labels:
                remover = SilenceRemover(n_cmp = cfg.lf0_dim, silence_pattern = cfg.silence_pattern, label_type=cfg.label_type)
                remover.remove_silence(in_file_list_dict['lf0'][cfg.train_file_number-20:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_lf0_list)
                # remover.remove_silence(in_file_list_dict['lf0'][cfg.train_file_number:cfg.train_file_number+cfg.valid_file_number+cfg.test_file_number], in_gen_label_align_file_list, ref_lf0_list)
            else:
                if cfg.vocoder_type == 'MAGPHASE':
                    ref_data_dir = os.path.join(data_dir, 'feats')
                else:
                    ref_data_dir = os.path.join(data_dir, 'lf0')
            train_f0_mse, train_f0_corr, train_vuv_error   = calculator.compute_distortion(train_file_id_list, ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)
            valid_f0_mse, valid_f0_corr, valid_vuv_error   = calculator.compute_distortion(valid_file_id_list, ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)
            test_f0_mse , test_f0_corr, test_vuv_error    = calculator.compute_distortion(test_file_id_list , ref_data_dir, gen_dir, cfg.lf0_ext, cfg.lf0_dim)


        if cfg.vocoder_type == 'MAGPHASE':
            logger.info('Develop: DNN -- MAG: %.3f dB; REAL: %.3f dB; IMAG: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                    %(valid_mag_mse, valid_real_mse, valid_imag_mse, valid_f0_mse, valid_f0_corr, valid_vuv_error*100.))
            logger.info('Test   : DNN -- MAG: %.3f dB; REAL: %.3f dB; IMAG: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                    %(test_mag_mse, test_real_mse, test_imag_mse , test_f0_mse , test_f0_corr, test_vuv_error*100.))
        else:
            logger.info('Train: DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                        % (train_spectral_distortion, train_bap_mse, train_f0_mse, train_f0_corr, train_vuv_error * 100.))
            logger.info('Develop: DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                    %(valid_spectral_distortion, valid_bap_mse, valid_f0_mse, valid_f0_corr, valid_vuv_error*100.))
            logger.info('Test   : DNN -- MCD: %.3f dB; BAP: %.3f dB; F0:- RMSE: %.3f Hz; CORR: %.3f; VUV: %.3f%%' \
                    %(test_spectral_distortion , test_bap_mse , test_f0_mse , test_f0_corr, test_vuv_error*100.))


if __name__ == '__main__':
    # these things should be done even before trying to parse the command line
    # create a configuration instance
    # and get a short name for this instance
    cfg=configuration.cfg
    # set up logging to use our custom class
    logging.setLoggerClass(LoggerPlotter)
    # get a logger for this main function
    logger = logging.getLogger("main")
    if len(sys.argv) != 2:
        logger.critical('usage: run_merlin.sh [config file name]')
        sys.exit(1)
    config_file = sys.argv[1]
    config_file = os.path.abspath(config_file)
    cfg.configure(config_file)
    main_function(cfg)
    sys.exit(0)
