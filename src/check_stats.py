"""
This file is for checking statistical properties of training data
You need to provide configuration file
"""
import pickle
import gzip
import os, sys, errno
# sys.path.insert(0, '../configuration')
# sys.path.insert(0, '../utils')
# sys.path.insert(0,"../
sys.path.insert(0,'../')
from io_funcs.binary_io import BinaryIOCollection
from configuration.configuration import configuration
from utils.utils import read_file_list, prepare_file_path_list
import time
import math

import subprocess
import logging
import  numpy as np
#import gnumpy as gnp
# we need to explicitly import this in some cases, not sure why this doesn't get imported with numpy itself
import numpy.distutils.__config__
# and only after that can we import theano
import pdb
if __name__ == '__main__':
    pdb.set_trace()
    cfg = configuration()
    # set up logging to use our custom class
    # logging.setLoggerClass(LoggerPlotter)
    # get a logger for this main function
    logger = logging.getLogger("main")
    if len(sys.argv) != 2:
        logger.critical('usage: run_merlin.sh [config file name]')
        sys.exit(1)
    config_file = sys.argv[1]
    config_file = os.path.abspath(config_file)
    cfg.configure(config_file,False)
    file_id_list = read_file_list(cfg.file_id_scp)
    in_file_list_dict = {}
    io_funcs = BinaryIOCollection()
    for feature_name in list(cfg.in_dir_dict.keys()):
        in_file_list_dict[feature_name] = prepare_file_path_list(file_id_list, cfg.in_dir_dict[feature_name],
                                                                 cfg.file_extension_dict[feature_name], False)
    file_dict=dict()
    global_mean_vector=np.zeros((1, 62))
    global_std_vector=np.zeros((1,62))
    all_frame_number = 0
    all_feature_list=[]
    all_file_mean_list=[]
    all_file_95_list=[]
    all_file_5_list=[]
    # feature_dict=dict()
    mgc_feature_list = []
    for num, file_id in enumerate(file_id_list):
        # file_dict[file_id]=[]
        mean_feature_list=[]
        p95_list=[]
        p5_list=[]
        for feature_name in list(cfg.in_dir_dict.keys()):
            in_file_name = in_file_list_dict[feature_name][num]
            in_feature_dim = cfg.in_dimension_dict[feature_name]
            logger.info("preprocessing ")
            features, frame_number = io_funcs.load_binary_file_frame(in_file_name, in_feature_dim)
            if feature_name=="lf0":
                features=features[features>0]
                features=np.reshape(features,[-1,1])
            if feature_name=="mgc":
                mgc_feature_list.append(features)
                # pdb.set_trace()
            p5_list.extend(np.percentile(features,5,axis=0))
            p95_list.extend(np.percentile(features,95,axis=0))
            mean_feature_list.extend(np.mean(features,0))
            # feature_dict[feature_name]=np.mean(features,1)
            # file_dict[file_id].append(feature_dict)
        all_file_mean_list.append(mean_feature_list)
        all_file_95_list.append(p95_list)
        all_file_5_list.append(p5_list)
    mgc_features=np.concatenate(mgc_feature_list,axis=0)
    global_mean_vector = np.mean(mgc_features,axis=0)
    global_std_vector = np.mean(mgc_features,axis=0)
    pdb.set_trace()
    all_file_mean_array=np.array(all_file_mean_list)
    all_file_5_array=np.array(all_file_5_list)
    all_file_95_array=np.array(all_file_95_list)

    # pdb.set_trace()
