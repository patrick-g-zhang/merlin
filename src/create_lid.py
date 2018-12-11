# this file just for add lid will be modified if works
from io_funcs.binary_io import BinaryIOCollection
from utils.utils import read_file_list, prepare_file_path_list
import numpy as np 
bin_io = BinaryIOCollection()
file_id_scp = "/home/gyzhang/merlin/egs/mixcc/s1/experiments/mixcc/acoustic_model/data/file_id_list_mixcc.scp"
new_feat_dir = "/home/gyzhang/merlin/egs/mixcc/s1/experiments/mixcc/acoustic_model/data/lid"
lf0_dir = "/home/gyzhang/merlin/egs/mixcc/s1/experiments/mixcc/acoustic_model/inter_module/nn_mgc_lf0_vuv_bap_187"
file_id_list = read_file_list(file_id_scp)
new_feature = "lid"
lf0_feature ="cmp"
new_feat_file_list = prepare_file_path_list(file_id_list, new_feat_dir, '.'+new_feature)
lf0_file_list = prepare_file_path_list(file_id_list, lf0_dir,'.cmp')
for index, file_name in enumerate(file_id_list):
	lf0_file = lf0_file_list[index]
	new_feat_file = new_feat_file_list[index]
	_, frame_numbers = bin_io.load_binary_file_frame(lf0_file, 187)
	if int(file_name) > 10000:
		# should be madarin
		lid_features = np.ones((frame_numbers,1),dtype=np.float32)
	else:
		lid_features = np.zeros((frame_numbers,1),dtype=np.float32)
	bin_io.array_to_binary_file(lid_features, new_feat_file_list[index])
