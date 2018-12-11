import glob
import re
import os
lab_dir = "/home/gyzhang/merlin/egs/namestory/s1/experiments/namestory/duration_model/data/label_phone_align"
scp_file = "/home/gyzhang/merlin/egs/namestory/s1/experiments/namestory/duration_model/data/file_id_list_namestory.scp"
with open(scp_file, 'r') as sfid:
	scp_lines = sfid.readlines()

lab_file_pre_list = []
for lab_file in glob.glob(lab_dir+'/*.lab'):
	 lab_name = os.path.basename(lab_file)
	 lab_pre = re.split('\.', lab_name)[0]
	 lab_file_pre_list.append(lab_pre)

for scp_line in scp_lines:
	if scp_line.strip() not in lab_file_pre_list:
		print(scp_line)