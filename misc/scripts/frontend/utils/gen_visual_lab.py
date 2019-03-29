import re
from praatio import tgio
import pdb
import os
def lab2praat(file_name, praat_align_file,state_number=5):
    """
    convert state alignment lab file to praat alignment file
    file_name should be state alignment file end with *.lab
    praat_align_file should be praat Textgrid file for visualization
    and state number are state number of one senone
    """
    fid = open(file_name)
    utt_labels = fid.readlines()
    fid.close()
    current_index = 0
    label_number = len(utt_labels)
    duration_phone_list=[]
    for line in utt_labels:
        line = line.strip()
        if len(line) < 1:
            continue
        temp_list = re.split('\s+', line)
        start_time = int(temp_list[0])
        end_time = int(temp_list[1])
        frame_number = int((end_time - start_time) / 50000)  # all frame number of this phone
        full_label = temp_list[2]
        full_label_length = len(full_label) - 3  # remove state information [k]
        state_index = full_label[full_label_length + 1]
        state_index = int(state_index) - 1
        full_label = full_label[0:full_label_length]
        match = re.match(r"^.*?\-(.*?)\+.*?$",full_label,re.M|re.I)
        phone_identity = match.group(1)
        if state_index == 1:
            phone_duration = frame_number
            for i in range(state_number - 1):
                line = utt_labels[current_index + i + 1].strip()
                temp_list = re.split('\s+', line)
                phone_duration += (int(temp_list[1]) - int(temp_list[0])) / 50000
            start_time = start_time/10000000.0
            end_time = start_time+phone_duration*0.005
            duration_phone_list.append((str(start_time),str(end_time),phone_identity))
        current_index+=1
    setTG = tgio.Textgrid()
    # pdb.set_trace()
    phoneTier = tgio.IntervalTier('phone', duration_phone_list)
    setTG.addTier(phoneTier)
    setTG.save(praat_align_file)

def mlf2praat(mlf, praat_align_file):
    "transform cuprosody mlf file to Textgrid file"
    """
        mlf format:
        0 4400000 sil -2160.365723 sil
        4400000 5200000 I_g -504.555634 gaa
        5200000 6600000 F_aa -960.479187
        6600000 7300000 I_j -543.072876 jau
        7300000 8500000 F_au -856.253418
        8500000 8900000 I_d -320.236786 daai
        8900000 10400000 F_aai -789.435547
        10400000 11300000 I_s -523.623901 si
    """
    fid = open(mlf, 'r')
    lines = fid.readlines()
    duration_phone_list = []
    for line in lines:
        tmp_split = re.split('\s+', line.strip())
        if len(tmp_split) == 5 and tmp_split[2] == tmp_split[4]:
            start_time = int(tmp_split[0])
            end_time = int(tmp_split[1])
            phone_identity = tmp_split[4]
            start_time = start_time / 10000000.0
            end_time = end_time / 10000000.0
            duration_phone_list.append((str(start_time), str(end_time), phone_identity))
        elif len(tmp_split) == 4:
            end_time = int(tmp_split[1])
            start_time = start_time / 10000000.0
            end_time = end_time / 10000000.0
            duration_phone_list.append((str(start_time), str(end_time), phone_identity))
        else:
            start_time = int(tmp_split[0])
            phone_identity = tmp_split[4]
    setTG = tgio.Textgrid()
    pdb.set_trace()
    phoneTier = tgio.IntervalTier('syllable', duration_phone_list)
    setTG.addTier(phoneTier)
    setTG.save(praat_align_file)


if __name__ == '__main__':
    file_name="/home/patrick/projects/merlin/egs/build_your_own_voice/s1/database/labels/label_state_align/arctic_a0003.lab"
    praat_align_file="1.TextGrid"
    mlf2praat("./1.mlf", praat_align_file)









