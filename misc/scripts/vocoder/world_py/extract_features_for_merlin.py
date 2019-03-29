import os
import sys
import shutil
import glob
import time
import multiprocessing as mp
import numpy as np
import pyworld
import pysptk
import librosa
import pdb
if len(sys.argv)!=5:
    print("Usage: ")
    print("python extract_features_for_merlin.py <path_to_merlin_dir> <path_to_wav_dir> <path_to_feat_dir> <sampling rate>")
    sys.exit(1)

# top merlin directory
merlin_dir = sys.argv[1]

# input audio directory
wav_dir = sys.argv[2]

# Output features directory
out_dir = sys.argv[3]

# initializations
fs = int(sys.argv[4])

# tools directory
world  = os.path.join(merlin_dir, "tools/bin/WORLD")
sptk   = os.path.join(merlin_dir, "tools/bin/SPTK-3.9")
reaper = os.path.join(merlin_dir, "tools/bin/REAPER")

sp_dir  = os.path.join(out_dir, 'sp' )
mgc_dir = os.path.join(out_dir, 'mgc')
ap_dir  = os.path.join(out_dir, 'ap' )
bap_dir = os.path.join(out_dir, 'bap')
f0_dir  = os.path.join(out_dir, 'f0' )
lf0_dir = os.path.join(out_dir, 'lf0')

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(sp_dir):
    os.mkdir(sp_dir)

if not os.path.exists(mgc_dir):
    os.mkdir(mgc_dir)

if not os.path.exists(bap_dir):
    os.mkdir(bap_dir)

if not os.path.exists(f0_dir):
    os.mkdir(f0_dir)

if not os.path.exists(lf0_dir):
    os.mkdir(lf0_dir)

if fs == 16000:
    nFFTHalf = 1024
    alpha = 0.41

elif fs == 22050:
    nFFTHalf = 1024
    alpha = 0.455

elif fs == 44100:
    nFFTHalf = 2048
    alpha = 0.76

elif fs == 48000:
    nFFTHalf = 2048
    alpha = 0.77

else:
    print("As of now, we don't support %d Hz sampling rate." %(fs))
    print("Please consider either downsampling to 16000 Hz or upsampling to 48000 Hz")
    sys.exit(1)

#bap order depends on sampling rate.
mcsize=59
b_use_reaper=False # If True: Reaper is used for f0 extraction. If False: The vocoder is used for f0 extraction.

def get_wav_filelist(wav_dir):
    wav_files = []
    for file in os.listdir(wav_dir):
        whole_filepath = os.path.join(wav_dir,file)
        if os.path.isfile(whole_filepath) and str(whole_filepath).endswith(".wav"):
            wav_files.append(whole_filepath)
        elif os.path.isdir(whole_filepath):
            wav_files += get_wav_filelist(whole_filepath)

    wav_files.sort()

    return wav_files


def read_binfile(filename, dim=60, dtype=np.float64):
    '''
    Reads binary file into numpy array.
    '''
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return  m_data

def write_binfile(m_data, filename, dtype=np.float64):
    '''
    Writes numpy array into binary file.
    '''
    m_data = np.array(m_data, dtype)
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return

def process(filename):
    '''
    The function decomposes a wav file into F0, mel-cepstral coefficients, and aperiodicity
    :param filename: path to wav file
    :return: .lf0, .mgc and .bap files
    '''

    file_id = os.path.basename(filename).split(".")[0]
    print('\n' + file_id)

    ### WORLD ANALYSIS -- extract vocoder parameters ###
    x, fs = librosa.core.load(filename, sr=16000)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.harvest(x, fs, frame_period=5, f0_floor=71.0, f0_ceil=700)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    f0 = f0[:, None]
    lf0 = f0.copy()
    lf0 = lf0.astype(np.float32)
    nonzero_indices = np.where(f0 != 0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    zero_indices = np.where(f0 == 0)
    lf0[zero_indices] = -1.0E+10
    write_binfile(lf0, os.path.join(lf0_dir, file_id + '.lf0'),dtype=np.float32)
    mc = pysptk.sp2mc(spectrogram, mcsize, alpha=alpha)
    mc = mc.astype(np.float32)
    write_binfile(mc, os.path.join(mgc_dir, file_id + '.mgc'),dtype=np.float32)
    bap = pyworld.code_aperiodicity(aperiodicity, fs)
    bap = bap.astype(np.float32)
    write_binfile(bap, os.path.join(bap_dir, file_id + '.bap'),dtype=np.float32)
    ### convert bapd to bap ###

print("--- Feature extraction started ---")
start_time = time.time()

# get wav files list
wav_files = get_wav_filelist(wav_dir)

# do multi-processing
pool = mp.Pool(10)
pool.map(process, wav_files)

# DEBUG:
# for nxf in range(len(wav_files)):
#    process(wav_files[nxf])

# clean temporal files
# keep temporal files for test
# shutil.rmtree(sp_dir, ignore_errors=True)
# shutil.rmtree(f0_dir, ignore_errors=True)


for zippath in glob.iglob(os.path.join(bap_dir, '*.bapd')):
    os.remove(zippath)

print("You should have your features ready in: "+out_dir)    

(m, s) = divmod(int(time.time() - start_time), 60)
print(("--- Feature extraction completion time: %d min. %d sec ---" % (m, s)))

