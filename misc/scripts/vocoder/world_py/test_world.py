# test difference between python and bash generated features
import os
import sys
import shutil
import glob
import time
import multiprocessing as mp
import numpy as np
import pyworld
import pysptk
from scipy.io import wavfile
import librosa
import scipy
import pdb
# initializations
fs = 16000
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

def load_binary_file_frame(file_name, dimension):
    fid_lab = open(file_name, 'rb')
    features = np.fromfile(fid_lab, dtype=np.float32)
    fid_lab.close()
    assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    frame_number = features.size // dimension
    features = features[:(dimension * frame_number)]
    features = features.reshape((-1, dimension))
    return  features, frame_number

def process(filename):
    '''
    The function decomposes a wav file into F0, mel-cepstral coefficients, and aperiodicity
    :param filename: path to wav file
    :return: .lf0, .mgc and .bap files
    '''

    # pdb.set_trace()
    file_id = os.path.basename(filename).split(".")[0]
    print('\n' + file_id)

    ### WORLD ANALYSIS -- extract vocoder parameters ###
    # x, fs = librosa.core.load(filename, sr=16000)
    fs, x = wavfile.read(filename)
    alpha = pysptk.util.mcepalpha(fs)
    print(alpha)
    hopesize=int(0.005*fs)
    # pdb.set_trace()
    f0 = pysptk.rapt(x.astype(np.float32), fs=fs, hopsize=hopesize, min=60, max=600, voice_bias=0.0, otype=1)
    f0=f0.astype(np.float64)
    x = x.astype(np.float64)/(2**15)
    _, timeaxis = pyworld.harvest(x, fs, frame_period=5, f0_floor=60.0, f0_ceil=600)
    pdb.set_trace()
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    # pdb.set_trace()
    f0 = f0[:, None]
    lf0 = f0.copy()
    lf0 = lf0.astype(np.float32)
    nonzero_indices = np.where(f0 != 0)
    lf0[nonzero_indices] = np.log(f0[nonzero_indices])
    zero_indices = np.where(f0 == 0)
    lf0[zero_indices] = -1.0E+10
    write_binfile(lf0, os.path.join('./', file_id + '.lf0'),dtype=np.float32)
    mc = pysptk.sp2mc(spectrogram, mcsize, alpha=alpha)
    mc = mc.astype(np.float32)
    write_binfile(mc, os.path.join('./', file_id + '.mgc'),dtype=np.float32)
    bap = pyworld.code_aperiodicity(aperiodicity, fs)
    bap = bap.astype(np.float32)
    write_binfile(bap, os.path.join('./', file_id + '.bap'),dtype=np.float32)
    ### convert bapd to bap ###

def synthesis():
    # pdb.set_trace()
    lf0_file = "p225_001.lf0"
    bap_file_name="p225_001.bap"
    mgc_file_name="p225_001.mgc"
    fl=4096
    sr=48000
    # pdb.set_trace()
    lf0 = read_binfile(lf0_file, dim=1, dtype=np.float32)
    zeros_index = np.where(lf0 == -1E+10)
    nonzeros_index = np.where(lf0 != -1E+10)
    f0 = lf0.copy()
    f0[zeros_index] = 0
    f0[nonzeros_index] = np.exp(lf0[nonzeros_index])
    f0 = f0.astype(np.float64)
    bap_dim = 5
    bap = read_binfile(bap_file_name, dim=bap_dim, dtype=np.float32)
    ap = pyworld.decode_aperiodicity(bap.astype(np.float64).reshape(-1, bap_dim), sr, fl)
    mc = read_binfile(mgc_file_name, dim=60, dtype=np.float32)
    alpha = pysptk.util.mcepalpha(sr)
    sp = pysptk.mc2sp(mc.astype(np.float64), fftlen=fl, alpha=alpha)
    wav = pyworld.synthesize(f0, sp, ap, sr, 5)
    x2 = wav * 32768
    x2 = x2.astype(np.int16)
    scipy.io.wavfile.write("resynthesis.wav", sr, x2)
    # os.chdir(cur_dir)

process("p225_001.wav")
synthesis()
# read_binfile("lc.f0")